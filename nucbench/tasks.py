"""
nucbench/tasks.py
-----------------
Task-runner functions for the three NucBench benchmarks.

Each runner:
  1. Loads its dataset (JSON for exams, image files for flow regime).
  2. Randomly samples *n_samples* items (optionally re-sampling each run).
  3. Repeats the sample *n_runs* times, calling the LiteLLM API once per
     item per run, with *delay_s* seconds between requests.
  4. Scores each response and returns a results payload ready for
     ``scoring.save_results()``.

Run modes
~~~~~~~~~
* ``unique_per_run=False`` (default): the same *n_samples* questions are
  used for every run — useful for measuring response variance on a fixed set.
* ``unique_per_run=True``: a fresh random sample is drawn before each run —
  useful for broader dataset coverage across repeated runs.

Progress is reported through an optional ``progress_cb`` callable that
accepts a float in [0.0, 1.0] so the UI can drive a progress bar.

All file I/O is intentionally straightforward — the JSON files and image
directories are expected to exist at the documented paths.
"""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from nucbench.prompts import build_exam_message, build_flow_regime_message
from nucbench.scoring import score_exam_response, score_flow_regime_response

# ---------------------------------------------------------------------------
# Repo-relative data paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent

UNDERGRADUATE_JSON = _REPO_ROOT / "exams" / "undergraduate" / "exam_questions.json"

OPERATOR_JSON_BWR = _REPO_ROOT / "exams" / "operator" / "BWR" / "bwr-bank.json"
OPERATOR_JSON_PWR = _REPO_ROOT / "exams" / "operator" / "PWR" / "pwr-bank.json"

# Actual directory names on disk (note the spaces and underscores)
IMAGE_DIRS: Dict[str, List[Path]] = {
    "bubbly":  [
        _REPO_ROOT / "images" / "Fluid 1_Air" / "bubbly",
        _REPO_ROOT / "images" / "Fluid 2_CO2" / "bubbly",
    ],
    "slug":    [
        _REPO_ROOT / "images" / "Fluid 1_Air" / "slug",
        _REPO_ROOT / "images" / "Fluid 2_CO2" / "slug",
    ],
    "churn":   [
        _REPO_ROOT / "images" / "Fluid 1_Air" / "churn",
        _REPO_ROOT / "images" / "Fluid 2_CO2" / "churn",
    ],
    "taylor":  [
        _REPO_ROOT / "images" / "Fluid 1_Air" / "taylor",
        _REPO_ROOT / "images" / "Fluid 2_CO2" / "taylor",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    """Load and return parsed JSON from *path*."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _encode_image(image_path: Path) -> Tuple[str, str]:
    """Return ``(base64_string, mime_type)`` for an image file."""
    suffix = image_path.suffix.lower()
    mime_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif":  "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")
    with image_path.open("rb") as fh:
        b64 = base64.standard_b64encode(fh.read()).decode("ascii")
    return b64, mime


def _call_llm(
    model: str,
    api_key: str,
    temperature: float,
    messages: List[Dict[str, Any]],
    api_base: Optional[str] = None,
) -> str:
    """Make a single LiteLLM completion call and return the response text.

    When *api_base* is set (local mode) the model name and base URL are
    normalised through ``resolve_local_model`` so that Ollama and any other
    OpenAI-compatible local server are routed correctly regardless of the
    provider prefix the user typed.

    Raises the original LiteLLM exception on failure so callers can decide
    how to handle rate limits, auth errors, etc.
    """
    import litellm
    from nucbench.models import resolve_local_model

    resolved_model = model
    resolved_base = api_base

    if api_base:
        resolved_model, resolved_base = resolve_local_model(model, api_base)

    kwargs: Dict[str, Any] = dict(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=256,
    )
    if api_key:
        kwargs["api_key"] = api_key
    elif api_base:
        # LiteLLM's openai provider requires a non-empty key; local servers ignore it.
        kwargs["api_key"] = "local"
    if resolved_base:
        kwargs["api_base"] = resolved_base

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""


def _collect_all_images() -> List[Tuple[Path, str]]:
    """Return a flat list of ``(image_path, true_label)`` for all regime dirs."""
    items: List[Tuple[Path, str]] = []
    for label, dirs in IMAGE_DIRS.items():
        for d in dirs:
            for img in sorted(d.glob("*.png")):
                items.append((img, label))
            for img in sorted(d.glob("*.jpg")):
                items.append((img, label))
            for img in sorted(d.glob("*.jpeg")):
                items.append((img, label))
    return items


# ---------------------------------------------------------------------------
# Task 1 — Two-phase flow regime image classification
# ---------------------------------------------------------------------------

def run_flow_regime_task(
    model: str,
    api_key: str,
    temperature: float,
    n_samples: int,
    delay_s: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
    api_base: Optional[str] = None,
    n_runs: int = 1,
    unique_per_run: bool = False,
) -> Dict[str, Any]:
    """Benchmark the model on two-phase flow regime image classification.

    Args:
        model:          LiteLLM model identifier.
        api_key:        Provider API key (empty string for local models).
        temperature:    Sampling temperature.
        n_samples:      Number of images to sample per run.
        delay_s:        Seconds to wait between consecutive API calls.
        progress_cb:    Optional callback receiving progress in [0.0, 1.0].
        api_base:       Optional base URL for local inference servers.
        n_runs:         How many times to repeat the sample (default 1).
        unique_per_run: When True, draw a fresh random sample for each run.
                        When False (default), every run uses the same images.

    Returns:
        A dict with keys ``scores``, ``details``, ``task_name``, ``model``,
        ``n_questions``, ``n_runs``.
    """
    all_images = _collect_all_images()
    if n_samples > len(all_images):
        n_samples = len(all_images)

    # Pre-sample once when every run should see the same questions.
    fixed_sample = None if unique_per_run else random.sample(all_images, n_samples)

    total_requests = n_samples * n_runs
    scores: List[int] = []
    details: List[Dict[str, Any]] = []
    request_idx = 0

    for run_idx in range(n_runs):
        sample = random.sample(all_images, n_samples) if unique_per_run else fixed_sample

        for img_path, true_label in sample:
            b64, mime = _encode_image(img_path)
            messages = build_flow_regime_message(b64, mime)

            try:
                response_text = _call_llm(model, api_key, temperature, messages, api_base)
                score = score_flow_regime_response(response_text, true_label)
            except Exception as exc:  # noqa: BLE001
                response_text = f"[ERROR] {exc}"
                score = 0

            scores.append(score)
            details.append(
                {
                    "run": run_idx + 1,
                    "image": img_path.name,
                    "fluid": img_path.parts[-3],
                    "true_label": true_label,
                    "response": response_text,
                    "score": score,
                }
            )

            request_idx += 1
            if progress_cb:
                progress_cb(request_idx / total_requests)
            if request_idx < total_requests:
                time.sleep(delay_s)

    return {
        "task_name": "Two-Phase Flow Regime Classification",
        "model": model,
        "n_questions": n_samples,
        "n_runs": n_runs,
        "scores": scores,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Task 2 — Undergraduate Nuclear Engineering Exam
# ---------------------------------------------------------------------------

def run_undergraduate_exam(
    model: str,
    api_key: str,
    temperature: float,
    n_samples: int,
    delay_s: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
    api_base: Optional[str] = None,
    n_runs: int = 1,
    unique_per_run: bool = False,
) -> Dict[str, Any]:
    """Benchmark the model on the undergraduate nuclear engineering exam.

    Args:
        model:          LiteLLM model identifier.
        api_key:        Provider API key (empty string for local models).
        temperature:    Sampling temperature.
        n_samples:      Number of questions to sample per run.
        delay_s:        Seconds to wait between consecutive API calls.
        progress_cb:    Optional callback receiving progress in [0.0, 1.0].
        api_base:       Optional base URL for local inference servers.
        n_runs:         How many times to repeat the sample (default 1).
        unique_per_run: When True, draw a fresh random sample for each run.
                        When False (default), every run uses the same questions.

    Returns:
        A dict with keys ``scores``, ``details``, ``task_name``, ``model``,
        ``n_questions``, ``n_runs``.
    """
    questions = _load_json(UNDERGRADUATE_JSON)

    if n_samples > len(questions):
        n_samples = len(questions)

    fixed_sample = None if unique_per_run else random.sample(questions, n_samples)

    total_requests = n_samples * n_runs
    scores: List[int] = []
    details: List[Dict[str, Any]] = []
    request_idx = 0

    for run_idx in range(n_runs):
        sample = random.sample(questions, n_samples) if unique_per_run else fixed_sample

        for question in sample:
            props = question.get("Properties", {})
            q_text = question.get("Question_Prompt", "")
            q_type = props.get("Question_Type", "Qualitative")
            key_answer = str(props.get("Key_Answer", ""))

            inline_images: List[Tuple[str, str]] = []
            for img_entry in question.get("images", []):
                inline_images.append((img_entry["data"], img_entry.get("mime", "image/png")))

            messages = build_exam_message(q_text, q_type, inline_images)

            try:
                response_text = _call_llm(model, api_key, temperature, messages, api_base)
                score = score_exam_response(response_text, key_answer)
            except Exception as exc:  # noqa: BLE001
                response_text = f"[ERROR] {exc}"
                score = 0

            scores.append(score)
            details.append(
                {
                    "run": run_idx + 1,
                    "question_id": props.get("Question_ID", ""),
                    "topic": props.get("Question_Topic", ""),
                    "type": q_type,
                    "key_answer": key_answer,
                    "response": response_text,
                    "score": score,
                }
            )

            request_idx += 1
            if progress_cb:
                progress_cb(request_idx / total_requests)
            if request_idx < total_requests:
                time.sleep(delay_s)

    return {
        "task_name": "Undergraduate Nuclear Engineering Exam",
        "model": model,
        "n_questions": n_samples,
        "n_runs": n_runs,
        "scores": scores,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Task 3 — GFE Reactor Operator Exam (BWR + PWR combined)
# ---------------------------------------------------------------------------

def run_operator_exam(
    model: str,
    api_key: str,
    temperature: float,
    n_samples: int,
    delay_s: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
    api_base: Optional[str] = None,
    n_runs: int = 1,
    unique_per_run: bool = False,
) -> Dict[str, Any]:
    """Benchmark the model on the GFE Reactor Operator exam question bank.

    Both BWR and PWR JSON files are merged before sampling so the draw
    represents the full 4,292-question bank proportionally.

    Args:
        model:          LiteLLM model identifier.
        api_key:        Provider API key (empty string for local models).
        temperature:    Sampling temperature.
        n_samples:      Number of questions to sample per run.
        delay_s:        Seconds to wait between consecutive API calls.
        progress_cb:    Optional callback receiving progress in [0.0, 1.0].
        api_base:       Optional base URL for local inference servers.
        n_runs:         How many times to repeat the sample (default 1).
        unique_per_run: When True, draw a fresh random sample for each run.
                        When False (default), every run uses the same questions.

    Returns:
        A dict with keys ``scores``, ``details``, ``task_name``, ``model``,
        ``n_questions``, ``n_runs``.
    """
    bwr_questions = _load_json(OPERATOR_JSON_BWR)
    pwr_questions = _load_json(OPERATOR_JSON_PWR)
    all_questions = bwr_questions + pwr_questions

    if n_samples > len(all_questions):
        n_samples = len(all_questions)

    fixed_sample = None if unique_per_run else random.sample(all_questions, n_samples)

    total_requests = n_samples * n_runs
    scores: List[int] = []
    details: List[Dict[str, Any]] = []
    request_idx = 0

    for run_idx in range(n_runs):
        sample = random.sample(all_questions, n_samples) if unique_per_run else fixed_sample

        for question in sample:
            props = question.get("Properties", {})
            q_text = question.get("Question_Prompt", "")
            q_type = props.get("Question_Type", "Qualitative")
            key_answer = str(props.get("Key_Answer", ""))

            inline_images: List[Tuple[str, str]] = []
            for img_entry in question.get("images", []):
                inline_images.append((img_entry["data"], img_entry.get("mime", "image/png")))

            messages = build_exam_message(q_text, q_type, inline_images)

            try:
                response_text = _call_llm(model, api_key, temperature, messages, api_base)
                score = score_exam_response(response_text, key_answer)
            except Exception as exc:  # noqa: BLE001
                response_text = f"[ERROR] {exc}"
                score = 0

            scores.append(score)
            details.append(
                {
                    "run": run_idx + 1,
                    "question_id": props.get("Question_ID", ""),
                    "reactor_type": props.get("Exam_Type", ""),
                    "topic": props.get("Question_Topic", ""),
                    "type": q_type,
                    "key_answer": key_answer,
                    "response": response_text,
                    "score": score,
                }
            )

            request_idx += 1
            if progress_cb:
                progress_cb(request_idx / total_requests)
            if request_idx < total_requests:
                time.sleep(delay_s)

    return {
        "task_name": "GFE Reactor Operator Exam",
        "model": model,
        "n_questions": n_samples,
        "n_runs": n_runs,
        "scores": scores,
        "details": details,
    }
