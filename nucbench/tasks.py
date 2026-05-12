"""
nucbench/tasks.py
-----------------
Task-runner functions for the three NucBench benchmarks.

Each runner:
  1. Loads its dataset (JSON for exams, image files for flow regime).
  2. Randomly samples *n_samples* items.
  3. Calls the LiteLLM API once per item, with *delay_s* seconds between
     requests to respect provider rate limits.
  4. Scores each response and returns a results payload ready for
     ``scoring.save_results()``.

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
from nucbench.scoring import extract_confidence_score, score_exam_response, score_flow_regime_response

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
) -> str:
    """Make a single LiteLLM completion call and return the response text.

    Raises the original LiteLLM exception on failure so callers can decide
    how to handle rate limits, auth errors, etc.
    """
    import litellm

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=256,
    )
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
    n_runs: int = 1,
    delay_s: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Benchmark the model on two-phase flow regime image classification.

    Args:
        model:       LiteLLM model identifier.
        api_key:     Provider API key.
        temperature: Sampling temperature.
        n_samples:   Number of unique images to sample from the dataset.
        n_runs:      Number of times to repeat the prompt for each image.
        delay_s:     Seconds to wait between consecutive API calls.
        progress_cb: Optional callback receiving progress in [0.0, 1.0].

    Returns:
        A dict with keys ``scores``, ``details``, ``task_name``, ``model``,
        ``n_questions``, ``n_runs``.
    """
    all_images = _collect_all_images()
    if n_samples > len(all_images):
        n_samples = len(all_images)

    sampled = random.sample(all_images, n_samples)
    total_requests = n_samples * n_runs

    scores: List[int] = []
    details: List[Dict[str, Any]] = []

    for i, (img_path, true_label) in enumerate(sampled):
        b64, mime = _encode_image(img_path)
        messages = build_flow_regime_message(b64, mime)

        for run_idx in range(n_runs):
            try:
                response_text = _call_llm(model, api_key, temperature, messages)
                score = score_flow_regime_response(response_text, true_label)
            except Exception as exc:  # noqa: BLE001
                response_text = f"[ERROR] {exc}"
                score = 0

            scores.append(score)
            details.append(
                {
                    "image": img_path.name,
                    "fluid": img_path.parts[-3],   # e.g. "Fluid 1_Air"
                    "true_label": true_label,
                    "response": response_text,
                    "score": score,
                    "run": run_idx + 1,
                }
            )

            request_num = i * n_runs + run_idx + 1
            if progress_cb:
                progress_cb(request_num / total_requests)

            if request_num < total_requests:
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
    n_runs: int = 1,
    delay_s: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Benchmark the model on the undergraduate nuclear engineering exam.

    Args:
        model:       LiteLLM model identifier.
        api_key:     Provider API key.
        temperature: Sampling temperature.
        n_samples:   Number of unique questions to sample from the dataset.
        n_runs:      Number of times to repeat the prompt for each question.
        delay_s:     Seconds to wait between consecutive API calls.
        progress_cb: Optional callback receiving progress in [0.0, 1.0].

    Returns:
        A dict with keys ``scores``, ``details``, ``task_name``, ``model``,
        ``n_questions``, ``n_runs``.
    """
    questions = _load_json(UNDERGRADUATE_JSON)

    if n_samples > len(questions):
        n_samples = len(questions)

    sampled = random.sample(questions, n_samples)
    total_requests = n_samples * n_runs

    scores: List[int] = []
    details: List[Dict[str, Any]] = []

    for i, question in enumerate(sampled):
        props = question.get("Properties", {})
        q_text = question.get("Question_Prompt", "")
        q_type = props.get("Question_Type", "Qualitative")
        key_answer = str(props.get("Key_Answer", ""))
        is_mcq = bool(props.get("MCQ", True))
        marks = props.get("Marks")

        # Inline images embedded in the question (base64 encoded in the JSON)
        inline_images: List[Tuple[str, str]] = []
        for img_entry in question.get("images", []):
            inline_images.append((img_entry["data"], img_entry.get("mime", "image/png")))

        messages = build_exam_message(q_text, q_type, inline_images, is_mcq=is_mcq)

        for run_idx in range(n_runs):
            try:
                response_text = _call_llm(model, api_key, temperature, messages)
                if is_mcq:
                    score = score_exam_response(response_text, key_answer)
                    confidence_score = None
                    human_grade: Optional[float] = float(score)
                else:
                    score = 0  # placeholder; replaced after human grading
                    confidence_score = extract_confidence_score(response_text)
                    human_grade = None
            except Exception as exc:  # noqa: BLE001
                response_text = f"[ERROR] {exc}"
                score = 0
                confidence_score = None
                human_grade = None if not is_mcq else 0.0

            scores.append(score)
            details.append(
                {
                    "question_id": props.get("Question_ID", ""),
                    "topic": props.get("Question_Topic", ""),
                    "type": q_type,
                    "key_answer": key_answer,
                    "question": q_text,
                    "response": response_text,
                    "score": score,
                    "run": run_idx + 1,
                    "format": "MCQ" if is_mcq else "Open-Ended",
                    "confidence_score": confidence_score,
                    "human_grade": human_grade,
                    "marks": marks,
                }
            )

            request_num = i * n_runs + run_idx + 1
            if progress_cb:
                progress_cb(request_num / total_requests)

            if request_num < total_requests:
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
    n_runs: int = 1,
    delay_s: float = 1.0,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Benchmark the model on the GFE Reactor Operator exam question bank.

    Both BWR and PWR JSON files are merged before sampling so the draw
    represents the full 4,292-question bank proportionally.

    Args:
        model:       LiteLLM model identifier.
        api_key:     Provider API key.
        temperature: Sampling temperature.
        n_samples:   Number of unique questions to sample from the dataset.
        n_runs:      Number of times to repeat the prompt for each question.
        delay_s:     Seconds to wait between consecutive API calls.
        progress_cb: Optional callback receiving progress in [0.0, 1.0].

    Returns:
        A dict with keys ``scores``, ``details``, ``task_name``, ``model``,
        ``n_questions``, ``n_runs``.
    """
    bwr_questions = _load_json(OPERATOR_JSON_BWR)
    pwr_questions = _load_json(OPERATOR_JSON_PWR)
    all_questions = bwr_questions + pwr_questions

    if n_samples > len(all_questions):
        n_samples = len(all_questions)

    sampled = random.sample(all_questions, n_samples)
    total_requests = n_samples * n_runs

    scores: List[int] = []
    details: List[Dict[str, Any]] = []

    for i, question in enumerate(sampled):
        props = question.get("Properties", {})
        q_text = question.get("Question_Prompt", "")
        q_type = props.get("Question_Type", "Qualitative")
        key_answer = str(props.get("Key_Answer", ""))
        is_mcq = bool(props.get("MCQ", True))
        marks = props.get("Marks")

        inline_images: List[Tuple[str, str]] = []
        for img_entry in question.get("images", []):
            inline_images.append((img_entry["data"], img_entry.get("mime", "image/png")))

        messages = build_exam_message(q_text, q_type, inline_images, is_mcq=is_mcq)

        for run_idx in range(n_runs):
            try:
                response_text = _call_llm(model, api_key, temperature, messages)
                if is_mcq:
                    score = score_exam_response(response_text, key_answer)
                    confidence_score = None
                    human_grade: Optional[float] = float(score)
                else:
                    score = 0  # placeholder; replaced after human grading
                    confidence_score = extract_confidence_score(response_text)
                    human_grade = None
            except Exception as exc:  # noqa: BLE001
                response_text = f"[ERROR] {exc}"
                score = 0
                confidence_score = None
                human_grade = None if not is_mcq else 0.0

            scores.append(score)
            details.append(
                {
                    "question_id": props.get("Question_ID", ""),
                    "reactor_type": props.get("Exam_Type", ""),
                    "topic": props.get("Question_Topic", ""),
                    "type": q_type,
                    "key_answer": key_answer,
                    "question": q_text,
                    "response": response_text,
                    "score": score,
                    "run": run_idx + 1,
                    "format": "MCQ" if is_mcq else "Open-Ended",
                    "confidence_score": confidence_score,
                    "human_grade": human_grade,
                    "marks": marks,
                }
            )

            request_num = i * n_runs + run_idx + 1
            if progress_cb:
                progress_cb(request_num / total_requests)

            if request_num < total_requests:
                time.sleep(delay_s)

    return {
        "task_name": "GFE Reactor Operator Exam",
        "model": model,
        "n_questions": n_samples,
        "n_runs": n_runs,
        "scores": scores,
        "details": details,
    }
