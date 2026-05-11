"""
nucbench/scoring.py
-------------------
Answer extraction, correctness scoring, and results persistence for NucBench.

Design notes
~~~~~~~~~~~~
* Exam answers are scored by extracting the first unambiguous multiple-choice
  letter (A–D) from the model's response and comparing it against the key.
  If the Key_Answer itself is not a single letter (e.g. ``"C. Reprocessing …"``),
  the leading letter is extracted first.
* For quantitative questions whose Key_Answer is not a letter, the function
  falls back to normalised substring matching followed by a numeric comparison
  with 1 % tolerance.
* Flow-regime labels are matched by case-insensitive substring search so that
  responses like ``"This appears to be a Bubbly flow"`` score correctly.
* Results are appended (not overwritten) to ``results.json`` so multiple runs
  accumulate in the same file.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All valid flow-regime label strings (lowercase).
FLOW_REGIME_LABELS: List[str] = ["bubbly", "slug", "churn", "taylor bubble", "taylor"]

#: Regex for a standalone MCQ letter (A–D), with common surrounding punctuation.
_MCQ_PATTERNS = [
    r"(?:answer|choice|option)[:\s]+([A-D])\b",   # "Answer: B"
    r"^\s*([A-D])[.)]\s",                          # "C) …" or "C. …" at line start
    r"\(([A-D])\)",                                # "(C)"
    r"\b([A-D])\b",                                # bare letter surrounded by word breaks
]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def extract_mcq_letter(text: str) -> str:
    """Extract the most likely MCQ answer letter (A–D) from a text string.

    Tries several patterns in decreasing specificity.  Returns the letter in
    uppercase, or an empty string when nothing matches.
    """
    for pattern in _MCQ_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def normalise_key_answer(key_answer: str) -> str:
    """Extract the canonical form of a Key_Answer for comparison.

    * ``"D"``                             → ``"D"``
    * ``"C. Reprocessing spent fuel"``    → ``"C"``
    * ``"42.5 MW"``                       → ``"42.5 MW"`` (returned as-is)
    """
    stripped = key_answer.strip()
    match = re.match(r"^([A-D])[.):\s]", stripped, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Single bare letter
    if re.match(r"^[A-D]$", stripped, re.IGNORECASE):
        return stripped.upper()
    return stripped


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_exam_response(model_response: str, key_answer: str) -> int:
    """Score a single exam response against the ground-truth key.

    For MCQ questions the model's answer letter is extracted and compared.
    For free-form / numerical questions a normalised substring match is tried,
    followed by a numerical comparison (±1 % tolerance).

    Returns:
        ``1`` for correct, ``0`` for incorrect.
    """
    expected = normalise_key_answer(key_answer)

    # --- MCQ path ---
    if re.match(r"^[A-D]$", expected):
        predicted = extract_mcq_letter(model_response)
        return 1 if predicted == expected else 0

    # --- Free-form / numerical path ---
    response_normalised = model_response.lower().strip()
    expected_lower = expected.lower()

    # Substring match (e.g. model echoes the full expected phrase)
    if expected_lower in response_normalised:
        return 1

    # Numeric comparison: extract first number from each string
    pred_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response_normalised)
    exp_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", expected_lower)
    if pred_nums and exp_nums:
        try:
            pred_val = float(pred_nums[0])
            exp_val = float(exp_nums[0])
            if abs(exp_val) < 1e-12:
                # Both must be zero
                return 1 if abs(pred_val) < 1e-12 else 0
            if abs(pred_val - exp_val) / abs(exp_val) < 0.01:
                return 1
        except ValueError:
            pass

    return 0


def score_flow_regime_response(model_response: str, true_label: str) -> int:
    """Score a two-phase flow regime classification response.

    Performs case-insensitive substring matching.  ``"taylor"`` matches both
    ``"Taylor"`` and ``"Taylor Bubble"`` in the model output.

    Returns:
        ``1`` for correct, ``0`` for incorrect.
    """
    response_lower = model_response.lower().strip()
    label_lower = true_label.lower().strip()

    if label_lower == "taylor":
        # Accept "taylor" or "taylor bubble"
        return 1 if "taylor" in response_lower else 0

    return 1 if label_lower in response_lower else 0


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(
    task_name: str,
    model: str,
    scores: List[int],
    details: List[Dict[str, Any]],
    output_path: str = "results.json",
) -> Path:
    """Append a benchmark-run record to ``results.json``.

    Loads any existing file content, appends the new run, and writes back
    atomically so partial failures do not corrupt the file.

    Args:
        task_name:   Human-readable task identifier.
        model:       LiteLLM model string used for the run.
        scores:      Per-question scores (each 0 or 1).
        details:     Per-question detail dictionaries (question, response, etc.).
        output_path: Destination file path (relative or absolute).

    Returns:
        The resolved ``Path`` of the written file.
    """
    mean_acc = statistics.mean(scores) if scores else 0.0
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

    run_record: Dict[str, Any] = {
        "task": task_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(scores),
        "mean_accuracy": round(mean_acc, 6),
        "std_deviation": round(std_dev, 6),
        "scores": scores,
        "details": details,
    }

    out = Path(output_path)

    # Load existing records safely
    existing: List[Dict[str, Any]] = []
    if out.exists():
        try:
            with out.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            existing = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            # Corrupted file — start fresh rather than crash
            existing = []

    existing.append(run_record)

    # Write to a temporary file then rename for atomicity
    tmp_path = out.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, ensure_ascii=False)
    tmp_path.replace(out)

    return out.resolve()
