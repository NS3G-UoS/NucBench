"""
nucbench/cli.py — NucBench Command-Line Interface
--------------------------------------------------
Run benchmarks without the Streamlit UI.

Usage examples
~~~~~~~~~~~~~~
# Undergraduate exam, 20 unique questions, 3 runs each, save to my_results.json
python -m nucbench.cli undergrad \\
    --model openai/gpt-4o \\
    --api-key sk-... \\
    --questions 20 \\
    --runs 3 \\
    --output my_results.json

# Operator exam (BWR + PWR), deterministic, 50 questions
python -m nucbench.cli operator \\
    --model openai/gpt-4o \\
    --api-key sk-... \\
    --questions 50 \\
    --temperature 0.0

# Two-phase flow classification, 10 images
python -m nucbench.cli flow \\
    --model openai/gpt-4o \\
    --api-key sk-... \\
    --questions 10

# Read the API key from an env-var instead of --api-key
NUCBENCH_API_KEY=sk-... python -m nucbench.cli undergrad --model openai/gpt-4o

Available tasks: undergrad, operator, flow
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Progress callback — prints to stdout without requiring Streamlit
# ---------------------------------------------------------------------------

def _make_progress_cb(label: str):
    """Return a callback that prints a simple ASCII progress bar."""
    last_pct = [-1]

    def _cb(frac: float) -> None:
        pct = int(frac * 100)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        filled = pct // 5
        bar = "█" * filled + "░" * (20 - filled)
        print(f"\r{label} [{bar}] {pct:3d}%", end="", flush=True)
        if pct >= 100:
            print()  # newline on completion

    return _cb


# ---------------------------------------------------------------------------
# Post-run summary
# ---------------------------------------------------------------------------

def _print_summary(payload: Dict[str, Any], out_path: str) -> None:
    scores: List[float] = payload["scores"]
    n = len(scores)
    mean_acc = statistics.mean(scores) if scores else 0.0
    std_dev = statistics.stdev(scores) if n > 1 else 0.0

    n_q = payload.get("n_questions", n)
    n_r = payload.get("n_runs", 1)

    print()
    print("=" * 60)
    print(f"  Task  : {payload['task_name']}")
    print(f"  Model : {payload['model']}")
    print(f"  Unique questions : {n_q}")
    print(f"  Runs per question: {n_r}")
    print(f"  Total requests   : {n}")
    print(f"  Mean accuracy    : {mean_acc:.2%}")
    print(f"  Std deviation    : {std_dev:.4f}")
    print(f"  Results saved to : {out_path}")

    # Warn about any open-ended responses that couldn't be auto-graded
    ungraded = [
        d for d in payload["details"]
        if d.get("format") == "Open-Ended" and d.get("human_grade") is None
    ]
    if ungraded:
        print()
        print(f"  ⚠  {len(ungraded)} open-ended response(s) have no human_grade.")
        print("     Re-open the Streamlit UI to assign manual grades, or use")
        print("     --grade-open-ended to be prompted interactively.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Interactive grading for open-ended questions (optional, --grade-open-ended)
# ---------------------------------------------------------------------------

def _interactive_grade(payload: Dict[str, Any], grade_all: bool = False) -> None:
    """Walk through responses and ask the user for a grade.

    Args:
        grade_all: When ``True``, prompt for every response regardless of
                   format.  When ``False`` (default), only open-ended
                   responses without an existing human_grade are shown.
    """
    details = payload["details"]
    scores = payload["scores"]

    for i, detail in enumerate(details):
        is_open_ended_ungraded = (
            detail.get("format") == "Open-Ended"
            and detail.get("human_grade") is None
        )
        if not grade_all and not is_open_ended_ungraded:
            continue

        print()
        print("-" * 60)
        q_id = detail.get("question_id") or f"#{i+1}"
        topic = detail.get("topic", "—")
        run = detail.get("run", 1)
        print(f"  Response #{i+1} | ID: {q_id} | Topic: {topic} | Run: {run}")
        print()
        # For --grade-all, also show the auto-score for context
        if grade_all:
            fmt = detail.get("format", "Auto")
            cur = detail.get("human_grade")
            if cur is None:
                cur_pct = float(detail.get("score", 0)) * 100.0
            else:
                cur_pct = float(cur)
            print(f"  Format: {fmt} | Current score: {cur_pct:.0f}%")
            print()
        print("  KEY ANSWER:")
        print(f"  {detail.get('key_answer', detail.get('true_label', '—'))}")
        print()
        conf = detail.get("confidence_score")
        if conf is not None:
            print(f"  Model confidence: {conf}%")
        print()
        print("  LLM RESPONSE:")
        for line in detail.get("response", "").splitlines():
            print(f"  {line}")
        print()

        while True:
            raw = input("  Manual Grade (0-100%): ").strip()
            try:
                grade = float(raw)
                if 0.0 <= grade <= 100.0:
                    break
                print("  Please enter a value between 0 and 100.")
            except ValueError:
                print("  Please enter a numeric value.")

        details[i]["human_grade"] = grade
        scores[i] = grade / 100.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nucbench",
        description="NucBench — Vision-LLM benchmarking for nuclear engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="task", metavar="TASK")
    sub.required = True

    # Shared arguments added to all sub-commands
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--model", "-m",
        required=True,
        help="LiteLLM model identifier, e.g. openai/gpt-4o",
    )
    shared.add_argument(
        "--api-key", "-k",
        default=None,
        help="Provider API key. Falls back to env var NUCBENCH_API_KEY.",
    )
    shared.add_argument(
        "--questions", "-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of unique questions/images to sample. (default: 10)",
    )
    shared.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        metavar="R",
        help="Number of times to repeat each question. (default: 1)",
    )
    shared.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="LLM sampling temperature. (default: 0.0)",
    )
    shared.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Delay between API calls in seconds. (default: 1.0)",
    )
    shared.add_argument(
        "--output", "-o",
        default="results.json",
        help="Path to the results JSON file. (default: results.json)",
    )
    shared.add_argument(
        "--grade-open-ended",
        action="store_true",
        help="Interactively prompt for a manual grade on open-ended responses.",
    )
    shared.add_argument(
        "--grade-all",
        action="store_true",
        help="Interactively prompt for a manual grade on every response (overrides --grade-open-ended).",
    )

    # Sub-commands
    sub.add_parser(
        "undergrad",
        parents=[shared],
        help="Undergraduate Nuclear Engineering Exam",
        description="Run the undergraduate nuclear engineering exam benchmark.",
    )
    sub.add_parser(
        "operator",
        parents=[shared],
        help="GFE Reactor Operator Exam (BWR + PWR)",
        description="Run the NRC GFE reactor operator exam benchmark.",
    )
    sub.add_parser(
        "flow",
        parents=[shared],
        help="Two-Phase Flow Regime Classification",
        description="Run the two-phase flow regime image classification benchmark.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve API key
    api_key = args.api_key or os.environ.get("NUCBENCH_API_KEY", "")
    if not api_key:
        print(
            "Error: no API key provided. Use --api-key or set NUCBENCH_API_KEY.",
            file=sys.stderr,
        )
        return 1

    # Select and run task
    task_map = {
        "undergrad": ("nucbench.tasks", "run_undergraduate_exam", "Undergraduate Exam"),
        "operator":  ("nucbench.tasks", "run_operator_exam",      "Operator Exam"),
        "flow":      ("nucbench.tasks", "run_flow_regime_task",    "Flow Classification"),
    }
    module_name, fn_name, label = task_map[args.task]

    import importlib
    task_fn = getattr(importlib.import_module(module_name), fn_name)

    total = args.questions * args.runs
    print(f"NucBench · {label}")
    print(f"  Model     : {args.model}")
    print(f"  Questions : {args.questions}  Runs/question: {args.runs}  "
          f"Total requests: {total}")
    print(f"  Temp      : {args.temperature}  Delay: {args.delay}s")
    print()

    progress_cb = _make_progress_cb(label)

    try:
        payload = task_fn(
            model=args.model,
            api_key=api_key,
            temperature=args.temperature,
            n_samples=args.questions,
            n_runs=args.runs,
            delay_s=args.delay,
            progress_cb=progress_cb,
        )
    except Exception as exc:
        print(f"\nRun failed: {exc}", file=sys.stderr)
        return 1

    # Optional interactive grading
    if args.grade_all:
        _interactive_grade(payload, grade_all=True)
    elif args.grade_open_ended:
        _interactive_grade(payload, grade_all=False)

    # Save results
    from nucbench.scoring import save_results
    out_path = save_results(
        task_name=payload["task_name"],
        model=payload["model"],
        scores=payload["scores"],
        details=payload["details"],
        output_path=args.output,
    )

    _print_summary(payload, str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
