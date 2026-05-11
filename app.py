"""
app.py — NucBench Streamlit application
----------------------------------------
Entry point for the NucBench benchmarking UI.  Run with:

    streamlit run app.py

The UI is divided into two columns:
  Left  — configuration sidebar (model, key, temperature, N, task buttons)
  Right — live results panel

All long-running benchmark calls are executed inside a ``st.spinner`` so the
UI remains responsive and progress is streamed via ``st.progress``.
"""

from __future__ import annotations

import statistics
import threading
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NucBench",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /* ── Global typography ──────────────────────────────────────── */
        html, body, [class*="css"] { font-family: "Inter", sans-serif; }

        /* ── App header ─────────────────────────────────────────────── */
        .nb-header {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 12px;
            padding: 28px 36px;
            margin-bottom: 24px;
        }
        .nb-header h1 {
            color: #e0f7fa;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin: 0;
        }
        .nb-header p {
            color: #b2dfdb;
            font-size: 1rem;
            margin: 6px 0 0;
        }

        /* ── Task buttons ────────────────────────────────────────────── */
        div[data-testid="stButton"] > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px 0;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        }

        /* ── Results card ────────────────────────────────────────────── */
        .nb-result-card {
            background: #f0f4f8;
            border-left: 5px solid #00897b;
            border-radius: 8px;
            padding: 18px 22px;
            margin-bottom: 16px;
        }
        .nb-result-card h3 { margin: 0 0 10px; color: #004d40; }

        /* ── Status badge ────────────────────────────────────────────── */
        .nb-badge-success {
            background: #e8f5e9; color: #1b5e20;
            border-radius: 20px; padding: 3px 12px;
            font-size: 0.82rem; font-weight: 700;
            display: inline-block;
        }
        .nb-badge-error {
            background: #ffebee; color: #b71c1c;
            border-radius: 20px; padding: 3px 12px;
            font-size: 0.82rem; font-weight: 700;
            display: inline-block;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="nb-header">
        <h1>⚛️ NucBench</h1>
        <p>Vision-LLM benchmarking for nuclear engineering exams
           and two-phase flow regime classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "key_validated": False,
    "validation_message": "",
    "vision_models": [],
    "models_loaded": False,
    "run_results": None,
    "run_error": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Sidebar — load vision models once per session
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # -- Model list ----------------------------------------------------------
    if not st.session_state.models_loaded:
        with st.spinner("Loading vision-capable models…"):
            from nucbench.models import get_vision_models
            st.session_state.vision_models = get_vision_models()
            st.session_state.models_loaded = True

    model_list: List[str] = st.session_state.vision_models

    if not model_list:
        st.warning("No vision models found — check your LiteLLM installation.")
        model_list = ["(none available)"]

    selected_model: str = st.selectbox(
        "Model",
        options=model_list,
        help="Only models with vision (image) support are listed.",
    )

    # -- API key -------------------------------------------------------------
    st.subheader("API Key")
    api_key_input: str = st.text_input(
        "Provider API Key",
        type="password",
        placeholder="sk-…",
        help="Key is sent only to your chosen provider via LiteLLM.",
    )

    validate_btn = st.button("🔑 Validate Key", use_container_width=True)

    if validate_btn:
        if not api_key_input.strip():
            st.error("Please enter an API key before validating.")
        else:
            with st.spinner("Validating…"):
                from nucbench.models import validate_api_key
                ok, msg = validate_api_key(selected_model, api_key_input.strip())
            st.session_state.key_validated = ok
            st.session_state.validation_message = msg

    if st.session_state.validation_message:
        if st.session_state.key_validated:
            st.success(st.session_state.validation_message)
        else:
            st.error(st.session_state.validation_message)

    st.divider()

    # -- Run parameters -------------------------------------------------------
    st.subheader("Run Parameters")

    temperature: float = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Sampling temperature for the LLM. 0 = deterministic.",
    )

    n_iterations: int = st.number_input(
        "N — samples per run",
        min_value=1,
        max_value=5000,
        value=10,
        step=1,
        help="Number of randomly sampled questions / images per benchmark run.",
    )

    delay_seconds: float = st.number_input(
        "Delay between requests (s)",
        min_value=0.0,
        max_value=60.0,
        value=1.0,
        step=0.5,
        help="Wait time inserted between consecutive API calls to avoid rate limits.",
    )

    st.divider()
    st.caption("Results are saved to **results.json** in the project root.")


# ---------------------------------------------------------------------------
# Helpers — run a task in the main thread and stream progress
# ---------------------------------------------------------------------------

def _run_task(task_fn, label: str, **kwargs) -> None:
    """Execute *task_fn* and store results / errors in session state."""
    st.session_state.run_results = None
    st.session_state.run_error = None

    progress_bar = st.progress(0.0, text=f"Running {label}…")

    def _progress(frac: float) -> None:
        progress_bar.progress(min(frac, 1.0), text=f"{label} — {int(frac * 100)}%")

    try:
        payload = task_fn(progress_cb=_progress, **kwargs)
        from nucbench.scoring import save_results
        out_path = save_results(
            task_name=payload["task_name"],
            model=payload["model"],
            scores=payload["scores"],
            details=payload["details"],
        )
        payload["results_path"] = str(out_path)
        st.session_state.run_results = payload
    except Exception as exc:  # noqa: BLE001
        st.session_state.run_error = str(exc)
    finally:
        progress_bar.empty()


# ---------------------------------------------------------------------------
# Main panel — task buttons + results
# ---------------------------------------------------------------------------

col_tasks, col_results = st.columns([1, 2], gap="large")

with col_tasks:
    st.subheader("📋 Benchmark Tasks")

    locked = not st.session_state.key_validated
    lock_msg = "Validate your API key first." if locked else ""

    # Task 1 — Undergrad exam
    if st.button(
        "🎓 Undergraduate NE Exam",
        disabled=locked,
        use_container_width=True,
        help=lock_msg or "108 questions across reactor physics, T-H, fuel cycle, etc.",
    ):
        from nucbench.tasks import run_undergraduate_exam
        _run_task(
            run_undergraduate_exam,
            label="Undergraduate Exam",
            model=selected_model,
            api_key=api_key_input.strip(),
            temperature=temperature,
            n_samples=n_iterations,
            delay_s=delay_seconds,
        )

    st.write("")

    # Task 2 — Operator exam
    if st.button(
        "🔬 GFE Reactor Operator Exam",
        disabled=locked,
        use_container_width=True,
        help=lock_msg or "4,292 NRC GFE multiple-choice questions (BWR + PWR).",
    ):
        from nucbench.tasks import run_operator_exam
        _run_task(
            run_operator_exam,
            label="Reactor Operator Exam",
            model=selected_model,
            api_key=api_key_input.strip(),
            temperature=temperature,
            n_samples=n_iterations,
            delay_s=delay_seconds,
        )

    st.write("")

    # Task 3 — Flow regime classification
    if st.button(
        "🌊 Two-Phase Flow Classification",
        disabled=locked,
        use_container_width=True,
        help=lock_msg or "Classify bubbly / slug / churn / Taylor Bubble images.",
    ):
        from nucbench.tasks import run_flow_regime_task
        _run_task(
            run_flow_regime_task,
            label="Flow Regime Classification",
            model=selected_model,
            api_key=api_key_input.strip(),
            temperature=temperature,
            n_samples=n_iterations,
            delay_s=delay_seconds,
        )

    if locked:
        st.info("🔒 Validate your API key in the sidebar to enable benchmarking.")


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

with col_results:
    st.subheader("📊 Results")

    if st.session_state.run_error:
        st.error(f"**Run failed:** {st.session_state.run_error}")

    elif st.session_state.run_results:
        res = st.session_state.run_results
        scores: List[int] = res["scores"]
        n = len(scores)
        mean_acc = statistics.mean(scores) if scores else 0.0
        std_dev = statistics.stdev(scores) if n > 1 else 0.0

        st.markdown(
            f"""
            <div class="nb-result-card">
                <h3>{res['task_name']}</h3>
                <b>Model:</b> {res['model']}<br>
                <b>Samples:</b> {n} &nbsp;|&nbsp;
                <b>Mean Accuracy:</b> {mean_acc:.1%} &nbsp;|&nbsp;
                <b>Std Dev:</b> {std_dev:.4f}<br>
                <b>Saved to:</b> <code>{res.get('results_path', 'results.json')}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Per-question breakdown in an expander to keep the page tidy
        with st.expander(f"Per-sample details ({n} items)", expanded=False):
            for idx, detail in enumerate(res["details"], start=1):
                correct = detail.get("score", 0) == 1
                badge = (
                    '<span class="nb-badge-success">✓ Correct</span>'
                    if correct
                    else '<span class="nb-badge-error">✗ Incorrect</span>'
                )

                # Build a short label depending on task type
                if "true_label" in detail:
                    label_line = (
                        f"**Image:** {detail['image']} &nbsp; "
                        f"**Fluid:** {detail['fluid']} &nbsp; "
                        f"**True:** {detail['true_label']}"
                    )
                else:
                    label_line = (
                        f"**ID:** {detail.get('question_id', idx)} &nbsp; "
                        f"**Topic:** {detail.get('topic', '—')} &nbsp; "
                        f"**Type:** {detail.get('type', '—')} &nbsp; "
                        f"**Key:** {detail.get('key_answer', '—')}"
                    )

                st.markdown(
                    f"**{idx}.** {label_line} &nbsp; {badge}",
                    unsafe_allow_html=True,
                )
                with st.container():
                    st.text_area(
                        label=f"Response {idx}",
                        value=detail.get("response", ""),
                        height=80,
                        disabled=True,
                        key=f"resp_{idx}",
                        label_visibility="collapsed",
                    )

    else:
        st.info("Run a benchmark task to see results here.")
