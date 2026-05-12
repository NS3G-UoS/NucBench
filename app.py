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
# TODO: score in results.json is always incorrect 0 for open ended responses
# just delete score tbh it's useless
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
    "pending_grading": False,
    "grading_complete": False,
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

    n_questions: int = st.number_input(
        "Unique Questions",
        min_value=1,
        max_value=5000,
        value=10,
        step=1,
        help="Number of unique items to pull from the dataset.",
    )

    n_runs: int = st.number_input(
        "Runs per Question",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help="Number of times to repeat the prompt for each unique question.",
    )

    st.info(
        f"**Total Requests** = {n_questions} × {n_runs} = **{n_questions * n_runs}**"
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
    st.session_state.pending_grading = False
    st.session_state.grading_complete = False

    progress_bar = st.progress(0.0, text=f"Running {label}…")

    def _progress(frac: float) -> None:
        progress_bar.progress(min(frac, 1.0), text=f"{label} — {int(frac * 100)}%")

    try:
        payload = task_fn(progress_cb=_progress, **kwargs)

        # Check whether any open-ended responses need human grading
        has_open_ended = any(
            d.get("format") == "Open-Ended" and d.get("human_grade") is None
            for d in payload.get("details", [])
        )

        if has_open_ended:
            # Defer save until human grades are submitted
            st.session_state.pending_grading = True
        else:
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
            n_samples=n_questions,
            n_runs=n_runs,
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
            n_samples=n_questions,
            n_runs=n_runs,
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
            n_samples=n_questions,
            n_runs=n_runs,
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

        # ── Human grading form (shown when open-ended responses are pending) ──
        if st.session_state.pending_grading and not st.session_state.grading_complete:
            open_ended_indices = [
                i for i, d in enumerate(res["details"])
                if d.get("format") == "Open-Ended" and d.get("human_grade") is None
            ]

            st.warning(
                f"⚠️ **{len(open_ended_indices)} open-ended response(s) require "
                "manual grading** before results can be saved."
            )

            with st.form("grading_form"):
                st.markdown("### ✍️ Human Grading")
                collected_grades: Dict[int, float] = {}

                for idx in open_ended_indices:
                    detail = res["details"][idx]
                    max_marks = detail.get("marks")
                    q_num = idx + 1

                    st.markdown(
                        f"---\n**Response #{q_num}** &nbsp;"
                        f"*(Run {detail.get('run', 1)} · "
                        f"ID: {detail.get('question_id', '—')} · "
                        f"Topic: {detail.get('topic', '—')})*"
                    )

                    # Show question text
                    q_text = detail.get("question", "")
                    if q_text:
                        with st.expander("Question", expanded=False):
                            st.write(q_text)

                    # Key answer for reference
                    st.markdown(
                        f"**Key Answer:** `{detail.get('key_answer', '—')}`"
                    )

                    # Confidence score (if extracted)
                    conf = detail.get("confidence_score")
                    if conf is not None:
                        st.markdown(f"**Model Confidence:** {conf}%")

                    # LLM response
                    st.text_area(
                        label="LLM Response",
                        value=detail.get("response", ""),
                        height=120,
                        disabled=True,
                        key=f"oe_resp_{idx}",
                    )

                    # Manual grade input
                    collected_grades[idx] = st.number_input(
                        "Manual Grade (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key=f"grade_{idx}",
                        help="Score as a percentage (0–100).",
                    )

                submitted = st.form_submit_button(
                    "✅ Submit Grades & Save Results",
                    use_container_width=True,
                )

                if submitted:
                    for idx, grade in collected_grades.items():
                        res["details"][idx]["human_grade"] = grade
                        res["details"][idx]["score"] = grade
                        res["scores"][idx] = grade

                    from nucbench.scoring import save_results
                    out_path = save_results(
                        task_name=res["task_name"],
                        model=res["model"],
                        scores=res["scores"],
                        details=res["details"],
                    )
                    res["results_path"] = str(out_path)
                    st.session_state.run_results = res
                    st.session_state.pending_grading = False
                    st.session_state.grading_complete = True
                    st.rerun()

        # ── Summary card (shown once grading is done or no open-ended items) ──
        if not st.session_state.pending_grading or st.session_state.grading_complete:
            scores: List[int] = res["scores"]
            n = len(scores)
            n_q = res.get("n_questions", n)
            n_r = res.get("n_runs", 1)
            mean_acc = statistics.mean(scores) if scores else 0.0
            std_dev = statistics.stdev(scores) if n > 1 else 0.0

            # Separate MCQ vs open-ended counts for display
            mcq_details = [d for d in res["details"] if d.get("format") == "MCQ"]
            oe_details = [d for d in res["details"] if d.get("format") == "Open-Ended"]
            format_line = ""
            if mcq_details or oe_details:
                format_line = (
                    f"<b>MCQ:</b> {len(mcq_details)} &nbsp;|&nbsp; "
                    f"<b>Open-Ended:</b> {len(oe_details)}<br>"
                )

            st.markdown(
                f"""
                <div class="nb-result-card">
                    <h3>{res['task_name']}</h3>
                    <b>Model:</b> {res['model']}<br>
                    <b>Unique Questions:</b> {n_q} &nbsp;|&nbsp;
                    <b>Runs per Question:</b> {n_r} &nbsp;|&nbsp;
                    <b>Total Requests:</b> {n}<br>
                    {format_line}
                    <b>Mean Accuracy:</b> {mean_acc:.1%} &nbsp;|&nbsp;
                    <b>Std Dev:</b> {std_dev:.4f}<br>
                    <b>Saved to:</b> <code>{res.get('results_path', 'results.json')}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Per-response breakdown in an expander to keep the page tidy
            with st.expander(f"Per-response details ({n} items)", expanded=False):
                for idx, detail in enumerate(res["details"], start=1):
                    fmt = detail.get("format", "")
                    is_oe = fmt == "Open-Ended"

                    if is_oe:
                        human_g = detail.get("human_grade")
                        badge = (
                            f'<span class="nb-badge-success">Grade: {human_g}%</span>'
                            if human_g is not None
                            else '<span class="nb-badge-error">Ungraded</span>'
                        )
                    else:
                        correct = detail.get("score", 0) == 1
                        badge = (
                            '<span class="nb-badge-success">✓ Correct</span>'
                            if correct
                            else '<span class="nb-badge-error">✗ Incorrect</span>'
                        )

                    # Build label line
                    if "true_label" in detail:
                        label_line = (
                            f"**Image:** {detail['image']} &nbsp; "
                            f"**Fluid:** {detail['fluid']} &nbsp; "
                            f"**True:** {detail['true_label']}"
                        )
                    else:
                        conf_str = ""
                        if is_oe and detail.get("confidence_score") is not None:
                            conf_str = f" &nbsp; **Conf:** {detail['confidence_score']}%"
                        label_line = (
                            f"**ID:** {detail.get('question_id', idx)} &nbsp; "
                            f"**Topic:** {detail.get('topic', '—')} &nbsp; "
                            f"**Format:** {fmt} &nbsp; "
                            f"**Key:** {detail.get('key_answer', '—')}"
                            f"{conf_str}"
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

            # ── Score correction form ─────────────────────────────────────────
            with st.expander("✏️ Manually Correct Scores", expanded=False):
                st.caption(
                    "Override any automated or human score. "
                    "All scores are expressed as percentages (0–100). "
                    "MCQ auto-scores are shown as 0% (incorrect) or 100% (correct)."
                )
                with st.form("correction_form"):
                    corrections: Dict[int, float] = {}
                    for i, detail in enumerate(res["details"]):
                        fmt = detail.get("format", "")
                        is_oe = fmt == "Open-Ended"

                        # Derive current percentage value for pre-fill
                        if is_oe:
                            cur_pct = float(detail.get("human_grade") or 0.0)
                        else:
                            # MCQ scores are 0/1 — convert to 0/100
                            cur_pct = float(detail.get("score", 0)) * 100.0

                        # Build a compact label
                        if "true_label" in detail:
                            entry_label = (
                                f"#{i+1} · {detail.get('fluid','—')} · "
                                f"{detail.get('image','—')} · "
                                f"True: {detail.get('true_label','—')}"
                            )
                        else:
                            entry_label = (
                                f"#{i+1} · {detail.get('question_id', i+1)} · "
                                f"{detail.get('topic','—')} · "
                                f"[{fmt or 'Auto'}] · "
                                f"Run {detail.get('run', 1)}"
                            )

                        corrections[i] = st.number_input(
                            entry_label,
                            min_value=0.0,
                            max_value=100.0,
                            value=cur_pct,
                            step=1.0,
                            key=f"corr_{i}",
                        )

                    save_corrections = st.form_submit_button(
                        "💾 Save Corrections",
                        use_container_width=True,
                    )

                if save_corrections:
                    for i, pct in corrections.items():
                        res["details"][i]["human_grade"] = pct
                        res["details"][i]["score_corrected"] = True
                        # Normalise score to 0-1 for aggregate stats
                        res["scores"][i] = pct / 100.0
                    from nucbench.scoring import save_results
                    out_path = save_results(
                        task_name=res["task_name"],
                        model=res["model"],
                        scores=res["scores"],
                        details=res["details"],
                    )
                    res["results_path"] = str(out_path)
                    st.session_state.run_results = res
                    st.success(f"Corrections saved to `{out_path}`.")
                    st.rerun()

    else:
        st.info("Run a benchmark task to see results here.")
