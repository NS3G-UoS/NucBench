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
    "run_counter": 0,
    "corrections_saved": False,
    "cloud_vision_supported": None,  # None=unknown, True/False after validation
    # Local / open-source model support
    "mode": "Cloud",
    "local_endpoint": "http://localhost:11434",
    "local_models": [],
    "local_models_loaded": False,
    "local_connected": False,
    "local_connection_message": "",
    "local_model_manual": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Sidebar — load vision models once per session
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # -- Mode selector -------------------------------------------------------
    mode: str = st.radio(
        "Mode",
        options=["Cloud", "Local"],
        index=0 if st.session_state.mode == "Cloud" else 1,
        horizontal=True,
        help=(
            "**Cloud** — use a hosted API provider via LiteLLM.  "
            "**Local** — connect to a local inference server "
            "(Ollama, llama.cpp, vLLM, LM Studio, etc.)."
        ),
    )
    # Reset validation state when the user switches modes
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.key_validated = False
        st.session_state.local_connected = False
        st.session_state.cloud_vision_supported = None

    if mode == "Cloud":
        # -- Cloud model list ------------------------------------------------
        if not st.session_state.models_loaded:
            with st.spinner("Loading models…"):
                from nucbench.models import get_all_models
                st.session_state.vision_models = get_all_models()
                st.session_state.models_loaded = True

        model_list: List[str] = st.session_state.vision_models

        if not model_list:
            st.warning("No models found — check your LiteLLM installation.")
            model_list = ["(none available)"]

        selected_model: str = st.selectbox(
            "Model",
            options=model_list,
            help="All LiteLLM-registered models. Vision support is checked when you validate your API key.",
        )
        api_base: Optional[str] = None

        # -- API key ---------------------------------------------------------
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
                    from nucbench.models import validate_api_key, model_supports_vision
                    ok, msg = validate_api_key(selected_model, api_key_input.strip())
                st.session_state.key_validated = ok
                st.session_state.validation_message = msg
                if ok:
                    st.session_state.cloud_vision_supported = model_supports_vision(selected_model)
                else:
                    st.session_state.cloud_vision_supported = None

        if st.session_state.validation_message:
            if st.session_state.key_validated:
                st.success(st.session_state.validation_message)
            else:
                st.error(st.session_state.validation_message)

        if st.session_state.key_validated and st.session_state.cloud_vision_supported is False:
            st.warning("⚠️ This model does not support image input — the Two-Phase Flow task will be disabled.")

    else:  # mode == "Local"
        # -- Local inference server ------------------------------------------
        st.subheader("Local Inference Server")
        local_endpoint: str = st.text_input(
            "Server URL",
            value=st.session_state.local_endpoint,
            placeholder="http://localhost:11434",
            help=(
                "Base URL of your local inference server — **no `/v1` needed**, "
                "it is appended automatically.\n\n"
                "Common defaults:\n"
                "- **Ollama**: `http://localhost:11434`\n"
                "- **LM Studio**: `http://localhost:1234`\n"
                "- **vLLM / llama.cpp / LocalAI**: `http://localhost:8080`\n"
                "- **Jan**: `http://localhost:1337`\n"
                "- **Text Gen WebUI**: `http://localhost:5000`"
            ),
        )
        if local_endpoint != st.session_state.local_endpoint:
            st.session_state.local_endpoint = local_endpoint
            st.session_state.local_models = []
            st.session_state.local_models_loaded = False
            st.session_state.local_connected = False

        if st.button("🔄 Refresh Models", use_container_width=True):
            with st.spinner("Querying local server…"):
                from nucbench.models import get_local_models
                st.session_state.local_models = get_local_models(
                    st.session_state.local_endpoint
                )
                st.session_state.local_models_loaded = True

        _local_list: List[str] = st.session_state.local_models

        # Always-visible text input — value persists in session state via key.
        # The user can type any model name here; it takes priority over the
        # dropdown below.
        st.text_input(
            "Model identifier",
            key="local_model_manual",
            placeholder="llama3:8b  or  ollama/llama3:8b",
            help=(
                "Any of these formats work — the app normalises them automatically:\n\n"
                "- Bare name: `llama3:8b`, `mistral`, `phi3`\n"
                "- Ollama prefix: `ollama/llama3:8b`\n"
                "- OpenAI-compat prefix: `openai/mistral`\n"
                "- vLLM HuggingFace path: `meta-llama/Llama-3-8B-Instruct`\n\n"
                "Leave blank to pick from the discovered-models dropdown below."
            ),
        )
        _manual_value: str = st.session_state.local_model_manual.strip()

        # Dropdown of auto-discovered models (optional convenience).
        _dropdown_choice = ""
        if _local_list:
            _dropdown_choice = st.selectbox(
                "Discovered models",
                options=[""] + _local_list,
                format_func=lambda x: "— select a discovered model —" if x == "" else x,
                help="Models auto-discovered from the local inference server. "
                     "Ignored when a model identifier is typed above.",
            )
        elif not st.session_state.local_models_loaded:
            st.info("Click **Refresh Models** to discover models on the server.")
        else:
            st.warning("No models found at that URL. Enter a model name above.")

        # Manual entry wins; fall back to dropdown selection.
        selected_model = _manual_value or _dropdown_choice

        api_base = st.session_state.local_endpoint
        api_key_input = ""  # no API key needed for local models

        # -- Connection test -------------------------------------------------
        test_btn = st.button("🔌 Test Connection", use_container_width=True)
        if test_btn:
            from nucbench.models import validate_model_identifier, test_local_connection
            fmt_ok, fmt_err = validate_model_identifier(selected_model)
            if not fmt_ok:
                st.error(fmt_err)
            else:
                with st.spinner("Testing connection…"):
                    ok, msg = test_local_connection(selected_model, api_base)
                st.session_state.local_connected = ok
                st.session_state.local_connection_message = msg

        if st.session_state.local_connection_message:
            if st.session_state.local_connected:
                st.success(st.session_state.local_connection_message)
            else:
                st.error(st.session_state.local_connection_message)

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
        "Unique questions (samples)",
        min_value=1,
        max_value=5000,
        value=10,
        step=1,
        help="How many questions / images to randomly draw from the dataset per run.",
    )

    n_runs: int = st.number_input(
        "Runs per question",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help=(
            "How many times each sampled question is sent to the model.  "
            "Total requests = Unique questions × Runs per question."
        ),
    )

    unique_per_run: bool = st.checkbox(
        "Unique questions per run",
        value=False,
        help=(
            "When **checked**: a fresh random sample is drawn before each run — "
            "good for broad dataset coverage across many runs.  "
            "When **unchecked** (default): every run repeats the same questions — "
            "good for measuring response variance on a fixed set."
        ),
    )

    total_requests = n_questions * n_runs
    st.caption(
        f"Total requests: {n_questions} questions × {n_runs} run{'s' if n_runs != 1 else ''}"
        f" = **{total_requests}**"
    )

    delay_seconds: float = st.number_input(
        "Delay between requests (s)",
        min_value=0.0,
        max_value=60.0,
        value=1.0,
        step=0.5,
        help="Wait time between consecutive API calls. Set to 0 for local models.",
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
        out_path, run_index = save_results(
            task_name=payload["task_name"],
            model=payload["model"],
            scores=payload["scores"],
            details=payload["details"],
        )
        payload["results_path"] = str(out_path)
        payload["results_index"] = run_index
        st.session_state.run_counter = st.session_state.get("run_counter", 0) + 1
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

    locked = (
        not st.session_state.key_validated
        if st.session_state.mode == "Cloud"
        else not st.session_state.local_connected
    )
    lock_msg = (
        "Validate your API key first."
        if st.session_state.mode == "Cloud" and locked
        else "Test your local connection first." if locked else ""
    )

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
            api_base=api_base,
            temperature=temperature,
            n_samples=n_questions,
            n_runs=n_runs,
            unique_per_run=unique_per_run,
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
            api_base=api_base,
            temperature=temperature,
            n_samples=n_questions,
            n_runs=n_runs,
            unique_per_run=unique_per_run,
            delay_s=delay_seconds,
        )

    st.write("")

    # Task 3 — Flow regime classification
    _flow_no_vision = (
        st.session_state.mode == "Cloud"
        and st.session_state.cloud_vision_supported is False
    )
    _flow_disabled = locked or _flow_no_vision
    _flow_help = (
        lock_msg
        or ("Model does not support image input." if _flow_no_vision else "")
        or "Classify bubbly / slug / churn / Taylor Bubble images."
    )
    if st.button(
        "🌊 Two-Phase Flow Classification",
        disabled=_flow_disabled,
        use_container_width=True,
        help=_flow_help,
    ):
        from nucbench.tasks import run_flow_regime_task
        _run_task(
            run_flow_regime_task,
            label="Flow Regime Classification",
            model=selected_model,
            api_key=api_key_input.strip(),
            api_base=api_base,
            temperature=temperature,
            n_samples=n_questions,
            n_runs=n_runs,
            unique_per_run=unique_per_run,
            delay_s=delay_seconds,
        )

    if locked:
        if st.session_state.mode == "Local":
            st.info("🔒 Test connection to local model in the sidebar to enable benchmarking.")
        else:
            st.info("🔒 Validate your API key in the sidebar to enable benchmarking.")
    elif _flow_no_vision:
        st.info("🔒 Selected model does not support image input — Two-Phase Flow task is disabled.")


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
        n_q = res.get("n_questions", n)
        n_r = res.get("n_runs", 1)
        mean_acc = statistics.mean(scores) if scores else 0.0
        std_dev = statistics.stdev(scores) if n > 1 else 0.0

        # Show post-correction success banner (set by the save handler below)
        if st.session_state.pop("corrections_saved", False):
            st.success("✓ Corrections saved to results.json")

        runs_line = (
            f"<b>Questions:</b> {n_q} &nbsp;|&nbsp;"
            f"<b>Runs per question:</b> {n_r} &nbsp;|&nbsp;"
            f"<b>Total requests:</b> {n}"
        )

        st.markdown(
            f"""
            <div class="nb-result-card">
                <h3>{res['task_name']}</h3>
                <b>Model:</b> {res['model']}<br>
                {runs_line}<br>
                <b>Mean Accuracy:</b> {mean_acc:.1%} &nbsp;|&nbsp;
                <b>Std Dev:</b> {std_dev:.4f}<br>
                <b>Saved to:</b> <code>{res.get('results_path', 'results.json')}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Per-response details ─────────────────────────────────────────────
        with st.expander(f"Per-sample details ({n} items)", expanded=False):
            for idx, detail in enumerate(res["details"], start=1):
                correct = detail.get("score", 0) == 1
                badge = (
                    '<span class="nb-badge-success">✓ Correct</span>'
                    if correct
                    else '<span class="nb-badge-error">✗ Incorrect</span>'
                )

                run_tag = (
                    f" &nbsp; **Run:** {detail.get('run', 1)}/{n_r}"
                    if n_r > 1 else ""
                )

                if "true_label" in detail:
                    label_line = (
                        f"**Image:** {detail['image']} &nbsp; "
                        f"**Fluid:** {detail['fluid']} &nbsp; "
                        f"**True:** {detail['true_label']}"
                        + run_tag
                    )
                else:
                    label_line = (
                        f"**ID:** {detail.get('question_id', idx)} &nbsp; "
                        f"**Topic:** {detail.get('topic', '—')} &nbsp; "
                        f"**Type:** {detail.get('type', '—')} &nbsp; "
                        f"**Key:** {detail.get('key_answer', '—')}"
                        + run_tag
                    )

                st.markdown(
                    f"**{idx}.** {label_line} &nbsp; {badge}",
                    unsafe_allow_html=True,
                )
                with st.container():
                    _resp = detail.get("response", "")
                    # Grow height to fit content: ~20px per line, min 80, max 600
                    _lines = max(1, _resp.count("\n") + 1)
                    _resp_height = min(max(_lines * 20, 80), 600)
                    st.text_area(
                        label=f"Response {idx}",
                        value=_resp,
                        height=_resp_height,
                        disabled=True,
                        key=f"resp_{st.session_state.run_counter}_{idx}",
                        label_visibility="collapsed",
                    )

        # ── Score correction editor ──────────────────────────────────────────
        rc = st.session_state.run_counter
        with st.expander(f"✏️ Correct Scores ({n} items)", expanded=False):
            st.caption(
                "Override the automated score for any response, "
                "then click **Save Corrections** — `results.json` will be updated in place."
            )

            new_scores: List[int] = []
            for idx, detail in enumerate(res["details"]):
                auto = detail.get("score", 0)
                col_lbl, col_sel = st.columns([4, 1])
                with col_lbl:
                    run_part = (
                        f" · Run {detail.get('run', 1)}/{n_r}" if n_r > 1 else ""
                    )
                    if "true_label" in detail:
                        st.markdown(
                            f"**{idx + 1}.** `{detail['image']}` · "
                            f"True: **{detail['true_label']}**{run_part}"
                        )
                    else:
                        st.markdown(
                            f"**{idx + 1}.** `{detail.get('question_id', idx + 1)}` · "
                            f"Key: **{detail.get('key_answer', '—')}**{run_part}"
                        )
                with col_sel:
                    sel = st.selectbox(
                        "score",
                        options=["✓ Correct", "✗ Incorrect"],
                        index=0 if auto == 1 else 1,
                        key=f"score_edit_{rc}_{idx}",
                        label_visibility="collapsed",
                    )
                new_scores.append(1 if sel == "✓ Correct" else 0)

            if st.button(
                "💾 Save Corrections",
                use_container_width=True,
                key=f"save_corr_{rc}",
            ):
                updated_details = []
                for i, (det, ns) in enumerate(zip(res["details"], new_scores)):
                    d = {**det, "score": ns}
                    if ns != res["scores"][i]:
                        d["human_grade"] = ns * 100
                        d["manually_corrected"] = True
                    updated_details.append(d)

                from nucbench.scoring import save_score_corrections
                try:
                    save_score_corrections(
                        output_path=res.get("results_path", "results.json"),
                        run_index=res.get("results_index", -1),
                        updated_scores=new_scores,
                        updated_details=updated_details,
                    )
                    # Update in-memory payload so summary card reflects changes
                    st.session_state.run_results["scores"] = new_scores
                    st.session_state.run_results["details"] = updated_details
                    st.session_state["corrections_saved"] = True
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to save corrections: {exc}")

    else:
        st.info("Run a benchmark task to see results here.")
