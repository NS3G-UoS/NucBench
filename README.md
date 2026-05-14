# NucBench: Benchmarking LLMs in Nuclear Engineering

**NucBench** is an open benchmark for evaluating multimodal Large Language Models (LLMs) in nuclear engineering.  
It covers three task types:

1. **Reactor Operator Generic Fundamentals Exam (GFE)** — ~4,292 MCQs (PWR + BWR)  
2. **Undergraduate Nuclear Engineering Exam (curated by authors)** — 100+ mixed quantitative/qualitative questions across six subdomains  
3. **Two-Phase Flow Regime Images** — Bubbly, Slug, Churn, Taylor (external dataset)
<br><br>

---

## 🖥️ Usage
NucBench has both a web UI and a CLI for running benchmarks.

### ⚙️ Setup
1. Request access to the HuggingFace repository by emailing our research supervisor Dr. Bassam (bkhuwaileh@sharjah.ac.ae). 

2. Upon fresh install run the following (registers both the UI and the `nucbench` CLI command):
```bash
pip install -e .
```

### 🌐 Web UI (Streamlit)
```bash
streamlit run app.py
```

The interface is split into a **sidebar** (configuration) and a **main panel** (results).

#### Sidebar — Configuration
| Control | Description |
|---------|-------------|
| **Mode** | Switch between **Cloud** (hosted API) and **Local** (on-device server) |
| **Model** | Cloud: dropdown of all vision-capable models detected by LiteLLM. Local: type a model name or pick from auto-discovered list |
| **Provider API Key** | Cloud only — password field; click **Validate Key** to test before running |
| **Temperature** | Slider 0.0 – 1.0 (0 = deterministic greedy decoding) |
| **Unique questions (samples)** | How many questions/images are randomly drawn from the dataset per run |
| **Runs per question** | How many times each sampled question is sent to the model (default 1). Total requests = questions × runs |
| **Unique questions per run** | Checkbox — see [Run Modes](#run-modes) below |
| **Delay between requests (s)** | Wait time between consecutive API calls. Set to 0 for local models; use 1–2 s for cloud APIs to respect rate limits |

#### Run Modes

Each benchmark run has two independent dimensions: **how many questions** to sample, and **how many times** to ask each one.

| Setting | Behaviour | Good for |
|---------|-----------|----------|
| **Runs per question = 1** (default) | Each sampled question is asked once | Quick single-pass evaluation |
| **Runs per question > 1, unique questions per run OFF** (default) | The same set of questions is repeated for every run | Measuring response variance / consistency on a fixed set |
| **Runs per question > 1, unique questions per run ON** | A fresh random sample is drawn at the start of each run | Broader dataset coverage — each run sees different questions |

**Example:** 20 questions, 3 runs, unique-per-run **off** → the same 20 questions are asked 3 times each → 60 total requests.  
**Example:** 20 questions, 3 runs, unique-per-run **on** → 3 independent samples of 20 questions → 60 total requests, 60 distinct question draws.

The sidebar always shows the live **total requests** count so you can budget API cost or time before starting.

#### Main Panel — Benchmark Tasks
Three task buttons are enabled once the API key is validated:
- **🎓 Undergraduate NE Exam** — mixed MCQ and open-ended questions across six nuclear engineering subdomains
- **🔬 GFE Reactor Operator Exam** — 4,292 NRC multiple-choice questions (BWR + PWR combined)
- **🌊 Two-Phase Flow Classification** — classify bubbly / slug / churn / Taylor Bubble flow images

A live progress bar tracks execution. Results are streamed as they complete.

#### Results Panel
After a run completes the panel shows:

**Summary card** — task name, model, unique questions, runs per question, total requests, MCQ vs open-ended counts, mean accuracy, and std deviation.

**Per-response details** *(expandable)* — for each response: question ID, topic, format (MCQ / Open-Ended), key answer, confidence score (open-ended), and the full LLM response text. Each row is badged ✓ Correct / ✗ Incorrect for MCQ, or shows the assigned grade % for open-ended.

**Human Grading form** *(shown automatically when open-ended responses are present)* — displays the LLM response, model-reported confidence score, and key answer side-by-side. A **Manual Grade (%)** input (0–100) must be filled for each open-ended response before results are saved.

**✏️ Correct Scores** *(expandable)* — override any automated or human score after the fact. All scores are shown as percentages; MCQ auto-scores are converted to 0% / 100%. A single **Save Corrections** button re-saves `results.json` with the updated values.


### 💻 CLI Tool
```bash
python -m nucbench.cli <task> [options]
```

Progress is shown as an ASCII bar, and a summary table prints at completion. Open-ended responses without a human grade print a warning pointing back to the Streamlit UI (or you can pass --grade-open-ended to be prompted inline).

Three sub-commands `<task>`: <Br>
undergrad — for the Undergraduate Nuclear Engineering Exam dataset,<Br>
operator — for the PWR & BWR GFE NRC dataset,<br>
flow — for the 2-phase flow image classification dataset 

Each sub-command accepts the same flags:

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | required | LiteLLM model ID (e.g. `openai/gpt-4o`, `llama3:8b`) |
| `--api-key` | `-k` | `$NUCBENCH_API_KEY` | Provider API key. Not needed for local models |
| `--api-base` | | — | Base URL of a local inference server (e.g. `http://localhost:11434`). See [Local Model Benchmarking](#-local-model-benchmarking) |
| `--questions` | `-n` | `10` | Number of questions/images randomly sampled from the dataset per run |
| `--runs` | `-r` | `1` | How many times each sampled question is sent to the model. Total requests = questions × runs |
| `--unique-per-run` | | off | When set, draw a fresh random sample for each run instead of repeating the same questions. See [Run Modes](#run-modes) |
| `--temperature` | `-t` | `0.0` | LLM sampling temperature (0 = deterministic) |
| `--delay` | | `1.0` | Seconds to wait between consecutive API calls |
| `--output` | `-o` | `results.json` | Path to the output results file |
| `--grade-open-ended` | | off | Interactively prompt for a manual grade on open-ended responses in the terminal |
| `--grade-all` | | off | Interactively prompt for a manual grade on every response, including MCQ |

### 🏠 Local Model Benchmarking

NucBench can benchmark any LLM running on your own machine — no API key required.  
All modern local inference servers expose an OpenAI-compatible REST API, and NucBench routes all local calls through that endpoint automatically.

#### Supported Servers

| Server | Default URL | Notes |
|--------|-------------|-------|
| **Ollama** | `http://localhost:11434` | Run `ollama serve`; models installed with `ollama pull <name>` |
| **LM Studio** | `http://localhost:1234` | Enable "Local Server" in the app |
| **vLLM** | `http://localhost:8000` | `python -m vllm.entrypoints.openai.api_server --model <name>` |
| **llama.cpp** | `http://localhost:8080` | `./server -m model.gguf --port 8080` |
| **LocalAI** | `http://localhost:8080` | Drop-in OpenAI-compatible backend |
| **Jan** | `http://localhost:1337` | Enable the local API server from Jan's settings |
| **Text Gen WebUI** | `http://localhost:5000` | Enable the OpenAI extension |
| **Koboldcpp** | `http://localhost:5001` | Start with `--openai` flag |

Any other server that exposes `/v1/chat/completions` will also work.

---

#### Web UI — Local Mode Workflow

1. Launch the app: `streamlit run app.py`
2. In the sidebar, switch **Mode** to **Local**.
3. Enter your **Server URL** (e.g. `http://localhost:11434`). No `/v1` needed — it is appended automatically.
4. Click **🔄 Refresh Models** to auto-discover available models.  
   If discovery fails (some servers don't list models), skip this step and type the name directly.
5. In **Model identifier**, type the model name or select one from the dropdown.  
   See the [Model Identifier Format](#model-identifier-format) table below.
6. Click **🔌 Test Connection** to verify the server responds correctly.  
   A green success message must appear before the benchmark buttons unlock.
7. Set your run parameters (temperature, number of samples, delay) and run a task.

> **Note:** The delay between requests can be set to **0** for local servers since there are no rate limits.

---

#### Model Identifier Format

The model identifier tells NucBench which model to call on the inference server.  
The app accepts several formats and normalises them automatically — you do not need to match the exact format the server uses internally.

| What you type | Server | Resolved internally as |
|---------------|--------|------------------------|
| `llama3:8b` | Ollama | `openai/llama3:8b → localhost:11434/v1` |
| `ollama/llama3:8b` | Ollama | `openai/llama3:8b → localhost:11434/v1` |
| `mistral` | Ollama / any | `openai/mistral → <your url>/v1` |
| `openai/mistral` | Any OpenAI-compat | `openai/mistral → <your url>/v1` |
| `meta-llama/Llama-3-8B-Instruct` | vLLM | `openai/meta-llama/Llama-3-8B-Instruct → <your url>/v1` |
| `phi3`, `codellama:13b`, `deepseek-r1` | Any | `openai/<name> → <your url>/v1` |

**Common mistakes caught before any network call is made:**

| Bad input | Problem | Fix |
|-----------|---------|-----|
| `http://localhost:11434` | URL entered in the wrong field | Paste into **Server URL**, not Model identifier |
| `ollama/` | Missing model name after the slash | `ollama/llama3:8b` |
| `/llama3` | Leading slash | `llama3` or `ollama/llama3` |
| `llama 3` | Space in the name | `llama3` or `llama3:8b` |
| `ollama//llama3` | Double slash | `ollama/llama3` |

> **Tip for Ollama users:** find exact model names with `ollama list` in your terminal. The name shown there (e.g. `llama3:8b`, `mistral:latest`) is what you paste into the Model identifier field.

---

#### CLI — Local Models

Pass `--api-base` to point the CLI at your local server.  
No `--api-key` is needed; omit it or leave it blank.

For example:
```bash
# Ollama running llama3 locally
python -m nucbench.cli undergrad \
  --model llama3:8b \
  --api-base http://localhost:11434 \
  --questions 20

# vLLM with a HuggingFace model
python -m nucbench.cli operator \
  --model meta-llama/Llama-3-8B-Instruct \
  --api-base http://localhost:8000 \
  --questions 50 \
  --temperature 0.1

# llama.cpp server
python -m nucbench.cli flow \
  --model my-model \
  --api-base http://localhost:8080 \
  --questions 30
```

The same model identifier formats accepted by the UI work on the CLI.

---

#### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "Cannot reach `http://…`" | Server not running | Start the server; confirm it responds at the URL shown |
| "Connection test failed: model not found" | Model name typo or not pulled | Check `ollama list`; run `ollama pull <name>` if missing |
| "Connection test failed: … /v1/chat/completions" | Server doesn't support OpenAI-compat API | Enable the OpenAI extension / start server with correct flags |
| Test passes but benchmark errors on every question | Model too slow / `max_tokens` too low | Increase **Delay between requests**; the model may need more time |
| Images tasks fail with local model | Model lacks vision support | Use a vision-capable model (e.g. `llava`, `llava:13b`, `moondream` in Ollama) |

---

### Results.json

The after you complete your benchmarking, whether through the UI or CLI, the program will save a results.json file with the following information: 

- task: the task you preformed (which benchmark was run)
- model: the model being tested
- timestamp: date & time of benchmarking
- n_samples: the number of sample questions pulled from the dataset
- mean_accuracy: The mean accuracy of the model's responses calculated using ___
- std_deviation: The standard deviation of the model's responses calculated using ___
- image: (for image classification tasks) it says the image name used for benchmark
- fluid: actual fluid type folder from dataset that image is pulled from
- true_label: (for image classification tasks) the correct classification for the image
- response: (for image classification tasks) the model's classification of the image
- score: correct/incorrect binary automated marking
- run: number n where it is the nth time this question has been tested in the current session 
- format: multiple choice or Open-Ended (not for image classification dataset)
- human_grade: user's marking
- marks: automated marking
- confidence_score: LLM is asked to return confidence score on it's response, if unsupported this is null


> Note: Automated marking is decently robust, marking is usually accurate even if different case used (capital vs lowercase) or if there is text response with the multiple choice answer somewhere within the text answer it is also detected. 
---

## 📚 Data Sources

- **Generic Fundamentals Examinations (GFE)**  
  Source: [U.S. Nuclear Regulatory Commission – Generic Fundamentals Examinations](https://www.nrc.gov/reactors/operator-licensing/history-rulemaking-activities/generic-fundamentals-examinations.html)

> Note: The GFE is completely multiple choice


- **Undergraduate Nuclear Engineering Exam**  
  Curated by the authors to represent core topics and skills expected of upper-level nuclear engineering students.  
  
  **Topics covered:**
  - Reactor Thermal Hydraulics (quantitative + qualitative)  
  - Reactor Physics (quantitative + qualitative)  
  - Fuel Cycle (quantitative + qualitative)  
  - Nuclear Materials (quantitative + qualitative)  
  - Radiation (quantitative + qualitative)  
  - General / Other (primarily qualitative)

  This set balances numerical problem-solving with conceptual understanding to reflect real academic evaluation standards.
> Note: The undergraduate exam is a mix of multiple choice and open-ended questions.


- **Two-Phase Flow Regime Images**  
  Dataset should be cited as:  
  > Manikonda, Kaushik; Obi, Chinemerem Edmond; Brahmane, Aarya Abhay; Rahman, Mohammad Azizur; Hasan, Abu Rashid (2025),  
  > *Vertical Two-Phase Flow Regimes in an Annulus Image Dataset – Texas A&M University*,  
  > Mendeley Data, V3, doi: [10.17632/nxncbzzz38.3](https://doi.org/10.17632/nxncbzzz38.3)

---

## 👥 Authors
- **Bassam A. Khuwaileh** — University of Sharjah  
- **Polina Matesha** — University of Sharjah  
- **Dina Elhanan** - University of Sharjah

---

## 📂 Repository Map
- `exams/` – curated exam datasets (operator, undergraduate)  
- `images/` – labeled regime images *(if redistribution not permitted, link provided above)*  
- `docs/` – GitHub Pages site (landing page)  
- `CITATION.cff` – machine-readable citation file  
- `LICENSE` – open-access license (CC BY 4.0)

---

## 🔓 License
Dataset text and metadata are released under **Creative Commons Attribution 4.0 (CC BY 4.0)**.  
See the [LICENSE](LICENSE) file for details.

---
