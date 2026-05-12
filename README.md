# NucBench: Benchmarking LLMs in Nuclear Engineering

**NucBench** is an open benchmark for evaluating multimodal Large Language Models (LLMs) in nuclear engineering.  
It covers three task types:

1. **Reactor Operator Generic Fundamentals Exam (GFE)** — ~4,292 MCQs (PWR + BWR)  
2. **Undergraduate Nuclear Engineering Exam (curated by authors)** — 100+ mixed quantitative/qualitative questions across six subdomains  
3. **Two-Phase Flow Regime Images** — Bubbly, Slug, Churn, Taylor (external dataset)
<br><br><br>

---

## 🖥️ Usage
NucBench has both a web UI and a CLI for running benchmarks.

### ⚙️ Setup
1. Clone this repo into your local machine

`git clone git@github.com:NS3G-UoS/NucBench.git`

2. Request access to the dataset files through emailing our research supervisor Dr. Bassam (bkhuwaileh@sharjah.ac.ae). You will recieve the following files & folders and they must be placed in the following locations within this repository. 

| File/Folder Name | Location to Move it to |
|------------------|-----------------------|
| exam_questions.json | exams/undergraduate/   (folder) |
| pwr_bank.json | exams/operator/PWR/    (folder) |
| bwr_bank.json | exams/operator/BWR/   (folder) |
| Fluid 1_Air/    (folder) | images/    (folder) |
| Fluid 2_CO2/    (folder) | images/    (folder) |


3. Upon fresh install run the following (registers both the UI and the `nucbench` CLI command):
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
| **Model** | Dropdown of all vision-capable models detected by LiteLLM |
| **Provider API Key** | Password field; click **Validate Key** to test before running |
| **Temperature** | Slider 0.0 – 1.0 (0 = deterministic) |
| **Unique Questions** | Number of distinct questions/images sampled from the dataset |
| **Runs per Question** | How many times each question is repeated (e.g. for stochasticity studies) |
| **Total Requests** | Live calculation: Unique Questions × Runs per Question |
| **Delay between requests (s)** | Throttle between consecutive API calls to respect rate limits |

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

|Flag	| Short	| Default	| Description|
|-----|-------|---------|------------|
|--model	| -m	| required	| LiteLLM model ID |
|--api-key	| -k	| $NUCBENCH_API_KEY	| Provider API key |
|--questions	| -n	| 10	| Unique questions to sample |
|--runs	| -r | 1	| Runs per question |
|--temperature	| -t	| 0.0	| Sampling temperature |
|--delay |		| 1.0 s	| Delay between requests |
|--output	| -o	|results.json	| Output file path |
|--grade-open-ended |		| off |	Interactively grade open-ended responses in the terminal |
|--grade-all |		| off |	Interactively grade every response (MCQ + open-ended) in the terminal |

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
