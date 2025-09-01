# NucBench: Benchmarking LLMs in Nuclear Engineering

**NucBench** is an open benchmark for evaluating multimodal Large Language Models (LLMs) in nuclear engineering.  
It covers three task types:

1. **Reactor Operator Generic Fundamentals Exam (GFE)** — ~4,292 MCQs (PWR + BWR)  
2. **Undergraduate Nuclear Engineering Exam (curated by authors)** — 100+ mixed quantitative/qualitative questions across six subdomains  
3. **Two-Phase Flow Regime Images** — Bubbly, Slug, Churn, Taylor (external dataset)

---

## 📚 Data Sources

- **Generic Fundamentals Examinations (GFE)**  
  Source: [U.S. Nuclear Regulatory Commission – Generic Fundamentals Examinations](https://www.nrc.gov/reactors/operator-licensing/history-rulemaking-activities/generic-fundamentals-examinations.html)

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

- **Two-Phase Flow Regime Images**  
  Dataset should be cited as:  
  > Manikonda, Kaushik; Obi, Chinemerem Edmond; Brahmane, Aarya Abhay; Rahman, Mohammad Azizur; Hasan, Abu Rashid (2025),  
  > *Vertical Two-Phase Flow Regimes in an Annulus Image Dataset – Texas A&M University*,  
  > Mendeley Data, V3, doi: [10.17632/nxncbzzz38.3](https://doi.org/10.17632/nxncbzzz38.3)

---

## 👥 Authors
- **Bassam A. Khuwaileh** — University of Sharjah  
- **Polina Matesha** — University of Sharjah  

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
