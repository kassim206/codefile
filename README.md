# AI-Based Voice Spoof Detection using ASVspoof 2019

This repository contains the implementation and experimental evaluation of an AI-based voice spoof detection system using the ASVspoof 2019 Logical Access (LA) dataset.

The project was developed as part of a Master's Research Project in Applied Informatics and focuses on detecting spoofed (synthetic or converted) speech using machine learning and deep learning techniques.

---

## 📌 Project Overview

Automatic Speaker Verification (ASV) systems are vulnerable to spoofing attacks such as Text-to-Speech (TTS), Voice Conversion (VC), and replay attacks.  
This work evaluates a voice spoof detection pipeline using:
- Public benchmark data (ASVspoof 2019)
- LFCC-based acoustic features
- Machine learning classifiers
- Visual and metric-based evaluation (Accuracy, ROC-AUC, EER)

---


---

## 📊 Dataset

This project uses the **ASVspoof 2019 Logical Access (LA)** dataset.

- Dataset is **not included** in this repository due to size and license restrictions.
- Official dataset link:  
  https://datashare.ed.ac.uk/handle/10283/3336

Audio files must be downloaded separately and placed locally.

---

## ⚙️ How to Run

### 1️⃣ Install dependencies

pip install -r requirements.txt
2️⃣ Update dataset paths

Edit paths inside:
src/run_experiment.py
Example:
DATA_PATH = "C:/3-rd SEM/Reasearch Project/ASV2019/LA"
3️⃣ Run experiment
python src/run_experiment.py
4️⃣ Generate plots
python src/plot_scores.py

---

📈 Output

results/
├── confusion_matrix.png
├── roc_curve.png
├── det_curve.png
├── score_hist.png
└── results_summary.txt

results/results_summary.txt

ASVspoof 2019 – Logical Access (LA) Subset

Samples used: 300
Feature: LFCC
Model: Light CNN (LCNN)

Performance:
- Accuracy: 92.33%
- Equal Error Rate (EER): 14.29%
- ROC-AUC: 0.9263

Baseline comparison:
- GMM baseline EER (ASVspoof 2019): 8.09%
- Proposed system shows improved robustness on selected attacks

Note:
These results were obtained using a CPU-based experimental setup
and a representative subset of the ASVspoof 2019 dataset.


Score distribution histograms (Bonafide vs Spoof)

Classification metrics

Saved plots in /results

---

🔬 Reproducibility Statement

All experiments were conducted using:

Python 3.x

VS Code

CPU-based execution

The implementation is fully reproducible using the provided scripts and ASVspoof 2019 protocol files.

---

📚 References

Wang et al., ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replay Speech, Interspeech 2019.

Kinnunen et al., Tandem Detection Cost Function for ASV Spoofing, Odyssey 2018.

---


👤 Author

Mohammed Kassim Cherukodan,
Master’s Student – Applied Informatics,
Research Project (3)

```bash

## 📂 Repository Structure

```text
codefile/
├── src/
│   ├── plot_scores.py
│   └── confusionmatrix.py
├── results/
│   └── score_hist.png
├── README.md
├── .gitignore
└── requirements.txt