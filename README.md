# AI-Based Voice Spoof Detection using ASVspoof 2019

This repository contains the implementation and experimental evaluation of an AI-based voice spoof detection system using the ASVspoof 2019 Logical Access (LA) dataset.

The project was developed as part of a Master’s Research Project in Applied Informatics.

---

## Project Overview

Automatic Speaker Verification (ASV) systems are vulnerable to spoofing attacks such as Text-to-Speech (TTS), Voice Conversion (VC), and replay attacks.  
This project implements a voice spoof detection pipeline using acoustic features and machine learning techniques to distinguish between bonafide and spoofed speech.

---

## Dataset

This project uses the **ASVspoof 2019 Logical Access (LA)** dataset.

- The dataset is **not included** in this repository due to size and license restrictions.
- Official dataset link:  
  https://datashare.ed.ac.uk/handle/10283/3336

---

## Repository Structure

project/
├── src/
│ ├── plot_scores.py
│ ├── confusionmatrix.py
├── results/
│ └── score_hist.png
├── README.md
├── .gitignore
└── requirements.txt


---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt

2. Update dataset paths

Edit dataset paths inside the Python scripts located in the src/ folder.
Example:
BASE_ROOT = "C:/path/to/ASVspoof2019/LA"

3. Run the experiment
python src/plot_scores.py

Output

Score distribution histogram (Bonafide vs Spoof)

ROC curve

Confusion matrix

All generated figures are saved locally.

Reproducibility

All experiments were conducted using Python and standard open-source libraries.
The implementation is fully reproducible using the provided scripts and the official ASVspoof 2019 dataset.

Author

Mohammed Kassim Cherukodan
Master’s Student – Applied Informatics
Research Project (3)