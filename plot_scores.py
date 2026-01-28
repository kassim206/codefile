import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG: CHANGE THIS PATH
# =========================
BASE_ROOT = r"C:\Users\moham\Downloads\LA"

# These must exist after you extract the ASVspoof2019 LA dataset
PROTO_DIR = os.path.join(BASE_ROOT, "ASVspoof2019_LA_cm_protocols")
FLAC_DIR  = os.path.join(BASE_ROOT, "ASVspoof2019_LA_eval", "flac")

N = 300  # smaller = faster

# =========================
# 1) FIND EVAL PROTOCOL FILE
# =========================
if not os.path.isdir(PROTO_DIR):
    raise FileNotFoundError(f"Protocol folder not found:\n{PROTO_DIR}")

proto_files = [
    f for f in os.listdir(PROTO_DIR)
    if f.lower().endswith(".txt") and "eval" in f.lower()
]
if not proto_files:
    raise FileNotFoundError(f"No EVAL protocol .txt file found inside:\n{PROTO_DIR}")

PROTO_PATH = os.path.join(PROTO_DIR, proto_files[0])
print("Using protocol file:", PROTO_PATH)

# =========================
# 2) READ IDS + LABELS
# y_true: 1=bonafide, 0=spoof
# =========================
file_ids, y_true = [], []

with open(PROTO_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        fid = parts[1]
        lab = parts[-1].lower()

        if lab not in ("bonafide", "spoof"):
            continue

        file_ids.append(fid)
        y_true.append(1 if lab == "bonafide" else 0)

file_ids = file_ids[:N]
y_true = np.array(y_true[:N], dtype=int)
print("Protocol entries used:", len(file_ids))

# =========================
# 3) FEATURE EXTRACTION
# =========================
if not os.path.isdir(FLAC_DIR):
    raise FileNotFoundError(f"FLAC folder not found:\n{FLAC_DIR}")

X, y = [], []
missing = 0

for fid, label in zip(file_ids, y_true):
    path = os.path.join(FLAC_DIR, fid + ".flac")
    if not os.path.exists(path):
        missing += 1
        continue

    audio, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feat = np.mean(mfcc, axis=1)

    X.append(feat)
    y.append(label)

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

print(f"Loaded samples: {len(X)} | Missing files: {missing}")
if len(X) < 5:
    raise RuntimeError("Too few samples loaded. Check dataset extraction and paths.")

# =========================
# 4) TRAIN + SCORES
# scores = P(bonafide)
# =========================
clf = LogisticRegression(max_iter=2000)
clf.fit(X, y)

scores = clf.predict_proba(X)[:, 1]

# =========================
# 5) PLOT SCORE HISTOGRAM
# =========================
bonafide_scores = scores[y == 1]
spoof_scores = scores[y == 0]

plt.figure()
plt.hist(bonafide_scores, bins=30, alpha=0.7, label="Bonafide")
plt.hist(spoof_scores, bins=30, alpha=0.7, label="Spoof")
plt.title("Score Distribution – Voice Spoof Detection")
plt.xlabel("Model Score (P(Bonafide))")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.savefig("score_hist.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: score_hist.png (in the src folder)")
