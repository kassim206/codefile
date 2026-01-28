import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# =========================
# DATASET PATH (FIXED)
# =========================
BASE_ROOT = r"C:\Users\moham\Downloads\LA"

PROTO_DIR = os.path.join(BASE_ROOT, "ASVspoof2019_LA_cm_protocols")
FLAC_DIR  = os.path.join(BASE_ROOT, "ASVspoof2019_LA_eval", "flac")

N = 300

# =========================
# READ PROTOCOL
# =========================
proto_file = [f for f in os.listdir(PROTO_DIR) if "eval" in f.lower()][0]
proto_path = os.path.join(PROTO_DIR, proto_file)

file_ids, y_true = [], []

with open(proto_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if parts[-1].lower() in ("bonafide", "spoof"):
            file_ids.append(parts[1])
            y_true.append(1 if parts[-1] == "bonafide" else 0)

file_ids = file_ids[:N]
y_true = np.array(y_true[:N])

# =========================
# FEATURE EXTRACTION
# =========================
X, y = [], []

for fid, label in zip(file_ids, y_true):
    path = os.path.join(FLAC_DIR, fid + ".flac")
    if not os.path.exists(path):
        continue

    audio, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    X.append(np.mean(mfcc, axis=1))
    y.append(label)

X = np.array(X)
y = np.array(y)

# =========================
# MODEL
# =========================
clf = LogisticRegression(max_iter=2000)
clf.fit(X, y)

scores = clf.predict_proba(X)[:, 1]

# =========================
# SCORE DISTRIBUTION PLOT
# =========================
bonafide_scores = scores[y == 1]
spoof_scores = scores[y == 0]

plt.figure()
plt.hist(bonafide_scores, bins=30, alpha=0.7, label="Bonafide")
plt.hist(spoof_scores, bins=30, alpha=0.7, label="Spoof")
plt.title("Score Distribution – Voice Spoof Detection")
plt.xlabel("Model Score")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.savefig("score_hist.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: score_hist.png")
