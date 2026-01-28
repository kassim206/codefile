import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

# =========================
# CONFIG (YOUR DATASET PATH)
# =========================
BASE_ROOT = r"C:\Users\moham\Downloads\LA"

PROTO_DIR = os.path.join(BASE_ROOT, "ASVspoof2019_LA_cm_protocols")
FLAC_DIR  = os.path.join(BASE_ROOT, "ASVspoof2019_LA_eval", "flac")

N = 300  # subset size (fast)

# =========================
# 1) FIND PROTOCOL FILE
# =========================
proto_files = [
    f for f in os.listdir(PROTO_DIR)
    if f.lower().endswith(".txt") and "eval" in f.lower()
]

if not proto_files:
    raise FileNotFoundError("No eval protocol file found.")

PROTO_PATH = os.path.join(PROTO_DIR, proto_files[0])
print("Using protocol file:", PROTO_PATH)

# =========================
# 2) READ LABELS
# =========================
file_ids, labels = [], []

with open(PROTO_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        fid = parts[1]              # file ID
        lab = parts[-1].lower()     # bonafide / spoof

        if lab not in ("bonafide", "spoof"):
            continue

        file_ids.append(fid)
        labels.append(1 if lab == "spoof" else 0)

print("Total protocol entries:", len(file_ids))

# subset
file_ids = file_ids[:N]
labels = labels[:N]

# =========================
# 3) FEATURE EXTRACTION
# =========================
X, y = [], []
missing = 0

for fid, label in zip(file_ids, labels):
    path = os.path.join(FLAC_DIR, fid + ".flac")

    if not os.path.exists(path):
        missing += 1
        continue

    audio, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feat = np.mean(mfcc, axis=1)

    X.append(feat)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Loaded samples:", len(X), "| Missing:", missing)

# =========================
# 4) TRAIN + SCORE
# =========================
clf = LogisticRegression(max_iter=2000)
clf.fit(X, y)

scores = clf.predict_proba(X)[:, 1]
y_pred = (scores >= 0.5).astype(int)

# =========================
# 5) METRICS
# =========================
acc = accuracy_score(y, y_pred)

fpr, tpr, _ = roc_curve(y, scores)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

auc = roc_auc_score(y, scores)

print(f"Accuracy: {acc*100:.2f}%")
print(f"EER: {eer*100:.2f}%")
print(f"ROC-AUC: {auc:.4f}")

# =========================
# 6) ROC CURVE
# =========================
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Voice Spoof Detection")
plt.grid(True)
plt.savefig("roc_voice.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 7) CONFUSION MATRIX
# =========================
cm = confusion_matrix(y, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix – Voice Spoof Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Bonafide", "Spoof"])
plt.yticks([0, 1], ["Bonafide", "Spoof"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.savefig("fig_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: roc_voice.png and fig_confusion_matrix.png")
