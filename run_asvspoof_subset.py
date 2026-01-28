import os
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# =========================
# CONFIG (your dataset root)
# =========================
BASE_ROOT = r"C:\Users\moham\Downloads\LA"

# Your folder names (as you listed)
PROTO_DIR = os.path.join(BASE_ROOT, "ASVspoof2019_LA_cm_protocols")
FLAC_DIR  = os.path.join(BASE_ROOT, "ASVspoof2019_LA_eval", "flac")

# subset size for fast CPU run
N = 300   # change to 200 if you want even faster

# =========================
# 1) Find eval protocol file
# =========================
if not os.path.isdir(PROTO_DIR):
    raise FileNotFoundError(f"Protocol folder not found: {PROTO_DIR}")

proto_files = [f for f in os.listdir(PROTO_DIR) if f.lower().endswith(".txt") and "eval" in f.lower()]
if not proto_files:
    raise FileNotFoundError("No eval protocol .txt file found in ASVspoof2019_LA_cm_protocols.")

PROTO_PATH = os.path.join(PROTO_DIR, proto_files[0])
print("Using protocol file:", PROTO_PATH)

# =========================
# 2) Read protocol labels
#    Usually: <spk> <file_id> <...> <label>
# =========================
file_ids, labels = [], []
with open(PROTO_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        fid = parts[1]                 # file ID often in 2nd column
        lab = parts[-1].lower()        # label usually last column: bonafide/spoof
        if lab not in ("bonafide", "spoof"):
            continue

        file_ids.append(fid)
        labels.append(1 if lab == "spoof" else 0)

total = len(file_ids)
print("Total protocol entries:", total)

if total == 0:
    raise RuntimeError("No labeled entries found in protocol file. Check protocol format.")

# subset
N = min(N, total)
file_ids = file_ids[:N]
y = np.array(labels[:N], dtype=int)
print("Using subset size:", N)

# =========================
# 3) Extract MFCC features
# =========================
if not os.path.isdir(FLAC_DIR):
    raise FileNotFoundError(f"Audio folder not found: {FLAC_DIR}")

X = []
valid_y = []
missing = 0

for fid, label in zip(file_ids, y):
    audio_path = os.path.join(FLAC_DIR, fid + ".flac")
    if not os.path.exists(audio_path):
        missing += 1
        continue

    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    feat = np.mean(mfcc, axis=1)  # 20-dim vector
    X.append(feat)
    valid_y.append(label)

X = np.array(X)
valid_y = np.array(valid_y, dtype=int)

print("Loaded samples:", len(X), "| Missing:", missing)

if len(X) < 10:
    raise RuntimeError("Too few samples loaded. Check FLAC folder + file IDs.")

# =========================
# 4) Train baseline classifier
# =========================
clf = LogisticRegression(max_iter=2000)
clf.fit(X, valid_y)
scores = clf.predict_proba(X)[:, 1]  # probability of spoof (1)

# =========================
# 5) Metrics: Accuracy, EER, ROC-AUC
# =========================
fpr, tpr, _ = roc_curve(valid_y, scores)
fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
auc = roc_auc_score(valid_y, scores)
acc = accuracy_score(valid_y, (scores >= 0.5).astype(int))

print(f"Accuracy: {acc*100:.2f}%")
print(f"EER: {eer*100:.2f}%")
print(f"ROC-AUC: {auc:.4f}")

# =========================
# 6) Save ROC curve figure
# =========================
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – ASVspoof 2019 LA (Subset)")
out_png = os.path.join(os.path.dirname(__file__), "roc_voice.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print("Saved ROC curve to:", out_png)
plt.show()
