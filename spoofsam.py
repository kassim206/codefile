import os
import matplotlib.pyplot as plt

BASE_ROOT = r"C:\Users\moham\Downloads\LA"
PROTO_DIR = os.path.join(BASE_ROOT, "ASVspoof2019_LA_cm_protocols")

def count_labels(protocol_path: str):
    bonafide = 0
    spoof = 0
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lab = parts[-1].lower()
            if lab == "bonafide":
                bonafide += 1
            elif lab == "spoof":
                spoof += 1
    return bonafide, spoof

# find protocol files
files = os.listdir(PROTO_DIR)
train_file = [f for f in files if "train" in f.lower() and f.endswith(".txt")]
dev_file   = [f for f in files if "dev" in f.lower() and f.endswith(".txt")]
eval_file  = [f for f in files if "eval" in f.lower() and f.endswith(".txt")]

if not (train_file and dev_file and eval_file):
    raise FileNotFoundError("Could not find train/dev/eval protocol txt files in cm_protocols folder.")

train_path = os.path.join(PROTO_DIR, train_file[0])
dev_path   = os.path.join(PROTO_DIR, dev_file[0])
eval_path  = os.path.join(PROTO_DIR, eval_file[0])

train_b, train_s = count_labels(train_path)
dev_b, dev_s     = count_labels(dev_path)
eval_b, eval_s   = count_labels(eval_path)

splits = ["Train", "Dev", "Eval"]
bonafide_counts = [train_b, dev_b, eval_b]
spoof_counts    = [train_s, dev_s, eval_s]

# plot
x = range(len(splits))
width = 0.35

plt.figure()
plt.bar([i - width/2 for i in x], bonafide_counts, width, label="bonafide")
plt.bar([i + width/2 for i in x], spoof_counts, width, label="spoof")
plt.xticks(list(x), splits)
plt.xlabel("Split")
plt.ylabel("Number of files")
plt.title("Bonafide vs. Spoof Counts by Split (ASVspoof 2019 LA)")
plt.legend()
plt.grid(True, axis="y")

out_path = os.path.join(os.getcwd(), "fig_dataset_counts.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", out_path)
print("Train:", train_b, train_s, "| Dev:", dev_b, dev_s, "| Eval:", eval_b, eval_s)
