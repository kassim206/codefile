import matplotlib.pyplot as plt

# Your real results
metrics = ["Accuracy (%)", "EER (%)", "ROC-AUC"]
values = [92.33, 14.29, 0.9263]

plt.figure()
plt.bar(metrics, values)
plt.title("Performance Metrics Summary (ASVspoof 2019 LA, N=300)")
plt.ylabel("Value")
plt.grid(True, axis="y")

out_path = "fig_metrics_bar.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", out_path)
