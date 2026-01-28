import matplotlib.pyplot as plt

# =========================
# Voice spoof detection results
# =========================
metrics = ["Accuracy (%)", "EER (%)", "ROC-AUC"]
values = [92.33, 14.29, 0.9263]

# =========================
# Create bar chart
# =========================
plt.figure(figsize=(8, 5))
plt.bar(metrics, values)
plt.title("Performance Metrics Summary\n(ASVspoof 2019 LA Evaluation Subset)")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.grid(True, axis="y")

# =========================
# Save figure
# =========================
output_file = "fig_metrics_bar.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("Bar chart saved as:", output_file)
