"""Draws the system design / data-flow diagram used in the report."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(8, 10.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")

def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                fc=color, ec="#333333", lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=10)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))

box(5, 13.2, 6, 0.9, "IMDB Movie Review Dataset\n(50,000 labelled reviews)", "#DCE6F2")
box(5, 11.7, 6, 1.1, "Preprocessing\nTop-10,000 vocabulary  •  word → integer ID\nPad / truncate to 200 tokens", "#E2EFDA")
box(5, 10.1, 6, 0.9, "Data Split\nTrain 20,000  |  Validation 5,000  |  Test 25,000", "#E2EFDA")
box(5, 8.7, 6, 0.8, "Embedding Layer (10,000 × 64)", "#FFF2CC")

xs = [1.6, 3.9, 6.1, 8.4]
names = ["SimpleRNN\n(64)", "LSTM\n(64)", "GRU\n(64)", "BiLSTM\n(64 × 2)"]
cols = ["#F4CCCC", "#CFE2F3", "#D9EAD3", "#D9D2E9"]
for x, n, c in zip(xs, names, cols):
    arrow(5, 8.3, x, 7.65)
    box(x, 7.1, 1.9, 0.95, n, c)
    arrow(x, 6.6, 5, 5.95)
ax.text(5, 7.9, "Recurrent layer (one per experiment)", ha="center", fontsize=9, style="italic", bbox=dict(fc="white", ec="none", pad=1.5))

box(5, 5.6, 6, 0.65, "Dropout (0.5)  →  Dense (1, sigmoid)", "#FFF2CC")
box(5, 4.3, 6, 1.0, "Training\nAdam  •  Binary cross-entropy  •  Batch 128\nEarly stopping on validation loss", "#FCE4D6")
box(5, 2.75, 6, 1.0, "Evaluation on Test Set\nAccuracy • Precision • Recall • F1 • Confusion matrix\nTraining time • Parameters", "#FCE4D6")
box(5, 1.2, 6, 0.9, "Comparative Analysis & Prediction\non new reviews (Positive / Negative)", "#DCE6F2")

for y1, y2 in [(12.75, 12.25), (11.15, 10.55), (9.65, 9.1), (5.27, 4.8), (3.8, 3.25), (2.25, 1.65)]:
    arrow(5, y1, 5, y2)

plt.tight_layout()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.savefig(os.path.join(BASE, "2_results", "fig5_system_design_dfd.png"), dpi=170)
print("saved")
