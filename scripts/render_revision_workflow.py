"""Render the two distinct evidential layers used in the revised article."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from figure_io import save_png_atomic

def render(directory):
    plt.rcParams.update({"svg.hashsalt": "aicw-v3"})
    fig, ax = plt.subplots(figsize=(10.8, 4.1))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 4.1); ax.axis("off")
    lanes = [
        (3.05, "Descriptive analysis", ["Define outcomes\nand timing assumptions", "Compare four learners\non shared 5 × 5 folds", "Estimate held-out\npermutation importance", "Inspect ranking and\noutcome sensitivity"]),
        (1.25, "Independent shortlist evaluation", ["Hold out candidate\nprojects and outcomes", "Learn weights and scales\ninside training projects", "Freeze scoring rule\nand rank candidates", "Compare observed lift\nwith simple baselines"]),
    ]
    width, height = 2.34, .84
    xpos = [.08, 2.86, 5.64, 8.42]
    for y, heading, labels in lanes:
        ax.text(.08, y + .63, heading, fontsize=12, weight="bold", va="center")
        for i, (x, text) in enumerate(zip(xpos, labels)):
            patch = FancyBboxPatch((x, y - height/2), width, height,
                                   boxstyle="round,pad=.02", facecolor="#F5F6F7",
                                   edgecolor="#444444", linewidth=1)
            ax.add_patch(patch)
            ax.text(x + width/2, y, text, ha="center", va="center", fontsize=9.8,
                    linespacing=1.35)
            if i < 3:
                ax.annotate("", xy=(xpos[i+1]-.07, y), xytext=(x+width+.07, y),
                            arrowprops=dict(arrowstyle="->", lw=1.1, color="#444444"))
    ax.text(5.4, .28, "Predictive signal, importance stability and shortlist value are evaluated separately",
            ha="center", fontsize=10.2)
    fig.tight_layout(pad=.2)
    fig.savefig(directory/"Figure1_workflow.svg", bbox_inches="tight", metadata={"Date": None})
    path=directory/"Figure1_workflow.png"
    save_png_atomic(fig, path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1]/"figures")
