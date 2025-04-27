import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
#                   Configuration
# ---------------------------------------------------------------------
RESULTS_DIR = Path("feature_shift_results")  
PROMPT_PAIRS = [
    ("truthful", "deceptive"),
    ("truthful", "neutral"),
    ("neutral", "deceptive"),
]
LAYERS = list(range(32))

# Metric labels
METRICS = ["l2", "cosine", "overlap"]
DISPLAY_NAMES = {
    "l2": "L2 Distance",
    "cosine": "Cosine Similarity",
    "overlap": "Overlap Ratio",
}

# ---------------------------------------------------------------------
#                   Load Data
# ---------------------------------------------------------------------
def load_metrics(pair):
    a, b = pair
    l1_all, l2_all, cosine_all, overlap_all = [], [], [], []

    for layer in LAYERS:
        path = RESULTS_DIR / f"{a}_vs_{b}" / f"layer_{layer}_shifts.npz"
        data = np.load(path)

        l1_all.append((data["l1"].mean(), data["l1"].std()))
        l2_all.append((data["l2"].mean(), data["l2"].std()))
        cosine_all.append((data["cosine"].mean(), data["cosine"].std()))
        overlap_all.append((data["overlap"].mean(), data["overlap"].std()))

    return {
        "l1": np.array(l1_all),
        "l2": np.array(l2_all),
        "cosine": np.array(cosine_all),
        "overlap": np.array(overlap_all),
    }


# ---------------------------------------------------------------------
#                     Plotting
# ---------------------------------------------------------------------
def plot_metrics_for_pair(pair, metrics):
    a, b = pair
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # main y axis (Cosine, Overlap)
    colors = ["green", "red"]
    for metric_name, color in zip(["cosine", "overlap"], colors):
        means = metrics[metric_name][:, 0]
        stds = metrics[metric_name][:, 1]
        ax1.plot(LAYERS, means, label=DISPLAY_NAMES[metric_name], color=color, marker='o')
        ax1.fill_between(LAYERS, means-stds, means+stds, color=color, alpha=0.2)

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Metric Value (Cosine / Overlap)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # minor y axis (L2)
    ax2 = ax1.twinx()
    means_l2 = metrics["l2"][:, 0]
    stds_l2 = metrics["l2"][:, 1]
    ax2.plot(LAYERS, means_l2, label=DISPLAY_NAMES["l2"], color="orange", marker='s')
    ax2.fill_between(LAYERS, means_l2-stds_l2, means_l2+stds_l2, color="orange", alpha=0.2)
    ax2.set_ylabel("L2 Distance Value", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"Feature Shift Metrics: {a} vs {b}", fontsize=14)
    plt.tight_layout()

    save_path = f"feature_shift_{a}_vs_{b}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")
    plt.close()

# ---------------------------------------------------------------------
#                         Main
# ---------------------------------------------------------------------
def main():
    for pair in PROMPT_PAIRS:
        metrics = load_metrics(pair)
        plot_metrics_for_pair(pair, metrics)

if __name__ == "__main__":
    main()

