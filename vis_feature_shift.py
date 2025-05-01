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
DATASETS = ["cities", "animal_class", "facts", "element_symb", "inventors", "sp_en_trans"]  

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
    l2_all, cosine_all, overlap_all = [], [], []

    for layer in LAYERS:
        l2_layer, cosine_layer, overlap_layer = [], [], []
        
        for dataset in DATASETS:
            path = RESULTS_DIR / f"user_end_{a}_vs_{b}" / dataset / f"layer_{layer}_shifts.npz"
            data = np.load(path)

            l2_layer.append(data["l2"])
            cosine_layer.append(data["cosine"])
            overlap_layer.append(data["overlap"])

        l2_all.append((np.concatenate(l2_layer).mean(), np.concatenate(l2_layer).std()))
        cosine_all.append((np.concatenate(cosine_layer).mean(), np.concatenate(cosine_layer).std()))
        overlap_all.append((np.concatenate(overlap_layer).mean(), np.concatenate(overlap_layer).std()))


    return {
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

    save_path = f"experimental_outputs/feature_shift_results/user_end_feature_shift_{a}_vs_{b}.png"
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

