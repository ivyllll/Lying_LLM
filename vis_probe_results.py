import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------------
PROMPT_TYPES = ["truthful", "neutral", "deceptive"]
MODEL_FAMILY = "Gemma2" # Options: Llama3.1, Gemma2
MODEL_SIZE = "9B"
MODEL_TYPE = "chat"
SAVE_DIR = Path("experimental_outputs/probing_and_visualization/accuracy_figures")
SAVE_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_probe_accuracies(prompt_types, model_family, model_size, model_type):
    data = {}

    for prompt_type in prompt_types:
        pkl_path = Path("experimental_outputs") / "probing_and_visualization" / \
            prompt_type / model_family / model_size / model_type / \
                f"{prompt_type}_logical_probe_accuracies_layerwise.pkl"
        with open(pkl_path, "rb") as f:
            layerwise_results = pickle.load(f)

        ttpd_means = [layer_result["TTPD"]["mean"] * 100 for layer_result in layerwise_results]
        ttpd_stds  = [layer_result["TTPD"]["std_dev"] * 100 for layer_result in layerwise_results]
        lr_means   = [layer_result["LRProbe"]["mean"] * 100 for layer_result in layerwise_results]
        lr_stds    = [layer_result["LRProbe"]["std_dev"] * 100 for layer_result in layerwise_results]

        data[prompt_type] = {
            "TTPD_mean": np.array(ttpd_means),
            "TTPD_std": np.array(ttpd_stds),
            "LR_mean": np.array(lr_means),
            "LR_std": np.array(lr_stds),
        }

    return data

def plot_probe_accuracy_comparison(data, model_family, model_size, model_type, save_dir):
    layers = np.arange(0, 32)

    plt.figure(figsize=(10, 6))

    prompt_colors = {
        "truthful": "#1f77b4",   # blue
        "neutral": "#ff7f0e",    # orange
        "deceptive": "#2ca02c",  # green
    }

    probe_styles = {
        "TTPD": "solid",
        "LR": "dashed",
    }

    for prompt_type in PROMPT_TYPES:
        color = prompt_colors[prompt_type]

        ttpd_mean = data[prompt_type]["TTPD_mean"]
        ttpd_std  = data[prompt_type]["TTPD_std"]
        lr_mean   = data[prompt_type]["LR_mean"]
        lr_std    = data[prompt_type]["LR_std"]

        plt.errorbar(
            layers,
            ttpd_mean,
            yerr=ttpd_std,
            fmt='o-',   # marker+line
            color=color,
            linestyle=probe_styles["TTPD"],
            label=f"{prompt_type.capitalize()} - TTPD",
            markersize=7,
            linewidth=2.5,
            elinewidth=1,
            capsize=3,
        )
        plt.errorbar(
            layers,
            lr_mean,
            yerr=lr_std,
            fmt='s-',  # square marker + line
            color=color,
            linestyle=probe_styles["LR"],
            label=f"{prompt_type.capitalize()} - LR",
            markersize=7,
            linewidth=2.5,
            elinewidth=1,
            capsize=3,
        )

        # plt.plot(
        #     layers, ttpd_mean,
        #     label=f"{prompt_type.capitalize()} - TTPD",
        #     linestyle=probe_styles["TTPD"],
        #     marker="o-",
        #     markersize=5,
        #     color=color,
        #     linewidth=2,
        # )
        # plt.fill_between(
        #     layers,
        #     ttpd_mean - ttpd_std,
        #     ttpd_mean + ttpd_std,
        #     color=color,
        #     alpha=0.2,
        # )

        # plt.plot(
        #     layers, lr_mean,
        #     label=f"{prompt_type.capitalize()} - LR",
        #     linestyle=probe_styles["LR"],
        #     marker="s",
        #     markersize=5,
        #     color=color,
        #     linewidth=2,
        #     alpha=0.8
        # )
        # plt.fill_between(
        #     layers,
        #     lr_mean - lr_std,
        #     lr_mean + lr_std,
        #     color=color,
        #     alpha=0.2,
        # )
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Probing Accuracy (%)", fontsize=14)
    plt.title(f"{model_family}-{model_size}-{model_type}: Logical Probing Accuracy", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(40, 100)
    plt.legend(fontsize=10, loc='lower right', ncol=2)
    plt.tight_layout()

    save_path = save_dir / f"{model_family}_{model_size}_{model_type}_logical_probing_accuracy_error_bar.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Plot saved to {save_path}")

def export_markdown_table(data, save_path):
    layers = np.arange(0, 32)

    header = "| Layer | " + " | ".join([f"{prompt.capitalize()}-TTPD (Mean ± Std) | {prompt.capitalize()}-LR (Mean ± Std)" for prompt in PROMPT_TYPES]) + " |"
    separator = "|:-----:|" + (":--------------------------:|:-----------------------:|" * len(PROMPT_TYPES))
    lines = [header, separator]

    for layer in layers:
        row = [str(layer)]
        for prompt_type in PROMPT_TYPES:
            ttpd_mean = data[prompt_type]["TTPD_mean"][layer]
            ttpd_std  = data[prompt_type]["TTPD_std"][layer]
            lr_mean   = data[prompt_type]["LR_mean"][layer]
            lr_std    = data[prompt_type]["LR_std"][layer]

            row.append(f"{ttpd_mean:.2f} ± {ttpd_std:.2f}")
            row.append(f"{lr_mean:.2f} ± {lr_std:.2f}")
        
        lines.append("| " + " | ".join(row) + " |")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown table saved to {save_path}")

def find_peak_layer(data):
    print("\nPeak layer summary:")
    for prompt_type in PROMPT_TYPES:
        ttpd_peak_layer = np.argmax(data[prompt_type]["TTPD_mean"])
        lr_peak_layer = np.argmax(data[prompt_type]["LR_mean"])
        print(f"- {prompt_type.capitalize()} Prompt:")
        print(f"  TTPD peak at Layer {ttpd_peak_layer} with {data[prompt_type]['TTPD_mean'][ttpd_peak_layer]:.2f}%")
        print(f"  LR peak at Layer {lr_peak_layer} with {data[prompt_type]['LR_mean'][lr_peak_layer]:.2f}%")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    data = load_probe_accuracies(PROMPT_TYPES, MODEL_FAMILY, MODEL_SIZE, MODEL_TYPE)
    
    plot_probe_accuracy_comparison(data, MODEL_FAMILY, MODEL_SIZE, MODEL_TYPE, SAVE_DIR)

    export_markdown_table(data, save_path=SAVE_DIR / "logical_probe_accuracy_table_usr_end.md")

    find_peak_layer(data)

if __name__ == "__main__":
    main()
