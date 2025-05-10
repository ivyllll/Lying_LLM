import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------
PROMPT_TYPES = ["truthful", "neutral", "deceptive"]
MODEL_FAMILY   = "Llama3.1"        #  Llama3.1  or  Gemma2
MODEL_SIZE     = "8B"
MODEL_TYPE     = "chat"

# 关键层：Gemma→22，Llama→14
KEY_LAYER = 22 if MODEL_FAMILY.startswith("Gemma") else 14

SAVE_DIR = Path(
    "experimental_outputs/probing_and_visualization/accuracy_figures"
)
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
def _read_pkl(pkl_path):
    """读取单个 pkl，返回 2×32 的 mean / std 向量（TTPD, LR）。"""
    with open(pkl_path, "rb") as f:
        layerwise = pickle.load(f)

    ttpd_mean = np.array([l["TTPD"]["mean"] * 100 for l in layerwise])
    ttpd_std  = np.array([l["TTPD"]["std_dev"] * 100 for l in layerwise])
    lr_mean   = np.array([l["LRProbe"]["mean"] * 100 for l in layerwise])
    lr_std    = np.array([l["LRProbe"]["std_dev"] * 100 for l in layerwise])
    return ttpd_mean, ttpd_std, lr_mean, lr_std


def load_data(model_family, model_size, model_type):
    """
    返回:
        results["curated"]["TTPD_mean"]  shape=(32,)
        results["logical"]["LR_std"]     shape=(32,)
    即把三种 prompt 先做平均 / 方差合并
    """
    results = {
        "curated": {"TTPD_mean": [], "TTPD_std": [], "LR_mean": [], "LR_std": []},
        "logical": {"TTPD_mean": [], "TTPD_std": [], "LR_mean": [], "LR_std": []},
    }

    # ------- ① 先收集每个 prompt 的向量 -------
    for prompt in PROMPT_TYPES:
        # (a) curated
        pkl_c = Path("experimental_outputs") / "probing_and_visualization" / \
                prompt / model_family / model_size / model_type / \
                "chat_probe_accuracies_layerwise.pkl"
        # (b) logical
        pkl_l = Path("experimental_outputs") / "probing_and_visualization" / \
                prompt / model_family / model_size / model_type / \
                f"{prompt}_logical_probe_accuracies_layerwise.pkl"

        for tag, pkl in zip(["curated", "logical"], [pkl_c, pkl_l]):
            ttpd_m, ttpd_s, lr_m, lr_s = _read_pkl(pkl)
            results[tag]["TTPD_mean"].append(ttpd_m)
            results[tag]["TTPD_std" ].append(ttpd_s)
            results[tag]["LR_mean"  ].append(lr_m)
            results[tag]["LR_std"   ].append(lr_s)

    # ------- ② 按 prompt 取均值 / 合并方差 -------
    for tag in ["curated", "logical"]:
        for key in ["TTPD_mean", "LR_mean"]:
            results[tag][key] = np.mean(results[tag][key], axis=0)

        # 合并 std :   sqrt( mean(σ²) + var(μ) )
        for key_mean, key_std in [("TTPD_mean", "TTPD_std"),
                                  ("LR_mean",   "LR_std")]:
            mu_list  = results[tag][key_mean]          # 已经是均值
            sigma_sq = np.mean(
                np.stack(results[tag][key_std]) ** 2, axis=0
            )
            var_mu   = np.var(
                np.stack(results[tag][key_mean]), axis=0, ddof=0
            )
            results[tag][key_std] = np.sqrt(sigma_sq + var_mu)

    return results


def plot_curves(res, model_family, model_size, model_type, save_dir):
    layers = np.arange(32)
    plt.figure(figsize=(9, 5.5))

    # 颜色：probe；线型：数据形态
    COLOR = {"LR": "#1f77b4", "TTPD": "#ff7f0e"}
    STYLE = {"curated": "solid", "logical": "dashed"}
    LABEL = {
        ("LR", "curated"):     "Curated-LR",
        ("LR", "logical"):     "Logical-LR",
        ("TTPD", "curated"):   "Curated-TTPD",
        ("TTPD", "logical"):   "Logical-TTPD",
    }

    for probe in ["LR", "TTPD"]:
        for dtype in ["curated", "logical"]:
            mean = res[dtype][f"{probe}_mean"]
            std  = res[dtype][f"{probe}_std"]
            plt.plot(
                layers, mean,
                linestyle=STYLE[dtype],
                color=COLOR[probe],
                linewidth=2.5,
                label=LABEL[(probe, dtype)],
                marker="o" if probe == "LR" else "s",
                markersize=4,
            )
            plt.fill_between(
                layers, mean-std, mean+std,
                color=COLOR[probe], alpha=0.15
            )

    # 关键层灰带
    plt.axvspan(KEY_LAYER-0.5, KEY_LAYER+0.5,
                color="gray", alpha=0.25, label=f"Key Layer {KEY_LAYER}")

    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Probing Accuracy (%)", fontsize=14)
    plt.title(f"{model_family}-{model_size}-{model_type}: Logical Probing Accuracy",
              fontsize=15)
    plt.ylim(45, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, ncol=2, loc="lower right")
    plt.tight_layout()

    save_path = save_dir / f"{model_family}_{model_size}_{model_type}_logic_vs_curated_peak.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure saved → {save_path}")


def find_peak(res):
    print("\n*** Peak-layer summary (means %) ***")
    for dtype in ["curated", "logical"]:
        for probe in ["LR", "TTPD"]:
            peak = np.argmax(res[dtype][f"{probe}_mean"])
            val  = res[dtype][f"{probe}_mean"][peak]
            print(f"{dtype.capitalize():8s} {probe:5s} : Layer {peak:2d} @ {val:5.2f}")


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main():
    res = load_data(MODEL_FAMILY, MODEL_SIZE, MODEL_TYPE)
    plot_curves(res, MODEL_FAMILY, MODEL_SIZE, MODEL_TYPE, SAVE_DIR)
    find_peak(res)

if __name__ == "__main__":
    main()
