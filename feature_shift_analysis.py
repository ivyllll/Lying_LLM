import os
import torch as t
from sae_utils import SAEWrapper
from pathlib import Path
import numpy as np
import tqdm
from multiprocessing import Process
import multiprocessing as mp
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------
#                   Configuration 
# ---------------------------------------------------------------------
MODEL_RELEASE = "llama_scope_lxr_32x"
LAYERS_TO_ANALYZE = list(range(32))
DEVICE = "cuda:0"
ACTS_DIR = Path("acts")
PROMPT_TYPES = ["truthful", "deceptive", "neutral"]
DATASETS = ["counterfact_true_false"]  # "common_claim_true_false"
# "cities", "animal_class", "element_symb", "facts", "inventors", "sp_en_trans"
OUTPUT_DIR = Path("feature_shift_results")
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPT_PAIRS = [
    ("truthful", "deceptive"),
    ("truthful", "neutral"),
    ("neutral", "deceptive"),
]

# ---------------------------------------------------------------------
#                         Helper Functions
# ---------------------------------------------------------------------
def load_batch(prompt_type: str, dataset: str, layer: int, idx: int) -> t.Tensor:
    path = ACTS_DIR / f"acts_{prompt_type}_prompt" / "Llama3.1" / "8B" / "chat" / dataset / f"layer_{layer}_{idx}.pt"
    return t.load(path, map_location=DEVICE).float()


def compute_cosine(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """
    Measures how aligned two SAE feature vectors are in feature space, 
    helping to evaluate semantic similarity across inputs (directional similarity)
    """
    x_norm = t.nn.functional.normalize(x.float(), dim=-1)
    y_norm = t.nn.functional.normalize(y.float(), dim=-1)
    return (x_norm * y_norm).sum(dim=-1)  # (batch,)

def compute_overlap(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    """
    Measures shared activations between SAE feature vectors
    """
    overlap = ((x != 0) & (y != 0)).float().sum(dim=1)
    union = ((x != 0) | (y != 0)).float().sum(dim=1)
    return overlap / (union + 1e-6)

def compute_l1(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return (x - y).abs().mean(dim=1)

def compute_l2(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return ((x - y) ** 2).sum(dim=1).sqrt()

# ---------------------------------------------------------------------
#                        Per-Layer Worker
# ---------------------------------------------------------------------
def run_single_layer(dataset: str, layer: int):
    print(f"\nAnalyzing Dataset {dataset} - Layer {layer}...")

    SAE_ID = f"l{layer}r_32x"
    sae = SAEWrapper(release=MODEL_RELEASE, sae_id=SAE_ID, device=DEVICE)

    sample_dir = ACTS_DIR / "acts_truthful_prompt" / "Llama3.1" / "8B" / "chat" / dataset
    indexes = sorted(int(p.stem.split("_")[-1]) for p in sample_dir.glob(f"layer_{layer}_*.pt"))

    for a, b in PROMPT_PAIRS:
        subdir = OUTPUT_DIR / f"{a}_vs_{b}" / dataset
        subdir.mkdir(exist_ok=True, parents=True)

        l2_list, cosine_list, overlap_list = [], [], []
        for idx in tqdm.tqdm(indexes, desc=f"Dataset {dataset} Layer {layer} [{a} vs {b}]"):
            acts_a = load_batch(a, dataset, layer, idx).float()
            acts_b = load_batch(b, dataset, layer, idx).float()

            z_a = sae.encode(acts_a).float()
            z_b = sae.encode(acts_b).float()

            l2 = compute_l2(z_a, z_b)
            cosine = compute_cosine(z_a, z_b)
            overlap = compute_overlap(z_a, z_b)

            l2_list.append(l2.detach().cpu().numpy())
            cosine_list.append(cosine.detach().cpu().numpy())
            overlap_list.append(overlap.detach().cpu().numpy())

        np.savez(
            subdir / f"layer_{layer}_shifts.npz",
            l2=np.concatenate(l2_list),
            cosine=np.concatenate(cosine_list),
            overlap=np.concatenate(overlap_list),
        )

    del sae
    t.cuda.empty_cache()

# ---------------------------------------------------------------------
#                         Main Launcher
# ---------------------------------------------------------------------
def main():
    for dataset in DATASETS:
        for layer in LAYERS_TO_ANALYZE:
            p = Process(target=run_single_layer, args=(dataset, layer))
            p.start()
            p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
