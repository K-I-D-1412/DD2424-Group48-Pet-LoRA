"""Representation analysis via 2D embeddings of test logits.

Penultimate-feature extraction would need to re-run the backbone,
but ``best_predictions.npz`` already stores the 37-dim test logits.
The logit geometry differs visibly across strategies (linear probe,
full fine-tune, LoRA, gradual unfreeze), which is enough to support
a "representation analysis" discussion.

Generates:
  * t-SNE side-by-side grid, coloured by ground-truth class.
  * t-SNE side-by-side grid, coloured by cat / dog (binary).
  * PCA equivalents (cheap sanity check + reproducible geometry).

Run: python -m scripts.plot_representation_analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "results" / "figures" / "representation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx",
}

RUNS = [
    ("resnet18_linear_probe_100_seed42",   "Linear probe  (0.846)"),
    ("resnet18_finetune_100_seed42",       "Full fine-tune  (0.868)"),
    ("resnet18_gradual_unfreeze_100_seed42","Gradual unfreeze  (0.874)"),
    ("resnet18_lora_layer4_r4_100_seed42", "Layer4-LoRA r=4  (0.850)"),
]

SEED = 42


def load_logits(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = np.load(EXP_DIR / name / "best_predictions.npz", allow_pickle=True)
    return (
        d["test_logits"].astype(np.float32),
        d["test_labels"].astype(np.int64),
        list(d["class_names"]),
    )


def _build_class_cmap(n: int) -> ListedColormap:
    base = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
    )
    return ListedColormap(base[:n])


def _scatter_by_class(ax, emb, labels, class_names, title):
    cmap = _build_class_cmap(len(class_names))
    ax.scatter(
        emb[:, 0], emb[:, 1],
        c=labels, cmap=cmap, s=8, alpha=0.75,
        linewidths=0, edgecolors="none",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def _scatter_by_species(ax, emb, labels, class_names, title):
    is_cat = np.array([class_names[i] in CAT_BREEDS for i in labels])
    ax.scatter(emb[is_cat, 0],  emb[is_cat, 1],
               c="#c44e52", s=10, alpha=0.6, label=f"Cat ({is_cat.sum()})",
               linewidths=0)
    ax.scatter(emb[~is_cat, 0], emb[~is_cat, 1],
               c="#4c72b0", s=10, alpha=0.6, label=f"Dog ({(~is_cat).sum()})",
               linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def embed_tsne(x: np.ndarray) -> np.ndarray:
    return TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto", random_state=SEED,
    ).fit_transform(x)


def embed_pca(x: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=SEED).fit_transform(x)


def make_grid(
    embeddings: list[tuple[np.ndarray, np.ndarray, list[str], str]],
    color_fn,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (emb, labels, class_names, sub_title) in zip(axes.flat, embeddings):
        color_fn(ax, emb, labels, class_names, sub_title)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(ROOT)}")


def main() -> None:
    print(f"Writing figures to {OUT_DIR.relative_to(ROOT)}/")
    tsne_embeddings, pca_embeddings = [], []
    for run, label in RUNS:
        logits, labels, class_names = load_logits(run)
        print(f"  embedding {run}  shape={logits.shape}")
        tsne_embeddings.append((embed_tsne(logits), labels, class_names, label))
        pca_embeddings.append((embed_pca(logits), labels, class_names, label))

    make_grid(
        tsne_embeddings, _scatter_by_class,
        "t-SNE of test logits — coloured by ground-truth class (37 classes, 100% labels)",
        OUT_DIR / "tsne_by_class.png",
    )
    make_grid(
        tsne_embeddings, _scatter_by_species,
        "t-SNE of test logits — coloured by species (cat vs dog, 100% labels)",
        OUT_DIR / "tsne_by_species.png",
    )
    make_grid(
        pca_embeddings, _scatter_by_class,
        "PCA of test logits — coloured by class (100% labels)",
        OUT_DIR / "pca_by_class.png",
    )
    make_grid(
        pca_embeddings, _scatter_by_species,
        "PCA of test logits — coloured by species (100% labels)",
        OUT_DIR / "pca_by_species.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
