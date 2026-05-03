"""
Create reproducible dataset splits for the Oxford-IIIT Pet experiments.

This script creates:
1. A stratified train/validation split from the official trainval split.
2. Stratified label-budget subsets: 100%, 10%, and 1%.
3. A class-imbalanced training split where cat breeds are reduced to 20%.

Run from the project root:

    python scripts/make_splits.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.data import get_oxford_pet_dataset, get_targets, get_class_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"


CAT_BREEDS = {
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def group_indices_by_class(indices, labels):
    grouped = defaultdict(list)
    for idx in indices:
        grouped[int(labels[idx])].append(int(idx))
    return grouped


def make_label_budget_subset(indices, labels, fraction, seed):
    """
    Create a stratified subset of the given indices.

    Each class keeps approximately the requested fraction. For very small
    fractions, at least one sample per class is kept if the class exists.
    """
    rng = random.Random(seed)
    grouped = group_indices_by_class(indices, labels)

    subset = []
    for class_id, class_indices in grouped.items():
        class_indices = list(class_indices)
        rng.shuffle(class_indices)

        if fraction >= 1.0:
            keep_count = len(class_indices)
        else:
            keep_count = max(1, int(round(len(class_indices) * fraction)))

        subset.extend(class_indices[:keep_count])

    subset = sorted(subset)
    return subset


def make_cat_imbalanced_subset(indices, labels, class_names, keep_fraction=0.20, seed=42):
    """
    Reduce cat-breed samples to a fixed fraction while keeping dog-breed samples.

    Args:
        indices: Training indices.
        labels: Integer labels for the full trainval dataset.
        class_names: List of class names.
        keep_fraction: Fraction of cat-breed training samples to keep.
        seed: Random seed.

    Returns:
        A sorted list of selected indices.
    """
    rng = random.Random(seed)
    grouped = group_indices_by_class(indices, labels)

    selected = []

    for class_id, class_indices in grouped.items():
        class_name = class_names[class_id]
        class_indices = list(class_indices)
        rng.shuffle(class_indices)

        if class_name in CAT_BREEDS:
            keep_count = max(1, int(round(len(class_indices) * keep_fraction)))
            selected.extend(class_indices[:keep_count])
        else:
            selected.extend(class_indices)

    return sorted(selected)


def summarize_split(name, indices, labels, class_names):
    grouped = group_indices_by_class(indices, labels)
    summary = {
        "split_name": name,
        "num_samples": len(indices),
        "per_class_counts": {
            class_names[class_id]: len(class_indices)
            for class_id, class_indices in sorted(grouped.items())
        },
    }
    return summary


def main():
    seed = 42
    val_fraction = 0.10
    label_budgets = [1.0, 0.1, 0.01]

    np.random.seed(seed)
    random.seed(seed)

    dataset = get_oxford_pet_dataset(
        root=str(PROJECT_ROOT / "data"),
        split="trainval",
        image_size=224,
        train=True,
        use_augmentation=False,
        download=True,
    )

    labels = np.array(get_targets(dataset))
    class_names = get_class_names(dataset)

    all_indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    train_indices = sorted([int(i) for i in train_indices])
    val_indices = sorted([int(i) for i in val_indices])

    save_json(train_indices, SPLIT_DIR / "train_seed42.json")
    save_json(val_indices, SPLIT_DIR / "val_seed42.json")

    summaries = []
    summaries.append(summarize_split("train", train_indices, labels, class_names))
    summaries.append(summarize_split("val", val_indices, labels, class_names))

    for fraction in label_budgets:
        subset = make_label_budget_subset(
            indices=train_indices,
            labels=labels,
            fraction=fraction,
            seed=seed,
        )

        if fraction == 1.0:
            name = "train_100_seed42"
        elif fraction == 0.1:
            name = "train_10_seed42"
        elif fraction == 0.01:
            name = "train_1_seed42"
        else:
            name = f"train_{fraction}_seed42"

        save_json(subset, SPLIT_DIR / f"{name}.json")
        summaries.append(summarize_split(name, subset, labels, class_names))

    imbalanced = make_cat_imbalanced_subset(
        indices=train_indices,
        labels=labels,
        class_names=class_names,
        keep_fraction=0.20,
        seed=seed,
    )

    save_json(imbalanced, SPLIT_DIR / "train_imbalanced_cat20_seed42.json")
    summaries.append(
        summarize_split("train_imbalanced_cat20_seed42", imbalanced, labels, class_names)
    )

    save_json(
        {
            "seed": seed,
            "validation_fraction": val_fraction,
            "class_names": class_names,
            "summaries": summaries,
        },
        SPLIT_DIR / "split_summary_seed42.json",
    )

    print(f"Saved splits to: {SPLIT_DIR}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print("Done.")


if __name__ == "__main__":
    main()