"""
Data loading utilities for the Oxford-IIIT Pet dataset.

Pengyu Wang is responsible for the data pipeline, including dataset loading,
stratified validation splits, limited-label subsets, class-imbalance setups,
and reusable PyTorch DataLoaders for all experiments.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import OxfordIIITPet

from src.transforms import get_train_transform, get_eval_transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SPLIT_DIR = DEFAULT_DATA_ROOT / "splits"


def get_oxford_pet_dataset(
    root: str = "./data",
    split: str = "trainval",
    image_size: int = 224,
    train: bool = True,
    use_augmentation: bool = True,
    download: bool = True,
):
    """
    Load the Oxford-IIIT Pet dataset.

    Args:
        root: Root directory for storing the dataset.
        split: Dataset split. For torchvision OxfordIIITPet, common values are
            "trainval" and "test".
        image_size: Input image size for the model.
        train: Whether this dataset is used for training.
        use_augmentation: Whether to use data augmentation for training.
        download: Whether to download the dataset if it is not found locally.

    Returns:
        A torchvision OxfordIIITPet dataset object.
    """
    root_path = Path(root)

    if train:
        transform = get_train_transform(
            image_size=image_size,
            use_augmentation=use_augmentation,
        )
    else:
        transform = get_eval_transform(image_size=image_size)

    dataset = OxfordIIITPet(
        root=str(root_path),
        split=split,
        target_types="category",
        transform=transform,
        download=download,
    )

    return dataset


def get_class_names(dataset) -> list[str]:
    """
    Return class names from the Oxford-IIIT Pet dataset.

    Args:
        dataset: OxfordIIITPet dataset object.

    Returns:
        A list of class names.
    """
    return list(dataset.classes)


def get_targets(dataset) -> list[int]:
    """
    Return integer class labels for all samples in the dataset.

    Args:
        dataset: OxfordIIITPet dataset object.

    Returns:
        A list of integer labels.
    """
    return [int(label) for label in dataset._labels]


def load_split_indices(split_name: str, split_dir: Optional[str | Path] = None) -> list[int]:
    """
    Load split indices from a JSON file.

    Args:
        split_name: Name of the split file without the .json extension.
            Example: "train_10_seed42".
        split_dir: Directory containing split JSON files.

    Returns:
        A list of integer dataset indices.
    """
    if split_dir is None:
        split_dir = DEFAULT_SPLIT_DIR

    split_path = Path(split_dir) / f"{split_name}.json"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Please run `python -m scripts.make_splits` first."
        )

    with split_path.open("r", encoding="utf-8") as f:
        indices = json.load(f)

    return [int(i) for i in indices]


def create_subset(dataset, split_name: str, split_dir: Optional[str | Path] = None) -> Subset:
    """
    Create a torch.utils.data.Subset from a named split file.

    Args:
        dataset: Dataset object.
        split_name: Name of the split file without the .json extension.
        split_dir: Directory containing split JSON files.

    Returns:
        A PyTorch Subset object.
    """
    indices = load_split_indices(split_name=split_name, split_dir=split_dir)
    return Subset(dataset, indices)


def get_subset_targets(dataset, subset: Subset) -> list[int]:
    """
    Get labels for all samples in a Subset.

    Args:
        dataset: The original full dataset.
        subset: A PyTorch Subset object.

    Returns:
        A list of integer labels for the subset.
    """
    labels = get_targets(dataset)
    return [int(labels[idx]) for idx in subset.indices]


def compute_class_weights(
    subset_targets: list[int],
    num_classes: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy.

    Args:
        subset_targets: Labels in the training subset.
        num_classes: Number of classes.
        normalize: Whether to normalize weights so their mean is close to 1.

    Returns:
        A tensor of shape [num_classes].
    """
    counts = Counter(subset_targets)
    total = len(subset_targets)

    weights = []
    for class_id in range(num_classes):
        class_count = counts.get(class_id, 0)
        if class_count == 0:
            weight = 0.0
        else:
            weight = total / (num_classes * class_count)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    if normalize:
        nonzero = weights > 0
        weights[nonzero] = weights[nonzero] / weights[nonzero].mean()

    return weights


def create_weighted_sampler(subset_targets: list[int]) -> WeightedRandomSampler:
    """
    Create a weighted sampler for oversampling minority classes.

    Args:
        subset_targets: Labels in the training subset.

    Returns:
        A WeightedRandomSampler.
    """
    counts = Counter(subset_targets)

    sample_weights = []
    for label in subset_targets:
        sample_weights.append(1.0 / counts[int(label)])

    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.

    Args:
        dataset: Dataset or Subset.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        sampler: Optional sampler. If a sampler is used, shuffle must be False.
        num_workers: Number of dataloader workers.

    Returns:
        A DataLoader object.
    """
    if sampler is not None and shuffle:
        raise ValueError("shuffle must be False when sampler is provided.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloaders_for_split(
    train_split_name: str = "train_100_seed42",
    val_split_name: str = "val_seed42",
    root: str | Path = DEFAULT_DATA_ROOT,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    use_augmentation: bool = True,
    use_oversampling: bool = False,
    download: bool = True,
):
    """
    Create train and validation dataloaders for a named training split.

    Args:
        train_split_name: Name of the training split JSON file without .json.
        val_split_name: Name of the validation split JSON file without .json.
        root: Dataset root directory.
        split_dir: Directory containing split JSON files.
        image_size: Image size.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.
        use_augmentation: Whether to apply training augmentation.
        use_oversampling: Whether to use weighted random sampling for training.
        download: Whether to download data if missing.

    Returns:
        A dictionary containing datasets, dataloaders, class names, and class weights.
    """
    trainval_train_transform = get_oxford_pet_dataset(
        root=str(root),
        split="trainval",
        image_size=image_size,
        train=True,
        use_augmentation=use_augmentation,
        download=download,
    )

    trainval_eval_transform = get_oxford_pet_dataset(
        root=str(root),
        split="trainval",
        image_size=image_size,
        train=False,
        use_augmentation=False,
        download=download,
    )

    class_names = get_class_names(trainval_train_transform)
    num_classes = len(class_names)

    train_subset = create_subset(
        dataset=trainval_train_transform,
        split_name=train_split_name,
        split_dir=split_dir,
    )

    val_subset = create_subset(
        dataset=trainval_eval_transform,
        split_name=val_split_name,
        split_dir=split_dir,
    )

    train_targets = get_subset_targets(trainval_train_transform, train_subset)
    class_weights = compute_class_weights(
        subset_targets=train_targets,
        num_classes=num_classes,
        normalize=True,
    )

    if use_oversampling:
        sampler = create_weighted_sampler(train_targets)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = create_dataloader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
    )

    return {
        "train_dataset": train_subset,
        "val_dataset": val_subset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "class_names": class_names,
        "num_classes": num_classes,
        "class_weights": class_weights,
    }


def get_test_dataloader(
    root: str | Path = DEFAULT_DATA_ROOT,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
):
    """
    Create a test dataloader for the official Oxford-IIIT Pet test split.

    Args:
        root: Dataset root directory.
        image_size: Image size.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.
        download: Whether to download data if missing.

    Returns:
        A dictionary containing the test dataset and dataloader.
    """
    test_dataset = get_oxford_pet_dataset(
        root=str(root),
        split="test",
        image_size=image_size,
        train=False,
        use_augmentation=False,
        download=download,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
    )

    return {
        "test_dataset": test_dataset,
        "test_loader": test_loader,
        "class_names": get_class_names(test_dataset),
        "num_classes": len(get_class_names(test_dataset)),
    }