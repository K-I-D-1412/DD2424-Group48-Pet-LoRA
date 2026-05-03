"""
Data loading utilities for the Oxford-IIIT Pet dataset.

Pengyu Wang is responsible for the data pipeline, including dataset loading,
stratified validation splits, limited-label subsets, and class-imbalance setups.
"""

from pathlib import Path
from typing import Optional

from torchvision.datasets import OxfordIIITPet

from src.transforms import get_train_transform, get_eval_transform


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
    return list(dataset._labels)