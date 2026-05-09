"""
Cat-vs-dog binary classification DataLoader.
Used only for the §1.1 sanity check experiment and does not affect the 37-class main process. 
The OxfordIIITPet dataset provided by torchvision supports the parameter "target_types" set to "binary-category", and it directly returns 0 (for cat) or 1 (for dog), without the need for manual mapping.
"""

from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from src.transforms import get_eval_transform, get_train_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def get_binary_dataloaders(
    root: str | Path = DEFAULT_DATA_ROOT,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    use_augmentation: bool = False,
    download: bool = False,
) -> dict:
    """
    Return the train/test DataLoader for the cat-vs-dog dataset.
    Use the official trainval set for training and the official test set for evaluation.
    """
    train_dataset = OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="binary-category",
        transform=get_train_transform(image_size, use_augmentation),
        download=download,
    )
    test_dataset = OxfordIIITPet(
        root=str(root),
        split="test",
        target_types="binary-category",
        transform=get_eval_transform(image_size),
        download=download,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "num_classes": 2,
        "class_names": ["cat", "dog"],
    }