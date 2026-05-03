"""
Image transformation pipelines for the Oxford-IIIT Pet experiments.

This file defines separate transforms for training, validation, and testing.
Training transforms may include data augmentation, while validation and test
transforms should remain deterministic.
"""

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224, use_augmentation: bool = True):
    """
    Return the transformation pipeline for training images.

    Args:
        image_size: Target input size for the pretrained ResNet model.
        use_augmentation: Whether to apply random data augmentation.

    Returns:
        A torchvision transform pipeline.
    """
    if use_augmentation:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = 224):
    """
    Return the deterministic transformation pipeline for validation and testing.

    Args:
        image_size: Target input size for the pretrained ResNet model.

    Returns:
        A torchvision transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )