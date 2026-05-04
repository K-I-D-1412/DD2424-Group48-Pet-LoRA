"""
Quick sanity check for the data loading pipeline.
"""

from src.data import get_dataloaders_for_split, get_test_dataloader


def main():
    data = get_dataloaders_for_split(
        train_split_name="train_10_seed42",
        val_split_name="val_seed42",
        batch_size=8,
        num_workers=0,
        use_augmentation=True,
        use_oversampling=False,
        download=False,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]

    images, labels = next(iter(train_loader))

    print("Train batch image shape:", images.shape)
    print("Train batch label shape:", labels.shape)
    print("Number of classes:", data["num_classes"])
    print("First 5 class names:", data["class_names"][:5])
    print("Class weights shape:", data["class_weights"].shape)

    val_images, val_labels = next(iter(val_loader))
    print("Validation batch image shape:", val_images.shape)
    print("Validation batch label shape:", val_labels.shape)

    test_data = get_test_dataloader(
        batch_size=8,
        num_workers=0,
        download=False,
    )

    test_images, test_labels = next(iter(test_data["test_loader"]))
    print("Test batch image shape:", test_images.shape)
    print("Test batch label shape:", test_labels.shape)

    print("Dataloader check passed.")


if __name__ == "__main__":
    main()