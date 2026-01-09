import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np

# Dataset constants
BATCH_SIZE = 32
DEFAULT_DATA_ROOT = "data"


class ThermalPalmDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_flipped=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.include_flipped = include_flipped  # Whether to include flipped versions

        # Store tuples of (img_path, label, should_flip)
        self.data = []

        # 0 = healthy, 1 = sick
        healthy_dir = self.root_dir / "healthy"
        sick_dir = self.root_dir / "sick"

        # Load original images
        if healthy_dir.exists():
            for img_path in list(healthy_dir.glob("*.jpg")) + list(
                healthy_dir.glob("*.png")
            ):
                # Add original image
                self.data.append(
                    (str(img_path), 0, False)
                )  # (path, label, should_flip)

                # Add flipped version if enabled
                if self.include_flipped:
                    self.data.append(
                        (str(img_path), 0, True)
                    )  # (path, label, should_flip)

        if sick_dir.exists():
            for img_path in list(sick_dir.glob("*.jpg")) + list(sick_dir.glob("*.png")):
                # Add original image
                self.data.append(
                    (str(img_path), 1, False)
                )  # (path, label, should_flip)

                # Add flipped version if enabled
                if self.include_flipped:
                    self.data.append(
                        (str(img_path), 1, True)
                    )  # (path, label, should_flip)

        original_count = len(self.data) // 2 if self.include_flipped else len(self.data)
        healthy_count = sum(1 for _, label, _ in self.data if label == 0)
        sick_count = sum(1 for _, label, _ in self.data if label == 1)
        print(
            f"Loaded {len(self.data)} images from {root_dir} ({original_count} original + {original_count if self.include_flipped else 0} flipped)"
        )
        print(f"Healthy: {healthy_count}, Sick: {sick_count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, should_flip = self.data[idx]

        image = Image.open(img_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply horizontal flip if this is a flipped version
        if should_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode="train"):
    transform_list = []

    if mode == "train":
        transform_list.extend(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Larger resize => allow random crop variation
                transforms.RandomCrop(224),  # Random crop (augmentation)
            ]
        )
    else:
        transform_list.append(transforms.Resize((224, 224)))  # Direct resize to 224x224

    # Common transforms for both train and val/test
    transform_list.extend(
        [
            transforms.ToTensor(),  # Convert to Tensor (0-1)
            transforms.Normalize(  # Normalization (ImageNet mean and std)
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return transforms.Compose(transform_list)


def get_dataloaders(data_root=DEFAULT_DATA_ROOT, batch_size=BATCH_SIZE, num_workers=4):
    # Create Datasets
    # Include flipped images only for training (augmentation)
    train_dataset = ThermalPalmDataset(
        root_dir=os.path.join(data_root, "train"),
        transform=get_transforms("train"),
        include_flipped=True,  # Double the training data with flipped images
    )

    # Validation also includes flipped images (same as training)
    val_dataset = ThermalPalmDataset(
        root_dir=os.path.join(data_root, "val"),
        transform=get_transforms("val"),
        include_flipped=True,  # Double the validation data with flipped images
    )

    # Test also includes flipped images (same as training)
    test_dataset = ThermalPalmDataset(
        root_dir=os.path.join(data_root, "test"),
        transform=get_transforms("test"),
        include_flipped=True,
    )

    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle in training
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,  # Faster on GPU
        )
    else:
        print(f"No training data found in {data_root}/train/")
        train_loader = None

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle in validation
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    else:
        print(f"No validation data found in {data_root}/val/")
        val_loader = None

    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    else:
        print(f"No test data found in {data_root}/test/")
        test_loader = None

    return train_loader, val_loader, test_loader
