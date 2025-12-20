import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np


class ThermalPalmDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.images = []
        self.labels = []

        # 0 = healthy, 1 = sick
        healthy_dir = self.root_dir / "healthy"
        sick_dir = self.root_dir / "sick"

        if healthy_dir.exists():
            for img_path in list(healthy_dir.glob("*.jpg")) + list(
                healthy_dir.glob("*.png")
            ):
                self.images.append(str(img_path))
                self.labels.append(0)  # healthy

        if sick_dir.exists():
            for img_path in list(sick_dir.glob("*.jpg")) + list(sick_dir.glob("*.png")):
                self.images.append(str(img_path))
                self.labels.append(1)  # sick

        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"Healthy: {self.labels.count(0)}, Sick: {self.labels.count(1)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode="train"):
    if mode == "train":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (256, 256)
                ),  # Larger resize => allow random crop variation
                transforms.RandomCrop(224),  # Random crop (augmentation)
                transforms.RandomHorizontalFlip(
                    p=0.5
                ),  # Horizontal flip (augmentation)
                transforms.RandomRotation(degrees=15),  # Light rotation (augmentation)
                transforms.ToTensor(),  # Convert to Tensor (0-1)
                transforms.Normalize(  # Normalization (ImageNet mean and std)
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Direct resize to 224x224
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


def get_dataloaders(data_root="data", batch_size=32, num_workers=4):
    # Create Datasets
    train_dataset = ThermalPalmDataset(
        root_dir=os.path.join(data_root, "train"), transform=get_transforms("train")
    )

    val_dataset = ThermalPalmDataset(
        root_dir=os.path.join(data_root, "val"), transform=get_transforms("val")
    )

    test_dataset = ThermalPalmDataset(
        root_dir=os.path.join(data_root, "test"), transform=get_transforms("test")
    )

    if len(train_dataset) == 0:
        raise ValueError(f"No training data found in {data_root}/train/")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle in training
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # Faster on GPU
    )

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
