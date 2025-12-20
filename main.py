import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from pathlib import Path
from datetime import datetime
from utils import save_checkpoint

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.palm_disease_detector import PalmDiseaseDetector
from data.dataset import get_dataloaders
from train import train


def main():
    # Default values
    data_dir = "data"
    epochs = 50
    batch_size = 32
    lr = 1e-4
    freeze_backbone = True  # Set to False for full fine-tuning

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders(
        data_root=data_dir,
        batch_size=batch_size,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    # Require validation data
    if val_loader is None:
        raise ValueError(
            "No validation data found. Validation data is required for training."
        )

    # Create model
    print("Creating model...")
    model = PalmDiseaseDetector(freeze_backbone=freeze_backbone)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    patience = 10  # Check for improvement every X epochs
    training_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        patience=patience,
    )

    # Save the best model at the end with timestamp and metrics in filename
    if training_results["best_model_state"] is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"best_model_{timestamp}_val{training_results['best_val_acc']:.2f}_train{training_results['best_train_acc']:.2f}.pth"

        # Create a temporary model to save the checkpoint
        temp_model = PalmDiseaseDetector(freeze_backbone=freeze_backbone)
        temp_model.load_state_dict(training_results["best_model_state"])
        temp_optimizer = optim.Adam(temp_model.parameters(), lr=lr)
        temp_optimizer.load_state_dict(training_results["best_optimizer_state"])

        save_checkpoint(
            temp_model,
            temp_optimizer,
            training_results["best_epoch"],
            training_results["best_val_loss"],
            training_results["best_val_acc"],
            checkpoint_filename,
        )
        print(f"Best model saved: {checkpoint_filename}")
    else:
        print("No best model found to save.")


if __name__ == "__main__":
    main()
