import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from datetime import datetime
from utils import save_checkpoint, get_device

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.palm_disease_detector import PalmDiseaseDetector
from data.dataset import get_dataloaders
from train import train, fine_tune


def main():
    # Device
    device = get_device()

    # Load data
    print("Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders(
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    # Validate required datasets exist
    if train_loader is None:
        raise ValueError("Training data is required but not found in data/train/")
    if val_loader is None:
        raise ValueError("Validation data is required but not found in data/val/")

    # Create model (backbone frozen by default)
    print("Creating model...")
    model = PalmDiseaseDetector()
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Train the model (with frozen backbone)
    print("Starting Training")
    training_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        patience=10,
    )

    # Fine-tuning phase: Unfreeze backbone and train for 10 more epochs with 1/10 learning rate
    if training_results["best_model_state"] is not None:
        print("Loading best model for fine-tuning...")

        # Load the best model state
        model.load_state_dict(training_results["best_model_state"])

        # Fine-tune with minimum 10 epochs and patience of 5
        fine_tune_results = fine_tune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            min_epochs=10,
            patience=10,
        )

        # Use the fine-tuned model
        final_model_state = fine_tune_results["best_model_state"]
        final_val_acc = fine_tune_results["best_val_acc"]
        final_val_loss = fine_tune_results["best_val_loss"]
    else:
        print("No best model found from initial training. Skipping fine-tuning.")
        final_model_state = None
        final_val_acc = 0.0
        final_val_loss = float("inf")

    # Save the final best model
    if final_model_state is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"best_model_{timestamp}_valAcc{final_val_acc:.2f}_valLoss{final_val_loss:.4f}.pth"

        # Load the final model state into the existing model
        model.load_state_dict(final_model_state)

        save_checkpoint(
            model,
            checkpoint_filename,
        )
        print(f"Final best model saved: {checkpoint_filename}")
    else:
        print("No best model found to save.")


if __name__ == "__main__":
    main()
