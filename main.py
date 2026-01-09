import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from datetime import datetime
from utils import save_checkpoint, get_device, prepare_thermal_dataset

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.palm_disease_detector import PalmDiseaseDetector
from data.dataset import get_dataloaders
from train import train, fine_tune, pick_better_model_results


def main():
    # Extract and split dataset (idempotent data preparation utility)
    prepare_thermal_dataset()

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

    # Fine-tuning phase: Unfreeze backbone and train for minimum 20 epochs with 1/10 learning rate
    if training_results["best_model_state"] is not None:
        # Load the best model state
        model.load_state_dict(training_results["best_model_state"])
        # Fine-tune with minimum 20 epochs and patience of 5
        fine_tune_results = fine_tune(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            min_epochs=20,
            patience=5,
        )

        # Compare both models and pick the better one using shared helper
        best_results, source = pick_better_model_results(
            primary_results=training_results,
            secondary_results=fine_tune_results,
        )

        if source == "primary":
            print("\nInitial training model is better. Using initial training model.")
        else:
            print("\nFine-tuned model is better. Using fine-tuned model.")

        final_model_state = best_results["best_model_state"]
        final_val_acc = best_results["best_val_acc"]
        final_val_loss = best_results["best_val_loss"]
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
        print(f"\nFinal best model saved: {checkpoint_filename}")
    else:
        print("No best model found to save.")


if __name__ == "__main__":
    main()
