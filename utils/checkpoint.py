import torch
import os
from pathlib import Path

# Default directories
SAVE_DIR = "checkpoints"
# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)


def save_checkpoint(model, filename):
    filepath = os.path.join(SAVE_DIR, filename)

    checkpoint = {
        "model_state_dict": model.state_dict(),  # All model weights
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find all checkpoint files matching the pattern
    checkpoints = list(checkpoint_path.glob("best_model_*.pth"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    # Sort by modification time (most recent first)
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

    return str(latest_checkpoint)
