import torch
import torch.nn as nn
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.palm_disease_detector import PalmDiseaseDetector
from data.dataset import get_dataloaders
from utils.checkpoint import get_latest_checkpoint


CLASS_NAMES = ["Healthy", "Sick"]


def get_device(device_preference="cuda"):
    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if device_preference == "cuda":
            print("[WARNING] CUDA not available, using CPU")
        else:
            print("[INFO] Using CPU")
    return device


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading model from: {checkpoint_path}")

    # Create model (same architecture as training)
    model = PalmDiseaseDetector(freeze_backbone=True)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode

    # Extract checkpoint info
    epoch = checkpoint.get("epoch", "N/A")
    accuracy = checkpoint.get("accuracy", None)

    print(f"[INFO] Model loaded successfully!")
    print(f"[INFO] Trained for {epoch} epochs")
    if accuracy is not None:
        print(f"[INFO] Best validation accuracy: {accuracy:.2f}%")

    return model, checkpoint


def test_model(model, test_loader, criterion, device):
    model.eval()  # Evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    # Per-class statistics
    class_correct = [0, 0]  # [healthy, sick]
    class_total = [0, 0]  # [healthy, sick]

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).long()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()

            # Get predicted class (0=healthy, 1=sick)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predictions[i].item() == label:
                    class_correct[label] += 1

            # Update progress bar
            current_loss = running_loss / len(test_loader)
            current_acc = 100 * correct / total
            pbar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
            )

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    # Per-class accuracies
    healthy_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
    sick_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0

    per_class_stats = {
        "healthy": {
            "correct": class_correct[0],
            "total": class_total[0],
            "accuracy": healthy_acc,
        },
        "sick": {
            "correct": class_correct[1],
            "total": class_total[1],
            "accuracy": sick_acc,
        },
    }

    return avg_loss, accuracy, per_class_stats


def main():
    # Default values
    data_dir = "data"
    batch_size = 32
    device_preference = "cuda"

    parser = argparse.ArgumentParser(
        description="Test Palm Disease Detector Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test using latest checkpoint
  python test_model.py
        """,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda, auto-falls back to CPU if CUDA unavailable)",
    )

    args = parser.parse_args()

    # Override defaults with command line arguments if provided
    if args.device is not None:
        device_preference = args.device

    # Get device
    device = get_device(device_preference)

    # Get latest checkpoint
    try:
        model_path = get_latest_checkpoint()
        print(f"[INFO] Using latest checkpoint: {model_path}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Load test dataset
    print(f"\n[INFO] Loading test dataset from: {data_dir}/test/")
    _, _, test_loader = get_dataloaders(
        data_root=data_dir,
        batch_size=batch_size,
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    if test_loader is None:
        print(f"[ERROR] No test data found in {data_dir}/test/")
        return

    # Load model
    model, checkpoint = load_model(model_path, device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Run test
    print(f"\n[INFO] Starting test evaluation...")
    print(f"{'='*60}\n")

    try:
        test_loss, test_acc, per_class_stats = test_model(
            model, test_loader, criterion, device
        )

        # Print results
        print(f"\n{'='*60}")
        print("Test Results:")
        print(f"{'='*60}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"\nPer-Class Accuracy:")
        print(
            f"  Healthy: {per_class_stats['healthy']['accuracy']:.2f}% ({per_class_stats['healthy']['correct']}/{per_class_stats['healthy']['total']})"
        )
        print(
            f"  Sick: {per_class_stats['sick']['accuracy']:.2f}% ({per_class_stats['sick']['correct']}/{per_class_stats['sick']['total']})"
        )
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
