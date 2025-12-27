import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.palm_disease_detector import PalmDiseaseDetector
from data.dataset import get_dataloaders
from utils.checkpoint import get_latest_checkpoint
from utils import get_device


def load_model(checkpoint_path, device):
    print(f"Loading model from: {checkpoint_path}")
    model = PalmDiseaseDetector()
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully!")
    return model


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

            # Forward pass: model returns raw logits (no softmax)
            # CrossEntropyLoss applies softmax internally - we don't apply it manually
            outputs = model(images)
            # CrossEntropyLoss internally applies softmax, then calculates loss
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()

            # Calculate accuracy: argmax on logits to get predicted class
            # (Note: argmax on logits gives same result as argmax on softmax(logits))
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
    parser = argparse.ArgumentParser(description="Test Palm Disease Detector Model")
    args = parser.parse_args()

    device = get_device()

    # Get latest checkpoint
    try:
        model_path = get_latest_checkpoint()
        print(f"Using latest checkpoint: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load test dataset
    print("\nLoading test dataset...")
    _, _, test_loader = get_dataloaders(
        num_workers=4 if torch.cuda.is_available() else 0,
    )

    if test_loader is None:
        print("Error: No test data found")
        return

    # Load model and run test
    model = load_model(model_path, device)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting test evaluation...")

    test_loss, test_acc, per_class_stats = test_model(
        model, test_loader, criterion, device
    )

    # Print results
    print("Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nPer-Class Accuracy:")
    print(
        f"  Healthy: {per_class_stats['healthy']['accuracy']:.2f}% ({per_class_stats['healthy']['correct']}/{per_class_stats['healthy']['total']})"
    )
    print(
        f"  Sick: {per_class_stats['sick']['accuracy']:.2f}% ({per_class_stats['sick']['correct']}/{per_class_stats['sick']['total']})"
    )


if __name__ == "__main__":
    main()
