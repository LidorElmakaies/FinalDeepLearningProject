import torch
import torch.optim as optim
import copy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Training constants
LEARNING_RATE = 5e-5


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()  # Training mode (allows gradient)

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")  # Progress bar

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).long()  # requires for CrossEntropyLoss

        optimizer.zero_grad()  # Reset gradients from previous iteration

        outputs = model(images)  # shape [batch_size, 2] - logits [healthy, sick]

        # CrossEntropyLoss internally applies softmax, then calculates loss
        loss = criterion(outputs, labels)

        # backwards propagation (calculate gradients)
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calculate accuracy: argmax on logits to get predicted class
        # (Note: argmax on logits gives same result as argmax on softmax(logits))
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()  # count correct predictions
        # total number of images that were processed in this batch (size of the batch)
        total += labels.size(0)

        # Update progress bar
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100 * (correct / total)
        pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})

    # Final averages
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, epoch):
    model.eval()  # Evaluation mode (doesn't allow gradient - saves memory)

    running_loss = 0.0  # sum of losses for this epoch
    correct = 0  # count correct predictions
    total = 0  # count total samples processed

    # Don't calculate gradients in validation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).long()

            # Forward pass only (no backward)
            # Model returns raw logits - CrossEntropyLoss applies softmax internally
            outputs = model(images)
            # CrossEntropyLoss internally applies softmax, then calculates loss
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()  # sum of losses for this epoch

            # Calculate accuracy: argmax on logits to get predicted class
            # (Note: argmax on logits gives same result as argmax on softmax(logits))
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)  # count total samples processed

            # Update progress bar
            current_loss = running_loss / len(val_loader)
            current_acc = 100 * correct / total
            pbar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"}
            )

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def plot_training_history(train_losses, val_losses, val_accs):
    epochs = range(1, len(train_losses) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss (Train vs Validation)
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    ax2.plot(epochs, val_accs, "g-", label="Val Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def pick_better_model_results(primary_results, secondary_results):
    if primary_results is None or primary_results.get("best_model_state") is None:
        return secondary_results, "secondary"
    if secondary_results is None or secondary_results.get("best_model_state") is None:
        return primary_results, "primary"

    primary_acc = primary_results["best_val_acc"]
    primary_loss = primary_results["best_val_loss"]
    secondary_acc = secondary_results["best_val_acc"]
    secondary_loss = secondary_results["best_val_loss"]

    # Print comparison
    print("Comparing Models")
    print(
        f"Initial Training - Val Acc: {primary_acc:.2f}%, Val Loss: {primary_loss:.4f}"
    )
    print(
        f"Fine-Tuning      - Val Acc: {secondary_acc:.2f}%, Val Loss: {secondary_loss:.4f}"
    )

    if secondary_acc > primary_acc:
        return secondary_results, "secondary"
    if secondary_acc < primary_acc:
        return primary_results, "primary"

    # Accuracies equal - use loss as tie-breaker (lower is better)
    if secondary_loss < primary_loss:
        return secondary_results, "secondary"
    return primary_results, "primary"


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    lr=LEARNING_RATE,
    patience=10,
    min_epochs=None,  # Minimum epochs before early stopping checks
):
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training-specific variables
    best_val_acc = 0.0  # Primary criterion: higher is better
    best_val_loss = float(
        "inf"
    )  # Tie-breaker: lower is better when accuracies are equal
    best_train_loss = float("inf")
    best_train_acc = 0.0
    best_model_state = None
    has_improvement_in_current_run = (
        False  # Track if there was improvement in current patience window
    )

    # History tracking for visualization
    train_losses = []
    val_losses = []
    val_accs = []

    # Training loop - can continue beyond initial epochs if improvement detected
    epoch = 0
    should_stop = False

    while not should_stop:
        epoch_start_time = time.time()

        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # Store history for visualization
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start_time

        # Print summary
        print(f"\n[EPOCH {epoch+1}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s\n")

        # Track best model state based on validation accuracy (primary), then validation loss (tie-breaker)
        should_update_best = False
        if val_acc > best_val_acc:
            # Better validation accuracy - this is the new best model
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_train_acc = train_acc
            should_update_best = True
        elif val_acc == best_val_acc:
            # Same validation accuracy - use validation loss as tie-breaker (lower is better)
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_train_acc = train_acc
                should_update_best = True

        if should_update_best:
            # Save the best model state for later (deep copy to avoid reference issues)
            best_model_state = copy.deepcopy(model.state_dict())
            has_improvement_in_current_run = True
            print(
                f"New best validation accuracy: {val_acc:.2f}% (Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%)\n"
            )

        # Early stopping logic - only check after minimum epochs if specified
        if min_epochs is None or (epoch + 1) >= min_epochs:
            if (epoch + 1) % patience == 0:
                if has_improvement_in_current_run:
                    # Improvement detected in this patience window - continue training
                    has_improvement_in_current_run = False  # Reset for next window
                    print(
                        f"Improvement detected at epoch {epoch+1} (best val acc: {best_val_acc:.2f}%). Continuing training.\n"
                    )
                else:
                    # No improvement in last X epochs - stop training
                    print(
                        f"No improvement in validation accuracy over the last {patience} epochs. Best was {best_val_acc:.2f}%. Stopping."
                    )
                    should_stop = True

        epoch += 1

    print("Training Completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Training Accuracy: {best_train_acc:.2f}%")
    print(f"Best Training Loss: {best_train_loss:.4f}")

    # Plot training history
    if len(train_losses) > 0:
        print("\nGenerating training history plots...")
        plot_training_history(train_losses, val_losses, val_accs)

    return {
        "best_model_state": best_model_state,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }


def fine_tune(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    lr=LEARNING_RATE,
    min_epochs=10,
    patience=5,
):
    print("Starting Fine-Tuning Phase (Backbone Unfrozen)")

    # Unfreeze the backbone for fine-tuning
    model.unfreeze_backbone()
    print("Backbone weights unfrozen - full fine-tuning mode enabled")

    # Use 1/10 of the learning rate for fine-tuning
    fine_tune_lr = lr / 10.0
    print(f"Using learning rate: {fine_tune_lr} (1/10 of {lr})")

    return train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        lr=fine_tune_lr,
        patience=patience,
        min_epochs=min_epochs,
    )
