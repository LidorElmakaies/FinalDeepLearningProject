import torch
import torch.optim as optim
import copy
import time
from tqdm import tqdm

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
            best_epoch = epoch
            should_update_best = True
        elif val_acc == best_val_acc:
            # Same validation accuracy - use validation loss as tie-breaker (lower is better)
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_epoch = epoch
                should_update_best = True

        if should_update_best:
            # Save the best model state for later (deep copy to avoid reference issues)
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
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
