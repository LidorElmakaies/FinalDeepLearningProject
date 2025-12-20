import torch
import copy
import time
from tqdm import tqdm


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

        loss = criterion(outputs, labels)  # Calculate Loss (CrossEntropyLoss)

        # backwards propagation (calculate gradients)
        loss.backward()  # Calculate gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calculate accuracy
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
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate Loss (CrossEntropyLoss)

            # Statistics
            running_loss += loss.item()  # sum of losses for this epoch

            # Get predicted class (0=healthy, 1=sick)
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
    optimizer,
    device,
    patience=10,
):
    # Training-specific variables
    best_val_acc = 0.0
    best_train_acc = 0.0  # Track best training accuracy for tie-breaking
    best_epoch = 0
    best_val_loss = float("inf")
    best_model_state = None
    best_optimizer_state = None
    has_improvement_in_current_run = (
        False  # Track if there was improvement in current patience window
    )

    print("Starting Training")

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

        # Track best model state
        should_update_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            best_val_loss = val_loss
            should_update_best = True
        elif val_acc == best_val_acc:
            # When validation accuracy is equal, check if training accuracy is better
            if train_acc > best_train_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_epoch = epoch
                best_val_loss = val_loss
                should_update_best = True

        if should_update_best:
            # Save the best model state for later (deep copy to avoid reference issues)
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            has_improvement_in_current_run = True
            print(
                f"New best validation accuracy: {val_acc:.2f}% (Train Acc: {train_acc:.2f}%)\n"
            )

        # Check for early stopping every X epochs
        if (epoch + 1) % patience == 0:
            if has_improvement_in_current_run:
                # Improvement detected in this patience window - continue training
                has_improvement_in_current_run = False  # Reset for next window
                print(
                    f"Improvement detected at epoch {epoch+1} (best: {best_val_acc:.2f}%). Continuing training.\n"
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
    print(f"Best Training Accuracy: {best_train_acc:.2f}%")

    return {
        "best_model_state": best_model_state,
        "best_optimizer_state": best_optimizer_state,
        "best_val_acc": best_val_acc,
        "best_train_acc": best_train_acc,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
