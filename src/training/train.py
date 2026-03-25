"""
YOUR CODE: The Training Loop

This is the most important file in the project. The training loop is the heart
of deep learning — understanding it means understanding how neural networks learn.

WHAT TO LEARN:
- An EPOCH is one full pass through the training data
- Each epoch: iterate batches → forward pass → compute loss → backward pass → update weights
- The LOSS function measures how wrong the model is (CrossEntropyLoss for classification)
- The OPTIMIZER adjusts weights to reduce the loss (Adam is a good default)
- The SCHEDULER reduces the learning rate over time for better convergence
- You must track training loss AND validation accuracy to detect overfitting

THE TRAINING LOOP (pseudocode):
    for epoch in range(num_epochs):
        model.train()                    # Enable dropout, batch norm in training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU
            optimizer.zero_grad()        # Clear old gradients
            outputs = model(images)      # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()              # Backward pass (compute gradients)
            optimizer.step()             # Update weights

        model.eval()                     # Disable dropout for evaluation
        validate(model, val_loader)      # Check accuracy on held-out data
        scheduler.step()                 # Adjust learning rate

OVERFITTING:
- If train accuracy goes up but val accuracy plateaus or drops → overfitting
- Solutions: more augmentation, more dropout, early stopping, less epochs
- This is why we track BOTH metrics

DEVICE:
- Use torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
- MPS = Apple Silicon GPU (your Mac), CUDA = NVIDIA GPU
- .to(device) moves tensors to the right hardware
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: cpu, cuda, or mps

    Returns:
        Dict with "loss" (average training loss) and "accuracy" (training accuracy)

    TODO:
    1. Set model to training mode: model.train()
    2. Initialize running_loss = 0.0 and correct = 0 and total = 0
    3. Loop over train_loader with tqdm for a progress bar:
       for images, labels in tqdm(train_loader, desc="Training"):
           a. Move to device: images, labels = images.to(device), labels.to(device)
           b. Zero gradients: optimizer.zero_grad()
           c. Forward pass: outputs = model(images)
           d. Compute loss: loss = criterion(outputs, labels)
           e. Backward pass: loss.backward()
           f. Update weights: optimizer.step()
           g. Track metrics:
              running_loss += loss.item() * images.size(0)
              _, predicted = outputs.max(1)
              total += labels.size(0)
              correct += predicted.eq(labels).sum().item()
    4. Return {"loss": running_loss / total, "accuracy": correct / total}
    """
    raise NotImplementedError("Implement me!")


@torch.no_grad()  # Disables gradient computation — saves memory during evaluation
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model on validation data.

    Args & Returns: Same pattern as train_one_epoch

    TODO:
    1. Set model to eval mode: model.eval()
       (This disables dropout and uses running stats for batch norm)
    2. Same loop as training BUT:
       - No optimizer.zero_grad()
       - No loss.backward()
       - No optimizer.step()
       - Just forward pass and track metrics
    3. Return {"loss": ..., "accuracy": ...}
    """
    raise NotImplementedError("Implement me!")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    learning_rate: float = 0.001,
    save_dir: str | Path = "models/checkpoints",
    device: torch.device | None = None,
) -> dict:
    """
    Full training pipeline.

    TODO:
    1. Set up device (cuda > mps > cpu)

    2. Move model to device: model = model.to(device)

    3. Set up loss function:
       criterion = nn.CrossEntropyLoss()
       (CrossEntropyLoss combines LogSoftmax + NLLLoss — standard for classification)

    4. Set up optimizer:
       optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
       NOTE: filter(requires_grad) ensures we only optimize unfrozen parameters

    5. Set up learning rate scheduler:
       scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
       (Reduces LR by 10x every 7 epochs — helps convergence)

    6. Training loop:
       best_val_acc = 0.0
       history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

       for epoch in range(num_epochs):
           train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
           val_metrics = validate(model, val_loader, criterion, device)
           scheduler.step()

           # Print progress
           print(f"Epoch {epoch+1}/{num_epochs}")
           print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}")
           print(f"  Val   Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}")

           # Save best model
           if val_metrics["accuracy"] > best_val_acc:
               best_val_acc = val_metrics["accuracy"]
               torch.save(model.state_dict(), save_dir / "best_model.pth")
               print(f"  Saved best model (val_acc={best_val_acc:.4f})")

           # Append to history
           ...

       return history

    7. BONUS: Save a training checkpoint with more info:
       torch.save({
           "epoch": epoch,
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
           "best_val_acc": best_val_acc,
           "history": history,
       }, save_dir / "checkpoint.pth")
    """
    raise NotImplementedError("Implement me!")


if __name__ == "__main__":
    # This block runs when you execute: python -m src.training.train
    # Wire everything together here once you've implemented the pieces above.
    #
    # THE TWO-PHASE TRAINING STRATEGY:
    #
    # Phase 1 — "Teach the new head" (freeze_backbone=True)
    #   The backbone already knows how to SEE (edges, textures, shapes, feathers).
    #   We freeze it and only train the classifier head so it learns to MAP those
    #   visual features to our 555 bird species. This is fast because only the
    #   final Linear layer has trainable weights.
    #
    #   What's happening inside:
    #     Image → [FROZEN backbone extracts features] → [NEW head learns species] → prediction
    #     Early layers see edges → middle layers see shapes → late layers see "parts"
    #     Only the head weights update. Backbone stays exactly as ImageNet trained it.
    #
    # Phase 2 — "Specialize the vision" (unfreeze layers 14+, lower LR)
    #   Now that the head is decent, we unfreeze the LATE backbone layers (14-17)
    #   and train with a LOWER learning rate (typically 1/10th of Phase 1).
    #   This lets those layers shift from detecting "generic object parts" to
    #   detecting "bird-specific features" like beak shapes and wing bars.
    #
    #   What's happening inside:
    #     Image → [FROZEN early layers: edges/textures] →
    #             [UNFROZEN late layers: adapting to bird parts] →
    #             [Trained head: bird species] → prediction
    #
    #   The lower LR is critical: these layers already have GOOD weights from
    #   ImageNet. We want to gently nudge them toward birds, not scramble them.
    #
    # After both phases, the model has undergone "catastrophic forgetting" —
    # it can no longer classify "school bus" or "pizza", but it CAN distinguish
    # a House Finch from a Purple Finch. That's exactly what we want.
    #
    # Rough outline:
    # 1. from src.training.dataset import NABirdsDataset
    # 2. from src.training.transforms import get_train_transforms, get_val_transforms
    # 3. from src.training.model import create_model, unfreeze_backbone
    # 4. Create datasets with transforms
    # 5. Create DataLoaders
    # 6. Create model with freeze_backbone=True
    # 7. Phase 1: Train for N epochs (classifier head only, lr=0.001)
    # 8. Phase 2: unfreeze_backbone(model), train for more epochs (lr=0.0001)
    # 9. Evaluate final model
    pass
