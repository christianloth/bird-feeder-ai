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

    TRAINING LOSS: forward pass → compute loss → backpropagate → update weights.
    The loss DRIVES learning — gradients flow backward and the optimizer adjusts
    weights to reduce it.

    Steps:
    1. Set model to training mode (enables dropout, batch norm training behavior)
    2. For each batch: forward pass → compute loss → backward pass → update weights
    3. Track running loss and accuracy across all batches
    4. Return averaged metrics for this epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {"loss": running_loss / total, "accuracy": correct / total}


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

    VALIDATION LOSS: forward pass → compute loss → that's it. No backprop,
    no weight updates. We're just MEASURING performance on unseen data.
    The @torch.no_grad() decorator above saves memory by skipping gradient tracking.

    WHY COMPUTE BOTH TRAIN AND VAL LOSS?
    Compare them to detect overfitting:
        Epoch 1:  train_loss=2.5  val_loss=2.6   ← both high, model is learning
        Epoch 10: train_loss=0.3  val_loss=0.5   ← both dropping, good progress
        Epoch 30: train_loss=0.05 val_loss=0.8   ← OVERFITTING: train drops, val rises
    When train loss keeps dropping but val loss rises, the model is memorizing
    training images instead of learning general bird features. That's your signal
    to stop training or add more augmentation.

    Steps:
    1. Set model to eval mode (disables dropout, uses running batch norm stats)
    2. Same loop as training but NO backprop — just forward pass and track metrics
    3. Return averaged metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(val_loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def _print_epoch_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
    top_confusions: int = 5,
) -> None:
    """Print detailed validation metrics after each epoch."""
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    from src.training.evaluate import get_predictions

    preds, labels, _ = get_predictions(model, val_loader, device)

    # Precision, Recall, F1 (macro-averaged across all species)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Top-5 accuracy: was the correct label in the model's top 5 guesses?
    # We need the raw outputs for this, so re-run with logits
    model.eval()
    top5_correct = 0
    total = 0
    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, top5_preds = outputs.topk(5, dim=1)
        for i in range(len(targets)):
            if targets[i].item() in top5_preds[i].cpu().tolist():
                top5_correct += 1
            total += 1
    top5_acc = top5_correct / total if total > 0 else 0.0

    print("  --- Validation Metrics ---")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")

    # Top-N most confused species pairs
    cm = confusion_matrix(labels, preds)
    np.fill_diagonal(cm, 0)
    flat_indices = np.argsort(cm, axis=None)[-top_confusions:][::-1]
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    print(f"  Top {top_confusions} confusions:")
    for true_idx, pred_idx in zip(rows, cols):
        count = cm[true_idx, pred_idx]
        if count == 0:
            break
        true_name = class_names[true_idx] if class_names else str(true_idx)
        pred_name = class_names[pred_idx] if class_names else str(pred_idx)
        print(f"    {true_name} → {pred_name}: {count}")


@torch.no_grad()
def _save_epoch_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    save_dir: Path,
    class_names: list[str] | None = None,
) -> None:
    """Generate and save a confusion matrix image for this epoch."""
    from src.training.evaluate import get_predictions, plot_confusion_matrix

    preds, labels, _ = get_predictions(model, val_loader, device)
    cm_dir = save_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        labels, preds,
        class_names=class_names,
        top_n=20,
        save_path=cm_dir / f"epoch_{epoch:02d}.png",
    )
    print(f"  Confusion matrix saved to {cm_dir / f'epoch_{epoch:02d}.png'}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    learning_rate: float = 0.001,
    save_dir: str | Path = "models/checkpoints",
    device: torch.device | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """
    Full training pipeline.

    Steps:
    1. Set up device, loss function, optimizer, and LR scheduler
    2. Run training loop: train one epoch → validate → save best model
    3. Return training history for plotting
    """
    # 1. Device setup (cuda > mps > cpu)
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    # 2. Move model to device
    model = model.to(device)

    # 3. Loss function — standard CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer — only optimize unfrozen parameters (filter by requires_grad)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # 5. Scheduler — reduce LR by 10x every 7 epochs for better convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 6. Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 7. Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Print progress
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs} (lr={current_lr:.6f})")
        print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}  Acc: {val_metrics['accuracy']:.4f}")

        # Print detailed metrics for this epoch
        _print_epoch_metrics(model, val_loader, device, class_names)

        # Save confusion matrix for this epoch
        _save_epoch_confusion_matrix(
            model, val_loader, device, epoch + 1, save_dir, class_names
        )

        # Save best model (by validation accuracy)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  Saved best model (val_acc={best_val_acc:.4f})")

        # Append to history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

    # Save final checkpoint with full state (allows resuming training)
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
    }, save_dir / "checkpoint.pth")

    return history


if __name__ == "__main__":
    # This block runs when you execute: python -m src.training.train
    # Wire everything together here once you've implemented the pieces above.
    #
    # GPU-OPTIMIZED TRAINING:
    #   Increase --batch-size to fill GPU memory (~90% utilization is ideal).
    #   Set --num-workers to match CPU core count for optimal data loading.
    #
    #   Tesla T4 (14.6 GB VRAM, 4 CPU cores):
    #     python -m src.training.train --batch-size 176 --num-workers 4
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
    import argparse

    from src.training.dataset import NABirdsDataset
    from src.training.transforms import get_train_transforms, get_val_transforms
    from src.training.model import create_model, unfreeze_backbone, count_parameters
    from src.training.evaluate import plot_training_history

    parser = argparse.ArgumentParser(description="Train MobileNetV2 bird classifier")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="data loading workers (default: 4)")
    args = parser.parse_args()

    DATA_DIR = Path("data/nabirds")
    SAVE_DIR = Path("models/checkpoints")
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    # --- Create datasets with transforms ---
    print("Loading datasets...")
    train_dataset = NABirdsDataset(DATA_DIR, split="train", transform=get_train_transforms())
    val_dataset = NABirdsDataset(DATA_DIR, split="test", transform=get_val_transforms())
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Classes: {train_dataset.num_classes}")

    # Build class names list (index → species name) for confusion matrix labels
    class_names = [
        train_dataset.get_species_name(i) for i in range(train_dataset.num_classes)
    ]

    # --- Create DataLoaders ---
    # pin_memory=True speeds up CPU→GPU transfer, but MPS doesn't support it
    import torch as _torch
    use_pin_memory = _torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_pin_memory,
    )

    # --- Phase 1: Train classifier head only (backbone frozen) ---
    print("\n=== Phase 1: Training classifier head (backbone frozen) ===")
    model = create_model(num_classes=train_dataset.num_classes, pretrained=True, freeze_backbone=True)
    params = count_parameters(model)
    print(f"  Trainable: {params['trainable']:,} / {params['total']:,} parameters")

    history_phase1 = train(
        model, train_loader, val_loader,
        num_epochs=10, learning_rate=0.001, save_dir=SAVE_DIR,
        class_names=class_names,
    )

    # --- Phase 2: Fine-tune with late backbone layers unfrozen ---
    # Same as run 1 but with 25 epochs instead of 15 (it was still improving when it stopped)
    print("\n=== Phase 2: Fine-tuning (layers 14+ unfrozen, lower LR) ===")
    unfreeze_backbone(model, unfreeze_from=14)
    params = count_parameters(model)
    print(f"  Trainable: {params['trainable']:,} / {params['total']:,} parameters")

    history_phase2 = train(
        model, train_loader, val_loader,
        num_epochs=25, learning_rate=0.0001, save_dir=SAVE_DIR,
        class_names=class_names,
    )

    # --- Plot combined training history ---
    combined_history = {
        key: history_phase1[key] + history_phase2[key]
        for key in history_phase1
    }
    plot_training_history(combined_history, save_path=SAVE_DIR / "training_history.png")
    print("\nTraining complete!")
