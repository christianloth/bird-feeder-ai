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

import signal
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Module-level flag: set by SIGTERM handler to request graceful shutdown after current epoch
_graceful_shutdown = False


def _worker_init_sigterm_ignore(worker_id: int) -> None:
    """Make DataLoader workers ignore SIGTERM so the main process controls shutdown."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def install_sigterm_handler() -> None:
    """Install a SIGTERM handler that requests graceful shutdown instead of crashing.

    Call this before training starts. When SIGTERM arrives (e.g., IDE quitting),
    the current epoch finishes, checkpoint is saved, then training exits cleanly.
    """
    def _handler(signum, frame):
        global _graceful_shutdown
        _graceful_shutdown = True
        print("\nSIGTERM received — will save checkpoint after current epoch...")

    signal.signal(signal.SIGTERM, _handler)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (Adam)
        device: cpu, cuda, or mps
        scaler: GradScaler for mixed precision (None = FP32 training)

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
    use_amp = scaler is not None

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    use_amp: bool = False,
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
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def _validate_and_predict(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> dict:
    """
    Run model on entire validation set in a SINGLE pass. Computes everything:
    loss, accuracy, predictions, and top-5 accuracy. No redundant passes needed.

    Returns dict with: loss, accuracy, preds (numpy), labels (numpy), top5_acc
    """
    import numpy as np

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0
    all_preds = []
    all_labels = []

    for images, targets in tqdm(val_loader, desc="Validating"):
        images, targets_dev = images.to(device), targets.to(device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets_dev)

        # Loss
        running_loss += loss.item() * images.size(0)

        # Top-1 predictions (argmax on logits — same as softmax then argmax)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets_dev).sum().item()
        total += targets.size(0)

        # Collect predictions for sklearn metrics
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.numpy())

        # Top-5 accuracy (vectorized)
        _, top5_preds = outputs.topk(5, dim=1)
        top5_correct += (targets_dev.unsqueeze(1) == top5_preds).any(dim=1).sum().item()

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "top5_acc": top5_correct / total if total > 0 else 0.0,
    }


def _print_and_save_epoch_metrics(
    preds,
    labels,
    top5_acc: float,
    epoch: int,
    save_dir: Path,
    class_names: list[str] | None = None,
    top_confusions: int = 5,
) -> None:
    """Print detailed validation metrics and save confusion matrix.

    Computes the confusion matrix ONCE and uses it for both the top-confusions
    printout and the saved confusion matrix image.
    """
    import numpy as np
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        precision_recall_fscore_support, confusion_matrix,
    )
    from src.training.evaluate import plot_confusion_matrix

    # Confusion matrix — computed ONCE, used for both printing and plotting
    cm = confusion_matrix(labels, preds)

    # Precision, Recall, F1 (macro-averaged across all species)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # Per-class precision, recall, F1 for finding worst classes
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0,
    )

    # Overall correct/wrong summary
    total_samples = len(labels)
    total_correct = int((preds == labels).sum())
    total_wrong = total_samples - total_correct

    print("  --- Validation Metrics ---")
    print(f"  Total: {total_samples} | Correct: {total_correct} | Wrong: {total_wrong}")
    print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")

    # Worst 5 classes by F1 (only classes with support > 0)
    classes_with_samples = np.where(per_support > 0)[0]
    worst_indices = classes_with_samples[np.argsort(per_f1[classes_with_samples])[:5]]

    print("  Worst 5 classes (by F1)  [TP=correct, FP=wrongly called this, FN=missed]:")
    for idx in worst_indices:
        name = class_names[idx] if class_names else str(idx)
        tp = int(per_recall[idx] * per_support[idx]) if per_recall[idx] > 0 else 0
        fn = int(per_support[idx] - tp)
        fp = int(((preds == idx) & (labels != idx)).sum())
        print(f"    {name}: TP={tp} FP={fp} FN={fn} "
              f"P={per_precision[idx]:.2f} R={per_recall[idx]:.2f} F1={per_f1[idx]:.2f} "
              f"(n={int(per_support[idx])})")

    # Top-N most confused species pairs (reuses cm computed above)
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    flat_indices = np.argsort(cm_offdiag, axis=None)[-top_confusions:][::-1]
    rows, cols = np.unravel_index(flat_indices, cm_offdiag.shape)

    print(f"  Top {top_confusions} confusions:")
    for true_idx, pred_idx in zip(rows, cols):
        count = cm_offdiag[true_idx, pred_idx]
        if count == 0:
            break
        true_name = class_names[true_idx] if class_names else str(true_idx)
        pred_name = class_names[pred_idx] if class_names else str(pred_idx)
        print(f"    {true_name} → {pred_name}: {count}")

    # Save confusion matrix image (passes pre-computed cm — no recomputation)
    cm_dir = save_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        labels, preds,
        class_names=class_names,
        top_n=20,
        save_path=cm_dir / f"epoch_{epoch:02d}.png",
        cm=cm,
    )
    print(f"  Confusion matrix saved to {cm_dir / f'epoch_{epoch:02d}.png'}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    learning_rate: float = 0.001,
    save_dir: str | Path = "models/bird-classifier",
    device: torch.device | None = None,
    class_names: list[str] | None = None,
    use_amp: bool = False,
    resume_from: str | Path | None = None,
    scheduler_type: str = "step_lr",
    patience: int | None = None,
) -> dict:
    """
    Full training pipeline.

    Args:
        scheduler_type: "step_lr" (reduce every 7 epochs — good for MobileNetV2 phases)
                        or "reduce_on_plateau" (reduce when val acc stalls — good for
                        end-to-end training like EfficientNet-B2)
        patience: Early stopping patience. If val accuracy doesn't improve for this many
                  epochs, stop training. None = no early stopping.

    Steps:
    1. Set up device, loss function, optimizer, and LR scheduler
    2. Optionally resume from a checkpoint
    3. Run training loop: train one epoch → validate → save best model
    4. Return training history for plotting
    """
    # 1. Device setup (cuda > mps > cpu)
    if device is None:
        from config.settings import get_device
        device = torch.device(get_device())
    print(f"Using device: {device}")

    # 2. Move model to device
    model = model.to(device)

    # 3. Loss function — standard CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer — only optimize unfrozen parameters (filter by requires_grad)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # 5. Scheduler — controls how the learning rate changes during training
    #    StepLR: predictable decay every N epochs (good for short, phased training)
    #    ReduceLROnPlateau: adaptive — only drops LR when model stops improving
    if scheduler_type == "step_lr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler_type == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=1, min_lr=1e-6,
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # 6. Mixed precision scaler (only works on CUDA — ignored on MPS/CPU)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled — using FP16 on Tensor Cores")

    # 7. Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 8. Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_scheduler = checkpoint.get("scheduler_type", "step_lr")
        if saved_scheduler == scheduler_type:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(f"  Scheduler changed ({saved_scheduler} → {scheduler_type}) — using fresh scheduler")
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint["history"]
        print(f"Resumed from epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")

    # 9. Early stopping setup
    epochs_without_improvement = 0

    # 10. Training loop — wrapped in try/finally so saves happen even on Ctrl+C
    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler=scaler,
            )

            # Single validation pass — computes loss, accuracy, predictions, and top-5
            val_results = _validate_and_predict(
                model, val_loader, criterion, device, use_amp=use_amp,
            )

            # Step the scheduler (ReduceLROnPlateau needs the metric, StepLR does not)
            if scheduler_type == "reduce_on_plateau":
                scheduler.step(val_results["accuracy"])
            else:
                scheduler.step()

            # Print progress
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}/{num_epochs} (lr={current_lr:.6f})")
            print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   Loss: {val_results['loss']:.4f}  Acc: {val_results['accuracy']:.4f}")

            # Print detailed metrics and save confusion matrix
            # (uses pre-computed predictions — no extra inference, confusion matrix computed once)
            _print_and_save_epoch_metrics(
                val_results["preds"], val_results["labels"],
                val_results["top5_acc"], epoch + 1, save_dir, class_names,
            )

            # Save best model for this run (by validation accuracy)
            if val_results["accuracy"] > best_val_acc:
                best_val_acc = val_results["accuracy"]
                epochs_without_improvement = 0
                tmp_path = save_dir / "best_model.pth.tmp"
                torch.save(model.state_dict(), tmp_path)
                tmp_path.rename(save_dir / "best_model.pth")
                print(f"  Saved best model (val_acc={best_val_acc:.4f})")
            else:
                epochs_without_improvement += 1

            # Append to history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_results["loss"])
            history["val_acc"].append(val_results["accuracy"])

            # Save checkpoint after EVERY epoch (atomic: write to tmp then rename)
            tmp_ckpt = save_dir / "checkpoint.pth.tmp"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scheduler_type": scheduler_type,
                "best_val_acc": best_val_acc,
                "history": history,
            }, tmp_ckpt)
            tmp_ckpt.rename(save_dir / "checkpoint.pth")

            # Early stopping check
            if patience is not None and epochs_without_improvement >= patience:
                print(f"\n  Early stopping: no improvement for {patience} epochs. "
                      f"Best val_acc={best_val_acc:.4f}")
                break

            # Graceful shutdown check (SIGTERM received — checkpoint already saved above)
            if _graceful_shutdown:
                print(f"\n  Graceful shutdown after epoch {epoch + 1}. "
                      f"Resume with --resume to continue.")
                break

    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted at epoch {epoch + 1}.")

    finally:
        # Clean up any leftover tmp files from interrupted saves
        for tmp in save_dir.glob("*.pth.tmp"):
            tmp.unlink()

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
    #   For maximum throughput, preprocess first to eliminate CPU bottleneck:
    #     python scripts/preprocess_dataset.py
    #     python -m src.training.train --batch-size 176 --preprocessed data/nabirds/preprocessed
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

    from src.training.dataset import NABirdsDataset, PreprocessedNABirdsDataset
    from src.training.transforms import get_train_transforms, get_val_transforms
    from src.training.model import (
        create_model, unfreeze_backbone, count_parameters, get_model_config,
    )
    from src.training.evaluate import plot_training_history

    parser = argparse.ArgumentParser(description="Train bird species classifier")
    parser.add_argument(
        "--model", type=str, default="efficientnet_b2",
        choices=["mobilenetv2", "efficientnet_b2"],
        help="model architecture (default: efficientnet_b2)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="data loading workers (default: 4)")
    parser.add_argument(
        "--preprocessed", type=Path, default=None,
        help="path to preprocessed dataset (from scripts/preprocess_dataset.py)",
    )
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (FP16)")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume training from checkpoint.pth (keeps optimizer/scheduler state)",
    )
    args = parser.parse_args()

    # Preprocessed data is baked at 224x224 — only works with MobileNetV2
    if args.preprocessed and args.model != "mobilenetv2":
        parser.error("--preprocessed only works with mobilenetv2 (images were preprocessed at 224x224)")

    # Look up model-specific config (input size, etc.)
    model_config = get_model_config(args.model)
    input_size = model_config["input_size"]

    from datetime import datetime

    DATA_DIR = Path("data/nabirds")
    MODEL_DIR = Path("models/bird-classifier") / args.model
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    # Each training run gets its own timestamped folder.
    # --resume finds the latest run's checkpoint automatically.
    if args.resume:
        # Find the latest run folder with a checkpoint
        run_dirs = sorted(MODEL_DIR.glob("*/checkpoint.pth"))
        if run_dirs:
            SAVE_DIR = run_dirs[-1].parent
            print(f"Resuming from: {SAVE_DIR}")
        else:
            print("Warning: --resume specified but no run with checkpoint.pth found. Starting fresh.")
            SAVE_DIR = MODEL_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M")
            args.resume = False  # no checkpoint to resume from
    else:
        SAVE_DIR = MODEL_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M")

    print(f"Model: {args.model} (input size: {input_size}x{input_size})")
    print(f"Run dir: {SAVE_DIR}")

    # --- Create datasets with transforms ---
    print("Loading datasets...")
    if args.preprocessed:
        print(f"  Using preprocessed data from {args.preprocessed}")
        train_dataset = PreprocessedNABirdsDataset(args.preprocessed, split="train")
        val_dataset = PreprocessedNABirdsDataset(args.preprocessed, split="test")
    else:
        train_dataset = NABirdsDataset(
            DATA_DIR, split="train", transform=get_train_transforms(input_size),
        )
        val_dataset = NABirdsDataset(
            DATA_DIR, split="test", transform=get_val_transforms(input_size),
        )
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
        persistent_workers=NUM_WORKERS > 0,  # reuse workers across epochs (avoids FD leaks)
        worker_init_fn=_worker_init_sigterm_ignore,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_pin_memory,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=_worker_init_sigterm_ignore,
    )

    # Install SIGTERM handler so IDE/terminal exits don't crash training
    install_sigterm_handler()

    # --- Training strategy depends on model architecture ---
    #
    # MobileNetV2: Two-phase (small model, needs careful training)
    #   Phase 1: Freeze backbone, train only classifier head
    #   Phase 2: Unfreeze late layers (14+), fine-tune with lower LR
    #
    # EfficientNet-B2: Single-phase (larger model, robust to end-to-end training)
    #   Train everything at once — no freezing needed
    #

    # Resolve --resume checkpoint path
    resume_path = (SAVE_DIR / "checkpoint.pth") if args.resume else None

    if args.model == "mobilenetv2":
        # --- Phase 1: Train classifier head only (backbone frozen) ---
        # Skip Phase 1 if resuming (checkpoint is from Phase 2)
        if resume_path:
            print("\n=== Resuming Phase 2 from checkpoint ===")
            model = create_model(
                num_classes=train_dataset.num_classes, pretrained=False,
                freeze_backbone=False, model_name="mobilenetv2",
            )
            unfreeze_backbone(model, unfreeze_from=14)
            history_phase1 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        else:
            print("\n=== Phase 1: Training classifier head (backbone frozen) ===")
            model = create_model(
                num_classes=train_dataset.num_classes, pretrained=True,
                freeze_backbone=True, model_name="mobilenetv2",
            )
            params = count_parameters(model)
            print(f"  Trainable: {params['trainable']:,} / {params['total']:,} parameters")

            history_phase1 = train(
                model, train_loader, val_loader,
                num_epochs=10, learning_rate=0.001, save_dir=SAVE_DIR,
                class_names=class_names, use_amp=args.amp,
            )

            # --- Phase 2: Fine-tune with late backbone layers unfrozen ---
            print("\n=== Phase 2: Fine-tuning (layers 14+ unfrozen, lower LR) ===")
            unfreeze_backbone(model, unfreeze_from=14)

        params = count_parameters(model)
        print(f"  Trainable: {params['trainable']:,} / {params['total']:,} parameters")

        history_phase2 = train(
            model, train_loader, val_loader,
            num_epochs=25, learning_rate=0.0001, save_dir=SAVE_DIR,
            class_names=class_names, use_amp=args.amp,
            resume_from=resume_path,
        )

        # Combine both phases into one history for plotting
        combined_history = {
            key: history_phase1[key] + history_phase2[key]
            for key in history_phase1
        }

    elif args.model == "efficientnet_b2":
        # --- Single phase: train end-to-end (no freezing) ---
        # EfficientNet-B2 is large enough to handle gradients from an untrained head.
        # This mirrors the approach used by Dennis Joostel's Birds-Classifier-EfficientNetB2
        # which achieved 99% accuracy on 525 species with this strategy.
        print("\n=== Training EfficientNet-B2 end-to-end ===")
        model = create_model(
            num_classes=train_dataset.num_classes, pretrained=True,
            freeze_backbone=False, model_name="efficientnet_b2",
        )
        params = count_parameters(model)
        print(f"  Trainable: {params['trainable']:,} / {params['total']:,} parameters")

        combined_history = train(
            model, train_loader, val_loader,
            num_epochs=30, learning_rate=0.001, save_dir=SAVE_DIR,
            class_names=class_names, use_amp=args.amp,
            resume_from=resume_path,
            scheduler_type="reduce_on_plateau",
            patience=5,
        )

    # --- Plot training history ---
    plot_training_history(combined_history, save_path=SAVE_DIR / "training_history.png")
    print("\nTraining complete!")
