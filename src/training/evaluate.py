"""
YOUR CODE: Model Evaluation and Metrics

After training, you need to understand HOW WELL the model works, and more
importantly, WHERE it fails. This is crucial for iterating and improving.

WHAT TO LEARN:
- Accuracy alone is misleading — a model could be 95% accurate but terrible at
  rare species (which is exactly what you want to detect in Frisco!)
- Confusion matrix shows which species get confused with each other
- Per-class metrics reveal weak spots in your model
- Precision: "Of all the times the model said Cardinal, how often was it right?"
- Recall: "Of all the actual Cardinals, how many did the model find?"

TOOLS:
- sklearn.metrics: classification_report, confusion_matrix
- matplotlib / seaborn: plotting
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions and true labels from a DataLoader.

    Returns:
        (all_predictions, all_labels, all_confidences)

    Steps:
    1. Set model to eval mode
    2. Run all data through the model, collecting predictions and confidences
    3. Return as numpy arrays for use with sklearn metrics
    """
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predicted = probabilities.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confs.extend(confidences.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_confs)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> None:
    """
    Print precision, recall, F1 for each species.

    The output shows per-species metrics. Look for species with low recall —
    those are the ones your model misses most often.
    """
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    top_n: int = 20,
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a confusion matrix for the top-N most confused species.

    With 555 species, the full matrix is unreadable. Instead:
    1. Find the top-N species with the most misclassifications
    2. Plot only those species in the matrix

    Steps:
    1. Compute full confusion matrix
    2. Find species with most misclassifications
    3. Extract and plot the sub-matrix for those species
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)

    # Find the top-N species with the most errors
    errors_per_class = cm.sum(axis=1) - cm.diagonal()
    top_confused = np.argsort(errors_per_class)[-top_n:]

    # Extract sub-matrix for just those species
    sub_cm = cm[np.ix_(top_confused, top_confused)]

    # Build labels for the axes
    if class_names is not None:
        sub_labels = [class_names[i] for i in top_confused]
    else:
        sub_labels = [str(i) for i in top_confused]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_cm, annot=True, fmt="d", xticklabels=sub_labels, yticklabels=sub_labels, ax=ax)
    ax.set_title(f"Top {top_n} Most Confused Species")
    ax.set_ylabel("True Species")
    ax.set_xlabel("Predicted Species")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history: dict, save_path: str | Path | None = None) -> None:
    """
    Plot training and validation loss/accuracy curves.

    This is your primary tool for diagnosing overfitting by comparing
    train vs val loss/accuracy side by side:
    - If train_acc >> val_acc: overfitting (model memorizes training data)
      → Fix: add augmentation, more dropout, early stopping
    - If both are low: underfitting (model hasn't learned enough)
      → Fix: train longer, unfreeze backbone, increase model capacity
    - If both rise steadily: good training, keep going
    - If val_loss starts rising while train_loss drops: overfitting has begun
      → The epoch just before val_loss rose is your best stopping point

    Steps:
    1. Left subplot: train_loss vs val_loss — watch for divergence (overfitting)
    2. Right subplot: train_acc vs val_acc — watch for gap widening (overfitting)
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Over Training")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Over Training")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    plt.close()


def find_worst_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    dataset,  # NABirdsDataset
    device: torch.device,
    n: int = 20,
) -> list[dict]:
    """
    Find the N predictions where the model was most confidently WRONG.

    These are the most informative errors — they tell you what the model
    is fundamentally confused about.

    Returns a list of dicts with image_path, true_species, predicted_species,
    and confidence. Sorted by confidence descending — the most confidently
    wrong predictions are the most informative for understanding failure modes.
    """
    model.eval()
    wrong_preds = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)

            for i in range(len(labels)):
                if predicted[i].item() != labels[i].item():
                    # Calculate the original index into the dataset
                    sample_idx = batch_idx * data_loader.batch_size + i
                    image_path, _ = dataset.samples[sample_idx]
                    wrong_preds.append({
                        "image_path": str(image_path),
                        "true_species": dataset.get_species_name(labels[i].item()),
                        "predicted_species": dataset.get_species_name(predicted[i].item()),
                        "confidence": confidences[i].item(),
                    })

    # Sort by confidence descending — most confident mistakes first
    wrong_preds.sort(key=lambda x: x["confidence"], reverse=True)
    return wrong_preds[:n]
