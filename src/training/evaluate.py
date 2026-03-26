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

    TODO:
    1. Set model to eval mode
    2. Collect all predictions, true labels, and confidence scores:
       all_preds, all_labels, all_confs = [], [], []
       for images, labels in data_loader:
           images = images.to(device)
           outputs = model(images)
           probabilities = torch.softmax(outputs, dim=1)
           confidences, predicted = probabilities.max(1)
           all_preds.extend(predicted.cpu().numpy())
           all_labels.extend(labels.numpy())
           all_confs.extend(confidences.cpu().numpy())
    3. Return as numpy arrays
    """
    raise NotImplementedError("Implement me!")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> None:
    """
    Print precision, recall, F1 for each species.

    TODO:
    1. from sklearn.metrics import classification_report
    2. Print classification_report(y_true, y_pred, target_names=class_names)

    The output shows per-species metrics. Look for species with low recall —
    those are the ones your model misses most often.
    """
    raise NotImplementedError("Implement me!")


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

    TODO:
    1. from sklearn.metrics import confusion_matrix
    2. cm = confusion_matrix(y_true, y_pred)
    3. Find species with most errors:
       errors_per_class = cm.sum(axis=1) - cm.diagonal()
       top_confused = np.argsort(errors_per_class)[-top_n:]
    4. Extract the sub-matrix for those species
    5. Plot with seaborn.heatmap:
       import seaborn as sns
       import matplotlib.pyplot as plt
       fig, ax = plt.subplots(figsize=(12, 10))
       sns.heatmap(sub_cm, annot=True, fmt="d", xticklabels=..., yticklabels=...)
       plt.title("Most Confused Species")
       plt.ylabel("True Species")
       plt.xlabel("Predicted Species")
    6. Save or show the plot
    """
    raise NotImplementedError("Implement me!")


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

    TODO:
    1. Create a figure with 2 subplots side by side
    2. Left: plot train_loss and val_loss vs epoch
    3. Right: plot train_acc and val_acc vs epoch
    4. Add legends, labels, title
    5. Save or show
    """
    raise NotImplementedError("Implement me!")


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

    TODO:
    Return a list of dicts with:
    - "image_path": path to the image
    - "true_species": actual species name
    - "predicted_species": what the model guessed
    - "confidence": how confident the model was (higher = worse mistake)

    This is a BONUS exercise. It's more advanced but very useful for
    understanding your model's failure modes.
    """
    raise NotImplementedError("Implement me (bonus)!")
