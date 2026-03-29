"""
Example script demonstrating how to load and use pre-trained models
without retraining. Includes accuracy and confusion matrix evaluation
on train/val/test splits (if masks exist).
"""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.elliptic import EllipticDataset
from src.utils.model_loader import load_all_models


def get_split_mask(data, split: str) -> torch.Tensor:
    """
    Returns a boolean mask for the given split AND labeled nodes (y != -1).
    If split masks do not exist, falls back to all labeled nodes.
    """
    labeled = (data.y != -1)
    attr = f"{split}_mask"

    if hasattr(data, attr):
        split_mask = getattr(data, attr)
        if split_mask.dtype != torch.bool:
            split_mask = split_mask.bool()
        return split_mask & labeled

    # fallback: no split masks available
    return labeled


def compute_confusion_matrix(model, data, split: str):
    """Compute confusion matrix for a model on a given split."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask = get_split_mask(data, split)
        pred = logits[mask].argmax(dim=1)
        y_true = data.y[mask]

    # Force 2x2 even if one class is absent in this split
    return confusion_matrix(
        y_true.cpu().numpy(),
        pred.cpu().numpy(),
        labels=[0, 1],
    )


def compute_metrics_from_cm(cm: np.ndarray):
    """Return (acc, precision, recall, f1, tn, fp, fn, tp) from a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return acc, precision, recall, f1, tn, fp, fn, tp


def plot_confusion_matrices(confusion_matrices, split: str, output_dir="results/figures"):
    """Plot and save confusion matrices for all models for a given split."""
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(confusion_matrices.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, model_name in enumerate(model_names):
        cm = confusion_matrices[model_name]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraudulent'])
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f'{model_name.upper()} ({split})')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'confusion_matrices_{split}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices ({split}) saved to {output_path}")
    plt.close()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset()
    data = dataset.get_data().to(device)
    print("✓ Dataset loaded")

    # Check masks exist
    has_masks = all(hasattr(data, k) for k in ["train_mask", "test_mask"])
    print(f"Has train/test masks: {has_masks}")

    if not has_masks:
        print("⚠ Split masks not found. Falling back to all labeled nodes (y != -1).")

    # Load all pre-trained models
    print("\nLoading pre-trained models...")
    models = load_all_models(
        num_features=data.num_features,
        num_classes=2,
        hidden_dim=64,
        device=device,
        model_dir="models"
    )

    if not models:
        print("\n⚠ No models found. Please train models first using:")
        print("  python scripts/train_models.py")
        return

    splits = ["train", "test"]

    print("\n" + "=" * 70)
    print("Model Evaluation Results (per split)")
    print("=" * 70)

    # confusion_matrices_by_split[split][model_name] = cm
    confusion_matrices_by_split = {s: {} for s in splits}

    for model_name, model in models.items():
        print(f"\n{model_name.upper()}")

        for split in splits:
            mask = get_split_mask(data, split)
            n_labeled = int(mask.sum().item())

            cm = compute_confusion_matrix(model, data, split)
            confusion_matrices_by_split[split][model_name] = cm

            acc, precision, recall, f1, tn, fp, fn, tp = compute_metrics_from_cm(cm)

            # Also show how many fraud predictions were made on this split
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                pred = logits[mask].argmax(dim=1)
                fraud_pred = int((pred == 1).sum().item())

            print(f"  [{split}] #labeled={n_labeled:5d} | Acc={acc:.4f} | P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} | FraudPred={fraud_pred:5d}")
            print(f"        TN={tn:5d}  FP={fp:5d}  FN={fn:5d}  TP={tp:5d}")

    print("=" * 70)

    # Plot per-split confusion matrices
    for split in splits:
        plot_confusion_matrices(confusion_matrices_by_split[split], split)


if __name__ == "__main__":
    main()
