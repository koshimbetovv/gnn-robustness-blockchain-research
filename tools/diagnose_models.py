"""
Diagnostic script to check what models are predicting.
"""
import sys
import os
import yaml
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.elliptic import EllipticDataset
from src.utils.model_loader import load_all_models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset()
    data = dataset.get_data()
    data = data.to(device)
    
    # Check label distribution
    mask = data.y != -1
    y_true = data.y[mask]
    print(f"\nLabel distribution in data:")
    print(f"  Class 0 (Legitimate): {(y_true == 0).sum().item()}")
    print(f"  Class 1 (Fraudulent): {(y_true == 1).sum().item()}")
    
    # Load all pre-trained models (configs loaded automatically)
    print("\nLoading pre-trained models...")
    models = load_all_models(
        num_features=data.num_features,
        num_classes=2,
        hidden_dim=64,
        device=device,
        model_dir="models",
        config_dir="config/models"
    )
    
    if not models:
        print("\n⚠ No models found.")
        return
    
    # Check predictions
    print("\n" + "="*60)
    print("Prediction Distribution Analysis")
    print("="*60)
    
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            mask = data.y != -1
            logits_filtered = logits[mask]
            
            # Check for NaN or Inf BEFORE softmax
            print(f"\n{model_name.upper()}")
            print(f"  Logits - min: {logits_filtered.min():.4f}, max: {logits_filtered.max():.4f}")
            print(f"  Logits - mean: {logits_filtered.mean():.4f}, std: {logits_filtered.std():.4f}")
            
            has_nan = torch.isnan(logits_filtered).any()
            has_inf = torch.isinf(logits_filtered).any()
            if has_nan or has_inf:
                print(f"  ⚠ Logits contain NaN={has_nan.item()}, Inf={has_inf.item()}")
                continue
            
            pred = logits_filtered.argmax(dim=1)
            probs = torch.softmax(logits_filtered, dim=1)
            
            print(f"  Predicted class 0: {(pred == 0).sum().item()}")
            print(f"  Predicted class 1: {(pred == 1).sum().item()}")
            print(f"  Avg prob class 0: {probs[:, 0].mean():.4f}")
            print(f"  Avg prob class 1: {probs[:, 1].mean():.4f}")
            
            # Check if model is always predicting same class
            if (pred == 0).sum().item() == len(pred) or (pred == 1).sum().item() == len(pred):
                print(f"  ⚠ WARNING: Model always predicts same class!")


if __name__ == "__main__":
    main()
