"""
Check if input data contains NaN or Inf values.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.elliptic import EllipticDataset

dataset = EllipticDataset()
data = dataset.get_data()

print("Checking input data for NaN/Inf values...")
print(f"Features (x) - NaN: {torch.isnan(data.x).any()}, Inf: {torch.isinf(data.x).any()}")
print(f"Labels (y) - NaN: {torch.isnan(data.y.float()).any()}, Inf: {torch.isinf(data.y.float()).any()}")
print(f"Edge Index - NaN: {torch.isnan(data.edge_index.float()).any()}, Inf: {torch.isinf(data.edge_index.float()).any()}")

print(f"\nFeature statistics:")
print(f"  Min: {data.x.min():.4f}, Max: {data.x.max():.4f}")
print(f"  Mean: {data.x.mean():.4f}, Std: {data.x.std():.4f}")

# Print unique labels
unique_labels = torch.unique(data.y)
print(f"\nUnique labels in dataset: {unique_labels.tolist()}")
