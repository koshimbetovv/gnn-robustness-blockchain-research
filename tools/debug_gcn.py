"""
Trace where NaN/Inf appears in GCN forward/backward (matches src/models/gcn.py exactly).
Uses train_mask if present, otherwise falls back to y!=-1.
"""
import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.elliptic import EllipticDataset
from src.models.gcn import GCN


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def report(name, t):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    tmin = t.min().item() if t.numel() > 0 else float("nan")
    tmax = t.max().item() if t.numel() > 0 else float("nan")
    print(f"{name:<22} shape={tuple(t.shape)}  NaN={has_nan} Inf={has_inf}  min={tmin:.6g} max={tmax:.6g}")
    return has_nan or has_inf


def get_train_mask(data):
    if hasattr(data, "train_mask"):
        return data.train_mask.bool() & (data.y != -1)
    return (data.y != -1)


def safe_class_weights(y_train, device):
    # 2-class (0 legit, 1 fraud)
    counts = torch.bincount(y_train, minlength=2).float()
    print(f"Train class counts: {counts.tolist()}")

    if (counts == 0).any():
        print("⚠ Missing class in train -> disabling class weights for this debug run.")
        return None

    n = float(y_train.numel())
    w = n / (2.0 * counts)
    return w.to(device)


def forward_trace_gcn(model, x, edge_index):
    """
    Exact trace of:
      conv -> (norm) -> relu -> dropout  for all but last
      last conv only
    """
    print("\n[Forward trace]")
    bad = report("x (input)", x)

    # hidden layers
    for i in range(len(model.convs) - 1):
        x = model.convs[i](x, edge_index)
        bad = report(f"after conv[{i}]", x) or bad

        if model.use_norm:
            x = model.norms[i](x)
            bad = report(f"after norm[{i}]", x) or bad

        x = F.relu(x)
        bad = report(f"after relu[{i}]", x) or bad

        x = F.dropout(x, p=model.dropout, training=model.training)
        bad = report(f"after drop[{i}]", x) or bad

    # logits layer
    x = model.convs[-1](x, edge_index)
    bad = report("logits (last conv)", x) or bad

    return x, bad


# ---------------- main ----------------
dataset = EllipticDataset()
data = dataset.get_data()

device = get_device()
print("Device:", device)
data = data.to(device)

print("\n[Data checks]")
report("data.x", data.x)
report("data.y(float)", data.y.float())
report("edge_index(float)", data.edge_index.float())

mask = get_train_mask(data)
print(f"\nUsing train mask. #train labeled = {int(mask.sum().item())}")
if mask.sum().item() == 0:
    raise ValueError("Train mask has 0 labeled nodes. Check dataset masks / label mapping.")

# Create model
model = GCN(data.num_features, 64, 2, num_layers=2, dropout=0.4, use_norm=True).to(device)
model.train()

# Class weights from TRAIN only (safe)
y_train = data.y[mask]
class_weights = safe_class_weights(y_train, device)
print(f"Class weights: {class_weights}" if class_weights is not None else "Class weights: None")

opt = Adam(model.parameters(), lr=0.001, eps=1e-5)

# -------- step 1 --------
print("\nRunning training step 1...")
logits, bad_fwd = forward_trace_gcn(model, data.x, data.edge_index)

print("\n[Loss computation]")
train_logits = logits[mask]
train_labels = data.y[mask]

report("train_logits", train_logits)
report("train_labels(float)", train_labels.float())

loss = F.cross_entropy(train_logits, train_labels, weight=class_weights)
print(f"loss = {loss.item():.6f}  NaN={torch.isnan(loss).item()} Inf={torch.isinf(loss).item()}")

if not torch.isfinite(loss):
    print("\nLoss is not finite -> extra checks:")
    loss_no_w = F.cross_entropy(train_logits, train_labels)
    print(f"loss_no_weights = {loss_no_w.item():.6f}  finite={torch.isfinite(loss_no_w).item()}")
    lp = F.log_softmax(train_logits, dim=1)
    report("log_softmax", lp)
else:
    print("\n[Backward]")
    opt.zero_grad(set_to_none=True)
    loss.backward()

    print("\n[Gradient checks]")
    bad_grad = False
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        has_nan = torch.isnan(p.grad).any().item()
        has_inf = torch.isinf(p.grad).any().item()
        if has_nan or has_inf:
            print(f"BAD GRAD -> {name}: NaN={has_nan} Inf={has_inf} "
                  f"min={p.grad.min().item():.6g} max={p.grad.max().item():.6g}")
            bad_grad = True
            break
    if not bad_grad:
        print("All gradients finite.")

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    print("Optimizer step done.")

# -------- step 2 --------
print("\nRunning training step 2...")
logits2, _ = forward_trace_gcn(model, data.x, data.edge_index)
loss2 = F.cross_entropy(logits2[mask], train_labels, weight=class_weights)
print(f"loss2 = {loss2.item():.6f}  NaN={torch.isnan(loss2).item()} Inf={torch.isinf(loss2).item()}")
