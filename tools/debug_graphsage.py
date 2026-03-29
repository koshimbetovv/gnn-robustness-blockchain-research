"""
Trace where NaN/Inf appears in GraphSAGE forward/backward.
Uses train_mask if present, otherwise falls back to y!=-1.
"""
import sys
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.elliptic import EllipticDataset
from src.models.graphsage import GraphSAGE


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
    print(f"{name:<26} shape={tuple(t.shape)}  NaN={has_nan} Inf={has_inf}  min={tmin:.6g} max={tmax:.6g}")
    return has_nan or has_inf


def get_train_mask(data):
    if hasattr(data, "train_mask"):
        return data.train_mask.bool() & (data.y != -1)
    return (data.y != -1)


def safe_class_weights(y_train, device):
    counts = torch.bincount(y_train, minlength=2).float()
    print(f"Train class counts: {counts.tolist()}")

    if (counts == 0).any():
        print("⚠ Missing class in train -> disabling class weights for this debug run.")
        return None

    n = float(y_train.numel())
    w = n / (2.0 * counts)
    return w.to(device)


def forward_trace_graphsage(model, x, edge_index):
    """
    Best-effort trace.
    If model exposes model.convs (ModuleList), we trace each layer.
    Otherwise we just trace model(x, edge_index).
    """
    print("\n[Forward trace]")
    bad = report("x (input)", x)

    # If model has conv stack
    if hasattr(model, "convs"):
        h = x
        n_layers = len(model.convs)

        dropout_p = getattr(model, "dropout", 0.0)
        use_norm = getattr(model, "use_norm", False)
        has_norms = hasattr(model, "norms")

        for i in range(n_layers - 1):
            h = model.convs[i](h, edge_index)
            bad = report(f"after conv[{i}]", h) or bad

            if use_norm and has_norms:
                h = model.norms[i](h)
                bad = report(f"after norm[{i}]", h) or bad

            h = F.relu(h)
            bad = report(f"after relu[{i}]", h) or bad

            h = F.dropout(h, p=dropout_p, training=model.training)
            bad = report(f"after drop[{i}]", h) or bad

        # last conv -> logits
        h = model.convs[-1](h, edge_index)
        bad = report("logits (last conv)", h) or bad
        return h, bad

    # Fallback
    out = model(x, edge_index)
    bad = report("logits (model output)", out) or bad
    return out, bad


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

# Create model (match your GraphSAGE signature!)
# Common signatures:
#   GraphSAGE(in_dim, hid_dim, out_dim, aggr="mean")
# Add dropout/norm if your class supports it.
model = GraphSAGE(data.num_features, 64, 2, aggr="mean").to(device)
model.train()

# Class weights from TRAIN only
y_train = data.y[mask]
class_weights = safe_class_weights(y_train, device)
print(f"Class weights: {class_weights}" if class_weights is not None else "Class weights: None")

opt = Adam(model.parameters(), lr=0.001, eps=1e-5)

# -------- step 1 --------
print("\nRunning training step 1...")
logits, _ = forward_trace_graphsage(model, data.x, data.edge_index)

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
logits2, _ = forward_trace_graphsage(model, data.x, data.edge_index)
loss2 = F.cross_entropy(logits2[mask], train_labels, weight=class_weights)
print(f"loss2 = {loss2.item():.6f}  NaN={torch.isnan(loss2).item()} Inf={torch.isinf(loss2).item()}")
