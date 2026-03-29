import os
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_paper_utils import (
    eval_split,
    forward_model,
    get_device,
    print_final_metrics,
    save_run,
    safe_class_weights,
    split_mask,
)
from src.datasets.chronowave_ellipticpp_actors import (
    ChronoWaveActorsConfig,
    ChronoWaveEllipticPPActorsDataset,
)
from src.models.chronowave_gnn import ChronoWaveGNN
from src.utils.logging import print_epoch_metrics
from src.utils.seed import seed_from_config


CONFIG = {
    "seed": 42,
    "data": {
        "feature_path": "data/raw/ellipticpp/actors/wallets_features.csv",
        "class_path": "data/raw/ellipticpp/actors/wallets_classes.csv",
        "edge_path": "data/raw/ellipticpp/actors/AddrAddr_edgelist.csv",
        "train_start": 1,
        "train_end": 34,
        "test_start": 35,
        "test_end": 49,
        "filter_unknown": True,
        "wavelet": "haar",
        "wavelet_level": 2,
    },
    "model": {
        "hidden_dim": 256,
        "time_dim": 8,
        "heads": 2,
        "num_layers": 3,
        "dropout": 0.4,
        "out_dim": 2,
    },
    "training": {
        "epochs": 300,
        "lr": 5e-3,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "log_every": 10,
        "use_class_weights": True,
    },
    "save": {
        "save_dir": "models",
        "save_run": True,
        "prefix": "chronowave_gnn_ellipticpp_actors",
    },
}


def main():
    seed_from_config(CONFIG)
    device = get_device()

    dataset = ChronoWaveEllipticPPActorsDataset(ChronoWaveActorsConfig(**CONFIG["data"]))
    data = dataset.get_data().to(device)

    train_mask = split_mask(data, "train")
    y_train = data.y[train_mask]
    if y_train.numel() == 0:
        raise ValueError("Train split has 0 labeled nodes after unknown filtering.")

    class_weights = None
    if CONFIG["training"].get("use_class_weights", False):
        class_weights = safe_class_weights(y_train, device)
    print(f"Class weights: {class_weights}" if class_weights is not None else "Class weights: None")

    model = ChronoWaveGNN(
        in_dim=data.x.size(1),
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        time_dim=CONFIG["model"]["time_dim"],
        heads=CONFIG["model"]["heads"],
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["training"]["epochs"])

    epochs = CONFIG["training"]["epochs"]
    log_every = CONFIG["training"].get("log_every", 10)
    grad_clip = CONFIG["training"].get("grad_clip", 1.0)
    label_smoothing = CONFIG["training"].get("label_smoothing", 0.1)

    print("\n=== Training ChronoWave-GNN on Elliptic++ Actors (train/test only) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = forward_model(model, data, "chronowave_gnn")
        loss = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or (epoch % log_every == 0) or epoch == epochs:
            train_metrics = eval_split(model, data, "train", "chronowave_gnn")
            test_metrics = eval_split(model, data, "test", "chronowave_gnn")
            current_lr = optimizer.param_groups[0]["lr"]
            print_epoch_metrics(epoch, float(loss.item()), train_metrics, test_metrics, lr=float(current_lr))

    test_metrics = eval_split(model, data, "test", "chronowave_gnn")
    print_final_metrics("ChronoWave-GNN (Elliptic++ Actors)", test_metrics)
    save_run(model, test_metrics, CONFIG, "chronowave_gnn_ellipticpp_actors")


if __name__ == "__main__":
    main()
