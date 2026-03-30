import os
import json
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.elliptic import EllipticConfig, EllipticDataset
from src.models.chronowave_gnn import ChronoWaveGNN
from src.utils.seed import seed_from_config
from scripts.training.train_paper_utils import (
    eval_split,
    get_device,
    print_epoch_metrics,
    print_final_metrics,
    save_run,
    split_mask,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "models", "train_chronowave_gnn.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def haar_level2_approximation(x: torch.Tensor) -> torch.Tensor:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "ChronoWave-GNN requires PyWavelets for level-2 Haar DWT. "
            "Install it with: pip install PyWavelets"
        ) from exc

    x_np = x.detach().cpu().numpy()
    cA2, *_ = pywt.wavedec(x_np, wavelet="haar", level=2, axis=1)
    return torch.from_numpy(cA2).to(dtype=x.dtype)


@torch.no_grad()
def standardize_from_train(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train_x = x[train_mask]
    if train_x.numel() == 0:
        raise ValueError("Train split has no labeled nodes for standardization.")
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def build_paper_features(data) -> None:
    raw_x = data.x.float().cpu()
    train_mask = split_mask(data, "train").cpu()

    wave_x = haar_level2_approximation(raw_x)

    raw_x = standardize_from_train(raw_x, train_mask)
    wave_x = standardize_from_train(wave_x, train_mask)

    data.x = torch.cat([raw_x, wave_x], dim=1)


def main():
    seed_from_config(CONFIG)
    device = get_device()

    data = EllipticDataset(EllipticConfig(**CONFIG["data"])).get_data()
    build_paper_features(data)
    data = data.to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    train_mask = split_mask(data, "train")
    _ = split_mask(data, "test")

    model = ChronoWaveGNN(
        in_dim=data.num_features,
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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["training"]["epochs"],
        eta_min=0.0,
    )

    print("Class weights: None (paper does not specify weighted CE)")
    print("\n=== Training ChronoWave-GNN (train/test only, unknown labels filtered by the dataset loader) ===")

    log_every = CONFIG["training"]["log_every"]
    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index, data.time_step)
        loss = F.cross_entropy(
            logits[train_mask],
            data.y[train_mask],
            label_smoothing=CONFIG["training"]["label_smoothing"],
        )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered at epoch {epoch}: {float(loss.item())}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["training"]["grad_clip"])
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == CONFIG["training"]["epochs"]:
            train_metrics = eval_split(model, data, "train", "chronowave_gnn")
            test_metrics = eval_split(model, data, "test", "chronowave_gnn")
            print_epoch_metrics(epoch, float(loss.item()), train_metrics, test_metrics)

    test_metrics = eval_split(model, data, "test", "chronowave_gnn")
    print_final_metrics("ChronoWave-GNN", test_metrics)
    save_run(model, test_metrics, CONFIG, "chronowave_gnn")


if __name__ == "__main__":
    main()
