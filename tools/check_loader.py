import os
import sys
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.utils.model_loader import load_model


MODELS_TO_TEST = ["gcn", "graphsage", "gat"]


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def split_mask(data, split: str):
    attr = f"{split}_mask"
    if not hasattr(data, attr):
        raise ValueError(f"Missing {attr}. Re-run preprocessing and ensure masks are saved/loaded.")
    return getattr(data, attr).bool() & (data.y != -1)


@torch.no_grad()
def eval_split(model, data, split: str):
    model.eval()
    mask = split_mask(data, split)
    logits = model(data.x, data.edge_index)[mask]
    y_true = data.y[mask].cpu()
    y_pred = logits.argmax(dim=1).cpu()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "n": int(mask.sum().item()),
        "acc": float(acc),
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "fraud_pred": int((y_pred == 1).sum().item()),
    }


def main():
    device = get_device()

    data = EllipticDataset().get_data().to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    # quick sanity
    for s in ["train", "test"]:
        _ = split_mask(data, s)

    print("\n=== Loader smoke test ===")

    for name in MODELS_TO_TEST:
        print("\n" + "-" * 60)
        print(f"Testing loader for: {name.upper()}")

        try:
            model = load_model(
                model_name=name,
                num_features=data.num_features,
                num_classes=2,
                device=device,
                model_dir="models",
                config_dir="config/models",
                run_id=None,   # latest
            )
        except Exception as e:
            print(f"✗ FAILED to load {name.upper()}: {repr(e)}")
            continue

        # Forward-pass sanity
        try:
            with torch.no_grad():
                out = model(data.x, data.edge_index)
            finite = torch.isfinite(out).all().item()
            print(f"✓ Forward pass OK. logits shape={tuple(out.shape)} finite={finite}")
        except Exception as e:
            print(f"✗ Forward pass FAILED for {name.upper()}: {repr(e)}")
            continue

        # Metric sanity
        tr = eval_split(model, data, "train")
        te = eval_split(model, data, "test")

        print(f"TRAIN: n={tr['n']} acc={tr['acc']:.4f} P={tr['precision_pos']:.4f} R={tr['recall_pos']:.4f} F1={tr['f1_pos']:.4f} fraud_pred={tr['fraud_pred']}")
        print(f"TEST : n={te['n']} acc={te['acc']:.4f} P={te['precision_pos']:.4f} R={te['recall_pos']:.4f} F1={te['f1_pos']:.4f} fraud_pred={te['fraud_pred']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
