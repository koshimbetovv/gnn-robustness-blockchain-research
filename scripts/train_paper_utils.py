import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.training.metrics import binary_classification_metrics
from src.utils.logging import print_epoch_metrics, print_final_metrics, save_run


def get_device(allow_cuda: bool = True, allow_mps: bool = True) -> torch.device:
    if allow_cuda and torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def split_mask(data, split: str):
    attr = f"{split}_mask"
    if not hasattr(data, attr):
        raise ValueError(f"Missing {attr} in data. Ensure the dataset loader creates masks correctly from the raw files.")
    return getattr(data, attr).bool() & (data.y != -1)


def safe_class_weights(y_train, device):
    counts = torch.bincount(y_train, minlength=2).float()
    if (counts == 0).any():
        print(f"⚠ Missing class in TRAIN split (counts={counts.tolist()}), disabling class weights.")
        return None
    n = float(y_train.numel())
    w = n / (2.0 * counts)
    return w.to(device)


def get_time_step_or_raise(data, model_name: str):
    if hasattr(data, "time_step"):
        return data.time_step

    temporal_models = {"chronowave_gnn", "recgnn"}
    if model_name in temporal_models:
        raise ValueError(
            f"{model_name} requires data.time_step, but the loaded dataset does not include it."
        )
    return None


def forward_model(model, data, model_name: str):
    time_step = get_time_step_or_raise(data, model_name)

    if model_name in {"chronowave_gnn", "recgnn"}:
        return model(data.x, data.edge_index, time_step)
    return model(data.x, data.edge_index)


@torch.no_grad()
def eval_split(model, data, split: str, model_name: str):
    model.eval()
    mask = split_mask(data, split)

    logits = forward_model(model, data, model_name)[mask]
    y_true = data.y[mask].cpu()
    y_pred = logits.argmax(dim=1).cpu()
    metrics = binary_classification_metrics(y_true, y_pred, split=split)
    return metrics


def train_standard_model(model, data, config: dict, model_name: str, title: str):
    device = next(model.parameters()).device
    train_mask = split_mask(data, "train")
    y_train = data.y[train_mask]
    if y_train.numel() == 0:
        raise ValueError("Train split has 0 labeled nodes (y!=-1). Check dataset masks / label mapping.")

    use_class_weights = config["training"].get("use_class_weights", True)
    class_weights = safe_class_weights(y_train, device) if use_class_weights else None
    print(f"Class weights: {class_weights}" if class_weights is not None else "Class weights: None")

    opt = Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        eps=config["training"].get("adam_eps", 1e-5),
    )

    log_every = config["training"].get("log_every", 50)

    print("\n=== Training (printing train/test P/R/F1 during training) ===")
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        logits = forward_model(model, data, model_name)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)

        if not torch.isfinite(loss):
            print(f"Epoch {epoch}: Loss is not finite (loss={loss.item()}). Stopping.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["training"].get("grad_clip", 1.0))
        opt.step()

        if epoch == 1 or (epoch % log_every == 0):
            train_metrics = eval_split(model, data, "train", model_name)
            test_metrics = eval_split(model, data, "test", model_name)
            print_epoch_metrics(epoch, float(loss.item()), train_metrics, test_metrics)

    test_metrics = eval_split(model, data, "test", model_name)
    print_final_metrics(title, test_metrics)
    save_run(model, test_metrics, config, model_name)
    return model, test_metrics
