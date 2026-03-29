from __future__ import annotations

import numpy as np
import torch

from src.models.evolvegcn import (
    EvolveGCNClassifier,
    EvolveGCNH,
    EvolveGCNNodeClassifier,
    EvolveGCNO,
)
from src.training.metrics import binary_classification_metrics


class WeightedCrossEntropy(torch.nn.Module):
    """Exact weighted cross-entropy used in the IBM EvolveGCN repo."""

    def __init__(self, class_weights: list[float], device: torch.device):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        nll = -torch.log(logits.softmax(dim=1).clamp_min(1e-12))
        weights = self.class_weights[labels.long()]
        loss = nll[torch.arange(labels.shape[0], device=labels.device), labels.long()] * weights
        return loss.mean()


class DualAdam:
    def __init__(self, model: EvolveGCNNodeClassifier, lr: float):
        self.backbone_opt = torch.optim.Adam(model.backbone.parameters(), lr=lr)
        self.classifier_opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    def zero_grad(self, set_to_none: bool = True):
        self.backbone_opt.zero_grad(set_to_none=set_to_none)
        self.classifier_opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.backbone_opt.step()
        self.classifier_opt.step()


def resolve_class_weights(train_samples, config: dict) -> list[float]:
    cfg_weights = config["training"].get("class_weights", "auto")
    if cfg_weights != "auto" and cfg_weights is not None:
        return [float(cfg_weights[0]), float(cfg_weights[1])]

    y_train = torch.cat([sample.label_vals for sample in train_samples], dim=0).long()
    n0 = int((y_train == 0).sum().item())
    n1 = int((y_train == 1).sum().item())
    if n0 == 0 or n1 == 0:
        raise ValueError(f"Cannot compute automatic class weights because class counts are n0={n0}, n1={n1}.")

    total = n0 + n1
    w0 = total / (2.0 * n0)
    w1 = total / (2.0 * n1)
    print(f"Train labeled counts: licit(0)={n0}, illicit(1)={n1}")
    return [float(w0), float(w1)]


def move_sparse_tensor(sp: torch.Tensor, device: torch.device) -> torch.Tensor:
    sp = sp.coalesce()
    return torch.sparse_coo_tensor(
        sp.indices().to(device),
        sp.values().to(device),
        size=sp.size(),
        dtype=sp.dtype,
        device=device,
    ).coalesce()


def move_sample(sample, device: torch.device):
    hist_adj_list = [move_sparse_tensor(adj, device) for adj in sample.hist_adj_list]
    hist_ndFeats_list = [x.to(device) for x in sample.hist_ndFeats_list]
    node_mask_list = [m.to(device) for m in sample.node_mask_list]
    label_idx = sample.label_idx.to(device)
    label_vals = sample.label_vals.to(device)
    return hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals


@torch.no_grad()
def evaluate_evolvegcn(model, samples, loss_fn, device: torch.device):
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    by_time = {}

    for sample in samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        logits = model(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx)
        loss = loss_fn(logits, label_vals)
        losses.append(float(loss.item()))

        y_true = label_vals.detach().cpu()
        y_pred = logits.argmax(dim=1).detach().cpu()
        all_true.append(y_true)
        all_pred.append(y_pred)

        time_metrics = binary_classification_metrics(y_true, y_pred)
        by_time[int(sample.current_time)] = {
            "precision_pos": float(time_metrics["precision_pos"]),
            "recall_pos": float(time_metrics["recall_pos"]),
            "f1_pos": float(time_metrics["f1_pos"]),
        }

    if not all_true:
        raise ValueError("No labeled windows found while evaluating EvolveGCN.")

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    metrics = binary_classification_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["f1_pos_by_timestep"] = by_time
    return metrics


def build_evolvegcn_model(num_features: int, config: dict) -> EvolveGCNNodeClassifier:
    variant = config["model"]["variant"].lower()
    if variant == "h":
        backbone = EvolveGCNH(
            in_dim=num_features,
            layer_1_feats=config["model"]["layer_1_feats"],
            layer_2_feats=config["model"]["layer_2_feats"],
            activation=torch.nn.RReLU(),
            skipfeats=config["model"].get("skipfeats", False),
        )
    elif variant == "o":
        backbone = EvolveGCNO(
            in_dim=num_features,
            layer_1_feats=config["model"]["layer_1_feats"],
            layer_2_feats=config["model"]["layer_2_feats"],
            activation=torch.nn.RReLU(),
            skipfeats=config["model"].get("skipfeats", False),
        )
    else:
        raise ValueError(f"Unknown EvolveGCN variant: {variant}")

    cls_in = config["model"]["layer_2_feats"] + (num_features if config["model"].get("skipfeats", False) else 0)
    classifier = EvolveGCNClassifier(in_dim=cls_in, hidden_dim=config["model"]["cls_feats"], out_dim=2)
    return EvolveGCNNodeClassifier(backbone=backbone, classifier=classifier)
