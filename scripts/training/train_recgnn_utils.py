from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.training.metrics import binary_classification_metrics


@torch.no_grad()
def evaluate_recgnn_sequence(model, graphs, device, prime_graphs=None):
    model.eval()
    model.reset_sequence_state(device)

    if prime_graphs is not None:
        for graph in prime_graphs:
            graph = graph.to(device)
            _ = model(graph.x.float(), graph.edge_index.long())
            model.detach_sequence_state()

    preds_all = []
    labels_all = []
    timestep_f1 = {}

    for graph in graphs:
        graph = graph.to(device)
        log_probs = model(graph.x.float(), graph.edge_index.long())
        model.detach_sequence_state()

        mask = graph.y != -1
        if int(mask.sum().item()) == 0:
            continue

        y_true = graph.y[mask].cpu()
        y_pred = log_probs[mask].argmax(dim=1).cpu()
        preds_all.append(y_pred)
        labels_all.append(y_true)

        by_time_metrics = binary_classification_metrics(y_true, y_pred)
        timestep_f1[int(graph.graph_timestep)] = float(by_time_metrics["f1_pos"])

    if not labels_all:
        raise ValueError("No labeled nodes found while evaluating RecGNN sequence.")

    y_true = torch.cat(labels_all)
    y_pred = torch.cat(preds_all)
    metrics = binary_classification_metrics(y_true, y_pred)
    metrics["f1_pos_by_timestep"] = timestep_f1
    return metrics


def train_recgnn_epoch(model, graphs, optimizer, device) -> float:
    model.train()
    model.reset_sequence_state(device)

    batch_losses = []
    for graph in graphs:
        graph = graph.to(device)
        mask = graph.y != -1
        if int(mask.sum().item()) == 0:
            _ = model(graph.x.float(), graph.edge_index.long())
            model.detach_sequence_state()
            continue

        optimizer.zero_grad(set_to_none=True)
        log_probs = model(graph.x.float(), graph.edge_index.long())
        loss = F.nll_loss(log_probs[mask], graph.y[mask])
        loss.backward()
        optimizer.step()
        model.detach_sequence_state()
        batch_losses.append(float(loss.item()))

    return float(np.mean(batch_losses)) if batch_losses else 0.0
