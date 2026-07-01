from __future__ import annotations

import math
from typing import Iterable

import torch


def degree_from_edge_index(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    row = edge_index[0].long()
    col = edge_index[1].long()
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))
    return deg.clamp_min_(1.0)


def degree_from_sparse_adj(num_nodes: int, adj: torch.Tensor) -> torch.Tensor:
    adj = adj.coalesce()
    idx = adj.indices()
    off_diag = idx[0] != idx[1]
    idx = idx[:, off_diag]
    if idx.numel() == 0:
        return torch.ones(num_nodes, device=adj.device, dtype=torch.float32)
    deg = torch.zeros(num_nodes, device=adj.device, dtype=torch.float32)
    deg.scatter_add_(0, idx[0].long(), torch.ones(idx.size(1), device=adj.device))
    deg.scatter_add_(0, idx[1].long(), torch.ones(idx.size(1), device=adj.device))
    return deg.clamp_min_(1.0)


def tdgia_defective_scores_from_probability(
    pv: torch.Tensor,
    degree: torch.Tensor,
    degree_limit: int,
    alpha_mu: float,
    k1: float,
    k2: float,
) -> torch.Tensor:
    pv = pv.float().clamp_min(1e-12)
    deg = degree.float().clamp_min(1.0)
    d = float(max(int(degree_limit), 1))
    lam = float(k1) / torch.sqrt(deg * d) + float(k2) / deg
    return (float(alpha_mu) * pv + (1.0 - float(alpha_mu))) * lam


def slot_scores_from_node_scores(
    node_scores: torch.Tensor,
    n_inject: int,
    degree_limit: int,
    *,
    max_slots: int | None = None,
) -> list[float]:
    if node_scores.numel() == 0 or int(n_inject) <= 0:
        return []

    sorted_scores = torch.sort(node_scores.detach().float().cpu(), descending=True).values
    chunk = max(1, int(degree_limit))
    slot_count = int(n_inject if max_slots is None else min(int(max_slots), int(n_inject)))
    scores: list[float] = []
    for slot in range(max(0, slot_count)):
        start = slot * chunk
        if start >= int(sorted_scores.numel()):
            start = max(0, int(sorted_scores.numel()) - chunk)
        segment = sorted_scores[start : start + chunk]
        if segment.numel() == 0:
            segment = sorted_scores
        scores.append(float(segment.mean().item()))
    return scores


def score_based_injection_schedule(
    eligible_timesteps: Iterable[int],
    timestep_slot_scores: dict[int, list[float]],
    n_inject: int,
    *,
    timestep_capacity: dict[int, int] | None = None,
) -> tuple[list[int], dict[int, int]]:
    eligible = [int(t) for t in eligible_timesteps]
    allocation = {int(t): 0 for t in eligible}
    if int(n_inject) <= 0 or not eligible:
        return [], allocation

    opportunities: list[tuple[float, int, int]] = []
    for t in eligible:
        slots = list(timestep_slot_scores.get(int(t), []))
        if timestep_capacity is not None:
            slots = slots[: max(0, int(timestep_capacity.get(int(t), 0)))]
        for slot_idx, score in enumerate(slots):
            if math.isfinite(float(score)):
                opportunities.append((float(score), int(t), int(slot_idx)))

    if len(opportunities) < int(n_inject):
        raise ValueError(
            f"Global injection budget N_INJECT={int(n_inject)} is infeasible for score-based scheduling: "
            f"only {len(opportunities)} scored injection slots are available."
        )

    opportunities.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = opportunities[: int(n_inject)]
    selected_timesteps = [int(t) for _, t, _ in selected]
    for t in selected_timesteps:
        allocation[int(t)] += 1
    return selected_timesteps, allocation

