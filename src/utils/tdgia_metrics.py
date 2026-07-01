from __future__ import annotations

from collections.abc import Iterable

import torch


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if int(denominator) == 0:
        return None
    return float(int(numerator) / int(denominator))


def attacked_target_ids_from_edges(
    injected_edges: Iterable[tuple[int, int]],
    selected_targets: torch.Tensor | Iterable[int],
) -> list[int]:
    if torch.is_tensor(selected_targets):
        selected = {int(v) for v in selected_targets.detach().cpu().view(-1).tolist()}
    else:
        selected = {int(v) for v in selected_targets}
    return sorted({int(dst) for _, dst in injected_edges if int(dst) in selected})


def mask_from_target_ids(num_nodes: int, target_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(int(num_nodes), dtype=torch.bool, device=device)
    ids = sorted({int(v) for v in target_ids if 0 <= int(v) < int(num_nodes)})
    if ids:
        mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def asr_object(
    y_true: torch.Tensor,
    pred_clean: torch.Tensor,
    pred_adv: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    mask = mask.bool() & (y_true != -1)
    y = y_true[mask].long()
    clean = pred_clean[mask].long()
    adv = pred_adv[mask].long()
    clean_correct = clean == y
    attempted = int(clean_correct.sum().item())
    success = int((clean_correct & (adv != y)).sum().item())
    return {
        "value": _safe_ratio(success, attempted),
        "success": success,
        "attempted_clean_correct": attempted,
    }


def attacked_target_outcome(
    *,
    y_true: torch.Tensor,
    victim_pred_clean: torch.Tensor,
    victim_pred_adv: torch.Tensor,
    surrogate_pred_clean: torch.Tensor,
    surrogate_pred_adv: torch.Tensor,
    attacked_mask: torch.Tensor,
    target_unit: str,
) -> dict:
    n_attacked = int((attacked_mask.bool() & (y_true != -1)).sum().item())
    return {
        "scope": "attacked_targets",
        "target_unit": target_unit,
        "n_attacked_targets": n_attacked,
        "asr": asr_object(y_true, victim_pred_clean, victim_pred_adv, attacked_mask),
        "surrogate_asr": asr_object(
            y_true, surrogate_pred_clean, surrogate_pred_adv, attacked_mask
        ),
    }


def aggregate_attacked_target_outcomes(outcomes: Iterable[dict], *, target_unit: str) -> dict:
    rows = [row for row in outcomes if row]
    n_attacked = int(sum(int(row.get("n_attacked_targets", 0)) for row in rows))
    victim_success = int(sum(int(row["asr"]["success"]) for row in rows))
    victim_attempted = int(sum(int(row["asr"]["attempted_clean_correct"]) for row in rows))
    surrogate_success = int(sum(int(row["surrogate_asr"]["success"]) for row in rows))
    surrogate_attempted = int(
        sum(int(row["surrogate_asr"]["attempted_clean_correct"]) for row in rows)
    )
    return {
        "scope": "attacked_targets",
        "target_unit": target_unit,
        "n_attacked_targets": n_attacked,
        "asr": {
            "value": _safe_ratio(victim_success, victim_attempted),
            "success": victim_success,
            "attempted_clean_correct": victim_attempted,
        },
        "surrogate_asr": {
            "value": _safe_ratio(surrogate_success, surrogate_attempted),
            "success": surrogate_success,
            "attempted_clean_correct": surrogate_attempted,
        },
    }


def coverage_metrics(
    *,
    n_selected_targets: int,
    n_budgeted_selected_targets: int,
    n_attacked_targets: int,
    target_unit: str,
) -> dict:
    return {
        "target_unit": target_unit,
        "n_selected_targets": int(n_selected_targets),
        "n_budgeted_selected_targets": int(n_budgeted_selected_targets),
        "n_attacked_targets": int(n_attacked_targets),
        "attacked_target_coverage": _safe_ratio(
            int(n_attacked_targets), int(n_budgeted_selected_targets)
        ),
    }


def budget_efficiency_metrics(
    *,
    n_attacked_targets: int,
    n_success: int,
    n_injected_nodes: int,
    n_logical_injected_edges: int,
) -> dict:
    return {
        "n_injected_nodes": int(n_injected_nodes),
        "n_logical_injected_edges": int(n_logical_injected_edges),
        "attacked_targets_per_injected_node": _safe_ratio(
            int(n_attacked_targets), int(n_injected_nodes)
        ),
        "attacked_targets_per_logical_injected_edge": _safe_ratio(
            int(n_attacked_targets), int(n_logical_injected_edges)
        ),
        "successes_per_injected_node": _safe_ratio(
            int(n_success), int(n_injected_nodes)
        ),
        "successes_per_logical_injected_edge": _safe_ratio(
            int(n_success), int(n_logical_injected_edges)
        ),
    }
