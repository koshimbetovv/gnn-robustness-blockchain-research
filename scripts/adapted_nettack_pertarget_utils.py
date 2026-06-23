import json
import os
from datetime import datetime
from typing import Optional

import torch

from src.training.metrics import (
    attack_success_rate,
    binary_classification_metrics,
    mean_confidence_drop,
    roc_auc_binary,
)


def make_pertarget_run_dir(model_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(
        repo_root,
        "attacks",
        f"{model_name}_adapted_nettack_pertarget_{ts}",
    )
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def nanmean(values):
    vals = [float(v) for v in values if float(v) == float(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def target_prob(logits: torch.Tensor, label: int) -> float:
    probs = torch.softmax(logits.detach().float(), dim=0)
    return float(probs[int(label)].item())


def _finite(values):
    return [float(v) for v in values if float(v) == float(v)]


def scalar_stats(values) -> dict:
    vals = _finite(values)
    if not vals:
        return {"n": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    if len(vals_sorted) % 2:
        median = vals_sorted[mid]
    else:
        median = 0.5 * (vals_sorted[mid - 1] + vals_sorted[mid])
    return {
        "n": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "median": float(median),
        "min": float(vals_sorted[0]),
        "max": float(vals_sorted[-1]),
    }


def int_histogram(values) -> list[dict]:
    counts = {}
    for value in values:
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    return [{"value": key, "count": counts[key]} for key in sorted(counts)]


def labeled_logits_summary(logits_parts: list[torch.Tensor], y_parts: list[torch.Tensor]) -> dict:
    """Aggregate labeled classification metrics from clean logits."""
    if not logits_parts or not y_parts:
        return {"n_labeled": 0, "roc_auc": float("nan")}
    logits = torch.cat([part.detach().cpu() for part in logits_parts], dim=0)
    y = torch.cat([part.detach().cpu() for part in y_parts], dim=0).long()
    mask = torch.ones(y.numel(), dtype=torch.bool)
    pred = logits.argmax(dim=1)
    return {
        **binary_classification_metrics(y, pred),
        "roc_auc": roc_auc_binary(logits, y, mask),
    }


def summarize_per_target_entries(entries: list[dict]) -> dict:
    """Compact summary of detailed per-target records.

    This intentionally avoids returning one row per target, keeping metrics.json
    small enough to scan and diff. The full records can still be written to a
    separate file by runner opt-in flags.
    """
    if not entries:
        return {
            "n_targets": 0,
            "victim": {},
            "surrogate": {},
            "perturbation": {},
            "success_by_directed_edges": [],
        }

    def branch_summary(branch: str) -> dict:
        clean_correct = sum(1 for item in entries if item[branch]["clean_correct"])
        success = sum(1 for item in entries if item[branch]["success"])
        return {
            "clean_correct": int(clean_correct),
            "success": int(success),
            "attempted": int(clean_correct),
            "asr": float(success / clean_correct) if clean_correct else 0.0,
            "confidence_drop": scalar_stats(
                item[branch]["confidence_drop"] for item in entries
            ),
            "true_prob_adv": scalar_stats(
                item[branch]["true_prob_adv"] for item in entries
            ),
        }

    directed_values = [
        item["perturbation"]["n_directed_edges_added"]
        for item in entries
    ]
    feature_l2 = [item["perturbation"]["feature_l2"] for item in entries]

    by_edges = {}
    for item in entries:
        key = int(item["perturbation"]["n_directed_edges_added"])
        row = by_edges.setdefault(
            key,
            {
                "n_targets": 0,
                "victim_success": 0,
                "victim_attempted": 0,
                "surrogate_success": 0,
                "surrogate_attempted": 0,
            },
        )
        row["n_targets"] += 1
        row["victim_success"] += int(item["victim"]["success"])
        row["victim_attempted"] += int(item["victim"]["clean_correct"])
        row["surrogate_success"] += int(item["surrogate"]["success"])
        row["surrogate_attempted"] += int(item["surrogate"]["clean_correct"])

    success_by_edges = []
    for key in sorted(by_edges):
        row = by_edges[key]
        row = {"n_directed_edges_added": key, **row}
        row["victim_asr"] = (
            float(row["victim_success"] / row["victim_attempted"])
            if row["victim_attempted"]
            else 0.0
        )
        row["surrogate_asr"] = (
            float(row["surrogate_success"] / row["surrogate_attempted"])
            if row["surrogate_attempted"]
            else 0.0
        )
        success_by_edges.append(row)

    label_counts = {}
    for item in entries:
        label = int(item["y"])
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1

    return {
        "n_targets": int(len(entries)),
        "label_counts": label_counts,
        "victim": branch_summary("victim"),
        "surrogate": branch_summary("surrogate"),
        "perturbation": {
            "feature_l2": scalar_stats(feature_l2),
            "n_directed_edges_added": {
                "stats": scalar_stats(directed_values),
                "histogram": int_histogram(directed_values),
            },
            "n_targets_with_edge_added": int(
                sum(
                    1
                    for item in entries
                    if item["perturbation"]["n_targets_with_edge_added"] > 0
                )
            ),
        },
        "success_by_directed_edges": success_by_edges,
    }


def _classification_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y = y.long().view(-1)
    logits = logits.float()
    row = torch.arange(y.numel(), device=logits.device)
    true_logits = logits[row, y]
    other_logits = logits.clone()
    other_logits[row, y] = float("-inf")
    max_other = other_logits.max(dim=1).values
    return true_logits - max_other


def margin_summary(logits_clean: torch.Tensor, logits_adv: torch.Tensor, y: torch.Tensor) -> dict:
    clean_margin = _classification_margin(logits_clean, y)
    adv_margin = _classification_margin(logits_adv, y)
    return {
        "clean": scalar_stats(clean_margin.detach().cpu().tolist()),
        "adv": scalar_stats(adv_margin.detach().cpu().tolist()),
        "drop": scalar_stats((clean_margin - adv_margin).detach().cpu().tolist()),
    }


def _asr_object(value: float, success: int, attempted: int) -> dict:
    return {
        "value": float(value) if attempted else None,
        "success": int(success),
        "attempted_clean_correct": int(attempted),
    }


def target_outcome_branch(
    *,
    y: torch.Tensor,
    pred_clean: torch.Tensor,
    pred_adv: torch.Tensor,
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    summary: dict,
    confidence_drop_value: float,
    confidence_drop_n: int,
) -> dict:
    asr, success, attempted = attack_success_rate(
        y,
        pred_clean,
        pred_adv,
        torch.ones(y.numel(), dtype=torch.bool, device=y.device),
    )
    return {
        "asr": _asr_object(asr, success, attempted),
        "mean_confidence_drop_independent_attacks": {
            "value": confidence_drop_value,
            "n_clean_correct": int(confidence_drop_n),
        },
        "confidence_drop_all_targets": summary["confidence_drop"],
        "true_prob_adv": summary["true_prob_adv"],
        "classification_margin": margin_summary(logits_clean, logits_adv, y),
    }


def transfer_summary(
    *,
    y: torch.Tensor,
    victim_pred_clean: torch.Tensor,
    victim_pred_adv: torch.Tensor,
    surrogate_pred_clean: torch.Tensor,
    surrogate_pred_adv: torch.Tensor,
    victim_asr: dict,
    surrogate_asr: dict,
) -> dict:
    victim_success = (victim_pred_clean == y) & (victim_pred_adv != y)
    surrogate_success = (surrogate_pred_clean == y) & (surrogate_pred_adv != y)
    joint_success = victim_success & surrogate_success
    surrogate_success_count = int(surrogate_success.sum().item())
    joint_success_count = int(joint_success.sum().item())
    return {
        "asr_gap_surrogate_minus_victim": (
            None
            if victim_asr["value"] is None or surrogate_asr["value"] is None
            else float(surrogate_asr["value"] - victim_asr["value"])
        ),
        "victim_success_given_surrogate_success": {
            "value": (
                float(joint_success_count / surrogate_success_count)
                if surrogate_success_count
                else None
            ),
            "victim_and_surrogate_success": joint_success_count,
            "surrogate_success": surrogate_success_count,
        },
    }


def perturbation_summary(
    *,
    per_target_summary: dict,
    structure: dict,
) -> dict:
    edge_stats = per_target_summary["perturbation"]["n_directed_edges_added"]
    structure_base = {
        key: value
        for key, value in structure.items()
        if key
        not in {
            "n_directed_edges_added_total",
            "mean_directed_edges_per_target",
        }
    }
    return {
        "feature": {
            "l2_all_targets": per_target_summary["perturbation"]["feature_l2"],
        },
        "structure": {
            **structure_base,
            "directed_edges_added": {
                "total": structure["n_directed_edges_added_total"],
                "mean_per_target": structure["mean_directed_edges_per_target"],
                "stats_per_target": edge_stats["stats"],
                "histogram": edge_stats["histogram"],
            },
        },
    }


def diagnostics_summary(per_target_summary: dict, **extra) -> dict:
    diagnostics = {
        "label_counts": per_target_summary["label_counts"],
    }
    by_edges = per_target_summary.get("success_by_directed_edges", [])
    if len(by_edges) > 1:
        diagnostics["by_directed_edges_added"] = by_edges
    diagnostics.update({k: v for k, v in extra.items() if v is not None})
    return diagnostics


class PerTargetResults:
    """Bookkeeping for independent per-target NETTACK runs."""

    def __init__(self):
        self.per_target = []
        self.target_y = []
        self.target_pred_clean = []
        self.target_pred_adv = []
        self.target_logits_clean = []
        self.target_logits_adv = []
        self.target_pred_surrogate_clean = []
        self.target_pred_surrogate_adv = []
        self.target_logits_surrogate_clean = []
        self.target_logits_surrogate_adv = []
        self.adv_f1_pos = []
        self.adv_recall_pos = []
        self.adv_f1_macro = []
        self.adv_roc_auc = []
        self.n_edges_orig = []
        self.n_directed_added_total = 0
        self.n_targets_with_edge_total = 0

    def __len__(self):
        return len(self.target_y)

    def add(
        self,
        *,
        context: dict,
        target: int,
        y_val: int,
        logits_clean_target: torch.Tensor,
        logits_adv_target: torch.Tensor,
        logits_surrogate_clean_target: torch.Tensor,
        logits_surrogate_adv_target: torch.Tensor,
        clean_row: torch.Tensor,
        adv_row: torch.Tensor,
        n_edges_orig: int,
        n_edges_adv: int,
        n_directed_added: int,
        n_targets_with_edge: int,
        adv_metrics: Optional[dict] = None,
    ) -> dict:
        pred_clean = int(logits_clean_target.argmax().item())
        pred_adv = int(logits_adv_target.argmax().item())
        pred_surrogate_clean = int(logits_surrogate_clean_target.argmax().item())
        pred_surrogate_adv = int(logits_surrogate_adv_target.argmax().item())
        clean_correct = pred_clean == int(y_val)
        surrogate_clean_correct = pred_surrogate_clean == int(y_val)
        success = clean_correct and pred_adv != int(y_val)
        surrogate_success = surrogate_clean_correct and pred_surrogate_adv != int(y_val)
        feature_l2 = float(
            torch.linalg.vector_norm((adv_row - clean_row).float(), ord=2).item()
        )

        prob_clean = target_prob(logits_clean_target, y_val)
        prob_adv = target_prob(logits_adv_target, y_val)
        surrogate_prob_clean = target_prob(logits_surrogate_clean_target, y_val)
        surrogate_prob_adv = target_prob(logits_surrogate_adv_target, y_val)

        self.target_y.append(int(y_val))
        self.target_pred_clean.append(pred_clean)
        self.target_pred_adv.append(pred_adv)
        self.target_logits_clean.append(logits_clean_target.detach().cpu())
        self.target_logits_adv.append(logits_adv_target.detach().cpu())
        self.target_pred_surrogate_clean.append(pred_surrogate_clean)
        self.target_pred_surrogate_adv.append(pred_surrogate_adv)
        self.target_logits_surrogate_clean.append(
            logits_surrogate_clean_target.detach().cpu()
        )
        self.target_logits_surrogate_adv.append(
            logits_surrogate_adv_target.detach().cpu()
        )
        self.n_edges_orig.append(int(n_edges_orig))
        self.n_directed_added_total += int(n_directed_added)
        self.n_targets_with_edge_total += int(n_targets_with_edge)

        if adv_metrics is not None:
            self.adv_f1_pos.append(adv_metrics.get("f1_pos", float("nan")))
            self.adv_recall_pos.append(adv_metrics.get("recall_pos", float("nan")))
            self.adv_f1_macro.append(adv_metrics.get("f1_macro", float("nan")))
            self.adv_roc_auc.append(adv_metrics.get("roc_auc", float("nan")))

        entry = {
            **context,
            "target": int(target),
            "y": int(y_val),
            "victim": {
                "pred_clean": pred_clean,
                "pred_adv": pred_adv,
                "clean_correct": bool(clean_correct),
                "success": bool(success),
                "true_prob_clean": prob_clean,
                "true_prob_adv": prob_adv,
                "confidence_drop": float(prob_clean - prob_adv),
            },
            "surrogate": {
                "pred_clean": pred_surrogate_clean,
                "pred_adv": pred_surrogate_adv,
                "clean_correct": bool(surrogate_clean_correct),
                "success": bool(surrogate_success),
                "true_prob_clean": surrogate_prob_clean,
                "true_prob_adv": surrogate_prob_adv,
                "confidence_drop": float(surrogate_prob_clean - surrogate_prob_adv),
            },
            "perturbation": {
                "feature_l2": feature_l2,
                "n_edges_orig": int(n_edges_orig),
                "n_edges_adv": int(n_edges_adv),
                "n_directed_edges_added": int(n_directed_added),
                "n_targets_with_edge_added": int(n_targets_with_edge),
            },
        }
        if adv_metrics is not None:
            entry["labeled_adv_metrics"] = adv_metrics
        self.per_target.append(entry)
        return entry

    def summarize(self) -> dict:
        if not self.target_y:
            return {
                "classification": None,
                "target_outcome": None,
                "perturbation": None,
                "diagnostics": diagnostics_summary(
                    summarize_per_target_entries(self.per_target)
                ),
            }

        y_t = torch.tensor(self.target_y, dtype=torch.long)
        pred_clean_t = torch.tensor(self.target_pred_clean, dtype=torch.long)
        pred_adv_t = torch.tensor(self.target_pred_adv, dtype=torch.long)
        logits_clean_t = torch.stack(self.target_logits_clean, dim=0)
        logits_adv_t = torch.stack(self.target_logits_adv, dim=0)
        pred_surrogate_clean_t = torch.tensor(
            self.target_pred_surrogate_clean, dtype=torch.long
        )
        pred_surrogate_adv_t = torch.tensor(
            self.target_pred_surrogate_adv, dtype=torch.long
        )
        logits_surrogate_clean_t = torch.stack(
            self.target_logits_surrogate_clean, dim=0
        )
        logits_surrogate_adv_t = torch.stack(
            self.target_logits_surrogate_adv, dim=0
        )
        target_mask = torch.ones(y_t.numel(), dtype=torch.bool)
        per_target_summary = summarize_per_target_entries(self.per_target)

        conf_drop, n_used = mean_confidence_drop(
            y_t, logits_clean_t, logits_adv_t, target_mask, only_clean_correct=True
        )

        surrogate_conf_drop, surrogate_n_used = mean_confidence_drop(
            y_t,
            logits_surrogate_clean_t,
            logits_surrogate_adv_t,
            target_mask,
            only_clean_correct=True,
        )

        classification = {
            "scope": "temporal_independent_per_target",
            "victim": {
                "labeled_adv_independent_mean": {
                    "n_runs": int(y_t.numel()),
                    "f1_pos": nanmean(self.adv_f1_pos),
                    "recall_pos": nanmean(self.adv_recall_pos),
                    "f1_macro": nanmean(self.adv_f1_macro),
                    "roc_auc": nanmean(self.adv_roc_auc),
                },
            },
        }

        victim_outcome = target_outcome_branch(
            y=y_t,
            pred_clean=pred_clean_t,
            pred_adv=pred_adv_t,
            logits_clean=logits_clean_t,
            logits_adv=logits_adv_t,
            summary=per_target_summary["victim"],
            confidence_drop_value=conf_drop,
            confidence_drop_n=n_used,
        )
        surrogate_outcome = target_outcome_branch(
            y=y_t,
            pred_clean=pred_surrogate_clean_t,
            pred_adv=pred_surrogate_adv_t,
            logits_clean=logits_surrogate_clean_t,
            logits_adv=logits_surrogate_adv_t,
            summary=per_target_summary["surrogate"],
            confidence_drop_value=surrogate_conf_drop,
            confidence_drop_n=surrogate_n_used,
        )
        target_outcome = {
            "n_targets": int(y_t.numel()),
            "victim": victim_outcome,
            "surrogate": surrogate_outcome,
            "transfer": transfer_summary(
                y=y_t,
                victim_pred_clean=pred_clean_t,
                victim_pred_adv=pred_adv_t,
                surrogate_pred_clean=pred_surrogate_clean_t,
                surrogate_pred_adv=pred_surrogate_adv_t,
                victim_asr=victim_outcome["asr"],
                surrogate_asr=surrogate_outcome["asr"],
            ),
        }

        perturbation = perturbation_summary(
            per_target_summary=per_target_summary,
            structure={
                "mode": "independent_per_target_sum",
                "n_edges_orig_min": int(min(self.n_edges_orig)),
                "n_edges_orig_max": int(max(self.n_edges_orig)),
                "n_edges_orig_mean": float(sum(self.n_edges_orig) / len(self.n_edges_orig)),
                "n_directed_edges_added_total": self.n_directed_added_total,
                "n_targets_with_edge_added_total": self.n_targets_with_edge_total,
                "mean_directed_edges_per_target": float(self.n_directed_added_total)
                / float(y_t.numel()),
            },
        )
        return {
            "classification": classification,
            "target_outcome": target_outcome,
            "perturbation": perturbation,
            "diagnostics": diagnostics_summary(per_target_summary),
        }
