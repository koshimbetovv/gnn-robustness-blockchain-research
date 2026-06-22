import json
import os
from datetime import datetime
from typing import Optional

import torch

from src.training.metrics import (
    attack_success_rate,
    asr_pos_neg,
    binary_classification_metrics,
    mean_confidence_drop,
    mean_perturbation_l2_on_success,
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
        self.clean_rows = []
        self.adv_rows = []
        self.adv_f1_pos = []
        self.adv_recall_pos = []
        self.adv_f1_macro = []
        self.adv_roc_auc = []
        self.n_edges_orig = []
        self.n_directed_added_total = 0
        self.n_unique_added_total = 0
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
        n_unique_added: int,
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
        self.clean_rows.append(clean_row.detach().cpu())
        self.adv_rows.append(adv_row.detach().cpu())
        self.n_edges_orig.append(int(n_edges_orig))
        self.n_directed_added_total += int(n_directed_added)
        self.n_unique_added_total += int(n_unique_added)
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
                "n_unique_edges_added": int(n_unique_added),
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
                "attack_effect": None,
                "surrogate": None,
                "per_target": self.per_target,
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
        clean_rows_t = torch.stack(self.clean_rows, dim=0)
        adv_rows_t = torch.stack(self.adv_rows, dim=0)
        target_mask = torch.ones(y_t.numel(), dtype=torch.bool)

        target_clean_m = binary_classification_metrics(y_t, pred_clean_t)
        target_adv_m = binary_classification_metrics(y_t, pred_adv_t)
        target_roc_clean = roc_auc_binary(logits_clean_t, y_t, target_mask)
        target_roc_adv = roc_auc_binary(logits_adv_t, y_t, target_mask)
        asr, ns, na = attack_success_rate(
            y_t, pred_clean_t, pred_adv_t, target_mask
        )
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(
            y_t, logits_clean_t, logits_adv_t, target_mask
        )
        conf_drop, n_used = mean_confidence_drop(
            y_t, logits_clean_t, logits_adv_t, target_mask, only_clean_correct=True
        )
        pert_l2_mean, pert_l2_n = mean_perturbation_l2_on_success(
            clean_rows_t, adv_rows_t, pred_clean_t, pred_adv_t, y_t
        )

        surrogate_target_clean_m = binary_classification_metrics(
            y_t, pred_surrogate_clean_t
        )
        surrogate_target_adv_m = binary_classification_metrics(
            y_t, pred_surrogate_adv_t
        )
        surrogate_asr, surrogate_ns, surrogate_na = attack_success_rate(
            y_t, pred_surrogate_clean_t, pred_surrogate_adv_t, target_mask
        )
        (
            surrogate_asr_p,
            surrogate_sp,
            surrogate_ap,
            surrogate_asr_n,
            surrogate_sn,
            surrogate_an,
        ) = asr_pos_neg(y_t, logits_surrogate_clean_t, logits_surrogate_adv_t, target_mask)
        surrogate_conf_drop, surrogate_n_used = mean_confidence_drop(
            y_t,
            logits_surrogate_clean_t,
            logits_surrogate_adv_t,
            target_mask,
            only_clean_correct=True,
        )

        classification = {
            "scope": "temporal_independent_per_target",
            "labeled_adv_independent_mean": {
                "n_runs": int(y_t.numel()),
                "f1_pos": nanmean(self.adv_f1_pos),
                "recall_pos": nanmean(self.adv_recall_pos),
                "f1_macro": nanmean(self.adv_f1_macro),
                "roc_auc": nanmean(self.adv_roc_auc),
            },
            "target_only": {
                "n": int(y_t.numel()),
                "f1_pos": {
                    "clean": target_clean_m["f1_pos"],
                    "adv": target_adv_m["f1_pos"],
                    "drop": float(target_clean_m["f1_pos"] - target_adv_m["f1_pos"]),
                },
                "recall_pos": {
                    "clean": target_clean_m["recall_pos"],
                    "adv": target_adv_m["recall_pos"],
                    "drop": float(
                        target_clean_m["recall_pos"] - target_adv_m["recall_pos"]
                    ),
                },
                "f1_macro": {
                    "clean": target_clean_m["f1_macro"],
                    "adv": target_adv_m["f1_macro"],
                    "drop": float(
                        target_clean_m["f1_macro"] - target_adv_m["f1_macro"]
                    ),
                },
                "roc_auc": {"clean": target_roc_clean, "adv": target_roc_adv},
                "clean_metrics": target_clean_m,
                "adv_metrics": target_adv_m,
            },
        }

        attack_effect = {
            "n_targets": int(y_t.numel()),
            "asr": {"value": asr, "success": ns, "attempted": na},
            "asr_pos_neg": {
                "asr_pos": asr_p,
                "succ_pos": sp,
                "attempted_pos": ap,
                "asr_neg": asr_n,
                "succ_neg": sn,
                "attempted_neg": an,
            },
            "mean_confidence_drop": {"value": conf_drop, "n": n_used},
            "perturbation_l2_on_success": {
                "value": pert_l2_mean,
                "n_flipped": pert_l2_n,
            },
            "structural": {
                "mode": "independent_per_target_sum",
                "n_edges_orig_min": int(min(self.n_edges_orig)),
                "n_edges_orig_max": int(max(self.n_edges_orig)),
                "n_edges_orig_mean": float(sum(self.n_edges_orig) / len(self.n_edges_orig)),
                "n_directed_edges_added_total": self.n_directed_added_total,
                "n_unique_edges_added_total": self.n_unique_added_total,
                "n_targets_with_edge_added_total": self.n_targets_with_edge_total,
                "mean_unique_edges_per_target": float(self.n_unique_added_total)
                / float(y_t.numel()),
            },
        }

        surrogate = {
            "classification": {
                "scope": "temporal_independent_per_target",
                "target_only": {
                    "n": int(y_t.numel()),
                    "f1_pos": {
                        "clean": surrogate_target_clean_m["f1_pos"],
                        "adv": surrogate_target_adv_m["f1_pos"],
                        "drop": float(
                            surrogate_target_clean_m["f1_pos"]
                            - surrogate_target_adv_m["f1_pos"]
                        ),
                    },
                    "clean_metrics": surrogate_target_clean_m,
                    "adv_metrics": surrogate_target_adv_m,
                },
            },
            "attack_effect": {
                "asr": {
                    "value": surrogate_asr,
                    "success": surrogate_ns,
                    "attempted": surrogate_na,
                },
                "asr_pos_neg": {
                    "asr_pos": surrogate_asr_p,
                    "succ_pos": surrogate_sp,
                    "attempted_pos": surrogate_ap,
                    "asr_neg": surrogate_asr_n,
                    "succ_neg": surrogate_sn,
                    "attempted_neg": surrogate_an,
                },
                "mean_confidence_drop": {
                    "value": surrogate_conf_drop,
                    "n": surrogate_n_used,
                },
            },
        }
        return {
            "classification": classification,
            "attack_effect": attack_effect,
            "surrogate": surrogate,
            "per_target": self.per_target,
        }
