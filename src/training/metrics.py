from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def get_split_mask(data, split: str) -> torch.Tensor:
    labeled = (data.y != -1)
    attr = f"{split}_mask"
    if hasattr(data, attr):
        m = getattr(data, attr)
        if m.dtype != torch.bool:
            m = m.bool()
        return m & labeled
    return labeled


@dataclass
class SplitMetrics:
    split: str
    n_labeled: int
    acc: float
    precision_pos: float
    recall_pos: float
    f1_pos: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    fraud_predictions: int
    confusion_matrix_2x2: list




def binary_classification_metrics(y_true, y_pred, split: str | None = None) -> dict:
    y_true = torch.as_tensor(y_true).detach().cpu().numpy()
    y_pred = torch.as_tensor(y_pred).detach().cpu().numpy()

    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty label set.")

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pM, rM, f1M, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    out = {
        "n_labeled": int(y_true.shape[0]),
        "acc": acc,
        "precision_pos": float(p[1]),
        "recall_pos": float(r[1]),
        "f1_pos": float(f1[1]),
        "precision_macro": float(pM),
        "recall_macro": float(rM),
        "f1_macro": float(f1M),
        "fraud_predictions": int((y_pred == 1).sum()),
        "confusion_matrix_2x2": cm.tolist(),
    }
    if split is not None:
        out["split"] = split
    return out


@torch.no_grad()
def evaluate_logits_on_split(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, split: str) -> SplitMetrics:
    y_true = y[mask].detach().cpu()
    y_pred = logits[mask].argmax(dim=1).detach().cpu()

    acc = float(accuracy_score(y_true, y_pred))

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pM, rM, f1M, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fraud_pred = int((y_pred == 1).sum())

    return SplitMetrics(
        split=split,
        n_labeled=int(mask.sum().item()),
        acc=acc,
        precision_pos=float(p[1]),
        recall_pos=float(r[1]),
        f1_pos=float(f1[1]),
        precision_macro=float(pM),
        recall_macro=float(rM),
        f1_macro=float(f1M),
        fraud_predictions=fraud_pred,
        confusion_matrix_2x2=cm.tolist(),
    )


def attack_success_rate(
    y_true: torch.Tensor,
    y_pred_clean: torch.Tensor,
    y_pred_adv: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, int, int]:
    """ASR = (# correct on clean AND wrong after attack) / (# correct on clean)."""
    y_true = y_true[mask]
    clean = y_pred_clean[mask]
    adv = y_pred_adv[mask]

    correct_before = (clean == y_true)
    attempted = int(correct_before.sum().item())
    if attempted == 0:
        return 0.0, 0, 0

    success = correct_before & (adv != y_true)
    n_success = int(success.sum().item())
    return float(n_success / attempted), n_success, attempted


@torch.no_grad()
def asr_pos_neg(
    y: torch.Tensor,
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    mask: torch.Tensor,
    pos_label: int = 1,
):
    """
    ASR_pos = (# clean-correct positives flipped) / (# clean-correct positives)
    ASR_neg = (# clean-correct negatives flipped) / (# clean-correct negatives)

    Returns:
      (asr_pos, succ_pos, attempted_pos, asr_neg, succ_neg, attempted_neg)
    """
    mask = mask.bool() & (y != -1)

    y_true = y[mask].long()
    pred_clean = logits_clean[mask].argmax(dim=1)
    pred_adv = logits_adv[mask].argmax(dim=1)

    clean_correct = (pred_clean == y_true)

    # positives (illicit)
    pos = (y_true == pos_label) & clean_correct
    attempted_pos = int(pos.sum().item())
    succ_pos = int((pred_adv[pos] != y_true[pos]).sum().item())
    asr_pos = (succ_pos / attempted_pos) if attempted_pos > 0 else 0.0

    # negatives (licit)
    neg = (y_true != pos_label) & clean_correct
    attempted_neg = int(neg.sum().item())
    succ_neg = int((pred_adv[neg] != y_true[neg]).sum().item())
    asr_neg = (succ_neg / attempted_neg) if attempted_neg > 0 else 0.0

    return asr_pos, succ_pos, attempted_pos, asr_neg, succ_neg, attempted_neg


@torch.no_grad()
def roc_auc_binary(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    pos_index: int = 1,
) -> float:
    """
    ROC-AUC for binary classification using P(class=pos_index).
    Returns NaN if split has only one class.
    """
    mask = mask.bool()
    y_true = y[mask].detach().cpu().numpy()

    # If labels include -1 (unlabeled), filter them out
    valid = (y_true != -1)
    y_true = y_true[valid]
    if y_true.size == 0:
        return float("nan")

    probs_pos = F.softmax(logits[mask], dim=1)[:, pos_index].detach().cpu().numpy()
    probs_pos = probs_pos[valid]

    # roc_auc_score requires both classes present
    try:
        return float(roc_auc_score(y_true, probs_pos))
    except ValueError:
        return float("nan")


@torch.no_grad()
def mean_confidence_drop(
    y: torch.Tensor,
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    mask: torch.Tensor,
    only_clean_correct: bool = True,
) -> tuple[float, int]:
    """
    Mean drop in true-class probability: E[p_clean(y) - p_adv(y)] over nodes in mask.
    If only_clean_correct=True, average only over nodes that are correct on clean.
    Returns (mean_drop, n_used). If n_used==0 -> (0.0, 0).
    """
    mask = mask.bool()
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return 0.0, 0

    y_true = y[idx].long()
    valid = (y_true != -1)
    idx = idx[valid]
    y_true = y_true[valid]
    if idx.numel() == 0:
        return 0.0, 0

    pred_clean = logits_clean[idx].argmax(dim=1)
    if only_clean_correct:
        keep = (pred_clean == y_true)
        idx = idx[keep]
        y_true = y_true[keep]
        if idx.numel() == 0:
            return 0.0, 0

    p_clean = F.softmax(logits_clean[idx], dim=1).gather(1, y_true.view(-1, 1)).squeeze(1)
    p_adv = F.softmax(logits_adv[idx], dim=1).gather(1, y_true.view(-1, 1)).squeeze(1)

    drop = (p_clean - p_adv).mean().item()
    return float(drop), int(idx.numel())
