from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def runtime_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "git_commit": _git_commit_hash(),
    }


def to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return to_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)


def print_epoch_metrics(
    epoch: int,
    loss: float,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    *,
    lr: float | None = None,
    loss_label: str = "Loss",
) -> None:
    lr_part = f" | LR={lr:.6f}" if lr is not None else ""
    print(
        f"Epoch {epoch:4d} | {loss_label}={loss:.6f}{lr_part} | "
        f"TRAIN: P={train_metrics['precision_pos']:.4f} R={train_metrics['recall_pos']:.4f} "
        f"F1={train_metrics['f1_pos']:.4f} (FraudPred={train_metrics['fraud_predictions']}) | "
        f"TEST: P={test_metrics['precision_pos']:.4f} R={test_metrics['recall_pos']:.4f} "
        f"F1={test_metrics['f1_pos']:.4f} (FraudPred={test_metrics['fraud_predictions']})"
    )


def print_eval_epoch_metrics(
    epoch: int,
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    *,
    loss_label: str = "Loss",
) -> None:
    print(
        f"Epoch {epoch:4d} | "
        f"TRAIN: {loss_label}={train_metrics['loss']:.6f} P={train_metrics['precision_pos']:.4f} "
        f"R={train_metrics['recall_pos']:.4f} F1={train_metrics['f1_pos']:.4f} "
        f"(FraudPred={train_metrics['fraud_predictions']}) | "
        f"TEST: {loss_label}={test_metrics['loss']:.6f} P={test_metrics['precision_pos']:.4f} "
        f"R={test_metrics['recall_pos']:.4f} F1={test_metrics['f1_pos']:.4f} "
        f"(FraudPred={test_metrics['fraud_predictions']})"
    )


def print_final_metrics(title: str, metrics: dict[str, Any]) -> None:
    cm = metrics["confusion_matrix_2x2"]
    split = metrics.get("split")
    split_prefix = f"[{split}] " if split is not None else ""
    print(f"\n=== {title} Final Test Evaluation ===")
    print(f"{split_prefix}#labeled={metrics['n_labeled']} | FraudPred={metrics['fraud_predictions']}")
    print(
        f"  Acc={metrics['acc']:.4f} | "
        f"P(pos)={metrics['precision_pos']:.4f} "
        f"R(pos)={metrics['recall_pos']:.4f} "
        f"F1(pos)={metrics['f1_pos']:.4f}"
    )
    print(
        f"  Macro: P={metrics['precision_macro']:.4f} "
        f"R={metrics['recall_macro']:.4f} "
        f"F1={metrics['f1_macro']:.4f}"
    )
    print(f"  CM [[TN,FP],[FN,TP]] = {cm}")


def save_run(
    model: torch.nn.Module,
    metrics: dict[str, Any],
    config: dict[str, Any],
    prefix: str,
    *,
    extra_metrics: dict[str, Any] | None = None,
) -> str | None:
    save_cfg = config.get("save", {})
    save_dir = Path(save_cfg.get("save_dir", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)

    if not save_cfg.get("save_run", True):
        model_path = save_dir / str(save_cfg.get("filename", "model.pt"))
        torch.save(model.state_dict(), model_path)
        print(f"\n✓ Saved model to {model_path}")
        return str(model_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = str(save_cfg.get("prefix", prefix))
    out_dir = save_dir / f"{run_prefix}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = runtime_metadata()
    torch.save(model.state_dict(), out_dir / "model.pt")

    metrics_payload: dict[str, Any] = {
        "test": to_serializable(metrics),
        "run_metadata": metadata,
    }
    if extra_metrics:
        for key, value in extra_metrics.items():
            metrics_payload[key] = to_serializable(value)
    save_json(out_dir / "metrics.json", metrics_payload)

    config_payload = dict(to_serializable(config)) if isinstance(config, dict) else {"config": to_serializable(config)}
    config_payload["_run"] = metadata
    save_json(out_dir / "config.json", config_payload)

    print(f"\n✓ Saved run to {out_dir}")
    return str(out_dir)
