from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


DEFAULT_SEED = 42


def resolve_seed(config: dict[str, Any] | None = None, default: int = DEFAULT_SEED) -> int:
    if config is None:
        return int(default)

    if isinstance(config, dict):
        if "seed" in config and config["seed"] is not None:
            return int(config["seed"])
        training_cfg = config.get("training", {})
        if isinstance(training_cfg, dict) and training_cfg.get("seed") is not None:
            return int(training_cfg["seed"])

    return int(default)


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = False) -> int:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark)

    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass

    return seed


def seed_from_config(
    config: dict[str, Any] | None,
    default: int = DEFAULT_SEED,
    deterministic: bool = False,
    benchmark: bool = False,
    verbose: bool = True,
) -> int:
    seed = resolve_seed(config, default=default)
    set_seed(seed, deterministic=deterministic, benchmark=benchmark)
    if verbose:
        print(f"Random seed set to {seed} (deterministic={bool(deterministic)})")
    return seed


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    g = torch.Generator(device=str(device))
    g.manual_seed(int(seed))
    return g
