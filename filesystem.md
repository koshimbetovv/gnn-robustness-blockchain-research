# Repository filesystem guide

This document explains what each folder/file is for, and which pieces are inputs vs generated outputs.

> Note: `.DS_Store` and `__MACOSX/` are macOS artifacts and can be ignored.

## High-level tree

```
gnn-adversarial-blockchain/
├── README.md
├── filesystem.md
├── requirements.txt
├── config/
│   ├── models/
│   │   ├── gcn.yaml
│   │   ├── gat.yaml
│   │   └── graphsage.yaml
│   └── datasets/
│       └── elliptic.yaml             # placeholder
├── data/
│   ├── raw/
│   │   └── elliptic/                 # user-provided raw CSVs (not in repo)
│   └── processed/                    # optional cached/generated artifacts
├── scripts/
│   ├── train_gcn.py
│   ├── train_gcn_focal.py
│   ├── train_gat.py
│   ├── train_graphsage.py
│   └── evaluate_models.py
├── src/
│   ├── main.py
│   ├── datasets/
│   │   └── elliptic.py
│   ├── models/
│   │   ├── gcn.py
│   │   ├── gat.py
│   │   └── graphsage.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── loss.py
│   │   └── metrics.py                # placeholder (empty)
│   ├── attacks/
│   │   ├── base_attack.py
│   │   ├── fgsm.py                   # placeholder (empty)
│   │   ├── pgd.py                    # placeholder (empty)
│   │   ├── random_edge_inject.py     # placeholder (empty)
│   │   ├── random_node_inject.py     # placeholder (empty)
│   │   └── node_inject_plus_fgsm.py  # placeholder (empty)
│   └── utils/
│       ├── model_loader.py
│       ├── graph_utils.py
│       ├── seed.py                   # placeholder (empty)
│       └── logging.py                # placeholder (empty)
├── tools/
│   ├── check_data.py
│   ├── check_loader.py
│   ├── diagnose_models.py
│   ├── debug_gcn.py
│   └── debug_graphsage.py
├── models/                           # generated checkpoints (may already contain runs)
└── results/                          # generated plots (e.g., confusion matrices)
```

## Top-level files

### `README.md`
Project overview + how to run training/evaluation.

### `requirements.txt`
Python dependencies (PyTorch, PyG, sklearn, pandas, etc.).

### `filesystem.md`
This file.

## `config/`
YAML configs. In the current state of the repo, most *training* scripts use in-file `CONFIG = {...}` dictionaries; the YAML configs are mainly used as:
- documentation for default hyperparameters
- fallback configs when loading checkpoints (see `src/utils/model_loader.py`)

Subfolders:
- `config/models/*.yaml` — model hyperparameters (hidden size, dropout, heads, etc.).
- `config/datasets/*.yaml` — dataset-related configs (currently placeholder).

## `data/`
Data is separated into:
- `data/raw/` — **user-provided** raw files (not shipped with the repo).
- `data/processed/` — optional cached/generated artifacts. The main Elliptic pipeline now loads raw CSVs directly.

### Elliptic layout
Expected raw files:
- `data/raw/elliptic/elliptic_txs_features.csv`
- `data/raw/elliptic/elliptic_txs_classes.csv`
- `data/raw/elliptic/elliptic_txs_edgelist.csv`

The repo now loads these raw CSVs directly through `src/datasets/elliptic.py`.
No standalone preprocessing script is required for the main training and attack workflows.

## `scripts/`
Runnable entry points (intended to be executed from the repo root).

- `train_gcn.py`, `train_gat.py`, `train_graphsage.py`
  - Train a baseline model on Elliptic.
  - Saves to `models/<model>_YYYYMMDD_HHMMSS/`.

- `train_gcn_focal.py`
  - GCN training with optional **Focal Loss** and optional threshold tuning on a dev subset.

- `evaluate_models.py`
  - Loads latest checkpoints for each model type and evaluates train/test.
  - Writes confusion-matrix plots to `results/figures/`.

## `src/`
The “library” code.

### `src/datasets/`
- `elliptic.py` — raw CSV loader for the Elliptic dataset; builds `x`, `edge_index`, `y`, `time_step`, `train_mask`, and `test_mask` directly from the original files.

### `src/models/`
- `gcn.py` — multi-layer GCN with optional LayerNorm.
- `gat.py` — multi-head GAT with optional LayerNorm.
- `graphsage.py` — simple 2-layer GraphSAGE.

### `src/training/`
- `trainer.py` — basic training loop with optional class weights.
- `evaluator.py` — simple accuracy evaluation on a given split.
- `loss.py` — focal loss implementation used by `scripts/train_gcn_focal.py`.
- `metrics.py` — placeholder for future metrics (currently empty).

### `src/attacks/`
- `base_attack.py` — attack interface.
- attack implementations for FGSM, PGD, node injection, MonTi/TDGIA-style injection, and adapted NETTACK.

### `src/utils/`
- `model_loader.py` — checkpoint discovery and model reconstruction; provides `load_model()` / `load_all_models()`.
- `graph_utils.py` — utilities (currently only `edge_index_to_adj`).
- `seed.py`, `logging.py` — shared utilities for reproducibility and run metadata/logging.

## `tools/`
Debug / diagnostics scripts.

- `check_data.py` — NaN/Inf checks + feature stats + label set.
- `check_loader.py` — loads latest checkpoints and runs a quick evaluation.
- `diagnose_models.py` — inspects prediction distributions.
- `debug_gcn.py`, `debug_graphsage.py` — forward/backward tracing to find NaN/Inf sources.

## Generated output folders

### `models/`
Created by training scripts. Each run is stored as:

- `model.pt` — state dict
- `metrics.json` — evaluation metrics dump
- `config.json` — the training config used for that run

### `results/`
Created by evaluation scripts (currently `results/figures/*.png`).
