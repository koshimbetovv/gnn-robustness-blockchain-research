# Robustness of Fraud-Detection Graph Neural Networks (Blockchain)

Codebase for **node-level fraud detection** with GNNs (GCN / GAT / GraphSAGE) and **robustness testing** under (mostly structural) adversarial attacks on transaction graphs.

At the moment, the main supported dataset is **Elliptic** (Bitcoin transaction graph). The project is structured so additional blockchain datasets/graphs can be added later.

## What’s implemented

### Models
- **GCN** (`src/models/gcn.py`) – multi-layer with optional `LayerNorm` + dropout.
- **GAT** (`src/models/gat.py`) – multi-head attention, optional `LayerNorm` + dropout.
- **GraphSAGE** (`src/models/graphsage.py`) – 2-layer SAGEConv.

### Training / evaluation
- Training scripts with precision/recall/F1 tracking:
  - `scripts/train_gcn.py`
  - `scripts/train_gat.py`
  - `scripts/train_graphsage.py`
- An extended GCN trainer with **Focal Loss** + optional **threshold tuning** on a dev split:
  - `scripts/train_gcn_focal.py`
- Loading and comparing saved checkpoints + confusion-matrix plots:
  - `scripts/evaluate_models.py`

### Attacks (current status)
Implemented:
- `src/attacks/nettack_local.py` — **greedy targeted edge-addition** (samples candidate non-neighbors and adds the edge that maximizes CE loss on the target node).
- `src/attacks/nettack.py` — a **toy NETTACK-style gradient attack** on a dense adjacency matrix (not scalable; toggles edges, so it may add/remove).

Placeholders (files exist but are currently empty):
- `fgsm.py`, `pgd.py`, `random_edge_inject.py`, `random_node_inject.py`, `node_inject_plus_fgsm.py`

## Setup

### 1) Install dependencies
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: `torch-geometric` often requires installing wheels matching your PyTorch/CUDA/MPS setup.

### 2) Prepare data (Elliptic)
This project expects the Elliptic raw CSVs in:

```
data/raw/elliptic/
  elliptic_txs_features.csv
  elliptic_txs_classes.csv
  elliptic_txs_edgelist.csv
```

Then run preprocessing:

```bash
python scripts/preprocess_elliptic.py
```

This creates:

```
data/processed/elliptic/data.pt
```

**Label convention** used throughout the project:
- `y = 0` → legitimate
- `y = 1` → fraudulent/illicit
- `y = -1` → unlabeled (ignored by masks)

**Time split** used in `preprocess_elliptic.py`:
- train: timesteps 1–41
- test: timesteps 42–49

Features are standardized using a `StandardScaler` fit on **train only**, then applied to all nodes.

## Run training

Train baseline models (each saves into `models/<name>_YYYYMMDD_HHMMSS/`):

```bash
python scripts/train_gcn.py
python scripts/train_gat.py
python scripts/train_graphsage.py
```

Train GCN with optional focal loss / threshold tuning:

```bash
python scripts/train_gcn_focal.py
```

Each training script prints:
- accuracy
- precision/recall/F1 for the positive class (fraud)
- macro precision/recall/F1
- 2×2 confusion matrix `[[TN, FP], [FN, TP]]`

## Evaluate saved checkpoints

To load the latest checkpoint of each model type in `models/` and plot confusion matrices:

```bash
python scripts/evaluate_models.py
```

Output figures are saved under `results/figures/`.

## Adversarial attack demo (NETTACK-Local)

A minimal usage pattern for `NettackLocalAttack` is shown in `src/main.py` (targeted attack on a single node).

Key points:
- Choose a **labeled** target node (`y != -1`).
- The attack, as implemented, **adds edges only** (no deletions).

If you want to run the demo, make sure your `data.pt` exists, then run:

```bash
python -m src.main
```

> `src/main.py` is a simple example entry point and may need small edits if you want a full “experiment runner” (the `config/experiments/` folder referenced in comments is not included).

## Checkpoints and outputs

Training runs are saved as:

```
models/
  gcn_YYYYMMDD_HHMMSS/
    model.pt
    metrics.json
    config.json
  gat_YYYYMMDD_HHMMSS/
    ...
  graphsage_YYYYMMDD_HHMMSS/
    ...
```

Utilities for loading checkpoints:
- `src/utils/model_loader.py` provides `load_model(...)` and `load_all_models(...)`.

## Debugging helpers

Useful scripts under `tools/`:
- `tools/check_data.py` – NaN/Inf checks + basic statistics + label set.
- `tools/check_loader.py` – smoke test: load latest checkpoints and evaluate train/test.
- `tools/diagnose_models.py` – inspect prediction distributions (helps detect degenerate models).
- `tools/debug_gcn.py`, `tools/debug_graphsage.py` – step-by-step forward/backward tracing for NaN/Inf.

## Known gaps / TODOs
- Most attacks (FGSM/PGD/random injections) are currently placeholders.
- `src/training/metrics.py`, `src/utils/seed.py`, and `src/utils/logging.py` are present but empty.
- There is no unified “experiment runner” yet (training is script-based).

