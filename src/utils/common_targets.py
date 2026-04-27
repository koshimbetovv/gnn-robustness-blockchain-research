"""Compute the set of clean-correct illicit Elliptic tx_ids per model.

Each model in this project consumes a different dataset shape and forward API
(static whole-graph, per-timestep RecGNN sequence, EvolveGCN sliding window,
CoSemiGNN per-slice). To choose attack targets that are fair across all models,
we evaluate each model on its native test split and return the set of tx_ids it
classifies correctly as illicit (y=1, pred=1). Intersecting these sets gives the
common-target pool used by run_pgd_attack_common_target.py.
"""

from __future__ import annotations

import json
import os
from typing import Set

import torch


def _open_ckpt_config(model_dir: str, model_name: str) -> tuple[str, dict]:
    from src.utils.model_loader import resolve_checkpoint
    ckpt_path, ckpt_run_dir = resolve_checkpoint(model_name, model_dir=model_dir)
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return ckpt_path, cfg


def static_clean_correct_illicit_txids(model_name: str, model_dir: str, device) -> Set[str]:
    """gcn / graphsage / gat / chronowave_gnn → Elliptic full graph forward."""
    from src.datasets.elliptic import EllipticDataset, EllipticConfig
    from src.utils.model_loader import load_model
    from src.attacks.model_forward import forward_logits
    from src.training.metrics import get_split_mask

    data = EllipticDataset(EllipticConfig(filter_unknown=False)).get_data()
    if model_name == "chronowave_gnn":
        from src.datasets.chronowave_features import build_paper_features
        build_paper_features(data)

    data = data.to(device)
    model = load_model(model_name, data.num_features, 2, device=device, model_dir=model_dir)
    time_step = getattr(data, "time_step", None)
    split_mask = get_split_mask(data, "test").to(device)

    with torch.no_grad():
        logits = forward_logits(model, data.x, data.edge_index, time_step=time_step)
    pred = logits.argmax(dim=1)

    sel = split_mask & (data.y == 1) & (pred == data.y)
    sel_idx = sel.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
    return {data.node_id[i] for i in sel_idx}


def recgnn_clean_correct_illicit_txids(model_dir: str, device) -> Set[str]:
    from src.datasets.recgnn_elliptic import RecGNNEllipticConfig, RecGNNEllipticDataset
    from src.utils.model_loader import load_model

    _, ckpt_cfg = _open_ckpt_config(model_dir, "recgnn")
    sequence = RecGNNEllipticDataset(RecGNNEllipticConfig(**ckpt_cfg["data"])).get_sequence()
    model = load_model(
        "recgnn", sequence.num_features, 2,
        device=device, model_dir=model_dir,
    )

    model.reset_sequence_state(device)
    with torch.no_grad():
        for g in sequence.train_graphs:
            gd = g.to(device)
            _ = model(gd.x.float(), gd.edge_index.long())
            model.detach_sequence_state()

    txids: Set[str] = set()
    with torch.no_grad():
        for g in sequence.test_graphs:
            gd = g.to(device)
            log_probs = model(gd.x.float(), gd.edge_index.long())
            model.detach_sequence_state()
            pred = log_probs.argmax(dim=1)
            ok = (gd.y == 1) & (pred == 1)
            for i in ok.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist():
                txids.add(g.tx_ids[i])
    return txids


def evolvegcn_clean_correct_illicit_txids(model_dir: str) -> Set[str]:
    """EvolveGCN-O uses RReLU (not implemented on MPS); we run it on CPU."""
    import pandas as pd
    from scripts.training.train_evolvegcn_utils import build_evolvegcn_model, move_sample
    from src.datasets.evolvegcn_elliptic import (
        EvolveGCNEllipticConfig, EvolveGCNEllipticDataset,
    )
    from src.attacks.pgd_evolvegcn import EvolveGCNPGDAttack

    use_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ckpt_path, ckpt_cfg = _open_ckpt_config(model_dir, "evolvegcn_o")
    sequence = EvolveGCNEllipticDataset(
        EvolveGCNEllipticConfig(**ckpt_cfg["data"])
    ).get_sequence()

    model = build_evolvegcn_model(sequence.num_features, ckpt_cfg).to(use_device)
    state = torch.load(ckpt_path, map_location=use_device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Loaded EVOLVEGCN_O from {ckpt_path}")

    atk = EvolveGCNPGDAttack(model, use_device, attack_start_col=1)

    # EvolveGCN preserves features.csv row order (no sort), so its global node id
    # is simply the row index in the original CSV.
    features_df = pd.read_csv(ckpt_cfg["data"]["feature_path"], header=None)
    tx_ids_unsorted = features_df.iloc[:, 0].astype(str).tolist()

    txids: Set[str] = set()
    for sample in sequence.test_samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(
            sample, use_device,
        )
        with torch.no_grad():
            logits = atk.forward_labels(
                hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx,
            ).detach()
        pred = logits.argmax(dim=1)
        ok = (label_vals == 1) & (pred == 1)
        for p in ok.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist():
            gidx = int(label_idx[p].item())
            txids.add(tx_ids_unsorted[gidx])
    return txids


def cosemignn_clean_correct_illicit_txids(model_dir: str, device) -> Set[str]:
    import pandas as pd
    from src.datasets.cosemignn_elliptic import load_cosemignn_elliptic
    from src.attacks.pgd_cosemignn import CoSemiPGDAttack
    from src.models.cosemignn import CoSemiGNN

    ckpt_path, ckpt_cfg = _open_ckpt_config(model_dir, "cosemignn")
    data_cfg = ckpt_cfg["data"]
    time_cfg = ckpt_cfg["time"]
    predict_start = int(time_cfg["predict_start"])
    predict_end = int(time_cfg["predict_end"])

    feature_list, adj_list, label_list, _ca_matrix_list, ca_weights_list, *_ = (
        load_cosemignn_elliptic(
            feature_path=data_cfg["feature_path"],
            class_path=data_cfg["class_path"],
            edge_path=data_cfg["edge_path"],
            semi_cache_dir=data_cfg["semi_cache_dir"],
            device=device,
            rebuild_semi=bool(data_cfg.get("rebuild_semi", False)),
        )
    )

    feature_in = None
    for ft in feature_list[1:]:
        if ft is not None and ft.numel() > 0:
            feature_in = ft.size(1)
            break
    if feature_in is None:
        raise RuntimeError("CoSemiGNN: no non-empty feature slices.")

    model_cfg = ckpt_cfg.get("cosemignn", {})
    model = CoSemiGNN(
        feature_in=feature_in,
        dim=int(model_cfg.get("dim", 128)),
        dim2=int(model_cfg.get("dim2", 256)),
        dim3=int(model_cfg.get("dim3", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Loaded COSEMIGNN from {ckpt_path}")
    atk = CoSemiPGDAttack(model, device, raw_feature_dim=feature_in - 6)

    # The CoSemiGNN loader filters to labeled rows then sorts each timestep slice
    # by txId. Replicate that ordering here so slice index → tx_id is consistent.
    features = pd.read_csv(data_cfg["feature_path"], header=None)
    classes = pd.read_csv(data_cfg["class_path"])
    features = features.rename(columns={0: "txId", 1: "time_step"})
    features["txId"] = features["txId"].astype(str)
    classes["class"] = classes["class"].astype(str).str.strip()
    classes["label"] = classes["class"].map({"1": 1, "2": 0}).fillna(-1).astype(int)
    classes["txId"] = classes["txId"].astype(str)
    df = features.merge(classes[["txId", "label"]], on="txId", how="left")
    df = df[df["label"] != -1].copy()

    txids: Set[str] = set()
    for t in range(predict_start, predict_end):
        if t >= len(feature_list):
            continue
        features_t = feature_list[t]
        adj_t = adj_list[t]
        labels_t = label_list[t]
        ca_weights_t = ca_weights_list[t]
        if features_t is None or labels_t is None or features_t.numel() == 0 or labels_t.numel() == 0:
            continue
        with torch.no_grad():
            logits = atk.forward_logits(features_t, adj_t, ca_weights_t)
        pred = logits.argmax(dim=1)
        ok = (labels_t == 1) & (pred == 1)
        sel_pos = ok.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
        if not sel_pos:
            continue
        slice_txids = (
            df[df["time_step"] == t].sort_values("txId").reset_index(drop=True)["txId"].tolist()
        )
        for p in sel_pos:
            txids.add(slice_txids[p])
    return txids


def compute_clean_correct_illicit_txids(model_name: str, model_dir: str, device) -> Set[str]:
    """Dispatch to the right per-architecture computation."""
    name = model_name.lower()
    if name in ("gcn", "graphsage", "gat", "chronowave_gnn"):
        return static_clean_correct_illicit_txids(name, model_dir, device)
    if name == "recgnn":
        return recgnn_clean_correct_illicit_txids(model_dir, device)
    if name == "evolvegcn_o":
        return evolvegcn_clean_correct_illicit_txids(model_dir)
    if name == "cosemignn":
        return cosemignn_clean_correct_illicit_txids(model_dir, device)
    raise ValueError(f"Unknown model_name: {model_name}")
