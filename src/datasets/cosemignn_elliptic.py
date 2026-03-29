from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

from src.datasets.cosemignn_pipeline import (
    TransactionSliceConfig,
    train_six_semi_supervised_predictions,
    build_cosemignn_slice_artifacts,
)


def load_cosemignn_elliptic(
    feature_path: str = "data/raw/elliptic/elliptic_txs_features.csv",
    class_path: str = "data/raw/elliptic/elliptic_txs_classes.csv",
    edge_path: str = "data/raw/elliptic/elliptic_txs_edgelist.csv",
    semi_cache_dir: str = "data/processed/cosemignn/elliptic/semi_supervised_results",
    device: str | torch.device = "cpu",
    rebuild_semi: bool = False,
):
    device = torch.device(device)

    features = pd.read_csv(feature_path, header=None)
    classes = pd.read_csv(class_path)
    edges = pd.read_csv(edge_path)

    features = features.rename(columns={0: "txId", 1: "time_step"})
    feat_cols = [f"f_{i}" for i in range(features.shape[1] - 2)]
    features.columns = ["txId", "time_step"] + feat_cols

    classes["class"] = classes["class"].astype(str).str.strip()
    # repo convention: illicit=1, licit=0, unknown=-1
    label_map = {"1": 1, "2": 0}
    classes["label"] = classes["class"].map(label_map).fillna(-1).astype(int)

    df = features.merge(classes[["txId", "label"]], on="txId", how="left")
    df = df[df["label"] != -1].copy()

    spec = TransactionSliceConfig(
        txid_col="txId",
        time_col="time_step",
        label_col="label",
        feature_cols=feat_cols,
        edge_src_col="txId1",
        edge_dst_col="txId2",
        label_values=(0, 1),
        semi_cache_dir=semi_cache_dir,
        build_cache=True,
    )

    feature_list = [None]
    adj_list = [None]
    label_list = [None]
    ca_matrix_list = [None]
    ca_weights_list = [None]
    semi_result_list = [None]
    fake_label_list = [None]
    ca_feature_list = [None]

    Path(semi_cache_dir).mkdir(parents=True, exist_ok=True)

    for t in range(1, 50):
        slice_df = df[df["time_step"] == t].sort_values("txId").reset_index(drop=True)

        if len(slice_df) == 0:
            feature_list.append(torch.empty((0, len(feat_cols) + 6), dtype=torch.float32, device=device))
            adj_list.append(torch.empty((2, 0), dtype=torch.long, device=device))
            label_list.append(torch.empty((0,), dtype=torch.long, device=device))
            ca_matrix_list.append([])
            ca_weights_list.append(torch.empty((0,), dtype=torch.float32, device=device))
            semi_result_list.append(torch.empty((0, 6), dtype=torch.float32, device=device))
            fake_label_list.append(torch.empty((0,), dtype=torch.float32, device=device))
            ca_feature_list.append(torch.empty((0, 7), dtype=torch.float32, device=device))
            continue

        semi_df = train_six_semi_supervised_predictions(
            slice_df=slice_df[["txId"] + feat_cols + ["label"]],
            txid_col="txId",
            label_col="label",
            cache_root=semi_cache_dir,
            cache_tag=f"t{t}",
            rebuild=rebuild_semi,
        )

        artifacts = build_cosemignn_slice_artifacts(
            slice_df=slice_df,
            edge_df=edges,
            spec=spec,
            semi_df=semi_df,
            device=device,
        )

        feature_list.append(artifacts["feature"])
        adj_list.append(artifacts["edge_index"])
        label_list.append(artifacts["labels"])
        ca_matrix_list.append(artifacts["ca_matrix"])
        ca_weights_list.append(artifacts["ca_weights"])
        semi_result_list.append(artifacts["semi_result"])
        fake_label_list.append(artifacts["fake_label"])
        ca_feature_list.append(artifacts["ca_feature"])

    return (
        feature_list,
        adj_list,
        label_list,
        ca_matrix_list,
        ca_weights_list,
        semi_result_list,
        fake_label_list,
        ca_feature_list,
    )