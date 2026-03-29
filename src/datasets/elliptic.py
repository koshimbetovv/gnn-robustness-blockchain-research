from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


@dataclass
class EllipticConfig:
    feature_path: str = "data/raw/elliptic/elliptic_txs_features.csv"
    class_path: str = "data/raw/elliptic/elliptic_txs_classes.csv"
    edge_path: str = "data/raw/elliptic/elliptic_txs_edgelist.csv"
    train_start: int = 1
    train_end: int = 34
    test_start: int = 35
    test_end: int = 49
    filter_unknown: bool = True


def _read_raw_tables(cfg: EllipticConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(cfg.feature_path, header=None)
    classes = pd.read_csv(cfg.class_path)
    edges = pd.read_csv(cfg.edge_path)
    return features, classes, edges


def _map_labels(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip().str.lower()
    mapped = raw.map({"1": 1, "2": 0, "unknown": -1, "3": -1, "nan": -1})
    return mapped.fillna(-1).astype(int)


def load_elliptic(cfg: EllipticConfig | None = None) -> Data:
    cfg = cfg or EllipticConfig()

    features, classes, edges = _read_raw_tables(cfg)
    features = features.sort_values(by=0).reset_index(drop=True)

    tx_ids = features.iloc[:, 0].astype(str).tolist()
    time_step = features.iloc[:, 1].astype(int).to_numpy()

    classes = classes.copy()
    classes["txId"] = classes["txId"].astype(str)
    classes["y"] = _map_labels(classes["class"])
    y_dict = dict(zip(classes["txId"], classes["y"]))

    feature_cols = list(features.columns[2:])
    feature_names = [f"feature_{i}" for i in range(len(feature_cols))]
    x = features.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = np.array([y_dict.get(tx, -1) for tx in tx_ids], dtype=np.int64)

    if cfg.filter_unknown:
        keep = y != -1
        if keep.sum() == 0:
            raise ValueError("No Elliptic nodes remain after filtering unknown labels.")
        tx_ids = [tx for tx, flag in zip(tx_ids, keep) if flag]
        time_step = time_step[keep]
        x = x[keep]
        y = y[keep]

    train_mask_np = (time_step >= cfg.train_start) & (time_step <= cfg.train_end)
    test_mask_np = (time_step >= cfg.test_start) & (time_step <= cfg.test_end)
    if train_mask_np.sum() == 0:
        raise ValueError("Train split is empty. Check train_start/train_end.")
    if test_mask_np.sum() == 0:
        raise ValueError("Test split is empty. Check test_start/test_end.")

    scaler = StandardScaler()
    scaler.fit(x[train_mask_np])
    x = scaler.transform(x).astype(np.float32)

    id2idx = {tx: i for i, tx in enumerate(tx_ids)}
    edge_pairs: list[tuple[int, int]] = []
    for u, v in edges[["txId1", "txId2"]].itertuples(index=False):
        su = str(u)
        sv = str(v)
        if su in id2idx and sv in id2idx:
            edge_pairs.append((id2idx[su], id2idx[sv]))

    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )
    data.time_step = torch.tensor(time_step, dtype=torch.long)
    data.train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask_np, dtype=torch.bool)
    data.node_id = tx_ids
    data.feature_names = feature_names
    return data


class EllipticDataset:
    def __init__(self, cfg: EllipticConfig | None = None):
        self.cfg = cfg or EllipticConfig()
        self.data = load_elliptic(self.cfg)

    def get_data(self) -> Data:
        return self.data
