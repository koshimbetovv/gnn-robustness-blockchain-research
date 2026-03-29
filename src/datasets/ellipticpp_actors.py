from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def _pick_column(columns: Iterable[str], preferred: list[str], contains: list[str] | None = None) -> str | None:
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in preferred:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if contains:
        for c in cols:
            cl = c.lower()
            if all(tok in cl for tok in contains):
                return c
    return None


def _detect_time_col(df: pd.DataFrame) -> str | None:
    return (
        _pick_column(df.columns, ["Time step", "time_step", "timeStep", "timestep", "ts"])
        or _pick_column(df.columns, [], contains=["time"])
    )


def _detect_label_col(df: pd.DataFrame) -> str:
    col = (
        _pick_column(df.columns, ["class", "Class", "label", "Label", "labels"])
        or _pick_column(df.columns, [], contains=["class"])
        or _pick_column(df.columns, [], contains=["label"])
    )
    if col is None:
        raise ValueError("Could not detect label/class column.")
    return col


def _detect_id_col(feature_df: pd.DataFrame, class_df: pd.DataFrame, time_col: str | None) -> str:
    common = [c for c in feature_df.columns if c in class_df.columns]
    common = [c for c in common if c != time_col and c.lower() not in {"class", "label", "labels"}]
    if not common:
        raise ValueError("Could not detect shared node-id column between features and classes CSVs.")

    preferred = [
        "address",
        "Address",
        "wallet",
        "walletId",
        "wallet_id",
        "walletAddress",
        "wallet_address",
        "addr",
        "addrId",
        "addr_id",
        "node",
        "node_id",
        "nodeId",
        "txId",
        "id",
    ]
    picked = _pick_column(common, preferred)
    return picked if picked is not None else common[0]


def _detect_edge_cols(edge_df: pd.DataFrame) -> tuple[str, str, str | None]:
    src = (
        _pick_column(edge_df.columns, ["source", "src", "from", "node1", "u", "txId1", "addr1", "address1"])
        or edge_df.columns[0]
    )
    remaining = [c for c in edge_df.columns if c != src]
    if not remaining:
        raise ValueError("Edge CSV must contain at least two columns.")
    dst = (
        _pick_column(remaining, ["target", "dst", "to", "node2", "v", "txId2", "addr2", "address2"])
        or remaining[0]
    )
    time_col = _detect_time_col(edge_df)
    return src, dst, time_col


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


def _detect_numeric_feature_cols(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    feature_cols: list[str] = []
    for c in df.columns:
        if c in excluded:
            continue
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().any():
            feature_cols.append(c)
    return feature_cols


def _aggregate_duplicate_actor_rows(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    label_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_spec = {c: "mean" for c in feature_cols}
    agg_spec[label_col] = "first"

    grouped = (
        work.groupby([id_col, time_col], as_index=False)
        .agg(agg_spec)
        .reset_index(drop=True)
    )
    return grouped


def _map_labels(series: pd.Series) -> pd.Series:
    mapped = pd.to_numeric(series, errors="coerce").map({1: 1, 2: 0, 3: -1})
    return mapped.astype("Int64")


@dataclass
class EllipticPPActorsConfig:
    feature_path: str = "data/raw/ellipticpp/actors/wallets_features.csv"
    class_path: str = "data/raw/ellipticpp/actors/wallets_classes.csv"
    edge_path: str = "data/raw/ellipticpp/actors/AddrAddr_edgelist.csv"
    train_start: int = 1
    train_end: int = 34
    test_start: int = 35
    test_end: int = 49
    filter_unknown: bool = True


def load_ellipticpp_actors(cfg: EllipticPPActorsConfig | None = None) -> Data:
    cfg = cfg or EllipticPPActorsConfig()

    feature_df = pd.read_csv(cfg.feature_path)
    class_df = pd.read_csv(cfg.class_path)
    edge_df = pd.read_csv(cfg.edge_path)

    time_col_feature = _detect_time_col(feature_df)
    time_col_class = _detect_time_col(class_df)
    time_col = time_col_feature or time_col_class
    if time_col is None:
        raise ValueError("Could not detect time-step column in Elliptic++ actors features/classes CSVs.")

    label_col = _detect_label_col(class_df)
    id_col = _detect_id_col(feature_df, class_df, time_col)
    src_col, dst_col, edge_time_col = _detect_edge_cols(edge_df)

    feature_df = feature_df.copy()
    class_df = class_df.copy()
    edge_df = edge_df.copy()

    if time_col_feature is not None and time_col_class is not None and time_col_feature != time_col_class:
        class_df = class_df.rename(columns={time_col_class: time_col_feature})
        time_col_class = time_col_feature
        time_col = time_col_feature

    feature_df[id_col] = feature_df[id_col].astype(str)
    class_df[id_col] = class_df[id_col].astype(str)
    edge_df[src_col] = edge_df[src_col].astype(str)
    edge_df[dst_col] = edge_df[dst_col].astype(str)

    merge_keys = [id_col]
    if time_col_feature is not None and time_col_class is not None and time_col_feature == time_col_class:
        merge_keys.append(time_col_feature)
    merged = pd.merge(feature_df, class_df, on=merge_keys, how="inner")

    label_col = _detect_label_col(merged)
    merged[time_col] = pd.to_numeric(merged[time_col], errors="coerce").astype("Int64")
    merged[label_col] = _map_labels(merged[label_col])
    merged = merged.dropna(subset=[time_col, label_col]).copy()
    merged[time_col] = merged[time_col].astype(int)

    if edge_time_col is not None:
        edge_df[edge_time_col] = pd.to_numeric(edge_df[edge_time_col], errors="coerce").astype("Int64")

    if cfg.filter_unknown:
        merged = merged[merged[label_col].isin([0, 1])].copy()

    excluded = set(merge_keys + [label_col, time_col])
    feature_cols = _detect_numeric_feature_cols(merged, excluded)
    if not feature_cols:
        raise ValueError("No numeric feature columns were detected for Elliptic++ actors.")

    merged = _aggregate_duplicate_actor_rows(
        df=merged,
        id_col=id_col,
        time_col=time_col,
        label_col=label_col,
        feature_cols=feature_cols,
    )
    merged = merged.sort_values([time_col, id_col]).reset_index(drop=True)
    if merged.empty:
        raise ValueError("No Elliptic++ actor nodes remain after preprocessing.")

    time_step = merged[time_col].astype(int).to_numpy()
    train_mask_np = (time_step >= cfg.train_start) & (time_step <= cfg.train_end)
    test_mask_np = (time_step >= cfg.test_start) & (time_step <= cfg.test_end)
    if train_mask_np.sum() == 0:
        raise ValueError("Train split is empty. Check train_start/train_end.")
    if test_mask_np.sum() == 0:
        raise ValueError("Test split is empty. Check test_start/test_end.")

    x = _safe_numeric_frame(merged[feature_cols]).to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(x[train_mask_np])
    x = scaler.transform(x).astype(np.float32)

    node_ids = merged[id_col].astype(str).tolist()
    global_index = {(int(t), str(nid)): i for i, (t, nid) in enumerate(zip(time_step, node_ids))}

    edge_pairs: set[tuple[int, int]] = set()
    unique_times = sorted(np.unique(time_step).tolist())
    for t in unique_times:
        if edge_time_col is not None:
            edge_slice = edge_df[edge_df[edge_time_col] == t]
        else:
            edge_slice = edge_df
        if edge_slice.empty:
            continue

        src_nodes = edge_slice[src_col].astype(str).tolist()
        dst_nodes = edge_slice[dst_col].astype(str).tolist()
        for u, v in zip(src_nodes, dst_nodes):
            ku = (int(t), u)
            kv = (int(t), v)
            if ku not in global_index or kv not in global_index:
                continue
            ui = global_index[ku]
            vi = global_index[kv]
            if ui == vi:
                continue
            edge_pairs.add((ui, vi))

    if edge_pairs:
        edge_index = torch.tensor(sorted(edge_pairs), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(merged[label_col].astype(int).to_numpy(), dtype=torch.long),
    )
    data.time_step = torch.tensor(time_step, dtype=torch.long)
    data.train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask_np, dtype=torch.bool)
    data.node_id = node_ids
    data.feature_names = feature_cols
    return data


class EllipticPPActorsDataset:
    def __init__(self, cfg: EllipticPPActorsConfig | None = None):
        self.cfg = cfg or EllipticPPActorsConfig()
        self.data = load_ellipticpp_actors(self.cfg)

    def get_data(self) -> Data:
        return self.data
