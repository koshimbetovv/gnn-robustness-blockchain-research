from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

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
    dst = (
        _pick_column(remaining, ["target", "dst", "to", "node2", "v", "txId2", "addr2", "address2"])
        or remaining[0]
    )
    time_col = _detect_time_col(edge_df)
    return src, dst, time_col


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)


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
class RecGNNEllipticPPActorsConfig:
    feature_path: str = "data/raw/ellipticpp/actors/wallets_features.csv"
    class_path: str = "data/raw/ellipticpp/actors/wallets_classes.csv"
    edge_path: str = "data/raw/ellipticpp/actors/AddrAddr_edgelist.csv"
    train_start: int = 1
    train_end: int = 34
    test_start: int = 35
    test_end: int = 49
    filter_unknown: bool = True


@dataclass
class RecGNNEllipticPPActorsSequence:
    all_graphs: List[Data]
    train_graphs: List[Data]
    test_graphs: List[Data]
    max_nodes: int
    num_features: int
    num_classes: int = 2


class RecGNNEllipticPPActorsDataset:
    """
    RecGNN sequence builder for Elliptic++ Actors.

    Adaptation to repo style:
    - one PyG graph per timestep
    - all numeric actor features are used as the local feature block
    - ANF adds two counts per node: incoming licit and incoming illicit neighbours
    - features are standardized using train timesteps only
    - optional filtering of unknown nodes, matching existing Elliptic++ actor scripts
    """

    def __init__(self, cfg: RecGNNEllipticPPActorsConfig | None = None):
        self.cfg = cfg or RecGNNEllipticPPActorsConfig()
        self.sequence = self._build_sequence(self.cfg)

    def get_sequence(self) -> RecGNNEllipticPPActorsSequence:
        return self.sequence

    @staticmethod
    def _build_sequence(cfg: RecGNNEllipticPPActorsConfig) -> RecGNNEllipticPPActorsSequence:
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

        non_feature = set(merge_keys + [label_col, time_col])
        feature_cols = [c for c in merged.columns if c not in non_feature]
        feature_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(pd.to_numeric(merged[c], errors="coerce"))
        ]
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

        local_x = _safe_numeric_frame(merged[feature_cols]).to_numpy(dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(local_x[train_mask_np])
        local_x = scaler.transform(local_x).astype(np.float32)

        node_ids = merged[id_col].astype(str).tolist()
        global_index = {(int(t), str(nid)): idx for idx, (t, nid) in enumerate(zip(time_step, node_ids))}
        timestep_to_nodes: dict[int, list[int]] = {}
        for idx, t in enumerate(time_step.tolist()):
            timestep_to_nodes.setdefault(int(t), []).append(idx)

        all_graphs: List[Data] = []
        max_nodes = 0

        unique_times = sorted(set(int(t) for t in time_step.tolist()))
        for t in unique_times:
            node_indices = timestep_to_nodes.get(t, [])
            if len(node_indices) == 0:
                continue

            local_map = {gidx: lidx for lidx, gidx in enumerate(node_indices)}
            node_id_set = {node_ids[gidx] for gidx in node_indices}

            if edge_time_col is not None:
                edge_slice = edge_df[edge_df[edge_time_col] == t]
            else:
                edge_slice = edge_df

            edge_pairs: list[tuple[int, int]] = []
            for u_id, v_id in edge_slice[[src_col, dst_col]].itertuples(index=False):
                if u_id not in node_id_set or v_id not in node_id_set:
                    continue
                gu = global_index.get((int(t), u_id))
                gv = global_index.get((int(t), v_id))
                if gu is None or gv is None:
                    continue
                lu = local_map[gu]
                lv = local_map[gv]
                if lu == lv:
                    continue
                edge_pairs.append((lu, lv))

            if edge_pairs:
                edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            y_t = merged[label_col].astype(int).to_numpy()[node_indices]
            x_local_t = local_x[node_indices]

            anf = np.zeros((len(node_indices), 2), dtype=np.float32)
            if edge_pairs:
                for src_local, dst_local in edge_pairs:
                    src_label = int(y_t[src_local])
                    if src_label == 0:
                        anf[dst_local, 0] += 1.0
                    elif src_label == 1:
                        anf[dst_local, 1] += 1.0

            x_t = np.concatenate([x_local_t, anf], axis=1).astype(np.float32)

            graph = Data(
                x=torch.tensor(x_t, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(y_t, dtype=torch.long),
            )
            graph.graph_timestep = int(t)
            graph.node_ids = [node_ids[gidx] for gidx in node_indices]
            graph.num_labeled = int((graph.y != -1).sum().item())
            all_graphs.append(graph)
            max_nodes = max(max_nodes, graph.num_nodes)

        train_graphs = [g for g in all_graphs if cfg.train_start <= g.graph_timestep <= cfg.train_end]
        test_graphs = [g for g in all_graphs if cfg.test_start <= g.graph_timestep <= cfg.test_end]

        if not train_graphs:
            raise ValueError("RecGNN Elliptic++ Actors train sequence is empty.")
        if not test_graphs:
            raise ValueError("RecGNN Elliptic++ Actors test sequence is empty.")

        num_features = int(all_graphs[0].num_features)
        if num_features < 3:
            raise ValueError(f"RecGNN Elliptic++ Actors expects at least 3 features including ANF, got {num_features}.")

        return RecGNNEllipticPPActorsSequence(
            all_graphs=all_graphs,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            max_nodes=max_nodes,
            num_features=num_features,
            num_classes=2,
        )
