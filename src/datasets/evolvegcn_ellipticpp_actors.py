from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


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

    grouped = work.groupby([id_col, time_col], as_index=False).agg(agg_spec).reset_index(drop=True)
    return grouped


def _map_labels(series: pd.Series) -> pd.Series:
    mapped = pd.to_numeric(series, errors="coerce").map({1: 1, 2: 0, 3: -1})
    return mapped.astype("Int64")


@dataclass
class EvolveGCNActorsConfig:
    feature_path: str = "data/raw/ellipticpp/actors/wallets_features.csv"
    class_path: str = "data/raw/ellipticpp/actors/wallets_classes.csv"
    edge_path: str = "data/raw/ellipticpp/actors/AddrAddr_edgelist.csv"
    num_hist_steps: int = 5
    adj_mat_time_window: int = 1
    train_start: int = 1
    train_end: int = 34
    test_start: int = 35
    test_end: int = 49
    filter_unknown: bool = True


@dataclass
class EvolveGCNWindowSample:
    current_time: int
    hist_adj_list: List[torch.Tensor]
    hist_ndFeats_list: List[torch.Tensor]
    node_mask_list: List[torch.Tensor]
    label_idx: torch.Tensor
    label_vals: torch.Tensor


@dataclass
class EvolveGCNActorsSequence:
    train_samples: List[EvolveGCNWindowSample]
    test_samples: List[EvolveGCNWindowSample]
    num_nodes: int
    num_features: int
    num_classes: int = 2


class EvolveGCNEllipticPPActorsDataset:
    """
    EvolveGCN window builder for Elliptic++ Actors, adapted to this repo style.

    Exact EvolveGCN parts preserved:
    - adjacency windowing by time
    - self-loop + symmetric normalization
    - current-time labels only
    - history-window training over sparse normalized adjacencies
    - exact H/O backbones can consume these samples unchanged

    Actors-specific adaptations:
    - a fixed node set is built from unique actor ids
    - node features are time-varying, so each history step has its own dense X_t
    - if an actor is absent at a timestep, its row in X_t is zero-filled
    - unknown actors can be filtered out by config to match the rest of this repo
    """

    def __init__(self, cfg: EvolveGCNActorsConfig | None = None):
        self.cfg = cfg or EvolveGCNActorsConfig()
        self.sequence = self._build_sequence(self.cfg)

    def get_sequence(self) -> EvolveGCNActorsSequence:
        return self.sequence

    @staticmethod
    def _normalize_adj(idx: torch.Tensor, vals: torch.Tensor, num_nodes: int) -> torch.Tensor:
        sp_tensor = torch.sparse_coo_tensor(
            idx.t(),
            vals.float(),
            size=(num_nodes, num_nodes),
            dtype=torch.float32,
        ).coalesce()

        eye_idx = torch.arange(num_nodes, dtype=torch.long)
        eye = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx], dim=0),
            torch.ones(num_nodes, dtype=torch.float32),
            size=(num_nodes, num_nodes),
            dtype=torch.float32,
        )
        sp_tensor = (sp_tensor + eye).coalesce()

        norm_idx = sp_tensor.indices()
        norm_vals = sp_tensor.values()
        degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
        di = degree[norm_idx[0]]
        dj = degree[norm_idx[1]]
        norm_vals = norm_vals * ((di * dj) ** -0.5)
        return torch.sparse_coo_tensor(norm_idx, norm_vals, size=(num_nodes, num_nodes), dtype=torch.float32).coalesce()

    @staticmethod
    def _build_node_mask(active_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        mask = torch.full((num_nodes, 1), -float("Inf"), dtype=torch.float32)
        if active_idx.numel() > 0:
            mask[active_idx.unique()] = 0.0
        return mask

    @classmethod
    def _build_sequence(cls, cfg: EvolveGCNActorsConfig) -> EvolveGCNActorsSequence:
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

        non_feature = set(merge_keys + [label_col])
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
            raise ValueError("No Elliptic++ actor rows remain after preprocessing.")

        time_values = merged[time_col].astype(int).to_numpy()
        train_mask_np = (time_values >= cfg.train_start) & (time_values <= cfg.train_end)
        test_mask_np = (time_values >= cfg.test_start) & (time_values <= cfg.test_end)
        if train_mask_np.sum() == 0:
            raise ValueError("Train split is empty. Check train_start/train_end.")
        if test_mask_np.sum() == 0:
            raise ValueError("Test split is empty. Check test_start/test_end.")

        feature_mat = _safe_numeric_frame(merged[feature_cols]).to_numpy(dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(feature_mat[train_mask_np])
        feature_mat = scaler.transform(feature_mat).astype(np.float32)

        actor_ids = sorted(merged[id_col].astype(str).unique().tolist())
        actor_to_idx = {aid: i for i, aid in enumerate(actor_ids)}
        num_nodes = len(actor_ids)
        num_features = int(feature_mat.shape[1])

        unique_times = sorted(set(int(t) for t in time_values.tolist()))
        rows_by_time: Dict[int, pd.DataFrame] = {
            int(t): merged[merged[time_col] == t].reset_index(drop=True) for t in unique_times
        }

        x_by_time: Dict[int, torch.Tensor] = {}
        y_by_time: Dict[int, torch.Tensor] = {}
        adj_by_time: Dict[int, torch.Tensor] = {}
        mask_by_time: Dict[int, torch.Tensor] = {}

        feature_lookup: Dict[tuple[int, str], np.ndarray] = {}
        label_lookup: Dict[tuple[int, str], int] = {}
        for i, row in merged.reset_index(drop=True).iterrows():
            key = (int(row[time_col]), str(row[id_col]))
            feature_lookup[key] = feature_mat[i]
            label_lookup[key] = int(row[label_col])

        for t in unique_times:
            x_t = np.zeros((num_nodes, num_features), dtype=np.float32)
            y_t = np.full((num_nodes,), -1, dtype=np.int64)

            active_feature_nodes: list[int] = []
            row_df = rows_by_time[t]
            for actor_id in row_df[id_col].astype(str).tolist():
                idx = actor_to_idx[actor_id]
                x_t[idx] = feature_lookup[(int(t), actor_id)]
                y_t[idx] = label_lookup[(int(t), actor_id)]
                active_feature_nodes.append(idx)

            if edge_time_col is not None:
                edge_slice = edge_df[(edge_df[edge_time_col] <= t) & (edge_df[edge_time_col] > (t - cfg.adj_mat_time_window))]
            else:
                edge_slice = edge_df

            edge_pairs: list[tuple[int, int]] = []
            for u_id, v_id in edge_slice[[src_col, dst_col]].itertuples(index=False):
                if u_id not in actor_to_idx or v_id not in actor_to_idx:
                    continue
                edge_pairs.append((actor_to_idx[u_id], actor_to_idx[v_id]))

            if edge_pairs:
                cur_idx = torch.tensor(edge_pairs, dtype=torch.long)
                active_adj_nodes = cur_idx.view(-1)
            else:
                cur_idx = torch.empty((0, 2), dtype=torch.long)
                active_adj_nodes = torch.empty((0,), dtype=torch.long)

            active_for_mask = active_adj_nodes if active_adj_nodes.numel() > 0 else torch.tensor(active_feature_nodes, dtype=torch.long)
            if active_for_mask.numel() == 0:
                raise ValueError(f"Timestep {t} has no active actor rows and no usable edges.")

            if cur_idx.numel() == 0:
                adj_by_time[t] = cls._normalize_adj(
                    torch.empty((0, 2), dtype=torch.long),
                    torch.empty((0,), dtype=torch.float32),
                    num_nodes,
                )
            else:
                vals = torch.ones(cur_idx.size(0), dtype=torch.float32)
                adj_by_time[t] = cls._normalize_adj(cur_idx, vals, num_nodes)

            mask_by_time[t] = cls._build_node_mask(active_for_mask, num_nodes)
            x_by_time[t] = torch.tensor(x_t, dtype=torch.float32)
            y_by_time[t] = torch.tensor(y_t, dtype=torch.long)

        def build_samples(time_subset: List[int]) -> List[EvolveGCNWindowSample]:
            samples: List[EvolveGCNWindowSample] = []
            for current_t in time_subset:
                if current_t - cfg.num_hist_steps < unique_times[0]:
                    continue

                hist_times = list(range(current_t - cfg.num_hist_steps, current_t + 1))
                if any(t not in adj_by_time for t in hist_times):
                    continue

                label_idx = torch.where(y_by_time[current_t] >= 0)[0]
                label_vals = y_by_time[current_t][label_idx]
                if label_idx.numel() == 0:
                    continue

                samples.append(
                    EvolveGCNWindowSample(
                        current_time=int(current_t),
                        hist_adj_list=[adj_by_time[t] for t in hist_times],
                        hist_ndFeats_list=[x_by_time[t] for t in hist_times],
                        node_mask_list=[mask_by_time[t] for t in hist_times],
                        label_idx=label_idx,
                        label_vals=label_vals,
                    )
                )
            return samples

        train_times = [t for t in unique_times if cfg.train_start <= t <= cfg.train_end]
        test_times = [t for t in unique_times if cfg.test_start <= t <= cfg.test_end]
        train_samples = build_samples(train_times)
        test_samples = build_samples(test_times)

        if not train_samples:
            raise ValueError("EvolveGCN Elliptic++ Actors train_samples is empty. Check split bounds or num_hist_steps.")
        if not test_samples:
            raise ValueError("EvolveGCN Elliptic++ Actors test_samples is empty. Check split bounds or num_hist_steps.")

        return EvolveGCNActorsSequence(
            train_samples=train_samples,
            test_samples=test_samples,
            num_nodes=num_nodes,
            num_features=num_features,
            num_classes=2,
        )
