from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch


@dataclass
class EvolveGCNEllipticConfig:
    feature_path: str = "data/raw/elliptic/elliptic_txs_features.csv"
    class_path: str = "data/raw/elliptic/elliptic_txs_classes.csv"
    edge_path: str = "data/raw/elliptic/elliptic_txs_edgelist.csv"
    num_hist_steps: int = 5
    adj_mat_time_window: int = 1
    train_start: int = 1
    train_end: int = 34
    test_start: int = 35
    test_end: int = 49
    filter_unknown: bool = False


@dataclass
class EvolveGCNWindowSample:
    current_time: int
    hist_adj_list: List[torch.Tensor]
    hist_ndFeats_list: List[torch.Tensor]
    node_mask_list: List[torch.Tensor]
    label_idx: torch.Tensor
    label_vals: torch.Tensor


@dataclass
class EvolveGCNEllipticSequence:
    train_samples: List[EvolveGCNWindowSample]
    test_samples: List[EvolveGCNWindowSample]
    num_nodes: int
    num_features: int
    num_classes: int = 2


class EvolveGCNEllipticDataset:
    """
    Exact-window Elliptic builder for IBM EvolveGCN node classification, adapted to this repo.

    Exact aspects preserved:
    - full-node feature matrix is used at every history step,
    - node features include the timestep column (like IBM preprocessing),
    - adjacency is built per timestep with time_window=1 by default,
    - self-loop + symmetric normalization matches the original repo,
    - labels are taken only from the current prediction timestep.
    """

    def __init__(self, cfg: EvolveGCNEllipticConfig | None = None):
        self.cfg = cfg or EvolveGCNEllipticConfig()
        self.sequence = self._build_sequence(self.cfg)

    def get_sequence(self) -> EvolveGCNEllipticSequence:
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
    def _build_node_mask(idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        mask = torch.full((num_nodes, 1), -float("Inf"), dtype=torch.float32)
        if idx.numel() > 0:
            non_zero = idx.unique()
            mask[non_zero] = 0.0
        return mask

    @classmethod
    def _build_sequence(cls, cfg: EvolveGCNEllipticConfig) -> EvolveGCNEllipticSequence:
        features = pd.read_csv(cfg.feature_path, header=None)
        classes = pd.read_csv(cfg.class_path)
        edges = pd.read_csv(cfg.edge_path)

        # Exact IBM preprocessing keeps the original row order and replaces txId by contiguous row ids.
        tx_ids = features.iloc[:, 0].astype(str).tolist()
        tx_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}

        # Exact IBM loader uses nodes[:,1:], i.e. timestep column is included in node features.
        node_features = torch.tensor(features.iloc[:, 1:].to_numpy(), dtype=torch.float32)
        raw_time = features.iloc[:, 1].astype(int).to_numpy()
        num_nodes = len(tx_ids)

        classes = classes.copy()
        classes["class"] = classes["class"].astype(str).str.strip()
        label_map = {"unknown": -1, "1": 1, "2": 0, "-1": -1}
        classes["y"] = classes["class"].map(label_map).fillna(-1).astype(int)

        labels_by_idx = torch.full((num_nodes,), -1, dtype=torch.long)
        for _, row in classes.iterrows():
            tx = str(row["txId"])
            if tx in tx_to_idx:
                labels_by_idx[tx_to_idx[tx]] = int(row["y"])

        if cfg.filter_unknown:
            keep = labels_by_idx >= 0
            if int(keep.sum().item()) == 0:
                raise ValueError("No nodes remain after filtering unknown labels in EvolveGCN Elliptic loader.")

            keep_np = keep.cpu().numpy().astype(bool)
            tx_ids = [tx for tx, k in zip(tx_ids, keep_np) if k]
            raw_time = raw_time[keep_np]
            node_features = node_features[keep]
            labels_by_idx = labels_by_idx[keep]

            tx_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}
            num_nodes = len(tx_ids)

        nodes_labels_times = []
        for idx in range(num_nodes):
            label = int(labels_by_idx[idx].item())
            if label >= 0:
                nodes_labels_times.append([idx, label, int(raw_time[idx])])
        if not nodes_labels_times:
            raise ValueError("No labeled nodes found in Elliptic classes file.")
        nodes_labels_times = torch.tensor(nodes_labels_times, dtype=torch.long)

        edge_rows = []
        for _, row in edges.iterrows():
            u = str(row.iloc[0])
            v = str(row.iloc[1])
            if u not in tx_to_idx or v not in tx_to_idx:
                continue
            ui = tx_to_idx[u]
            vi = tx_to_idx[v]
            t_u = int(raw_time[ui])
            t_v = int(raw_time[vi])
            if t_u != t_v:
                raise ValueError(
                    f"Elliptic edge ({u}, {v}) spans different timesteps ({t_u}, {t_v}); "
                    "the exact EvolveGCN temporal preprocessing assumes same-timestep transaction edges."
                )
            edge_rows.append([ui, vi, t_u])
            edge_rows.append([vi, ui, t_u])

        if not edge_rows:
            raise ValueError("No valid edges found while building the EvolveGCN Elliptic sequence.")
        timed_edges = torch.tensor(edge_rows, dtype=torch.long)

        unique_times = sorted(int(t) for t in set(raw_time.tolist()))
        adj_by_time: Dict[int, torch.Tensor] = {}
        mask_by_time: Dict[int, torch.Tensor] = {}
        for t in unique_times:
            subset = (timed_edges[:, 2] <= t) & (timed_edges[:, 2] > (t - cfg.adj_mat_time_window))
            cur_idx = timed_edges[subset][:, :2]
            cur_vals = torch.ones(cur_idx.size(0), dtype=torch.float32)
            if cur_idx.numel() == 0:
                raise ValueError(f"Timestep {t} has no active edges; exact EvolveGCN TopK summarization would fail.")

            cur_adj = torch.sparse_coo_tensor(cur_idx.t(), cur_vals, size=(num_nodes, num_nodes)).coalesce()
            idx = cur_adj.indices().t().contiguous()
            vals = cur_adj.values().float()

            mask_by_time[t] = cls._build_node_mask(idx, num_nodes)
            adj_by_time[t] = cls._normalize_adj(idx, vals, num_nodes)

        def build_samples(time_values: List[int]) -> List[EvolveGCNWindowSample]:
            samples: List[EvolveGCNWindowSample] = []
            for current_t in time_values:
                if current_t - cfg.num_hist_steps < unique_times[0]:
                    continue
                hist_times = list(range(current_t - cfg.num_hist_steps, current_t + 1))
                if any(t not in adj_by_time for t in hist_times):
                    continue

                label_subset = nodes_labels_times[:, 2] == current_t
                label_idx = nodes_labels_times[label_subset, 0]
                label_vals = nodes_labels_times[label_subset, 1]
                if label_idx.numel() == 0:
                    continue

                samples.append(
                    EvolveGCNWindowSample(
                        current_time=int(current_t),
                        hist_adj_list=[adj_by_time[t] for t in hist_times],
                        hist_ndFeats_list=[node_features for _ in hist_times],
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
            raise ValueError("EvolveGCN train_samples is empty. Check split bounds or num_hist_steps.")
        if not test_samples:
            raise ValueError("EvolveGCN test_samples is empty. Check split bounds or num_hist_steps.")

        return EvolveGCNEllipticSequence(
            train_samples=train_samples,
            test_samples=test_samples,
            num_nodes=num_nodes,
            num_features=int(node_features.size(1)),
            num_classes=2,
        )
