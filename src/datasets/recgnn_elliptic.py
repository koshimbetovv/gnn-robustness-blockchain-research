from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


@dataclass
class RecGNNEllipticConfig:
    feature_path: str = "data/raw/elliptic/elliptic_txs_features.csv"
    class_path: str = "data/raw/elliptic/elliptic_txs_classes.csv"
    edge_path: str = "data/raw/elliptic/elliptic_txs_edgelist.csv"
    train_start: int = 1
    train_end: int = 34
    test_start: int = 35
    test_end: int = 49
    filter_unknown: bool = False


@dataclass
class RecGNNEllipticSequence:
    all_graphs: List[Data]
    train_graphs: List[Data]
    test_graphs: List[Data]
    max_nodes: int
    num_features: int
    num_classes: int = 2


class RecGNNEllipticDataset:
    """
    Paper-faithful Elliptic sequence builder for RecGNN.

    Each timestep becomes one PyG graph. Node features are:
      [local features only (93 dims, excluding timestep) || ANF (2 dims)]
    so the final node dimension is 95, matching the paper.
    """

    def __init__(self, cfg: RecGNNEllipticConfig | None = None):
        self.cfg = cfg or RecGNNEllipticConfig()
        self.sequence = self._build_sequence(self.cfg)

    def get_sequence(self) -> RecGNNEllipticSequence:
        return self.sequence

    @staticmethod
    def _build_sequence(cfg: RecGNNEllipticConfig) -> RecGNNEllipticSequence:
        features = pd.read_csv(cfg.feature_path, header=None)
        classes = pd.read_csv(cfg.class_path)
        edges = pd.read_csv(cfg.edge_path)

        features = features.sort_values(by=0).reset_index(drop=True)
        classes = classes.copy()
        classes["class"] = classes["class"].astype(str).str.strip()
        classes["y"] = classes["class"].map({"1": 1, "2": 0}).fillna(-1).astype(int)

        tx_ids = features.iloc[:, 0].astype(str).tolist()
        time_step = features.iloc[:, 1].astype(int).to_numpy()

        # Paper: use only local features, excluding timestep.
        # First 94 Elliptic columns are [timestep + 93 local features].
        local_x = features.iloc[:, 2:95].to_numpy(dtype=np.float32)
        if local_x.shape[1] != 93:
            raise ValueError(
                f"Expected 93 local Elliptic features for RecGNN, got {local_x.shape[1]}."
            )

        y_map = dict(zip(classes["txId"].astype(str), classes["y"].astype(int)))
        y_all = np.asarray([y_map.get(tx_id, -1) for tx_id in tx_ids], dtype=np.int64)

        if cfg.filter_unknown:
            keep = y_all != -1
            if keep.sum() == 0:
                raise ValueError("No nodes remain after filtering unknown labels in RecGNN Elliptic loader.")
            tx_ids = [tx for tx, k in zip(tx_ids, keep) if k]
            time_step = time_step[keep]
            local_x = local_x[keep]
            y_all = y_all[keep]

        # Standardize local features using TRAIN nodes only.
        train_mask_np = (time_step >= cfg.train_start) & (time_step <= cfg.train_end)
        if train_mask_np.sum() == 0:
            raise ValueError("RecGNN Elliptic train split is empty after filtering. Check split bounds.")
        scaler = StandardScaler()
        scaler.fit(local_x[train_mask_np])
        local_x = scaler.transform(local_x).astype(np.float32)

        global_index = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}

        edges = edges.copy()
        edges["txId1"] = edges["txId1"].astype(str)
        edges["txId2"] = edges["txId2"].astype(str)

        timestep_to_nodes: dict[int, list[int]] = {}
        for idx, t in enumerate(time_step.tolist()):
            timestep_to_nodes.setdefault(int(t), []).append(idx)

        all_graphs: List[Data] = []
        max_nodes = 0

        for t in range(cfg.train_start, cfg.test_end + 1):
            node_indices = timestep_to_nodes.get(t, [])
            if len(node_indices) == 0:
                continue

            local_map = {gidx: lidx for lidx, gidx in enumerate(node_indices)}
            node_tx_ids = {tx_ids[gidx] for gidx in node_indices}

            edge_pairs: list[tuple[int, int]] = []
            for u_tx, v_tx in edges[["txId1", "txId2"]].itertuples(index=False):
                if u_tx not in node_tx_ids or v_tx not in node_tx_ids:
                    continue
                gu = global_index[u_tx]
                gv = global_index[v_tx]
                edge_pairs.append((local_map[gu], local_map[gv]))

            if edge_pairs:
                edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            y_t = y_all[node_indices]
            x_local_t = local_x[node_indices]

            # ANF = counts of labeled incoming antecedent neighbours (licit / illicit).
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
            graph.tx_ids = [tx_ids[gidx] for gidx in node_indices]
            graph.num_labeled = int((graph.y != -1).sum().item())
            all_graphs.append(graph)
            max_nodes = max(max_nodes, graph.num_nodes)

        train_graphs = [g for g in all_graphs if cfg.train_start <= g.graph_timestep <= cfg.train_end]
        test_graphs = [g for g in all_graphs if cfg.test_start <= g.graph_timestep <= cfg.test_end]

        if not train_graphs:
            raise ValueError("RecGNN train sequence is empty.")
        if not test_graphs:
            raise ValueError("RecGNN test sequence is empty.")

        num_features = int(all_graphs[0].num_features)
        if num_features != 95:
            raise ValueError(f"RecGNN expects 95 input features (LF+ANF), got {num_features}.")

        return RecGNNEllipticSequence(
            all_graphs=all_graphs,
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            max_nodes=max_nodes,
            num_features=num_features,
            num_classes=2,
        )
