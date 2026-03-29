import math
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class EvolveGCNClassifier(nn.Module):
    """
    Exact IBM EvolveGCN classifier head for node classification:
    Linear(in_dim -> cls_feats) + ReLU + Linear(cls_feats -> out_dim)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _TopK(nn.Module):
    """
    Exact TopK summarizer used by the IBM EvolveGCN implementation.
    """

    def __init__(self, feats: int, k: int):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self._reset_param_rows(self.scorer)
        self.k = int(k)

    @staticmethod
    def _reset_param_rows(t: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    @staticmethod
    def _pad_with_last_val(vect: torch.Tensor, k: int) -> torch.Tensor:
        if vect.numel() == 0:
            raise ValueError("TopK received no valid indices; current timestep adjacency is empty.")
        pad = torch.ones(k - vect.size(0), dtype=torch.long, device=vect.device) * vect[-1]
        return torch.cat([vect, pad], dim=0)

    def forward(self, node_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = self._pad_with_last_val(topk_indices, self.k)

        out = node_embs[topk_indices] * torch.tanh(scores[topk_indices].view(-1, 1))
        return out.t()


class _MatGRUGate(nn.Module):
    """
    Exact matrix-GRU gate from the IBM EvolveGCN implementation.
    """

    def __init__(self, rows: int, cols: int, activation: nn.Module):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.zeros(rows, cols))
        self._reset_param(self.W)
        self._reset_param(self.U)

    @staticmethod
    def _reset_param(t: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)


class _MatGRUCellH(nn.Module):
    """
    Exact IBM repo version for EvolveGCN-H.
    """

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.update = _MatGRUGate(rows, cols, nn.Sigmoid())
        self.reset = _MatGRUGate(rows, cols, nn.Sigmoid())
        self.htilda = _MatGRUGate(rows, cols, nn.Tanh())
        self.choose_topk = _TopK(feats=rows, k=cols)

    def forward(self, prev_Q: torch.Tensor, prev_Z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z_topk = self.choose_topk(prev_Z, mask)
        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)
        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)
        new_Q = (1.0 - update) * prev_Q + update * h_cap
        return new_Q


class _MatGRUCellO(nn.Module):
    """
    Exact IBM repo version for EvolveGCN-O.

    Note: the paper text describes an LSTM-style evolution for -O, but the released
    IBM implementation uses this GRU-like matrix cell with prev_Q as both input and hidden.
    This class mirrors the released code exactly.
    """

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.update = _MatGRUGate(rows, cols, nn.Sigmoid())
        self.reset = _MatGRUGate(rows, cols, nn.Sigmoid())
        self.htilda = _MatGRUGate(rows, cols, nn.Tanh())

    def forward(self, prev_Q: torch.Tensor) -> torch.Tensor:
        z_topk = prev_Q
        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)
        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)
        new_Q = (1.0 - update) * prev_Q + update * h_cap
        return new_Q


class _GRCUH(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, activation: nn.Module):
        super().__init__()
        self.activation = activation
        self.evolve_weights = _MatGRUCellH(rows=in_feats, cols=out_feats)
        self.GCN_init_weights = Parameter(torch.Tensor(in_feats, out_feats))
        self._reset_param(self.GCN_init_weights)

    @staticmethod
    def _reset_param(t: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(
        self,
        A_list: List[torch.Tensor],
        node_embs_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            out_seq.append(node_embs)
        return out_seq


class _GRCUO(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, activation: nn.Module):
        super().__init__()
        self.activation = activation
        self.evolve_weights = _MatGRUCellO(rows=in_feats, cols=out_feats)
        self.GCN_init_weights = Parameter(torch.Tensor(in_feats, out_feats))
        self._reset_param(self.GCN_init_weights)

    @staticmethod
    def _reset_param(t: torch.Tensor) -> None:
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list: List[torch.Tensor], node_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t, Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            GCN_weights = self.evolve_weights(GCN_weights)
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))
            out_seq.append(node_embs)
        return out_seq


class EvolveGCNH(nn.Module):
    """
    Exact EvolveGCN-H backbone ported from IBM/EvolveGCN.

    The original repo stores submodules in a Python list and overwrites self._parameters,
    which worked on older PyTorch internals but breaks on modern PyTorch. Here we keep the
    exact computation and layer ordering, but register layers with ModuleList so .to(),
    state_dict(), and optimizers work correctly.
    """

    def __init__(self, in_dim: int, layer_1_feats: int, layer_2_feats: int, activation: nn.Module, skipfeats: bool = False):
        super().__init__()
        feats = [in_dim, layer_1_feats, layer_2_feats]
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            grcu_i = _GRCUH(in_feats=feats[i - 1], out_feats=feats[i], activation=activation)
            self.GRCU_layers.append(grcu_i)

    def forward(self, A_list: List[torch.Tensor], Nodes_list: List[torch.Tensor], nodes_mask_list: List[torch.Tensor]) -> torch.Tensor:
        node_feats = Nodes_list[-1]
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list)
        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)
        return out


class EvolveGCNO(nn.Module):
    """
    Exact EvolveGCN-O backbone ported from IBM/EvolveGCN repo code.

    The original repo stores submodules in a Python list and overwrites self._parameters,
    which worked on older PyTorch internals but breaks on modern PyTorch. Here we keep the
    exact computation and layer ordering, but register layers with ModuleList so .to(),
    state_dict(), and optimizers work correctly.
    """

    def __init__(self, in_dim: int, layer_1_feats: int, layer_2_feats: int, activation: nn.Module, skipfeats: bool = False):
        super().__init__()
        feats = [in_dim, layer_1_feats, layer_2_feats]
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            grcu_i = _GRCUO(in_feats=feats[i - 1], out_feats=feats[i], activation=activation)
            self.GRCU_layers.append(grcu_i)

    def forward(self, A_list: List[torch.Tensor], Nodes_list: List[torch.Tensor], nodes_mask_list: List[torch.Tensor]) -> torch.Tensor:
        node_feats = Nodes_list[-1]
        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)
        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out, node_feats), dim=1)
        return out


class EvolveGCNNodeClassifier(nn.Module):
    """
    Small repo-style wrapper: exact EvolveGCN backbone + exact IBM MLP classifier head.
    """

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(
        self,
        A_list: List[torch.Tensor],
        Nodes_list: List[torch.Tensor],
        nodes_mask_list: List[torch.Tensor],
        node_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_embs = self.backbone(A_list, Nodes_list, nodes_mask_list)
        if node_indices is None:
            cls_input = node_embs
        else:
            cls_input = node_embs[node_indices]
        return self.classifier(cls_input)


__all__ = [
    "EvolveGCNH",
    "EvolveGCNO",
    "EvolveGCNClassifier",
    "EvolveGCNNodeClassifier",
]
