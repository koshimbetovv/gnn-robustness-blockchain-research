import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class ProjectedSinusoidalTimeEncoding(nn.Module):
    """
    Fixed sinusoidal timestamp encoding followed by a learnable linear projection,
    matching Eqs. (2)-(3) from the ChronoWave-GNN paper.
    """

    def __init__(self, time_dim: int = 8):
        super().__init__()
        if time_dim <= 0:
            raise ValueError("time_dim must be positive")
        self.time_dim = time_dim
        self.proj = nn.Linear(time_dim, time_dim)

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        t = time_step.float().view(-1, 1)
        device = t.device
        dtype = t.dtype

        pe = torch.zeros(t.size(0), self.time_dim, device=device, dtype=dtype)
        even_idx = torch.arange(0, self.time_dim, 2, device=device, dtype=dtype)
        div_term = torch.pow(10000.0, even_idx / float(self.time_dim))
        angles = t / div_term.view(1, -1)

        pe[:, 0::2] = torch.sin(angles)
        if self.time_dim > 1:
            pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])

        return self.proj(pe)


class ChronoWaveGNN(nn.Module):
    """
    Paper-faithful core ChronoWave-GNN backbone:
      - input = [normalized raw features || normalized level-2 Haar cA2 || projected sinusoidal time encoding]
      - 3-layer TransformerConv backbone
      - ELU activations and dropout between layers
      - linear classifier on final node embeddings

    Notes:
      * The paper does not report hidden dimension or number of attention heads.
        Those remain configurable repo-side defaults.
      * Later gate/dynamic-fusion and edge-semantic extensions in the paper are not
        specified well enough to reproduce verbatim here without inventing details.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        time_dim: int = 8,
        heads: int = 1,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        if num_layers != 3:
            raise ValueError("ChronoWave-GNN paper specifies a 3-layer TransformerConv backbone.")
        if heads <= 0:
            raise ValueError("heads must be positive")

        self.dropout = dropout
        self.time_encoder = ProjectedSinusoidalTimeEncoding(time_dim=time_dim)

        input_dim = in_dim + time_dim
        self.convs = nn.ModuleList(
            [
                TransformerConv(input_dim, hidden_dim, heads=heads, concat=False, dropout=dropout),
                TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout),
                TransformerConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout),
            ]
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_step: torch.Tensor,
        return_embeddings: bool = False,
    ):
        if time_step is None:
            raise ValueError("ChronoWaveGNN requires time_step.")

        t_emb = self.time_encoder(time_step)
        h = torch.cat([x, t_emb], dim=-1)

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.elu(h)
            if layer_idx < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)
        if return_embeddings:
            return logits, h
        return logits
