import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    GAT for node classification (logits output).
    Hidden layers: multi-head with concat=True
    Last layer: heads=1 with concat=False (produces out_dim logits)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.6,
        use_norm: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 2, "GAT needs at least 2 layers (input->hidden->output)."
        assert heads >= 1, "heads must be >= 1"

        self.dropout = dropout
        self.use_norm = use_norm
        self.heads = heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=True,
                dropout=dropout,          # attention dropout
                negative_slope=negative_slope,
            )
        )
        if use_norm:
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            if use_norm:
                self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Last layer (logits)
        self.convs.append(
            GATConv(
                in_channels=hidden_dim * heads,
                out_channels=out_dim,
                heads=1,
                concat=False,              # output shape: [N, out_dim]
                dropout=dropout,
                negative_slope=negative_slope,
            )
        )

    def forward(self, x, edge_index):
        # feature dropout at input (as in GAT paper)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # hidden layers
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)  # [N, hidden_dim*heads]
            if self.use_norm:
                x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # logits
        x = self.convs[-1](x, edge_index)     # [N, out_dim]
        return x
