import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, use_norm=True):
        super().__init__()
        assert num_layers >= 2, "GCN needs at least 2 layers (input->hidden->output)."

        self.dropout = dropout
        self.use_norm = use_norm

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # first
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # middle
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # last
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        # all but last layer
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            if self.use_norm:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # last layer (logits)
        x = self.convs[-1](x, edge_index)
        return x
