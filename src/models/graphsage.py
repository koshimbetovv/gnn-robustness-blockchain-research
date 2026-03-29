import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, aggr="mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim, aggr=aggr)
        self.conv2 = SAGEConv(hid_dim, out_dim, aggr=aggr)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
