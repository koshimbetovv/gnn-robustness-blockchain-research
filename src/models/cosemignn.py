import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric_temporal import EvolveGCNO


class MA(nn.Module):
    def __init__(self, feature_in, embed_mid, embed_out, num_heads=4):
        super(MA, self).__init__()
        self.embedding_position1 = nn.Linear(feature_in, embed_mid)
        self.embedding_position2 = nn.Linear(embed_mid, embed_out)

        self.linear_q = nn.Linear(embed_out, embed_out)
        self.linear_k = nn.Linear(embed_out, embed_out)
        self.linear_v = nn.Linear(embed_out, embed_out)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_out,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_trans_tail = nn.LayerNorm(embed_out, eps=1e-12)

    def forward(self, x):
        p = F.leaky_relu(self.embedding_position1(x))
        p = F.leaky_relu(self.embedding_position2(p))

        q = self.linear_q(p).unsqueeze(1)
        k = self.linear_k(p).unsqueeze(1)
        v = self.linear_v(p).unsqueeze(1)

        attention_output, _ = self.attention(q, k, v)
        p = attention_output.squeeze(1)

        p = self.norm_trans_tail(p)
        return p


class CoSemiGNN(nn.Module):
    def __init__(self, feature_in, dim=128, dim2=256, dim3=128, num_heads=4):
        super(CoSemiGNN, self).__init__()

        self.embedding_feature1 = nn.Linear(feature_in, dim)
        self.gnn = GATv2Conv(feature_in, dim, heads=4, concat=False, dropout=0)

        self.attention = MA(
            feature_in=dim * 2,
            embed_mid=dim,
            embed_out=dim2,
            num_heads=num_heads,
        )
        self.feature_norm1 = LayerNorm(dim2, eps=1e-12)

        self.evolve_gcn = EvolveGCNO(in_channels=dim2)
        self.norm_evolveGCN_tail = LayerNorm(dim2, eps=1e-12)

        self.fc2gat1 = nn.Linear(dim2, dim3)
        self.fc = nn.Linear(dim3, dim3)
        self.norm_gat_tail = LayerNorm(dim3, eps=1e-12)

        self.classifier = nn.Linear(dim3, 1)

    def forward(self, feature, adj=None, ca_weights=None):
        x0 = self.embedding_feature1(feature)
        x = F.leaky_relu(self.gnn(feature, adj))
        x = torch.cat((x0, x), dim=1)

        x0 = x
        x = F.leaky_relu(self.attention(x))
        x = self.feature_norm1(x0 + x)

        x0 = x

        # FIX: avoid carrying autograd graph across time steps / batches
        self.evolve_gcn.reinitialize_weight()

        x = F.leaky_relu(self.evolve_gcn(x, adj))
        x = x + x0
        x = self.norm_evolveGCN_tail(x)

        x = F.leaky_relu(self.fc2gat1(x))
        out = F.leaky_relu(self.fc(x))
        out = self.norm_gat_tail(out)

        out_line = self.classifier(out).squeeze(-1)
        return out_line, out

