import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CMOS(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size0=128,
        hidden_size1=256,
        output_size=1,
        num_layers=1,
        bidirectional=False,
        dropout=0,
        state_rows=3000,
        device=None,
    ):
        super(CMOS, self).__init__()

        self.fc0 = nn.Linear(input_size, hidden_size0)
        self.GNN1 = GCNConv(input_size, hidden_size0)
        self.dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size1,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.state_rows = state_rows
        self.device_ref = device

        self.h0 = None
        self.c0 = None

        self.classifier = nn.Linear(hidden_size1, output_size)

    def _device(self, x):
        return x.device if self.device_ref is None else self.device_ref

    def _ensure_state(self, device):
        if self.h0 is None or self.c0 is None or self.h0.device != device:
            self.h0 = torch.zeros(self.num_layers, self.state_rows, self.hidden_size1, device=device)
            self.c0 = torch.zeros(self.num_layers, self.state_rows, self.hidden_size1, device=device)

    def expanded(self, x):
        original_rows = x.size(0)
        num_new_rows = self.state_rows - original_rows
        if num_new_rows < 0:
            raise ValueError(
                f"CMOS state_rows={self.state_rows} is smaller than current slice size={original_rows}."
            )
        new_rows = torch.zeros(num_new_rows, x.size(1), device=x.device, dtype=x.dtype)
        return torch.cat((x, new_rows), dim=0)

    def forward(self, x, adj, weight):
        device = self._device(x)
        self._ensure_state(device)

        x0 = F.leaky_relu(self.fc0(x))
        x1 = F.leaky_relu(self.GNN1(x, adj, edge_weight=weight))
        x1 = self.dropout(x1)
        x = torch.cat([x0, x1], dim=1)

        x_len = x.size(0)
        x = self.expanded(x)
        x = x.unsqueeze(1)
        x0 = x

        x, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))

        # preserve original repo behavior
        self.h0 = self.h0.detach()
        self.c0 = self.c0.detach()

        x = x0 + x
        x = x.squeeze(1)[:x_len, :]

        out = self.classifier(x)
        return out.squeeze()


def create_cmos_model(input_size, state_rows, device):
    return CMOS(input_size=input_size, state_rows=state_rows, device=device).to(device)