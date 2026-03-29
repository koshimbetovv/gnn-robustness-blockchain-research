import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightEvolutionLSTM(nn.Module):
    """
    Paper-faithful EvolveLinear helper.

    W_{t+1} = LSTM(W_t)
    H_{t+1} = W_{t+1} * H_t

    Since the paper gives the matrix form but not the exact tensorization,
    we evolve the weight matrix row-wise with an LSTMCell.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.base_weight)

        self.row_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self._row_h = None
        self._row_c = None
        self._current_weight = None

    def reset_state(self, device: torch.device | None = None):
        dev = device if device is not None else self.base_weight.device
        self._row_h = torch.zeros(self.hidden_dim, self.hidden_dim, device=dev)
        self._row_c = torch.zeros(self.hidden_dim, self.hidden_dim, device=dev)
        self._current_weight = self.base_weight.clone().to(dev)

    def detach_state(self):
        if self._row_h is not None:
            self._row_h = self._row_h.detach()
        if self._row_c is not None:
            self._row_c = self._row_c.detach()
        if self._current_weight is not None:
            self._current_weight = self._current_weight.detach()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self._current_weight is None or self._row_h is None or self._row_c is None:
            self.reset_state(h.device)

        new_h, new_c = self.row_lstm(self._current_weight, (self._row_h, self._row_c))
        self._row_h, self._row_c = new_h, new_c
        self._current_weight = new_h
        return h @ self._current_weight.t()


class PaperMLSTMCell(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.w_x_i = nn.Linear(in_dim, hidden_dim, bias=True)
        self.w_x_f = nn.Linear(in_dim, hidden_dim, bias=True)
        self.w_x_o = nn.Linear(in_dim, hidden_dim, bias=True)
        self.w_x_c = nn.Linear(in_dim, hidden_dim, bias=True)

        self.w_h_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_h_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_h_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_h_c = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.w_c_i = nn.Parameter(torch.zeros(hidden_dim))
        self.w_c_f = nn.Parameter(torch.zeros(hidden_dim))
        self.w_c_o = nn.Parameter(torch.zeros(hidden_dim))

        self.evolve_linear = WeightEvolutionLSTM(hidden_dim)

    def reset_state(self, device: torch.device | None = None):
        self.evolve_linear.reset_state(device)

    def detach_state(self):
        self.evolve_linear.detach_state()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        i_t = torch.sigmoid(
            self.w_x_i(x_t) + self.w_h_i(h_prev) + c_prev * self.w_c_i.view(1, -1)
        )
        f_t = torch.sigmoid(
            self.w_x_f(x_t) + self.w_h_f(h_prev) + c_prev * self.w_c_f.view(1, -1)
        )

        m_t = F.relu(self.evolve_linear(self.w_h_c(h_prev)))
        c_tilde = torch.tanh(self.w_x_c(x_t) + m_t)
        c_t = f_t * c_prev + (1.0 - f_t) * c_tilde

        o_t = torch.sigmoid(
            self.w_x_o(x_t) + self.w_h_o(h_prev) + c_t * self.w_c_o.view(1, -1)
        )
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class PaperGNNLayer(nn.Module):
    """
    Exact paper equation (3):
      x'_i = Theta_1 x_i + Theta_2 sum_{j in N(i)} x_j

    Aggregation is performed over incoming (antecedent) neighbours.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.theta_self = nn.Linear(in_dim, out_dim, bias=False)
        self.theta_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.theta_self(x)
        if edge_index.numel() == 0:
            return out

        src, dst = edge_index
        agg = x.new_zeros(x.size())
        agg.index_add_(0, dst, x[src])
        out = out + self.theta_neigh(agg)
        return out


class PaperMLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, state_rows: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_rows = int(state_rows)
        self.cell = PaperMLSTMCell(in_dim, hidden_dim)
        self._h_state = None
        self._c_state = None

    def reset_state(self, device: torch.device | None = None):
        dev = device if device is not None else next(self.parameters()).device
        self._h_state = torch.zeros(self.state_rows, self.hidden_dim, device=dev)
        self._c_state = torch.zeros(self.state_rows, self.hidden_dim, device=dev)
        self.cell.reset_state(dev)

    def detach_state(self):
        if self._h_state is not None:
            self._h_state = self._h_state.detach()
        if self._c_state is not None:
            self._c_state = self._c_state.detach()
        self.cell.detach_state()

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        if self._h_state is None or self._c_state is None:
            self.reset_state(x_t.device)

        n = x_t.size(0)
        if n > self.state_rows:
            raise ValueError(
                f"Current timestep has {n} nodes, but state_rows={self.state_rows}."
            )

        if n < self.state_rows:
            pad = x_t.new_zeros((self.state_rows - n, x_t.size(1)))
            x_pad = torch.cat([x_t, pad], dim=0)
        else:
            x_pad = x_t

        h_new, c_new = self.cell(x_pad, self._h_state, self._c_state)
        self._h_state, self._c_state = h_new, c_new
        return h_new[:n]


class RecGNN(nn.Module):
    """
    Paper-faithful RecGNN backbone for Elliptic:
      X1_t = ReLU(M-LSTM(X_t))
      X2_t = ReLU(GNN(X1_t))
      X3_t = softmax(Linear(X2_t))

    forward(...) returns log-softmax so training can use NLLLoss exactly as in the paper.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, state_rows: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = float(dropout)
        self.m_lstm = PaperMLSTM(in_dim=in_dim, hidden_dim=hidden_dim, state_rows=state_rows)
        self.gnn = PaperGNNLayer(in_dim=hidden_dim, out_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def reset_sequence_state(self, device: torch.device | None = None):
        self.m_lstm.reset_state(device)

    def detach_sequence_state(self):
        self.m_lstm.detach_state()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time_step: torch.Tensor | None = None):
        h = F.relu(self.m_lstm(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gnn(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        return F.log_softmax(logits, dim=-1)
