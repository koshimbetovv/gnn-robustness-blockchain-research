import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_mean(x: torch.Tensor, index: torch.Tensor, num_segments: int | None = None) -> torch.Tensor:
    """
    Mean aggregation of x by integer segment ids in index.
    x: [N, D]
    index: [N]
    returns: [S, D]
    """
    if index.numel() == 0:
        s = 0 if num_segments is None else num_segments
        return x.new_zeros((s, x.size(-1)))

    if num_segments is None:
        num_segments = int(index.max().item()) + 1

    out = x.new_zeros((num_segments, x.size(-1)))
    cnt = x.new_zeros((num_segments, 1))
    out.index_add_(0, index, x)
    cnt.index_add_(0, index, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    cnt = cnt.clamp_min_(1.0)
    return out / cnt


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        time_step = time_step.float().view(-1, 1)
        device = time_step.device
        half = max(1, self.dim // 2)
        freq = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        angles = time_step * freq.view(1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.size(1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(1)))
        return emb[:, : self.dim]


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim * 2, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.lin(torch.cat([a, b], dim=-1)))
        return gate * a + (1.0 - gate) * b


def haar_multiscale_differences(seq: torch.Tensor, max_level: int = 2) -> torch.Tensor:
    """
    Simple Haar-style temporal enrichment on a sequence [T, D].
    Returns a sequence with the same shape.
    """
    if seq.size(0) <= 1 or max_level <= 0:
        return seq

    enriched = seq.clone()
    T = seq.size(0)
    for level in range(max_level):
        stride = 2 ** level
        if stride >= T:
            break
        prev = torch.roll(seq, shifts=stride, dims=0)
        prev[:stride] = seq[:stride]
        enriched = enriched + (seq - prev) / math.sqrt(float(stride + 1))
    return enriched