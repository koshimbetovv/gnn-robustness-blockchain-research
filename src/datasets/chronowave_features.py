from __future__ import annotations

import torch
import torch.nn.functional as F


def haar_level2_pywt(x: torch.Tensor) -> torch.Tensor:
    """Haar level-2 approximation via pywt (non-differentiable; used at preprocessing)."""
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "ChronoWave-GNN requires PyWavelets. Install: pip install PyWavelets"
        ) from exc

    x_np = x.detach().cpu().numpy()
    cA2, *_ = pywt.wavedec(x_np, wavelet="haar", level=2, axis=1)
    return torch.from_numpy(cA2).to(device=x.device, dtype=x.dtype)


def _haar_level1_torch(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Haar level-1 approximation. Replicate-pad odd-length signals
    to approximate pywt's default 'symmetric' boundary behavior."""
    if x.size(-1) % 2 == 1:
        x = F.pad(x, (0, 1), mode="replicate")
    return (x[..., 0::2] + x[..., 1::2]) / (2.0 ** 0.5)


def haar_level2_torch(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Haar level-2 approximation for attack-time gradient flow."""
    return _haar_level1_torch(_haar_level1_torch(x))


@torch.no_grad()
def _fit_standardize(x: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = x[train_mask]
    if train_x.numel() == 0:
        raise ValueError("Train split has no labeled rows for standardization.")
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, std


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def unstandardize(x_std: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x_std * std + mean


def standardize_from_train(x: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper used by the ChronoWave training script for backwards compat."""
    mean, std = _fit_standardize(x, train_mask)
    return standardize(x, mean, std)


def build_paper_features(data) -> None:
    """Replace data.x in-place with [standardized_raw || standardized_haar_level2].

    Also stores, for downstream attack code, all stats needed to rebuild the
    concatenated feature vector from a perturbed raw slice:
      data.raw_feature_dim, data.wave_feature_dim
      data.raw_mean, data.raw_std, data.wave_mean, data.wave_std
    """
    raw_x = data.x.float().cpu()
    train_mask = data.train_mask.bool().cpu() & (data.y.cpu() != -1)

    wave_x = haar_level2_pywt(raw_x)

    raw_mean, raw_std = _fit_standardize(raw_x, train_mask)
    wave_mean, wave_std = _fit_standardize(wave_x, train_mask)

    raw_std_x = standardize(raw_x, raw_mean, raw_std)
    wave_std_x = standardize(wave_x, wave_mean, wave_std)

    data.x = torch.cat([raw_std_x, wave_std_x], dim=1)
    data.raw_feature_dim = int(raw_x.size(1))
    data.wave_feature_dim = int(wave_x.size(1))
    data.raw_mean = raw_mean
    data.raw_std = raw_std
    data.wave_mean = wave_mean
    data.wave_std = wave_std


def make_consistent_rebuild(data):
    """Return a callable f(x_full) -> x_full that, given a full-feature tensor with
    a perturbed raw slice, recomputes the wavelet slice so the two branches stay
    consistent (i.e., wavelet features = Haar(raw) after un-standardize/re-standardize).

    Differentiable end-to-end; gradient flows through the wavelet pipeline into
    the raw slice."""
    raw_dim = data.raw_feature_dim
    device = data.x.device
    raw_mean = data.raw_mean.to(device)
    raw_std = data.raw_std.to(device)
    wave_mean = data.wave_mean.to(device)
    wave_std = data.wave_std.to(device)

    def rebuild(x_full: torch.Tensor) -> torch.Tensor:
        raw_std_x = x_full[:, :raw_dim]
        raw_x = unstandardize(raw_std_x, raw_mean, raw_std)
        wave_x = haar_level2_torch(raw_x)
        wave_std_x = standardize(wave_x, wave_mean, wave_std)
        return torch.cat([raw_std_x, wave_std_x], dim=1)

    return rebuild
