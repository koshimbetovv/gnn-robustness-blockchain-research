import torch

def pick_target_nodes(data, logits_clean, split_mask, *, only_illicit: bool, fraction: float,
                      only_clean_correct: bool, seed: int, device: torch.device) -> torch.Tensor:
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"ATTACK_FRACTION must be in (0, 1], got {fraction}")
    mask = split_mask.bool() & (data.y != -1)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return idx.to(device)
    if only_illicit:
        idx = idx[data.y[idx] == 1]
    if only_clean_correct and idx.numel() > 0:
        pred_clean = logits_clean.argmax(dim=1)
        idx = idx[pred_clean[idx] == data.y[idx]]
    if idx.numel() == 0:
        return idx.to(device)
    n = int(round(float(fraction) * float(idx.numel())))
    n = max(1, min(n, int(idx.numel())))
    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    idx_cpu = idx.detach().cpu()
    perm = torch.randperm(idx_cpu.numel(), generator=g)
    return idx_cpu[perm[:n]].to(device)