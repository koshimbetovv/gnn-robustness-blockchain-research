import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification. Works for binary too.
    alpha: None or list/tensor of shape [C]
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        else:
            alpha_t = torch.tensor(alpha, dtype=torch.float)
            self.register_buffer("alpha", alpha_t)

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)                # [B,C]
        probs = torch.exp(log_probs)                            # [B,C]

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # [B]

        focal = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            at = self.alpha.gather(0, targets)                  # [B]
            loss = -at * focal * log_pt
        else:
            loss = -focal * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss