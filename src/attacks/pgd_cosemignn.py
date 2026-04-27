from typing import Optional

import torch
import torch.nn.functional as F


class CoSemiPGDAttack:
    """L_inf PGD for CoSemiGNN, operating on one timestep slice at a time.

    CoSemiGNN consumes per-timestep slices whose feature vector is
    `[raw_tx_features (165) || semi_supervised_predictions (6)]`. The 6 semi-pred
    columns are outputs of auxiliary non-differentiable classifiers, so we treat
    them as a fixed oracle: the attacker cannot perturb or recompute them. Only
    the first `raw_feature_dim` columns participate in `delta`.

    CoSemiGNN's forward returns `(out_line, _)` with `out_line` of shape `(N,)`
    (BCE-style). We lift to `(N, 2)` as `[0, s]` so cross-entropy and argmax
    behave like BCE-with-logits on `s`.
    """

    def __init__(
        self,
        model,
        device,
        raw_feature_dim: int = 165,
        clamp: Optional[tuple[float, float]] = None,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.raw_dim = int(raw_feature_dim)
        self.clamp = clamp

    def forward_logits(self, features, adj, ca_weights=None):
        out_line, _ = self.model(features, adj, ca_weights)
        return torch.stack([torch.zeros_like(out_line), out_line], dim=1)

    def _perturb_targets(self, features, target_idx, delta):
        """Return a full feature tensor with `delta` applied to the raw slice of
        `target_idx` rows and the semi slice preserved."""
        if self.raw_dim == features.size(1):
            target_new = features[target_idx] + delta
        else:
            raw = features[target_idx, : self.raw_dim] + delta
            semi = features[target_idx, self.raw_dim :]
            target_new = torch.cat([raw, semi], dim=1)
        return features.index_copy(0, target_idx, target_new)

    def _project_clamp(self, features, target_idx, delta, eps):
        """Project resulting features into [clamp_min, clamp_max] on the
        perturbable raw slice and re-clip delta to the L_inf eps ball."""
        if self.clamp is None:
            return delta.detach()
        x_tmp = self._perturb_targets(features, target_idx, delta)
        x_tmp = torch.clamp(x_tmp, min=self.clamp[0], max=self.clamp[1])
        new_delta = x_tmp[target_idx, : self.raw_dim] - features[target_idx, : self.raw_dim]
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    def attack_slice(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        target_idx: torch.Tensor,
        labels_true: torch.Tensor,
        eps: float = 0.01,
        alpha: float = 0.002,
        steps: int = 10,
        random_start: bool = True,
        ca_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step L_inf PGD on one slice.

        Returns `(x_adv, logits_adv)` where both are detached.
        """
        target_idx = target_idx.to(self.device).long().view(-1)
        target_idx = torch.unique(target_idx)
        if target_idx.numel() == 0:
            with torch.no_grad():
                return features.clone(), self.forward_logits(features, adj, ca_weights).detach()

        direction = +1.0

        if random_start:
            delta = (2 * torch.rand((target_idx.numel(), self.raw_dim),
                                    device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((target_idx.numel(), self.raw_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(features, target_idx, delta, eps)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            x_adv = self._perturb_targets(features, target_idx, delta)
            logits = self.forward_logits(x_adv, adj, ca_weights)
            loss = F.cross_entropy(logits[target_idx], labels_true)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(features, target_idx, delta, eps)

        x_out = self._perturb_targets(features, target_idx, delta)
        if self.clamp is not None:
            x_out = torch.clamp(x_out, min=self.clamp[0], max=self.clamp[1])

        with torch.no_grad():
            logits_out = self.forward_logits(x_out, adj, ca_weights)

        return x_out.detach(), logits_out.detach()
