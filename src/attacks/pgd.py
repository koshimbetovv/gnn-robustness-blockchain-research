from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack
from src.attacks.model_forward import forward_logits


class PGDAttack(BaseAttack):
    """L_inf PGD feature attack for node classification on a fixed graph.

    Perturbs node features for selected target nodes under an L_infinity budget
    using projected gradient descent over `steps` iterations.

    Scalability: optimize a small `delta` only for attacked nodes (not full data.x).

    Threat-model parameters:
      attack_dim  : number of leading feature columns the attacker can perturb.
                    When `data.x` is a concatenation like `[raw || derived]`, set
                    `attack_dim = raw_dim` so only the raw slice is attacked.
                    Defaults to the full feature width.
      rebuild_fn  : optional `Callable[[Tensor], Tensor]` that takes the full
                    feature tensor (after raw perturbation) and returns a
                    consistent full feature tensor (e.g. recomputing the derived
                    slice from the perturbed raw slice). Must be differentiable
                    for the gradient to flow through the derived path.
    """

    def __init__(
        self,
        model,
        data,
        device,
        clamp: tuple[float, float] | None = None,
        attack_dim: Optional[int] = None,
        rebuild_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, data, device)
        self.x = data.x.to(device).detach()
        self.y = data.y.to(device)
        self.edge_index = data.edge_index.to(device)
        ts = getattr(data, "time_step", None)
        self.time_step = ts.to(device) if torch.is_tensor(ts) else None
        self.clamp = clamp
        self.attack_dim = int(attack_dim) if attack_dim is not None else int(self.x.size(1))
        if not (1 <= self.attack_dim <= self.x.size(1)):
            raise ValueError(f"attack_dim must be in [1, {self.x.size(1)}], got {self.attack_dim}")
        self.rebuild_fn = rebuild_fn

    def _apply_delta(self, delta: torch.Tensor, target_nodes: torch.Tensor) -> torch.Tensor:
        """Add `delta` to the first `attack_dim` columns at `target_nodes` and
        return a full-feature tensor. If a rebuild_fn is set, run it to recompute
        any derived columns so the graph input stays internally consistent."""
        if self.attack_dim == self.x.size(1):
            target_new = self.x[target_nodes] + delta
        else:
            raw = self.x[target_nodes, :self.attack_dim] + delta
            other = self.x[target_nodes, self.attack_dim:]
            target_new = torch.cat([raw, other], dim=1)

        x_full = self.x.index_copy(0, target_nodes, target_new)
        if self.rebuild_fn is not None:
            x_full = self.rebuild_fn(x_full)
        return x_full

    def _project_clamp(self, delta: torch.Tensor, target_nodes: torch.Tensor, eps: float) -> torch.Tensor:
        """Project `delta` in raw-feature space.

        When `rebuild_fn` is set, derived columns must be recomputed from the
        final feasible raw features. So projection/clamping happens only on the
        perturbable raw slice here; the full feature rebuild is deferred to
        `_apply_delta`.
        """
        if self.clamp is None:
            return delta.detach()
        raw_base = self.x[target_nodes, :self.attack_dim]
        raw_adv = raw_base + delta
        raw_adv = torch.clamp(raw_adv, min=self.clamp[0], max=self.clamp[1])
        new_delta = raw_adv - raw_base
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    def attack(
        self,
        target_nodes: torch.Tensor,
        eps: float = 0.01,
        alpha: float = 0.002,
        steps: int = 10,
        random_start: bool = True,
    ) -> torch.Tensor:
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)
        target_nodes = torch.unique(target_nodes)

        # filter unlabeled
        labeled_mask = self.y[target_nodes] != -1
        target_nodes = target_nodes[labeled_mask]
        if target_nodes.numel() == 0:
            return self.x.clone()

        labels = self.y[target_nodes].long()
        direction = +1.0  # maximize loss on true label

        if random_start:
            delta = (2 * torch.rand((target_nodes.numel(), self.attack_dim),
                                    device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((target_nodes.numel(), self.attack_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(delta, target_nodes, eps)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            x_adv = self._apply_delta(delta, target_nodes)
            logits = forward_logits(self.model, x_adv, self.edge_index, time_step=self.time_step)
            loss = F.cross_entropy(logits[target_nodes], labels)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(delta, target_nodes, eps)

        return self._apply_delta(delta, target_nodes).detach()
