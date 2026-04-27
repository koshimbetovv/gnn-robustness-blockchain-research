from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack
from src.attacks.model_forward import forward_logits


class FGSMAttack(BaseAttack):
    """FGSM feature attack for node classification on a fixed graph.

    Perturbs node features for selected target nodes under an L_infinity budget.

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

    def attack(
        self,
        target_nodes: torch.Tensor,
        eps: float = 0.01,
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

        delta = torch.zeros((target_nodes.numel(), self.attack_dim),
                            device=self.device, requires_grad=True)

        x_adv = self._apply_delta(delta, target_nodes)
        logits = forward_logits(self.model, x_adv, self.edge_index, time_step=self.time_step)
        loss = F.cross_entropy(logits[target_nodes], labels)

        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        delta_adv = direction * float(eps) * grad.sign()

        x_out = self._apply_delta(delta_adv.detach(), target_nodes)

        if self.clamp is not None:
            x_out = torch.clamp(x_out, min=self.clamp[0], max=self.clamp[1])

        return x_out.detach()
