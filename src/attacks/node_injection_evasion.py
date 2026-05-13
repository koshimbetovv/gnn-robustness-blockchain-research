import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack
from src.attacks.model_forward import forward_logits


@dataclass
class NodeInjectionResult:
    x_adv: torch.Tensor
    edge_index_adv: torch.Tensor
    y_adv: torch.Tensor
    time_step_adv: Optional[torch.Tensor]
    x_injected_base: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]  # (src, dst)


class NodeInjectionEvasionAttack(BaseAttack):
    """L_inf node-injection evasion attack for node classification on a fixed graph.

    Injects `n_inject` new nodes, connects them with directed edges
    `injected -> target`, and optimizes the injected nodes' features under an
    L_infinity budget via projected gradient descent. Existing-node features are
    untouched; the perturbation is the difference between the optimized injected
    features and their initialization.

    Threat-model parameters (mirroring `PGDAttack`):
      attack_dim  : number of leading feature columns the attacker can perturb on
                    injected nodes. When `data.x` is a concatenation like
                    `[raw || derived]`, set `attack_dim = raw_dim` so only the raw
                    slice is optimized.
                    Defaults to the full feature width.
      rebuild_fn  : optional `Callable[[Tensor], Tensor]` that takes the full
                    feature tensor (existing + injected, after raw perturbation)
                    and returns a consistent full feature tensor (e.g. recomputing
                    the derived slice from the perturbed raw slice). Must be
                    differentiable for the gradient to flow through the derived
                    path.

    For ChronoWaveGNN, each injected node's `time_step` is inherited from the
    first target it connects to (i.e. the injected node "appears" at the time of
    the message it sends).
    """

    def __init__(
        self,
        model,
        data,
        device,
        clamp: tuple[float, float] | None = None,
        attack_dim: Optional[int] = None,
        rebuild_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: int = 0,
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
        self.rng = random.Random(seed)

    def _init_injected_features(
        self,
        n_inject: int,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialize injected-node features from existing nodes.

        init:
          - "mean" : repeat the mean of `reference_nodes` (or all nodes).
          - "randn": gaussian around mean/std of reference nodes.
        """
        if reference_nodes is None or reference_nodes.numel() == 0:
            ref = self.x
        else:
            ref = self.x[reference_nodes]

        if init == "mean":
            return ref.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        if init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True).clamp_min(1e-6)
            return (mu + torch.randn((n_inject, self.x.size(1)), device=self.device) * std).detach()
        raise ValueError(f"Unknown init='{init}'")

    @staticmethod
    def _build_injection_edges(
        injected_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        edges_per_injected: int,
        connect_strategy: str = "round_robin",
    ) -> list[tuple[int, int]]:
        """Build (src, dst) edges from injected nodes to targets.

        connect_strategy:
          - "round_robin": spread targets across injected nodes.
          - "all_to_all" : each injected connects to up to `edges_per_injected` targets.
        """
        injected = injected_ids.view(-1).tolist()
        targets = target_nodes.view(-1).tolist()
        if not injected or not targets:
            return []

        if connect_strategy == "round_robin":
            edges: list[tuple[int, int]] = []
            t_ptr = 0
            for inj in injected:
                for _ in range(edges_per_injected):
                    edges.append((inj, targets[t_ptr % len(targets)]))
                    t_ptr += 1
            return edges

        if connect_strategy == "all_to_all":
            return [
                (inj, t)
                for inj in injected
                for t in targets[: min(edges_per_injected, len(targets))]
            ]

        raise ValueError(f"Unknown connect_strategy='{connect_strategy}'")

    def _apply_delta(self, base_inj: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Build the full feature tensor `[existing || perturbed_injected]` and
        run `rebuild_fn` if set so derived columns stay consistent."""
        if self.attack_dim == self.x.size(1):
            inj = base_inj + delta
        else:
            raw = base_inj[:, :self.attack_dim] + delta
            other = base_inj[:, self.attack_dim:]
            inj = torch.cat([raw, other], dim=1)

        x_full = torch.cat([self.x, inj], dim=0)
        if self.rebuild_fn is not None:
            x_full = self.rebuild_fn(x_full)
        return x_full

    def _project_clamp(self, base_inj: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
        """Project `delta` in raw-feature space (mirrors `PGDAttack._project_clamp`).

        When `rebuild_fn` is set, derived columns are recomputed from the final
        feasible raw features in `_apply_delta`; projection/clamping happens only
        on the perturbable raw slice here.
        """
        if self.clamp is None:
            return delta.detach()
        raw_base = base_inj[:, :self.attack_dim]
        raw_adv = raw_base + delta
        raw_adv = torch.clamp(raw_adv, min=self.clamp[0], max=self.clamp[1])
        new_delta = raw_adv - raw_base
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    def _build_time_step(self, injected_edges: list[tuple[int, int]], n_inject: int, n0: int) -> Optional[torch.Tensor]:
        """Extend `data.time_step` with one entry per injected node.

        Each injected node inherits the time_step of the first target it connects
        to. Injected nodes with no outgoing edge get time_step 0 (no message will
        carry their feature anyway).
        """
        if self.time_step is None:
            return None
        first_target_ts = [0] * n_inject
        seen = [False] * n_inject
        for src, dst in injected_edges:
            i = int(src) - n0
            if 0 <= i < n_inject and not seen[i]:
                first_target_ts[i] = int(self.time_step[int(dst)].item())
                seen[i] = True
        ts_inj = torch.tensor(first_target_ts, device=self.device, dtype=self.time_step.dtype)
        return torch.cat([self.time_step, ts_inj], dim=0)

    def attack(
        self,
        target_nodes: torch.Tensor,
        n_inject: int = 1,
        edges_per_injected: int = 1,
        eps: float = 0.05,
        alpha: float = 0.01,
        steps: int = 30,
        random_start: bool = True,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
        connect_strategy: str = "round_robin",
    ) -> NodeInjectionResult:
        """Multi-step L_inf PGD over injected-node features (untargeted)."""
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)
        target_nodes = torch.unique(target_nodes)

        # filter unlabeled
        labeled_mask = self.y[target_nodes] != -1
        target_nodes = target_nodes[labeled_mask]
        if target_nodes.numel() == 0:
            return NodeInjectionResult(
                x_adv=self.x.clone(),
                edge_index_adv=self.edge_index.clone(),
                y_adv=self.y.clone(),
                time_step_adv=None if self.time_step is None else self.time_step.clone(),
                x_injected_base=self.x.new_empty((0, self.x.size(1))),
                injected_node_ids=[],
                injected_edges=[],
            )

        labels = self.y[target_nodes].long()
        direction = +1.0  # maximize loss on true label

        n0 = self.x.size(0)
        n_inject = int(n_inject)
        injected_ids = torch.arange(n0, n0 + n_inject, device=self.device, dtype=torch.long)

        base_inj = self._init_injected_features(n_inject, init=init, reference_nodes=reference_nodes)

        new_edges = self._build_injection_edges(
            injected_ids, target_nodes, int(edges_per_injected), connect_strategy
        )
        if new_edges:
            add_ei = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
            edge_index_adv = torch.cat([self.edge_index, add_ei], dim=1)
        else:
            edge_index_adv = self.edge_index.clone()

        y_adv = torch.cat(
            [self.y, torch.full((n_inject,), -1, device=self.device, dtype=self.y.dtype)], dim=0
        )
        time_step_adv = self._build_time_step(new_edges, n_inject, n0)

        if random_start:
            delta = (2 * torch.rand((n_inject, self.attack_dim), device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((n_inject, self.attack_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(base_inj, delta, eps)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            x_adv = self._apply_delta(base_inj, delta)
            logits = forward_logits(self.model, x_adv, edge_index_adv, time_step=time_step_adv)
            loss = F.cross_entropy(logits[target_nodes], labels)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(base_inj, delta, eps)

        x_adv = self._apply_delta(base_inj, delta).detach()

        return NodeInjectionResult(
            x_adv=x_adv,
            edge_index_adv=edge_index_adv.detach(),
            y_adv=y_adv.detach(),
            time_step_adv=None if time_step_adv is None else time_step_adv.detach(),
            x_injected_base=base_inj.detach(),
            injected_node_ids=injected_ids.detach().cpu().tolist(),
            injected_edges=new_edges,
        )
