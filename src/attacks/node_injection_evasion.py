import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack


@dataclass
class NodeInjectionResult:
    x_adv: torch.Tensor
    edge_index_adv: torch.Tensor
    y_adv: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]  # (src, dst)

# Node Injection Evasion Attack (adds new nodes + edges; optimizes injected-node features via PGD) — 
# inspired by single-node injection evasion work (G-NIA / SNIA).
# Attackers can create new wallets/addresses (nodes) cheaply, then connect strategically.

class NodeInjectionEvasionAttack(BaseAttack):
    """
    Node Injection (evasion) attack:
      - Inject m new nodes (features are learnable/optimized).
      - Add directed edges from injected nodes -> target nodes (incoming to target).
      - Optimize only injected-node features with PGD under L_inf budget.

    Works well for directed PyG message passing when you want to influence targets via incoming edges.
    """

    def __init__(
        self,
        model,
        data,
        device,
        clamp: Optional[tuple[float, float]] = None,
        seed: int = 0,
    ):
        super().__init__(model, data, device)
        self.x = data.x.to(device).detach()
        self.y = data.y.to(device)
        self.edge_index = data.edge_index.to(device)
        self.clamp = clamp
        self.rng = random.Random(seed)

    def _init_injected_features(
        self,
        n_inject: int,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        init:
          - "mean": start from mean of reference_nodes (or all nodes)
          - "randn": gaussian around mean/std of reference nodes
        """
        if reference_nodes is None or reference_nodes.numel() == 0:
            ref = self.x
        else:
            ref = self.x[reference_nodes]

        if init == "mean":
            base = ref.mean(dim=0, keepdim=True).repeat(n_inject, 1)
            return base
        if init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True).clamp_min(1e-6)
            return mu + torch.randn((n_inject, self.x.size(1)), device=self.device) * std
        raise ValueError(f"Unknown init='{init}'")

    def _build_injection_edges(
        self,
        injected_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        edges_per_injected: int,
        connect_strategy: str = "round_robin",
    ) -> list[tuple[int, int]]:
        """
        Returns edges (src, dst). For incoming influence: injected -> target.

        connect_strategy:
          - "round_robin": spread targets across injected nodes
          - "all_to_all": each injected connects to as many targets as budget allows
        """
        injected_ids = injected_ids.view(-1).tolist()
        targets = target_nodes.view(-1).tolist()
        edges: list[tuple[int, int]] = []

        if len(targets) == 0 or len(injected_ids) == 0:
            return edges

        if connect_strategy == "round_robin":
            t_ptr = 0
            for inj in injected_ids:
                for _ in range(edges_per_injected):
                    if t_ptr >= len(targets):
                        t_ptr = 0
                    edges.append((inj, targets[t_ptr]))
                    t_ptr += 1
            return edges

        if connect_strategy == "all_to_all":
            for inj in injected_ids:
                chosen = targets[: min(edges_per_injected, len(targets))]
                for t in chosen:
                    edges.append((inj, t))
            return edges

        raise ValueError(f"Unknown connect_strategy='{connect_strategy}'")

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
        targeted: bool = False,
        target_label: Optional[int] = None,
        early_stop: bool = True,
        connect_strategy: str = "round_robin",
    ) -> NodeInjectionResult:
        """
        Returns NodeInjectionResult with expanded x/y and new edge_index.

        - For untargeted: maximize CE on true labels of target_nodes
        - For targeted: minimize CE toward target_label
        """
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)

        # keep labeled targets only
        target_nodes = target_nodes[self.y[target_nodes] != -1]
        if target_nodes.numel() == 0:
            # nothing to attack
            return NodeInjectionResult(
                x_adv=self.x.clone(),
                edge_index_adv=self.edge_index.clone(),
                y_adv=self.y.clone(),
                injected_node_ids=[],
                injected_edges=[],
            )

        if targeted:
            if target_label is None:
                raise ValueError("target_label must be provided when targeted=True")
            labels = torch.full((target_nodes.numel(),), int(target_label), device=self.device, dtype=torch.long)
            direction = -1.0
        else:
            labels = self.y[target_nodes].long()
            direction = +1.0

        n0 = self.x.size(0)
        injected_ids = torch.arange(n0, n0 + int(n_inject), device=self.device, dtype=torch.long)

        # init injected features
        base_inj = self._init_injected_features(int(n_inject), init=init, reference_nodes=reference_nodes).detach()
        if random_start:
            delta = (2 * torch.rand_like(base_inj) - 1.0) * float(eps)
        else:
            delta = torch.zeros_like(base_inj)
        delta = delta.clamp(-float(eps), float(eps)).detach()

        # build injection edges (fixed during optimization)
        new_edges = self._build_injection_edges(injected_ids, target_nodes, int(edges_per_injected), connect_strategy)
        add_ei = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
        edge_index_adv = torch.cat([self.edge_index, add_ei], dim=1)

        # expanded y (injected nodes unlabeled)
        y_adv = torch.cat([self.y, torch.full((int(n_inject),), -1, device=self.device, dtype=self.y.dtype)], dim=0)

        # optimize injected features
        for _ in range(int(steps)):
            delta.requires_grad_(True)

            x_inj = base_inj + delta
            x_adv = torch.cat([self.x, x_inj], dim=0)

            logits = self.model(x_adv, edge_index_adv)
            loss = F.cross_entropy(logits[target_nodes], labels)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(-float(eps), float(eps))

            if self.clamp is not None:
                x_tmp = torch.cat([self.x, (base_inj + delta)], dim=0)
                x_tmp = torch.clamp(x_tmp, min=self.clamp[0], max=self.clamp[1])
                delta = (x_tmp[n0:] - base_inj).detach().clamp(-float(eps), float(eps))

            if early_stop:
                with torch.no_grad():
                    pred = logits[target_nodes].argmax(dim=1)
                    if targeted:
                        if (pred == int(target_label)).all():
                            break
                    else:
                        # Evasion success: all fraud targets classified as benign (class 0)
                        if (pred == 0).all():
                            break

        x_adv = torch.cat([self.x, (base_inj + delta)], dim=0)
        if self.clamp is not None:
            x_adv = torch.clamp(x_adv, min=self.clamp[0], max=self.clamp[1])

        return NodeInjectionResult(
            x_adv=x_adv.detach(),
            edge_index_adv=edge_index_adv.detach(),
            y_adv=y_adv.detach(),
            injected_node_ids=injected_ids.detach().cpu().tolist(),
            injected_edges=new_edges,
        )
