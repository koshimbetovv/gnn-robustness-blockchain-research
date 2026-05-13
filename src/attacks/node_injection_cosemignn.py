from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CoSemiNodeInjectionResult:
    x_adv: torch.Tensor              # (n_existing + n_inject, raw_dim + 6)
    edge_index_adv: torch.Tensor     # (2, E_existing + 2 * E_injected)
    logits_adv: torch.Tensor         # (n_existing, 2) — sliced to original nodes
    logits_clean: torch.Tensor       # (n_existing, 2)
    x_injected_base: torch.Tensor    # (n_inject, raw_dim + 6) before PGD on injected rows
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]


class CoSemiGNNNodeInjectionAttack:
    """L_inf node-injection evasion attack for CoSemiGNN, one timestep slice at a time.

    Threat model (per spec):
      - Injected nodes exist ONLY at the attack slice. CoSemiGNN's
        `evolve_gcn.reinitialize_weight()` is called every forward, so each
        slice is independent and injected nodes do not persist into future
        slices.
      - The 6 trailing semi-supervised columns are outputs of auxiliary,
        non-differentiable classifiers run on the labeled training data of the
        slice; an attacker injecting a *new* node has no oracle predictions
        for it, so we set the 6 semi columns to zero on injected rows and
        exclude them from `delta`.
      - Only the first `raw_feature_dim` columns of injected nodes participate
        in `delta`.

    Threat-model parameters mirror `CoSemiPGDAttack`:
      raw_feature_dim : number of leading raw feature columns. Defaults to 165
                        for Elliptic; pass 55 for Elliptic++ actors.

    The dataset stores `edge_index` symmetrically (each undirected edge appears
    in both directions). We mirror that for injected edges so the attack does
    not introduce a directionality artifact.
    """

    def __init__(
        self,
        model,
        device,
        raw_feature_dim: int = 165,
        clamp: Optional[Tuple[float, float]] = None,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.raw_dim = int(raw_feature_dim)
        self.clamp = clamp

    def forward_logits(self, features, adj, ca_weights=None):
        out_line, _ = self.model(features, adj, ca_weights)
        return torch.stack([torch.zeros_like(out_line), out_line], dim=1)

    def _init_injected_features(
        self,
        features: torch.Tensor,
        n_inject: int,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialize injected rows: raw cols from reference, semi cols zero."""
        feat_dim = int(features.size(1))
        semi_dim = feat_dim - self.raw_dim
        if reference_nodes is None or reference_nodes.numel() == 0:
            ref_raw = features[:, : self.raw_dim]
        else:
            ref_raw = features[reference_nodes, : self.raw_dim]

        if init == "mean":
            raw_init = ref_raw.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        elif init == "randn":
            mu = ref_raw.mean(dim=0, keepdim=True)
            std = ref_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
            raw_init = (mu + torch.randn((n_inject, self.raw_dim), device=self.device) * std).detach()
        else:
            raise ValueError(f"Unknown init='{init}'")

        semi_init = torch.zeros((n_inject, semi_dim), device=self.device, dtype=features.dtype)
        return torch.cat([raw_init.to(features.dtype), semi_init], dim=1)

    @staticmethod
    def _build_injection_edges(
        injected_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        edges_per_injected: int,
        connect_strategy: str = "round_robin",
    ) -> list[tuple[int, int]]:
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

    def _augment_edge_index(
        self, adj: torch.Tensor, injected_edges: list[tuple[int, int]]
    ) -> torch.Tensor:
        if not injected_edges:
            return adj.clone()
        ei = torch.tensor(injected_edges, dtype=torch.long, device=adj.device).t()
        ei_rev = torch.stack([ei[1], ei[0]], dim=0)
        return torch.cat([adj, ei, ei_rev], dim=1)

    def _apply_delta(
        self, features: torch.Tensor, base_inj: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        if self.raw_dim == features.size(1):
            inj = base_inj + delta
        else:
            raw = base_inj[:, : self.raw_dim] + delta
            semi = base_inj[:, self.raw_dim :]
            inj = torch.cat([raw, semi], dim=1)
        return torch.cat([features, inj], dim=0)

    def _project_clamp(
        self, base_inj: torch.Tensor, delta: torch.Tensor, eps: float
    ) -> torch.Tensor:
        if self.clamp is None:
            return delta.detach()
        raw_base = base_inj[:, : self.raw_dim]
        raw_adv = raw_base + delta
        raw_adv = torch.clamp(raw_adv, min=self.clamp[0], max=self.clamp[1])
        new_delta = raw_adv - raw_base
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    def attack_slice(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        target_nodes: torch.Tensor,
        labels_true: torch.Tensor,
        n_inject: int = 1,
        edges_per_injected: int = 1,
        eps: float = 0.05,
        alpha: float = 0.01,
        steps: int = 30,
        random_start: bool = True,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
        connect_strategy: str = "round_robin",
        ca_weights: Optional[torch.Tensor] = None,
    ) -> CoSemiNodeInjectionResult:
        """Multi-step L_inf node-injection PGD on one slice.

        Returns a `CoSemiNodeInjectionResult` whose `logits_adv` is sliced to
        the original node set so callers can evaluate against the same node
        index space as `logits_clean`.
        """
        target_nodes = target_nodes.to(self.device).long().view(-1)
        target_nodes = torch.unique(target_nodes)
        n_existing = int(features.size(0))

        with torch.no_grad():
            logits_clean = self.forward_logits(features, adj, ca_weights).detach()

        if target_nodes.numel() == 0:
            return CoSemiNodeInjectionResult(
                x_adv=features.clone(),
                edge_index_adv=adj.clone(),
                logits_adv=logits_clean.clone(),
                logits_clean=logits_clean,
                x_injected_base=features.new_empty((0, features.size(1))),
                injected_node_ids=[],
                injected_edges=[],
            )

        n_inject = int(n_inject)
        injected_ids = torch.arange(n_existing, n_existing + n_inject, device=self.device, dtype=torch.long)
        base_inj = self._init_injected_features(features, n_inject, init=init, reference_nodes=reference_nodes)

        new_edges = self._build_injection_edges(
            injected_ids, target_nodes, int(edges_per_injected), connect_strategy
        )
        edge_index_adv = self._augment_edge_index(adj, new_edges)

        direction = +1.0

        if random_start:
            delta = (2 * torch.rand((n_inject, self.raw_dim), device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((n_inject, self.raw_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(base_inj, delta, eps)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            x_aug = self._apply_delta(features, base_inj, delta)
            logits_full = self.forward_logits(x_aug, edge_index_adv, ca_weights)
            loss = F.cross_entropy(logits_full[target_nodes], labels_true)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(base_inj, delta, eps)

        x_out = self._apply_delta(features, base_inj, delta).detach()
        with torch.no_grad():
            logits_adv_full = self.forward_logits(x_out, edge_index_adv, ca_weights).detach()

        return CoSemiNodeInjectionResult(
            x_adv=x_out,
            edge_index_adv=edge_index_adv.detach(),
            logits_adv=logits_adv_full[:n_existing],
            logits_clean=logits_clean,
            x_injected_base=base_inj.detach(),
            injected_node_ids=injected_ids.detach().cpu().tolist(),
            injected_edges=new_edges,
        )
