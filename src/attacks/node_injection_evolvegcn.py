from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class EvolveGCNNodeInjectionResult:
    hist_ndFeats_list: List[torch.Tensor]
    hist_adj_list: List[torch.Tensor]
    node_mask_list: List[torch.Tensor]
    label_idx: torch.Tensor             # passed through unchanged (existing node ids)
    logits_adv: torch.Tensor            # logits at `label_idx` (existing-only)
    x_injected_base: torch.Tensor       # (n_inject, F_last) before PGD on injected rows
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]


class EvolveGCNNodeInjectionAttack:
    """L_inf node-injection evasion attack for EvolveGCN-O, one test window at a time.

    Threat model (per spec):
      - Injected nodes appear ONLY in the last (current) step's feature matrix
        and adjacency `A_list[-1]` / `node_mask_list[-1]`. Earlier history steps
        are untouched. For EvolveGCN-O this is mathematically equivalent to
        replicating the same perturbation across all K steps because:
          * weight evolution `W_t = GRU(W_{t-1})` does not depend on features,
          * intermediate `node_embs` is overwritten inside the GRCU loop,
        so only the last step's features/adjacency reach the classifier.
      - Injected nodes do not persist into future windows (each window is
        attacked independently).

    Threat-model parameters mirror `EvolveGCNPGDAttack`:
      attack_start_col : index of the first perturbable feature column on the
                         injected nodes. Set to 1 for Elliptic (column 0 is
                         IBM time_step metadata) and 0 for Elliptic++ actors.
    """

    def __init__(
        self,
        model,
        device,
        attack_start_col: int = 1,
        clamp: Optional[Tuple[float, float]] = None,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.attack_start_col = int(attack_start_col)
        self.clamp = clamp

    def _forward(self, hist_adj_list, hist_ndFeats_list, node_mask_list, node_indices):
        return self.model(hist_adj_list, hist_ndFeats_list, node_mask_list, node_indices)

    def forward_labels(self, hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx):
        return self._forward(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx)

    @staticmethod
    def _dedupe_targets_with_labels(
        target_nodes: torch.Tensor,
        labels_true: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Drop duplicate targets while preserving the caller's original order."""
        target_nodes = target_nodes.view(-1)
        if target_nodes.numel() == 0:
            return target_nodes, labels_true

        keep_pos: list[int] = []
        seen: set[int] = set()
        for pos, node_id in enumerate(target_nodes.detach().cpu().tolist()):
            if int(node_id) in seen:
                continue
            seen.add(int(node_id))
            keep_pos.append(pos)

        keep_idx = torch.tensor(keep_pos, dtype=torch.long, device=target_nodes.device)
        target_nodes = target_nodes[keep_idx]

        if labels_true is None:
            return target_nodes, None
        labels_true = labels_true.view(-1)[keep_idx]
        return target_nodes, labels_true

    def _init_injected_features(
        self,
        base_x: torch.Tensor,
        n_inject: int,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if reference_nodes is None or reference_nodes.numel() == 0:
            ref = base_x
        else:
            ref = base_x[reference_nodes]
        if init == "mean":
            return ref.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        if init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True).clamp_min(1e-6)
            return (mu + torch.randn((n_inject, base_x.size(1)), device=self.device) * std).detach()
        raise ValueError(f"Unknown init='{init}'")

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

    def _expanded_features(
        self, base_x: torch.Tensor, base_inj: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        col_start = self.attack_start_col
        attack_end = col_start + delta.size(1)
        if col_start == 0 and attack_end == base_x.size(1):
            inj = base_inj + delta
        else:
            pre = base_inj[:, :col_start]
            mid = base_inj[:, col_start:attack_end] + delta
            post = base_inj[:, attack_end:]
            inj = torch.cat([pre, mid, post], dim=1)
        return torch.cat([base_x, inj], dim=0)

    def _project_clamp(
        self, base_inj: torch.Tensor, delta: torch.Tensor, eps: float
    ) -> torch.Tensor:
        if self.clamp is None:
            return delta.detach()
        col_start = self.attack_start_col
        attack_end = col_start + delta.size(1)
        raw_base = base_inj[:, col_start:attack_end]
        raw_adv = raw_base + delta
        raw_adv = torch.clamp(raw_adv, min=self.clamp[0], max=self.clamp[1])
        new_delta = raw_adv - raw_base
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    @staticmethod
    def _augment_adj(
        A_norm: torch.Tensor,
        n_existing: int,
        n_inject: int,
        injected_edges: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Build a re-normalized (N+m)x(N+m) adjacency that adds the injected
        edges (in both directions, matching the dataset's symmetric edge
        convention) and a fresh self-loop set."""
        n_total = n_existing + n_inject
        A_coal = A_norm.coalesce()
        idx = A_coal.indices()
        # Drop the self-loops added by the dataset's `_normalize_adj`; we'll
        # re-add them across the full (N+m) node set below.
        off_diag = idx[0] != idx[1]
        base_idx = idx[:, off_diag]

        if injected_edges:
            ei = torch.tensor(injected_edges, dtype=torch.long, device=A_norm.device).t()
            ei_rev = torch.stack([ei[1], ei[0]], dim=0)
            new_idx = torch.cat([base_idx, ei, ei_rev], dim=1)
        else:
            new_idx = base_idx

        new_vals = torch.ones(new_idx.size(1), dtype=torch.float32, device=A_norm.device)

        sp = torch.sparse_coo_tensor(new_idx, new_vals, size=(n_total, n_total)).coalesce()
        eye_idx = torch.arange(n_total, dtype=torch.long, device=A_norm.device)
        eye = torch.sparse_coo_tensor(
            torch.stack([eye_idx, eye_idx], dim=0),
            torch.ones(n_total, dtype=torch.float32, device=A_norm.device),
            size=(n_total, n_total),
        )
        sp = (sp + eye).coalesce()

        norm_idx = sp.indices()
        norm_vals = sp.values()
        degree = torch.sparse.sum(sp, dim=1).to_dense()
        di = degree[norm_idx[0]]
        dj = degree[norm_idx[1]]
        norm_vals = norm_vals * ((di * dj) ** -0.5)
        return torch.sparse_coo_tensor(
            norm_idx, norm_vals, size=(n_total, n_total), dtype=torch.float32
        ).coalesce()

    @staticmethod
    def _expanded_mask(node_mask: torch.Tensor, n_inject: int) -> torch.Tensor:
        """Append zero rows so injected nodes pass the TopK active-node filter."""
        pad = torch.zeros((n_inject, node_mask.size(1)), dtype=node_mask.dtype, device=node_mask.device)
        return torch.cat([node_mask, pad], dim=0)

    def attack_window(
        self,
        hist_adj_list: List[torch.Tensor],
        hist_ndFeats_list: List[torch.Tensor],
        node_mask_list: List[torch.Tensor],
        label_idx: torch.Tensor,
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
    ) -> EvolveGCNNodeInjectionResult:
        """Multi-step L_inf node-injection PGD on one window.

        Only the last history step is augmented (per the -O spec). The earlier
        history steps remain unchanged in shape and content.
        """
        target_nodes = target_nodes.to(self.device).long().view(-1)
        labels_true = labels_true.to(self.device).long().view(-1)
        target_nodes, labels_true = self._dedupe_targets_with_labels(target_nodes, labels_true)

        base_last = hist_ndFeats_list[-1]
        base_adj_last = hist_adj_list[-1]
        base_mask_last = node_mask_list[-1]
        n_existing = int(base_last.size(0))

        if target_nodes.numel() == 0:
            with torch.no_grad():
                logits = self.forward_labels(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
            return EvolveGCNNodeInjectionResult(
                hist_ndFeats_list=[x.clone() for x in hist_ndFeats_list],
                hist_adj_list=list(hist_adj_list),
                node_mask_list=[m.clone() for m in node_mask_list],
                label_idx=label_idx,
                logits_adv=logits,
                x_injected_base=base_last.new_empty((0, base_last.size(1))),
                injected_node_ids=[],
                injected_edges=[],
            )

        n_inject = int(n_inject)
        attack_dim = base_last.size(1) - self.attack_start_col
        if attack_dim <= 0:
            raise ValueError(
                f"attack_start_col={self.attack_start_col} leaves no feature columns to perturb "
                f"(features have {base_last.size(1)} columns)."
            )

        injected_ids = torch.arange(n_existing, n_existing + n_inject, device=self.device, dtype=torch.long)
        base_inj = self._init_injected_features(base_last, n_inject, init=init, reference_nodes=reference_nodes)

        new_edges = self._build_injection_edges(
            injected_ids, target_nodes, int(edges_per_injected), connect_strategy
        )

        adj_last_aug = self._augment_adj(base_adj_last, n_existing, n_inject, new_edges)
        mask_last_aug = self._expanded_mask(base_mask_last, n_inject)
        hist_adj_aug = list(hist_adj_list[:-1]) + [adj_last_aug]
        hist_mask_aug = list(node_mask_list[:-1]) + [mask_last_aug]

        direction = +1.0

        if random_start:
            delta = (2 * torch.rand((n_inject, attack_dim), device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((n_inject, attack_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(base_inj, delta, eps)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            x_last_aug = self._expanded_features(base_last, base_inj, delta)
            hist_feats_aug = list(hist_ndFeats_list[:-1]) + [x_last_aug]
            # Use `target_nodes` as node_indices so the gradient is attributable
            # exactly to the attacked rows (mirrors the PGD driver).
            logits_attack = self._forward(hist_adj_aug, hist_feats_aug, hist_mask_aug, target_nodes)
            loss = F.cross_entropy(logits_attack, labels_true)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(base_inj, delta, eps)

        x_last_out = self._expanded_features(base_last, base_inj, delta).detach()
        hist_feats_out = list(hist_ndFeats_list[:-1]) + [x_last_out]

        with torch.no_grad():
            logits_adv = self.forward_labels(hist_adj_aug, hist_feats_out, hist_mask_aug, label_idx).detach()

        return EvolveGCNNodeInjectionResult(
            hist_ndFeats_list=hist_feats_out,
            hist_adj_list=hist_adj_aug,
            node_mask_list=hist_mask_aug,
            label_idx=label_idx,
            logits_adv=logits_adv,
            x_injected_base=base_inj.detach(),
            injected_node_ids=injected_ids.detach().cpu().tolist(),
            injected_edges=new_edges,
        )
