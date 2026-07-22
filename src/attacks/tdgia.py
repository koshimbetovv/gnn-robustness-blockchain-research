from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack
from src.attacks.model_forward import forward_logits


@dataclass
class TDGIAResult:
    x_adv: torch.Tensor
    edge_index_adv: torch.Tensor
    y_adv: torch.Tensor
    time_step_adv: Optional[torch.Tensor]
    x_injected_base: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]


class TDGIAAttack(BaseAttack):
    """TDGIA-style graph injection attack for static node-classification models.

    This adapts the paper's two main stages to the repo's static setting:
      1. topological defective edge selection
      2. smooth adversarial optimization of injected-node features

    The attack is sequential: injected nodes are added in batches, each batch is
    connected to the currently most vulnerable targets, and then its features are
    optimized while keeping existing-node features fixed.
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

    def _feature_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clamp is not None:
            feat_min = torch.full(
                (self.attack_dim,), float(self.clamp[0]), device=self.device, dtype=self.x.dtype
            )
            feat_max = torch.full(
                (self.attack_dim,), float(self.clamp[1]), device=self.device, dtype=self.x.dtype
            )
            return feat_min, feat_max

        x_attack = self.x[:, : self.attack_dim]
        feat_min = x_attack.min(dim=0).values
        feat_max = x_attack.max(dim=0).values
        invalid = (~torch.isfinite(feat_min)) | (~torch.isfinite(feat_max)) | (feat_min == feat_max)
        if invalid.any():
            feat_min = feat_min.clone()
            feat_max = feat_max.clone()
            center = torch.where(torch.isfinite(feat_min), feat_min, torch.zeros_like(feat_min))
            feat_min[invalid] = center[invalid] - 1.0
            feat_max[invalid] = center[invalid] + 1.0
        return feat_min, feat_max

    def _init_injected_features(
        self,
        n_inject: int,
        *,
        init: str = "randn",
        reference_nodes: Optional[torch.Tensor] = None,
        sigma_scale: float = 1.0,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
    ) -> torch.Tensor:
        if reference_nodes is None or reference_nodes.numel() == 0:
            ref = self.x
        else:
            ref = self.x[reference_nodes]

        if init == "zeros":
            base = torch.zeros((n_inject, self.x.size(1)), device=self.device)
        elif init == "mean":
            base = ref.mean(dim=0, keepdim=True).repeat(n_inject, 1)
        elif init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            base = mu + float(sigma_scale) * torch.randn((n_inject, self.x.size(1)), device=self.device) * std
        else:
            raise ValueError(f"Unknown init={init!r}")

        base[:, : self.attack_dim] = base[:, : self.attack_dim].clamp(min=feat_min, max=feat_max)
        if self.attack_dim < self.x.size(1):
            suffix_ref = ref[:, self.attack_dim :]
            sample_idx = torch.randint(
                int(suffix_ref.size(0)),
                (int(n_inject),),
                device=self.device,
            )
            base[:, self.attack_dim :] = suffix_ref[sample_idx].to(base.dtype)
        return base.detach()

    def _apply_injected_raw(self, x_existing: torch.Tensor, raw_inj: torch.Tensor, base_inj: torch.Tensor) -> torch.Tensor:
        if self.attack_dim == self.x.size(1):
            inj = raw_inj
        else:
            other = base_inj[:, self.attack_dim:]
            inj = torch.cat([raw_inj, other], dim=1)
        x_full = torch.cat([x_existing, inj], dim=0)
        if self.rebuild_fn is not None:
            x_full = self.rebuild_fn(x_full)
        return x_full

    @staticmethod
    def _smoothmap(latent: torch.Tensor, feat_min: torch.Tensor, feat_max: torch.Tensor) -> torch.Tensor:
        mid = 0.5 * (feat_max + feat_min)
        amp = 0.5 * (feat_max - feat_min)
        return mid + amp * torch.sin(latent)

    @staticmethod
    def _inv_smoothmap(x: torch.Tensor, feat_min: torch.Tensor, feat_max: torch.Tensor) -> torch.Tensor:
        mid = 0.5 * (feat_max + feat_min)
        amp = (0.5 * (feat_max - feat_min)).clamp_min(1e-6)
        scaled = ((x - mid) / amp).clamp(min=-0.999999, max=0.999999)
        return torch.asin(scaled)

    @staticmethod
    def _build_injection_edges(
        injected_ids: torch.Tensor,
        target_nodes_sorted: torch.Tensor,
        degree_limit: int,
        time_step: Optional[torch.Tensor] = None,
    ) -> list[tuple[int, int]]:
        if injected_ids.numel() == 0 or target_nodes_sorted.numel() == 0 or degree_limit <= 0:
            return []

        if time_step is not None:
            target_ts = time_step[target_nodes_sorted.long()].detach().cpu().tolist()
            targets_by_ts: dict[int, list[int]] = {}
            seed_ts_order: list[int] = []
            for node, ts in zip(target_nodes_sorted.detach().cpu().tolist(), target_ts):
                ts_key = int(ts)
                targets_by_ts.setdefault(ts_key, []).append(int(node))
                seed_ts_order.append(ts_key)

            group_ptr: dict[int, int] = {ts: 0 for ts in targets_by_ts}
            edges: list[tuple[int, int]] = []
            for inj_pos, inj in enumerate(injected_ids.detach().cpu().tolist()):
                ts_key = seed_ts_order[inj_pos % len(seed_ts_order)]
                same_ts_targets = targets_by_ts[ts_key]
                ptr = group_ptr[ts_key]
                for offset in range(int(degree_limit)):
                    dst = same_ts_targets[(ptr + offset) % len(same_ts_targets)]
                    edges.append((int(inj), int(dst)))
                group_ptr[ts_key] = ptr + int(degree_limit)
            return edges

        total_needed = int(injected_ids.numel()) * int(degree_limit)
        reps = (total_needed + int(target_nodes_sorted.numel()) - 1) // int(target_nodes_sorted.numel())
        tiled_targets = target_nodes_sorted.repeat(reps)[:total_needed]

        edges: list[tuple[int, int]] = []
        ptr = 0
        for inj in injected_ids.tolist():
            for _ in range(int(degree_limit)):
                dst = int(tiled_targets[ptr].item())
                edges.append((int(inj), dst))
                ptr += 1
        return edges

    def _edge_destination_targets_and_labels(
        self,
        injected_edges: list[tuple[int, int]],
        target_nodes: torch.Tensor,
        surrogate_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not injected_edges:
            return (
                target_nodes.new_empty((0,)),
                surrogate_labels.new_empty((0,)),
            )

        label_by_target = {
            int(node): int(label)
            for node, label in zip(
                target_nodes.detach().cpu().tolist(),
                surrogate_labels.detach().cpu().tolist(),
            )
        }
        selected_nodes: list[int] = []
        selected_labels: list[int] = []
        seen: set[int] = set()
        for _, dst in injected_edges:
            dst = int(dst)
            if dst in seen or dst not in label_by_target:
                continue
            seen.add(dst)
            selected_nodes.append(dst)
            selected_labels.append(label_by_target[dst])

        return (
            torch.tensor(selected_nodes, dtype=torch.long, device=self.device),
            torch.tensor(selected_labels, dtype=surrogate_labels.dtype, device=self.device),
        )

    def _extend_time_step(
        self,
        time_step_curr: Optional[torch.Tensor],
        injected_edges: list[tuple[int, int]],
        injected_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if time_step_curr is None:
            return None

        inj_to_ts: dict[int, int] = {}
        for src, dst in injected_edges:
            if src not in inj_to_ts:
                inj_to_ts[src] = int(time_step_curr[int(dst)].item())

        ts_new = []
        for inj in injected_ids.tolist():
            ts_new.append(inj_to_ts.get(int(inj), 0))
        ts_inj = torch.tensor(ts_new, device=self.device, dtype=time_step_curr.dtype)
        return torch.cat([time_step_curr, ts_inj], dim=0)

    @staticmethod
    def _degree_total(num_nodes: int, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
        row = edge_index[0].long()
        col = edge_index[1].long()
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))
        return deg.clamp_min_(1.0)

    def _defective_scores(
        self,
        *,
        x_curr: torch.Tensor,
        edge_index_curr: torch.Tensor,
        time_step_curr: Optional[torch.Tensor],
        target_nodes: torch.Tensor,
        surrogate_labels: torch.Tensor,
        degree_limit: int,
        alpha_mu: float,
        k1: float,
        k2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = forward_logits(self.model, x_curr, edge_index_curr, time_step=time_step_curr)
            probs = F.softmax(logits[target_nodes], dim=1)
            pv = probs.gather(1, surrogate_labels.view(-1, 1)).view(-1).clamp_min(1e-12)

        deg = self._degree_total(x_curr.size(0), edge_index_curr, x_curr.device)[target_nodes]
        d = float(max(int(degree_limit), 1))
        lam = float(k1) / torch.sqrt(deg * d) + float(k2) / deg
        mu = (float(alpha_mu) * pv + (1.0 - float(alpha_mu))) * lam
        return mu, pv

    def _optimize_batch_features(
        self,
        *,
        x_existing: torch.Tensor,
        base_inj: torch.Tensor,
        edge_index_adv: torch.Tensor,
        time_step_adv: Optional[torch.Tensor],
        target_nodes: torch.Tensor,
        surrogate_labels: torch.Tensor,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
        eps_feature: Optional[float],
        steps: int,
        lr: float,
        smooth_r: float,
    ) -> torch.Tensor:
        if target_nodes.numel() == 0:
            return self._apply_injected_raw(
                x_existing, base_inj[:, : self.attack_dim], base_inj
            ).detach()

        if eps_feature is None:
            opt_min = feat_min
            opt_max = feat_max
        else:
            eps = float(eps_feature)
            base_raw = base_inj[:, : self.attack_dim]
            opt_min = torch.maximum(base_raw - eps, feat_min)
            opt_max = torch.minimum(base_raw + eps, feat_max)

        latent = self._inv_smoothmap(base_inj[:, : self.attack_dim], opt_min, opt_max).detach()
        latent.requires_grad_(True)
        opt = torch.optim.Adam([latent], lr=float(lr))

        for _ in range(int(steps)):
            opt.zero_grad()
            raw_inj = self._smoothmap(latent, opt_min, opt_max)
            x_adv = self._apply_injected_raw(x_existing, raw_inj, base_inj)
            logits = forward_logits(self.model, x_adv, edge_index_adv, time_step=time_step_adv)
            probs = F.softmax(logits[target_nodes], dim=1)
            pv = probs.gather(1, surrogate_labels.view(-1, 1)).view(-1).clamp_min(1e-12)
            loss = torch.relu(float(smooth_r) + torch.log(pv)).pow(2).mean()
            latent.grad = torch.autograd.grad(loss, latent, retain_graph=False, create_graph=False)[0]
            opt.step()

        raw_final = self._smoothmap(latent.detach(), opt_min, opt_max)
        return self._apply_injected_raw(x_existing, raw_final, base_inj).detach()

    def attack(
        self,
        target_nodes: torch.Tensor,
        *,
        attack_labels: Optional[torch.Tensor] = None,
        n_inject: int = 5,
        degree_limit: int = 20,
        batch_size: int = 1,
        steps: int = 30,
        lr: float = 0.05,
        smooth_r: float = 0.5,
        alpha_mu: float = 0.5,
        k1: float = 1.0,
        k2: float = 1.0,
        init: str = "randn",
        reference_nodes: Optional[torch.Tensor] = None,
        sigma_scale: float = 1.0,
        eps_feature: Optional[float] = None,
    ) -> TDGIAResult:
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)
        if attack_labels is not None:
            if not torch.is_tensor(attack_labels):
                attack_labels = torch.tensor(attack_labels, dtype=torch.long)
            attack_labels = attack_labels.to(self.device).long().view(-1)
            if attack_labels.numel() != target_nodes.numel():
                raise ValueError(
                    "attack_labels must align one-to-one with target_nodes "
                    f"({attack_labels.numel()} labels for {target_nodes.numel()} targets)."
                )

        keep_pos: list[int] = []
        seen: set[int] = set()
        for pos, node_id in enumerate(target_nodes.detach().cpu().tolist()):
            if int(node_id) in seen:
                continue
            seen.add(int(node_id))
            keep_pos.append(pos)
        if keep_pos:
            keep_idx = torch.tensor(keep_pos, dtype=torch.long, device=self.device)
            target_nodes = target_nodes[keep_idx]
            if attack_labels is not None:
                attack_labels = attack_labels[keep_idx]

        labeled_mask = self.y[target_nodes] != -1
        target_nodes = target_nodes[labeled_mask]
        if attack_labels is not None:
            attack_labels = attack_labels[labeled_mask]
            valid_attack_labels = attack_labels != -1
            target_nodes = target_nodes[valid_attack_labels]
            attack_labels = attack_labels[valid_attack_labels]
        if target_nodes.numel() == 0 or int(n_inject) <= 0:
            return TDGIAResult(
                x_adv=self.x.clone(),
                edge_index_adv=self.edge_index.clone(),
                y_adv=self.y.clone(),
                time_step_adv=None if self.time_step is None else self.time_step.clone(),
                x_injected_base=self.x.new_empty((0, self.x.size(1))),
                injected_node_ids=[],
                injected_edges=[],
            )

        feat_min, feat_max = self._feature_range()

        with torch.no_grad():
            logits_clean = forward_logits(self.model, self.x, self.edge_index, time_step=self.time_step)
            if attack_labels is None:
                surrogate_labels = logits_clean[target_nodes].argmax(dim=1)
            else:
                surrogate_labels = attack_labels
            if (
                surrogate_labels.numel() > 0
                and (
                    int(surrogate_labels.min().item()) < 0
                    or int(surrogate_labels.max().item()) >= int(logits_clean.size(1))
                )
            ):
                raise ValueError(
                    "attack_labels contain class ids outside the model output range "
                    f"[0, {int(logits_clean.size(1)) - 1}]."
                )

        x_curr = self.x.clone()
        edge_index_curr = self.edge_index.clone()
        y_curr = self.y.clone()
        time_step_curr = None if self.time_step is None else self.time_step.clone()

        all_base_inj: list[torch.Tensor] = []
        injected_node_ids: list[int] = []
        injected_edges: list[tuple[int, int]] = []

        remaining = int(n_inject)
        batch_size = max(1, int(batch_size))
        degree_limit = max(1, int(degree_limit))
        if int(steps) < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        if eps_feature is not None and float(eps_feature) < 0.0:
            raise ValueError(f"eps_feature must be non-negative or None, got {eps_feature}")

        while remaining > 0:
            bseq = min(batch_size, remaining)

            mu, _ = self._defective_scores(
                x_curr=x_curr,
                edge_index_curr=edge_index_curr,
                time_step_curr=time_step_curr,
                target_nodes=target_nodes,
                surrogate_labels=surrogate_labels,
                degree_limit=degree_limit,
                alpha_mu=alpha_mu,
                k1=k1,
                k2=k2,
            )
            order = torch.argsort(mu, descending=True)
            sorted_targets = target_nodes[order]

            n0 = x_curr.size(0)
            inj_ids = torch.arange(n0, n0 + bseq, device=self.device, dtype=torch.long)
            base_inj = self._init_injected_features(
                bseq,
                init=init,
                reference_nodes=reference_nodes,
                sigma_scale=sigma_scale,
                feat_min=feat_min,
                feat_max=feat_max,
            )
            batch_edges = self._build_injection_edges(
                inj_ids, sorted_targets, degree_limit, time_step_curr
            )
            if batch_edges:
                add_ei = torch.tensor(batch_edges, dtype=torch.long, device=self.device).t().contiguous()
                add_ei_rev = torch.stack([add_ei[1], add_ei[0]], dim=0)
                edge_index_adv = torch.cat([edge_index_curr, add_ei, add_ei_rev], dim=1)
            else:
                edge_index_adv = edge_index_curr.clone()

            y_adv = torch.cat(
                [y_curr, torch.full((bseq,), -1, device=self.device, dtype=y_curr.dtype)], dim=0
            )
            time_step_adv = self._extend_time_step(time_step_curr, batch_edges, inj_ids)
            opt_targets, opt_labels = self._edge_destination_targets_and_labels(
                batch_edges, target_nodes, surrogate_labels
            )

            x_adv = self._optimize_batch_features(
                x_existing=x_curr,
                base_inj=base_inj,
                edge_index_adv=edge_index_adv,
                time_step_adv=time_step_adv,
                target_nodes=opt_targets,
                surrogate_labels=opt_labels,
                feat_min=feat_min,
                feat_max=feat_max,
                eps_feature=eps_feature,
                steps=steps,
                lr=lr,
                smooth_r=smooth_r,
            )

            x_curr = x_adv
            edge_index_curr = edge_index_adv
            y_curr = y_adv
            time_step_curr = time_step_adv

            all_base_inj.append(base_inj.detach())
            injected_node_ids.extend(inj_ids.detach().cpu().tolist())
            injected_edges.extend(batch_edges)
            remaining -= bseq

        x_injected_base = (
            torch.cat(all_base_inj, dim=0)
            if all_base_inj
            else self.x.new_empty((0, self.x.size(1)))
        )

        return TDGIAResult(
            x_adv=x_curr.detach(),
            edge_index_adv=edge_index_curr.detach(),
            y_adv=y_curr.detach(),
            time_step_adv=None if time_step_curr is None else time_step_curr.detach(),
            x_injected_base=x_injected_base.detach(),
            injected_node_ids=injected_node_ids,
            injected_edges=injected_edges,
        )
