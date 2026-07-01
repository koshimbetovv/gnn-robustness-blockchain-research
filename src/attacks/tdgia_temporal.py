from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _dedupe_targets(target_nodes: torch.Tensor) -> torch.Tensor:
    target_nodes = target_nodes.long().view(-1)
    if target_nodes.numel() == 0:
        return target_nodes

    keep: list[int] = []
    seen: set[int] = set()
    for pos, node_id in enumerate(target_nodes.detach().cpu().tolist()):
        if int(node_id) in seen:
            continue
        seen.add(int(node_id))
        keep.append(pos)
    return target_nodes[torch.tensor(keep, dtype=torch.long, device=target_nodes.device)]


def _dedupe_targets_and_labels(
    target_nodes: torch.Tensor,
    attack_labels: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not torch.is_tensor(target_nodes):
        target_nodes = torch.tensor(target_nodes, dtype=torch.long)
    target_nodes = target_nodes.to(device).long().view(-1)

    if attack_labels is not None:
        if not torch.is_tensor(attack_labels):
            attack_labels = torch.tensor(attack_labels, dtype=torch.long)
        attack_labels = attack_labels.to(device).long().view(-1)
        if attack_labels.numel() != target_nodes.numel():
            raise ValueError(
                "attack_labels must align one-to-one with target_nodes "
                f"({attack_labels.numel()} labels for {target_nodes.numel()} targets)."
            )

    if target_nodes.numel() == 0:
        return target_nodes, attack_labels

    keep: list[int] = []
    seen: set[int] = set()
    for pos, node_id in enumerate(target_nodes.detach().cpu().tolist()):
        if int(node_id) in seen:
            continue
        seen.add(int(node_id))
        keep.append(pos)

    keep_idx = torch.tensor(keep, dtype=torch.long, device=device)
    target_nodes = target_nodes[keep_idx]
    if attack_labels is not None:
        attack_labels = attack_labels[keep_idx]
    return target_nodes, attack_labels


def _validate_attack_labels(attack_labels: torch.Tensor, num_classes: int) -> None:
    if attack_labels.numel() == 0:
        return
    if int(attack_labels.min().item()) < 0 or int(attack_labels.max().item()) >= int(num_classes):
        raise ValueError(
            "attack_labels contain class ids outside the model output range "
            f"[0, {int(num_classes) - 1}]."
        )


def _smoothmap(latent: torch.Tensor, feat_min: torch.Tensor, feat_max: torch.Tensor) -> torch.Tensor:
    mid = 0.5 * (feat_max + feat_min)
    amp = 0.5 * (feat_max - feat_min)
    return mid + amp * torch.sin(latent)


def _inv_smoothmap(x: torch.Tensor, feat_min: torch.Tensor, feat_max: torch.Tensor) -> torch.Tensor:
    mid = 0.5 * (feat_max + feat_min)
    amp = (0.5 * (feat_max - feat_min)).clamp_min(1e-6)
    scaled = ((x - mid) / amp).clamp(min=-0.999999, max=0.999999)
    return torch.asin(scaled)


def _feature_range(
    x: torch.Tensor,
    col_start: int,
    attack_dim: int,
    clamp: Optional[Tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if clamp is not None:
        feat_min = torch.full((attack_dim,), float(clamp[0]), device=x.device, dtype=x.dtype)
        feat_max = torch.full((attack_dim,), float(clamp[1]), device=x.device, dtype=x.dtype)
        return feat_min, feat_max

    x_attack = x[:, col_start : col_start + attack_dim]
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


def _degree_from_edge_index(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    row = edge_index[0].long()
    col = edge_index[1].long()
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))
    return deg.clamp_min_(1.0)


def _degree_from_sparse_adj(num_nodes: int, adj: torch.Tensor) -> torch.Tensor:
    adj = adj.coalesce()
    idx = adj.indices()
    off_diag = idx[0] != idx[1]
    idx = idx[:, off_diag]
    if idx.numel() == 0:
        return torch.ones(num_nodes, device=adj.device, dtype=torch.float32)
    deg = torch.zeros(num_nodes, device=adj.device, dtype=torch.float32)
    deg.scatter_add_(0, idx[0].long(), torch.ones(idx.size(1), device=adj.device))
    deg.scatter_add_(0, idx[1].long(), torch.ones(idx.size(1), device=adj.device))
    return deg.clamp_min_(1.0)


def _tdgia_order(
    log_prob_or_logits: torch.Tensor,
    target_nodes: torch.Tensor,
    surrogate_labels: torch.Tensor,
    degree: torch.Tensor,
    degree_limit: int,
    alpha_mu: float,
    k1: float,
    k2: float,
    *,
    input_is_log_prob: bool = False,
) -> torch.Tensor:
    if input_is_log_prob:
        log_pv = log_prob_or_logits[target_nodes].gather(1, surrogate_labels.view(-1, 1)).view(-1)
        pv = log_pv.exp().clamp_min(1e-12)
    else:
        probs = F.softmax(log_prob_or_logits[target_nodes], dim=1)
        pv = probs.gather(1, surrogate_labels.view(-1, 1)).view(-1).clamp_min(1e-12)

    deg = degree[target_nodes].clamp_min(1.0)
    d = float(max(int(degree_limit), 1))
    lam = float(k1) / torch.sqrt(deg * d) + float(k2) / deg
    mu = (float(alpha_mu) * pv + (1.0 - float(alpha_mu))) * lam
    return target_nodes[torch.argsort(mu, descending=True)]


def _build_injection_edges(
    injected_ids: torch.Tensor,
    target_nodes_sorted: torch.Tensor,
    degree_limit: int,
) -> list[tuple[int, int]]:
    if injected_ids.numel() == 0 or target_nodes_sorted.numel() == 0 or degree_limit <= 0:
        return []

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


@dataclass
class CoSemiTDGIAResult:
    x_adv: torch.Tensor
    edge_index_adv: torch.Tensor
    logits_adv: torch.Tensor
    logits_clean: torch.Tensor
    x_injected_base: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]


class CoSemiGNNTDGIAAttack:
    """TDGIA-style non-persistent node injection for one CoSemiGNN slice."""

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
        init: str,
        reference_nodes: Optional[torch.Tensor],
        sigma_scale: float,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
    ) -> torch.Tensor:
        semi_dim = int(features.size(1)) - self.raw_dim
        ref = (
            features
            if reference_nodes is None or reference_nodes.numel() == 0
            else features[reference_nodes.long()]
        )
        ref_raw = ref[:, : self.raw_dim]
        if init == "zeros":
            raw = torch.zeros((n_inject, self.raw_dim), device=self.device, dtype=features.dtype)
        elif init == "mean":
            raw = ref_raw.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        elif init == "randn":
            mu = ref_raw.mean(dim=0, keepdim=True)
            std = ref_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            raw = (mu + float(sigma_scale) * torch.randn((n_inject, self.raw_dim), device=self.device) * std).detach()
        else:
            raise ValueError(f"Unknown init={init!r}")
        raw = raw.clamp(min=feat_min, max=feat_max)
        if semi_dim > 0:
            ref_semi = ref[:, self.raw_dim :]
            sample_idx = torch.randint(
                int(ref_semi.size(0)),
                (int(n_inject),),
                device=features.device,
            )
            semi = ref_semi[sample_idx].detach().to(features.dtype)
        else:
            semi = features.new_empty((int(n_inject), 0))
        return torch.cat([raw.to(features.dtype), semi], dim=1)

    @staticmethod
    def _augment_edge_index(adj: torch.Tensor, injected_edges: list[tuple[int, int]]) -> torch.Tensor:
        if not injected_edges:
            return adj.clone()
        ei = torch.tensor(injected_edges, dtype=torch.long, device=adj.device).t().contiguous()
        ei_rev = torch.stack([ei[1], ei[0]], dim=0)
        return torch.cat([adj, ei, ei_rev], dim=1)

    def _apply_raw(self, features: torch.Tensor, raw_inj: torch.Tensor, base_inj: torch.Tensor) -> torch.Tensor:
        if self.raw_dim == features.size(1):
            inj = raw_inj
        else:
            inj = torch.cat([raw_inj, base_inj[:, self.raw_dim :]], dim=1)
        return torch.cat([features, inj], dim=0)

    def _optimize_batch(
        self,
        features: torch.Tensor,
        base_inj: torch.Tensor,
        edge_index_adv: torch.Tensor,
        target_nodes: torch.Tensor,
        surrogate_labels: torch.Tensor,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
        eps_feature: Optional[float],
        steps: int,
        lr: float,
        smooth_r: float,
        ca_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if eps_feature is None:
            opt_min, opt_max = feat_min, feat_max
        else:
            eps = float(eps_feature)
            base_raw = base_inj[:, : self.raw_dim]
            opt_min = torch.maximum(base_raw - eps, feat_min)
            opt_max = torch.minimum(base_raw + eps, feat_max)

        latent = _inv_smoothmap(base_inj[:, : self.raw_dim], opt_min, opt_max).detach()
        latent.requires_grad_(True)
        opt = torch.optim.Adam([latent], lr=float(lr))

        for _ in range(int(steps)):
            opt.zero_grad()
            raw = _smoothmap(latent, opt_min, opt_max)
            x_adv = self._apply_raw(features, raw, base_inj)
            logits = self.forward_logits(x_adv, edge_index_adv, ca_weights)
            probs = F.softmax(logits[target_nodes], dim=1)
            pv = probs.gather(1, surrogate_labels.view(-1, 1)).view(-1).clamp_min(1e-12)
            loss = torch.relu(float(smooth_r) + torch.log(pv)).pow(2).mean()
            latent.grad = torch.autograd.grad(loss, latent, retain_graph=False, create_graph=False)[0]
            opt.step()

        raw = _smoothmap(latent.detach(), opt_min, opt_max)
        return self._apply_raw(features, raw, base_inj).detach()

    def attack_slice(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        target_nodes: torch.Tensor,
        n_inject: int = 1,
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
        ca_weights: Optional[torch.Tensor] = None,
        attack_labels: Optional[torch.Tensor] = None,
    ) -> CoSemiTDGIAResult:
        target_nodes, attack_labels = _dedupe_targets_and_labels(target_nodes, attack_labels, self.device)
        n_existing = int(features.size(0))
        with torch.no_grad():
            logits_clean = self.forward_logits(features, adj, ca_weights).detach()

        if target_nodes.numel() == 0 or int(n_inject) <= 0:
            return CoSemiTDGIAResult(features.clone(), adj.clone(), logits_clean.clone(), logits_clean, features.new_empty((0, features.size(1))), [], [])

        feat_min, feat_max = _feature_range(features, 0, self.raw_dim, self.clamp)
        surrogate_labels = logits_clean[target_nodes].argmax(dim=1) if attack_labels is None else attack_labels
        _validate_attack_labels(surrogate_labels, int(logits_clean.size(1)))
        x_curr = features.clone()
        edge_curr = adj.clone()
        all_base: list[torch.Tensor] = []
        all_ids: list[int] = []
        all_edges: list[tuple[int, int]] = []

        remaining = int(n_inject)
        degree_limit = max(1, int(degree_limit))
        batch_size = max(1, int(batch_size))
        if int(steps) < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        if eps_feature is not None and float(eps_feature) < 0.0:
            raise ValueError(f"eps_feature must be non-negative or None, got {eps_feature}")
        while remaining > 0:
            bsz = min(batch_size, remaining)
            with torch.no_grad():
                logits_curr = self.forward_logits(x_curr, edge_curr, ca_weights).detach()
            degree = _degree_from_edge_index(x_curr.size(0), edge_curr)
            sorted_targets = _tdgia_order(
                logits_curr, target_nodes, surrogate_labels, degree, degree_limit,
                alpha_mu, k1, k2,
            )
            inj_ids = torch.arange(x_curr.size(0), x_curr.size(0) + bsz, device=self.device)
            base_inj = self._init_injected_features(
                features, bsz, init, reference_nodes, sigma_scale, feat_min, feat_max
            )
            edges = _build_injection_edges(inj_ids, sorted_targets, degree_limit)
            edge_adv = self._augment_edge_index(edge_curr, edges)
            x_curr = self._optimize_batch(
                x_curr, base_inj, edge_adv, target_nodes, surrogate_labels,
                feat_min, feat_max, eps_feature, steps, lr, smooth_r, ca_weights,
            )
            edge_curr = edge_adv
            all_base.append(base_inj.detach())
            all_ids.extend(inj_ids.detach().cpu().tolist())
            all_edges.extend(edges)
            remaining -= bsz

        with torch.no_grad():
            logits_adv = self.forward_logits(x_curr, edge_curr, ca_weights).detach()[:n_existing]

        return CoSemiTDGIAResult(
            x_adv=x_curr.detach(),
            edge_index_adv=edge_curr.detach(),
            logits_adv=logits_adv,
            logits_clean=logits_clean,
            x_injected_base=torch.cat(all_base, dim=0).detach() if all_base else features.new_empty((0, features.size(1))),
            injected_node_ids=all_ids,
            injected_edges=all_edges,
        )


@dataclass
class EvolveGCNTDGIAResult:
    hist_ndFeats_list: List[torch.Tensor]
    hist_adj_list: List[torch.Tensor]
    node_mask_list: List[torch.Tensor]
    label_idx: torch.Tensor
    logits_adv: torch.Tensor
    x_injected_base: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]


class EvolveGCNTDGIAAttack:
    """TDGIA-style non-persistent node injection for one EvolveGCN-O window."""

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

    def _init_injected_features(
        self,
        base_x: torch.Tensor,
        n_inject: int,
        init: str,
        reference_nodes: Optional[torch.Tensor],
        sigma_scale: float,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
    ) -> torch.Tensor:
        col = self.attack_start_col
        attack_dim = int(base_x.size(1) - col)
        ref = base_x if reference_nodes is None or reference_nodes.numel() == 0 else base_x[reference_nodes]
        if init == "zeros":
            base = torch.zeros((n_inject, base_x.size(1)), device=self.device, dtype=base_x.dtype)
        elif init == "mean":
            base = ref.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        elif init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            base = (mu + float(sigma_scale) * torch.randn((n_inject, base_x.size(1)), device=self.device) * std).detach()
        else:
            raise ValueError(f"Unknown init={init!r}")
        base[:, col : col + attack_dim] = base[:, col : col + attack_dim].clamp(min=feat_min, max=feat_max)
        return base

    @staticmethod
    def _augment_adj(
        A_norm: torch.Tensor,
        n_existing: int,
        n_inject: int,
        injected_edges: list[tuple[int, int]],
    ) -> torch.Tensor:
        n_total = n_existing + n_inject
        A_coal = A_norm.coalesce()
        idx = A_coal.indices()
        off_diag = idx[0] != idx[1]
        base_idx = idx[:, off_diag]

        if injected_edges:
            ei = torch.tensor(injected_edges, dtype=torch.long, device=A_norm.device).t().contiguous()
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
        degree = torch.sparse.sum(sp, dim=1).to_dense()
        di = degree[norm_idx[0]]
        dj = degree[norm_idx[1]]
        norm_vals = sp.values() * ((di * dj) ** -0.5)
        return torch.sparse_coo_tensor(norm_idx, norm_vals, size=(n_total, n_total), dtype=torch.float32).coalesce()

    @staticmethod
    def _expanded_mask(node_mask: torch.Tensor, n_inject: int) -> torch.Tensor:
        pad = torch.zeros((n_inject, node_mask.size(1)), dtype=node_mask.dtype, device=node_mask.device)
        return torch.cat([node_mask, pad], dim=0)

    def _expanded_features(self, base_x: torch.Tensor, base_inj: torch.Tensor, raw_inj: torch.Tensor) -> torch.Tensor:
        col = self.attack_start_col
        attack_end = col + raw_inj.size(1)
        if col == 0 and attack_end == base_x.size(1):
            inj = raw_inj
        else:
            inj = torch.cat([base_inj[:, :col], raw_inj, base_inj[:, attack_end:]], dim=1)
        return torch.cat([base_x, inj], dim=0)

    def _optimize_batch(
        self,
        hist_adj_aug: List[torch.Tensor],
        hist_feats_prefix: List[torch.Tensor],
        hist_mask_aug: List[torch.Tensor],
        base_last: torch.Tensor,
        base_inj: torch.Tensor,
        target_nodes: torch.Tensor,
        surrogate_labels: torch.Tensor,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
        eps_feature: Optional[float],
        steps: int,
        lr: float,
        smooth_r: float,
    ) -> torch.Tensor:
        col = self.attack_start_col
        attack_dim = int(base_last.size(1) - col)
        if eps_feature is None:
            opt_min, opt_max = feat_min, feat_max
        else:
            eps = float(eps_feature)
            base_raw = base_inj[:, col : col + attack_dim]
            opt_min = torch.maximum(base_raw - eps, feat_min)
            opt_max = torch.minimum(base_raw + eps, feat_max)

        latent = _inv_smoothmap(base_inj[:, col : col + attack_dim], opt_min, opt_max).detach()
        latent.requires_grad_(True)
        opt = torch.optim.Adam([latent], lr=float(lr))

        for _ in range(int(steps)):
            opt.zero_grad()
            raw = _smoothmap(latent, opt_min, opt_max)
            x_last_aug = self._expanded_features(base_last, base_inj, raw)
            hist_feats_aug = hist_feats_prefix + [x_last_aug]
            logits = self._forward(hist_adj_aug, hist_feats_aug, hist_mask_aug, target_nodes)
            probs = F.softmax(logits, dim=1)
            pv = probs.gather(1, surrogate_labels.view(-1, 1)).view(-1).clamp_min(1e-12)
            loss = torch.relu(float(smooth_r) + torch.log(pv)).pow(2).mean()
            latent.grad = torch.autograd.grad(loss, latent, retain_graph=False, create_graph=False)[0]
            opt.step()

        raw = _smoothmap(latent.detach(), opt_min, opt_max)
        return self._expanded_features(base_last, base_inj, raw).detach()

    def attack_window(
        self,
        hist_adj_list: List[torch.Tensor],
        hist_ndFeats_list: List[torch.Tensor],
        node_mask_list: List[torch.Tensor],
        label_idx: torch.Tensor,
        target_nodes: torch.Tensor,
        n_inject: int = 1,
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
        attack_labels: Optional[torch.Tensor] = None,
    ) -> EvolveGCNTDGIAResult:
        target_nodes, attack_labels = _dedupe_targets_and_labels(target_nodes, attack_labels, self.device)
        base_last_clean = hist_ndFeats_list[-1]
        n_existing = int(base_last_clean.size(0))

        if target_nodes.numel() == 0 or int(n_inject) <= 0:
            with torch.no_grad():
                logits = self.forward_labels(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
            return EvolveGCNTDGIAResult([x.clone() for x in hist_ndFeats_list], list(hist_adj_list), [m.clone() for m in node_mask_list], label_idx, logits, base_last_clean.new_empty((0, base_last_clean.size(1))), [], [])

        attack_dim = int(base_last_clean.size(1) - self.attack_start_col)
        if attack_dim <= 0:
            raise ValueError(
                f"attack_start_col={self.attack_start_col} leaves no feature columns to perturb "
                f"(features have {base_last_clean.size(1)} columns)."
            )

        with torch.no_grad():
            clean_target_logits = self._forward(hist_adj_list, hist_ndFeats_list, node_mask_list, target_nodes).detach()
        surrogate_labels = clean_target_logits.argmax(dim=1) if attack_labels is None else attack_labels
        _validate_attack_labels(surrogate_labels, int(clean_target_logits.size(1)))
        feat_min, feat_max = _feature_range(base_last_clean, self.attack_start_col, attack_dim, self.clamp)

        hist_adj_curr = list(hist_adj_list)
        hist_mask_curr = list(node_mask_list)
        base_last_curr = base_last_clean.clone()
        all_base: list[torch.Tensor] = []
        all_ids: list[int] = []
        all_edges: list[tuple[int, int]] = []

        remaining = int(n_inject)
        degree_limit = max(1, int(degree_limit))
        batch_size = max(1, int(batch_size))
        if int(steps) < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        if eps_feature is not None and float(eps_feature) < 0.0:
            raise ValueError(f"eps_feature must be non-negative or None, got {eps_feature}")
        while remaining > 0:
            bsz = min(batch_size, remaining)
            with torch.no_grad():
                curr_target_logits = self._forward(
                    hist_adj_curr,
                    list(hist_ndFeats_list[:-1]) + [base_last_curr],
                    hist_mask_curr,
                    target_nodes,
                ).detach()
            degree = _degree_from_sparse_adj(base_last_curr.size(0), hist_adj_curr[-1])
            probs = F.softmax(curr_target_logits, dim=1)
            pv = probs.gather(1, surrogate_labels.view(-1, 1)).view(-1).clamp_min(1e-12)
            deg = degree[target_nodes].clamp_min(1.0)
            d = float(max(int(degree_limit), 1))
            lam = float(k1) / torch.sqrt(deg * d) + float(k2) / deg
            mu = (float(alpha_mu) * pv + (1.0 - float(alpha_mu))) * lam
            sorted_targets = target_nodes[torch.argsort(mu, descending=True)]

            inj_ids = torch.arange(base_last_curr.size(0), base_last_curr.size(0) + bsz, device=self.device)
            base_inj = self._init_injected_features(
                base_last_clean, bsz, init, reference_nodes, sigma_scale, feat_min, feat_max
            )
            edges = _build_injection_edges(inj_ids, sorted_targets, degree_limit)
            adj_last_aug = self._augment_adj(hist_adj_curr[-1], base_last_curr.size(0), bsz, edges)
            mask_last_aug = self._expanded_mask(hist_mask_curr[-1], bsz)
            hist_adj_aug = list(hist_adj_curr[:-1]) + [adj_last_aug]
            hist_mask_aug = list(hist_mask_curr[:-1]) + [mask_last_aug]
            base_last_curr = self._optimize_batch(
                hist_adj_aug, list(hist_ndFeats_list[:-1]), hist_mask_aug, base_last_curr,
                base_inj, target_nodes, surrogate_labels, feat_min, feat_max,
                eps_feature, steps, lr, smooth_r,
            )
            hist_adj_curr = hist_adj_aug
            hist_mask_curr = hist_mask_aug
            all_base.append(base_inj.detach())
            all_ids.extend(inj_ids.detach().cpu().tolist())
            all_edges.extend(edges)
            remaining -= bsz

        hist_feats_out = list(hist_ndFeats_list[:-1]) + [base_last_curr.detach()]
        with torch.no_grad():
            logits_adv = self.forward_labels(hist_adj_curr, hist_feats_out, hist_mask_curr, label_idx).detach()

        return EvolveGCNTDGIAResult(
            hist_ndFeats_list=hist_feats_out,
            hist_adj_list=hist_adj_curr,
            node_mask_list=hist_mask_curr,
            label_idx=label_idx,
            logits_adv=logits_adv,
            x_injected_base=torch.cat(all_base, dim=0).detach() if all_base else base_last_clean.new_empty((0, base_last_clean.size(1))),
            injected_node_ids=all_ids,
            injected_edges=all_edges,
        )


@dataclass
class RecGNNTDGIAResult:
    x_adv: torch.Tensor
    edge_index_adv: torch.Tensor
    log_probs_adv: torch.Tensor
    log_probs_clean: torch.Tensor
    x_injected_base: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]


class RecGNNTDGIAAttack:
    """TDGIA-style non-persistent node injection for one RecGNN timestep."""

    def __init__(
        self,
        model,
        device,
        attack_dim: int = 93,
        clamp: Optional[Tuple[float, float]] = None,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.attack_dim = int(attack_dim)
        self.clamp = clamp

    def _forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.model(x, edge_index)

    def _save_state(self):
        ml = self.model.m_lstm
        ev = ml.cell.evolve_linear

        def _c(t):
            return t.detach().clone() if t is not None else None

        return {
            "h": _c(ml._h_state),
            "c": _c(ml._c_state),
            "ev_h": _c(ev._row_h),
            "ev_c": _c(ev._row_c),
            "ev_w": _c(ev._current_weight),
        }

    def _restore_state(self, snap) -> None:
        ml = self.model.m_lstm
        ev = ml.cell.evolve_linear

        def _c(t):
            return t.detach().clone() if t is not None else None

        ml._h_state = _c(snap["h"])
        ml._c_state = _c(snap["c"])
        ev._row_h = _c(snap["ev_h"])
        ev._row_c = _c(snap["ev_c"])
        ev._current_weight = _c(snap["ev_w"])

    def _zero_inject_rows(self, n_existing: int, n_inject: int) -> None:
        ml = self.model.m_lstm
        if ml._h_state is not None:
            ml._h_state[n_existing : n_existing + n_inject].zero_()
        if ml._c_state is not None:
            ml._c_state[n_existing : n_existing + n_inject].zero_()

    @torch.no_grad()
    def prime(self, graphs) -> None:
        self.model.reset_sequence_state(self.device)
        for g in graphs:
            g = g.to(self.device)
            _ = self._forward(g.x.float(), g.edge_index.long())
            self.model.detach_sequence_state()

    def _init_injected_features(
        self,
        x: torch.Tensor,
        n_inject: int,
        init: str,
        reference_nodes: Optional[torch.Tensor],
        sigma_scale: float,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
    ) -> torch.Tensor:
        ref = x if reference_nodes is None or reference_nodes.numel() == 0 else x[reference_nodes]
        if init == "zeros":
            base = torch.zeros((n_inject, x.size(1)), device=self.device, dtype=x.dtype)
        elif init == "mean":
            base = ref.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        elif init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            base = (mu + float(sigma_scale) * torch.randn((n_inject, x.size(1)), device=self.device) * std).detach()
        else:
            raise ValueError(f"Unknown init={init!r}")
        base[:, : self.attack_dim] = base[:, : self.attack_dim].clamp(min=feat_min, max=feat_max)
        return base

    def _apply_raw(self, x: torch.Tensor, base_inj: torch.Tensor, raw_inj: torch.Tensor) -> torch.Tensor:
        if self.attack_dim == x.size(1):
            inj = raw_inj
        else:
            inj = torch.cat([raw_inj, base_inj[:, self.attack_dim :]], dim=1)
        return torch.cat([x, inj], dim=0)

    def _optimize_batch(
        self,
        x_base: torch.Tensor,
        base_inj: torch.Tensor,
        edge_index_adv: torch.Tensor,
        target_nodes: torch.Tensor,
        surrogate_labels: torch.Tensor,
        snap_pre,
        n_existing_clean: int,
        n_total_injected: int,
        feat_min: torch.Tensor,
        feat_max: torch.Tensor,
        eps_feature: Optional[float],
        steps: int,
        lr: float,
        smooth_r: float,
    ) -> torch.Tensor:
        if eps_feature is None:
            opt_min, opt_max = feat_min, feat_max
        else:
            eps = float(eps_feature)
            base_raw = base_inj[:, : self.attack_dim]
            opt_min = torch.maximum(base_raw - eps, feat_min)
            opt_max = torch.minimum(base_raw + eps, feat_max)

        latent = _inv_smoothmap(base_inj[:, : self.attack_dim], opt_min, opt_max).detach()
        latent.requires_grad_(True)
        opt = torch.optim.Adam([latent], lr=float(lr))

        for _ in range(int(steps)):
            opt.zero_grad()
            self._restore_state(snap_pre)
            self._zero_inject_rows(n_existing_clean, n_total_injected)
            raw = _smoothmap(latent, opt_min, opt_max)
            x_adv = self._apply_raw(x_base, base_inj, raw)
            log_probs = self._forward(x_adv, edge_index_adv)
            log_pv = log_probs[target_nodes].gather(1, surrogate_labels.view(-1, 1)).view(-1)
            loss = torch.relu(float(smooth_r) + log_pv).pow(2).mean()
            latent.grad = torch.autograd.grad(loss, latent, retain_graph=False, create_graph=False)[0]
            opt.step()

        raw = _smoothmap(latent.detach(), opt_min, opt_max)
        return self._apply_raw(x_base, base_inj, raw).detach()

    def attack_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_nodes: torch.Tensor,
        n_inject: int = 1,
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
        attack_labels: Optional[torch.Tensor] = None,
    ) -> RecGNNTDGIAResult:
        target_nodes, attack_labels = _dedupe_targets_and_labels(target_nodes, attack_labels, self.device)
        n_existing = int(x.size(0))
        snap_pre = self._save_state()

        with torch.no_grad():
            log_probs_clean = self._forward(x, edge_index).detach()
            self.model.detach_sequence_state()
        snap_post = self._save_state()

        if target_nodes.numel() == 0 or int(n_inject) <= 0:
            return RecGNNTDGIAResult(x.clone(), edge_index.clone(), log_probs_clean.clone(), log_probs_clean, x.new_empty((0, x.size(1))), [], [])

        n_inject = int(n_inject)
        state_rows = int(self.model.m_lstm.state_rows)
        if n_existing + n_inject > state_rows:
            raise ValueError(
                f"n_existing + n_inject = {n_existing + n_inject} exceeds m-LSTM "
                f"state_rows = {state_rows}. Reduce n_inject or use a model with larger state_rows."
            )

        surrogate_labels = log_probs_clean[target_nodes].argmax(dim=1) if attack_labels is None else attack_labels
        _validate_attack_labels(surrogate_labels, int(log_probs_clean.size(1)))
        feat_min, feat_max = _feature_range(x, 0, self.attack_dim, self.clamp)
        x_curr = x.clone()
        edge_curr = edge_index.clone()
        all_base: list[torch.Tensor] = []
        all_ids: list[int] = []
        all_edges: list[tuple[int, int]] = []

        remaining = n_inject
        degree_limit = max(1, int(degree_limit))
        batch_size = max(1, int(batch_size))
        if int(steps) < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        if eps_feature is not None and float(eps_feature) < 0.0:
            raise ValueError(f"eps_feature must be non-negative or None, got {eps_feature}")
        while remaining > 0:
            bsz = min(batch_size, remaining)
            self._restore_state(snap_pre)
            self._zero_inject_rows(n_existing, len(all_ids))
            with torch.no_grad():
                log_probs_curr = self._forward(x_curr, edge_curr).detach()
                self.model.detach_sequence_state()
            degree = _degree_from_edge_index(x_curr.size(0), edge_curr)
            sorted_targets = _tdgia_order(
                log_probs_curr, target_nodes, surrogate_labels, degree, degree_limit,
                alpha_mu, k1, k2, input_is_log_prob=True,
            )
            inj_ids = torch.arange(x_curr.size(0), x_curr.size(0) + bsz, device=self.device)
            base_inj = self._init_injected_features(x, bsz, init, reference_nodes, sigma_scale, feat_min, feat_max)
            edges = _build_injection_edges(inj_ids, sorted_targets, degree_limit)
            if edges:
                add_ei = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
                edge_adv = torch.cat([edge_curr, add_ei], dim=1)
            else:
                edge_adv = edge_curr.clone()
            x_curr = self._optimize_batch(
                x_curr, base_inj, edge_adv, target_nodes, surrogate_labels,
                snap_pre, n_existing, len(all_ids) + bsz,
                feat_min, feat_max, eps_feature, steps, lr, smooth_r,
            )
            edge_curr = edge_adv
            all_base.append(base_inj.detach())
            all_ids.extend(inj_ids.detach().cpu().tolist())
            all_edges.extend(edges)
            remaining -= bsz

        self._restore_state(snap_pre)
        self._zero_inject_rows(n_existing, len(all_ids))
        with torch.no_grad():
            log_probs_adv_full = self._forward(x_curr, edge_curr).detach()
            self.model.detach_sequence_state()
        self._restore_state(snap_post)

        return RecGNNTDGIAResult(
            x_adv=x_curr.detach(),
            edge_index_adv=edge_curr.detach(),
            log_probs_adv=log_probs_adv_full[:n_existing],
            log_probs_clean=log_probs_clean,
            x_injected_base=torch.cat(all_base, dim=0).detach() if all_base else x.new_empty((0, x.size(1))),
            injected_node_ids=all_ids,
            injected_edges=all_edges,
        )
