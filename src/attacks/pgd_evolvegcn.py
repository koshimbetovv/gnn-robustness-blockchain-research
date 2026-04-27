from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class EvolveGCNPGDAttack:
    """L_inf PGD for EvolveGCN-O, operating one test window at a time.

    EvolveGCN-O consumes `(A_list, Nodes_list, mask_list, node_indices)` where
    `Nodes_list` is a K-step history of feature matrices. In variant -O the
    weight evolution `W_t = GRU(W_{t-1})` does not depend on features, and the
    per-step `node_embs` is overwritten inside the GRCU loop — only the last
    step's feature matrix reaches the classifier. We therefore perturb only
    `hist_ndFeats_list[-1]` (the current-time features), which is both the
    realistic threat model (an attacker cannot rewrite history) and
    mathematically equivalent to replicating the same perturbation across all
    K steps for -O.

    Threat model: the attacker perturbs feature columns
    `[attack_start_col : attack_start_col + attack_dim)` of target rows.
    The IBM Elliptic feature matrix has `time_step` in column 0 as metadata;
    we protect it from perturbation via `attack_start_col=1`. Set
    `attack_start_col=0` to allow perturbing the entire feature vector
    (Elliptic++ actors, which does not carry a metadata column).
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
        """Clean forward on the full K-step history. `hist_ndFeats_list` must be
        the per-step feature list as produced by the dataset loader."""
        return self._forward(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx)

    def _apply_delta(
        self,
        base_x: torch.Tensor,
        target_nodes: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        col_start = self.attack_start_col
        attack_end = col_start + delta.size(1)
        target_slice = base_x[target_nodes]
        if col_start == 0 and attack_end == base_x.size(1):
            row_vals = target_slice + delta
        else:
            pre = target_slice[:, :col_start]
            perturbed = target_slice[:, col_start:attack_end] + delta
            post = target_slice[:, attack_end:]
            row_vals = torch.cat([pre, perturbed, post], dim=1)
        return base_x.index_copy(0, target_nodes, row_vals)

    def _build_perturbed_list(
        self,
        hist_ndFeats_list: List[torch.Tensor],
        target_nodes: torch.Tensor,
        delta: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply `delta` to target rows at the current (last) history step;
        earlier steps are forwarded unchanged."""
        x_last_adv = self._apply_delta(hist_ndFeats_list[-1], target_nodes, delta)
        return list(hist_ndFeats_list[:-1]) + [x_last_adv]

    def _project_clamp(self, base_last, target_nodes, delta, eps):
        """Project last-step features into the clamp box on the perturbable
        column slice and re-clip delta to the L_inf eps ball."""
        if self.clamp is None:
            return delta.detach()
        col_start = self.attack_start_col
        attack_end = col_start + delta.size(1)
        x_tmp = self._apply_delta(base_last, target_nodes, delta)
        x_tmp = torch.clamp(x_tmp, min=self.clamp[0], max=self.clamp[1])
        new_delta = (
            x_tmp[target_nodes, col_start:attack_end]
            - base_last[target_nodes, col_start:attack_end]
        )
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    def attack_window(
        self,
        hist_adj_list: List[torch.Tensor],
        hist_ndFeats_list: List[torch.Tensor],
        node_mask_list: List[torch.Tensor],
        label_idx: torch.Tensor,
        target_nodes: torch.Tensor,
        labels_true: torch.Tensor,
        eps: float = 0.01,
        alpha: float = 0.002,
        steps: int = 10,
        random_start: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Multi-step L_inf PGD on one window.

        Returns `(hist_ndFeats_list_adv, logits_adv_labels)` — the perturbed
        feature history (list of length K, with the same shapes as the input
        list) and post-attack logits evaluated at all labeled nodes of the
        window. Both returns are detached.
        """
        target_nodes = target_nodes.to(self.device).long().view(-1)
        target_nodes = torch.unique(target_nodes)

        base_last = hist_ndFeats_list[-1]

        if target_nodes.numel() == 0:
            with torch.no_grad():
                logits = self.forward_labels(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
            return [x.clone() for x in hist_ndFeats_list], logits

        direction = +1.0
        attack_dim = base_last.size(1) - self.attack_start_col
        if attack_dim <= 0:
            raise ValueError(
                f"attack_start_col={self.attack_start_col} leaves no feature columns to perturb "
                f"(current-step features have {base_last.size(1)} columns)."
            )

        if random_start:
            delta = (2 * torch.rand((target_nodes.numel(), attack_dim),
                                    device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((target_nodes.numel(), attack_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(base_last, target_nodes, delta, eps)

        for _ in range(int(steps)):
            delta.requires_grad_(True)
            hist_adv = self._build_perturbed_list(hist_ndFeats_list, target_nodes, delta)
            # Use `target_nodes` as node_indices so the classifier runs only on the
            # targets; this makes the loss (and its gradient wrt delta) attributable
            # exactly to the attacked rows without leaking gradient through other
            # labeled nodes.
            logits_attack = self._forward(hist_adj_list, hist_adv, node_mask_list, target_nodes)
            loss = F.cross_entropy(logits_attack, labels_true)

            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(base_last, target_nodes, delta, eps)

        hist_out = self._build_perturbed_list(hist_ndFeats_list, target_nodes, delta.detach())
        if self.clamp is not None:
            hist_out = [torch.clamp(x, min=self.clamp[0], max=self.clamp[1]) for x in hist_out]

        with torch.no_grad():
            logits_adv_labels = self.forward_labels(hist_adj_list, hist_out, node_mask_list, label_idx).detach()

        return [x.detach() for x in hist_out], logits_adv_labels
