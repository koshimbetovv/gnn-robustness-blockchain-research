"""Adapted NETTACK for temporal/sequence GNN victims (RecGNN, EvolveGCN-O, CoSemiGNN).

Threat model differences vs. the static driver:
- Per-timestep: each test slice is its own graph. NETTACK runs independently on
  each slice (rebuilds A_hat^2, candidate set, chi^2 stats per slice).
- Surrogate is trained ONCE on the union of train timesteps (each train slice
  contributes its own A_hat^2 X but they share a single W in R^{D x K}). The
  trained W is reused across all test slices.
- The attack itself only queries the (stateless, linearized GCN) surrogate, so
  the victim's sequence state is never touched during NETTACK. State save/restore
  around the victim's clean and adversarial forwards is the driver's job.
- Edge ADDITIONS only (no deletions), matching the static driver. For temporal
  victims the addition is interpreted as a new edge appearing at the current
  timestep; earlier history is left unchanged.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attacks.nettack_adapted import (
    AdaptedNettackAttack, _build_A_hat, _make_undirected_no_self_loops,
)


def train_surrogate_on_train_slices(
    train_slices: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    num_features: int,
    num_classes: int,
    device,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    verbose: bool = True,
) -> torch.Tensor:
    """Train a single linearized-GCN surrogate W in R^{D x K} on the union of train slices.

    Each slice is `(x, edge_index, y, train_mask)`:
      - `x`           : (N_t, D) feature matrix at slice t
      - `edge_index`  : (2, |E_t|) directed/undirected pairs at slice t
      - `y`           : (N_t,) labels at slice t (-1 for unlabeled)
      - `train_mask`  : (N_t,) bool mask of nodes used to fit W

    For each epoch we sum cross-entropy across all train slices' masked nodes,
    so the surrogate jointly fits the temporally-varying graph distribution
    rather than memorizing a single slice. `A_hat^2 X` is precomputed per slice.

    Returns: detached W tensor on `device`, shape (D, K).
    """
    cached: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for x, edge_index, y, train_mask in train_slices:
        x = x.to(device).float()
        edge_index = edge_index.to(device).long()
        y = y.to(device).long()
        train_mask = train_mask.to(device).bool() & (y != -1)
        if int(train_mask.sum().item()) == 0:
            continue
        N = int(x.size(0))
        edges_no_sl = _make_undirected_no_self_loops(edge_index, N)
        A_hat, _ = _build_A_hat(edges_no_sl, N, device)
        AAX = torch.sparse.mm(A_hat, torch.sparse.mm(A_hat, x))
        cached.append((AAX, y, train_mask))

    if not cached:
        raise ValueError("train_surrogate_on_train_slices: no slices contributed labeled nodes.")

    W = torch.empty(num_features, num_classes, device=device)
    nn.init.xavier_uniform_(W)
    W.requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr, weight_decay=weight_decay)

    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        n_total = 0
        for AAX, y, train_mask in cached:
            logits = AAX @ W
            n_t = int(train_mask.sum().item())
            if n_t == 0:
                continue
            loss = loss + F.cross_entropy(
                logits[train_mask], y[train_mask], reduction="sum"
            )
            n_total += n_t
        loss = loss / max(n_total, 1)
        loss.backward()
        opt.step()
        if verbose and ((ep + 1) % max(1, epochs // 5) == 0 or ep == 0):
            print(f"  [surrogate] epoch {ep + 1}/{epochs}  loss={float(loss.item()):.4f}  n={n_total}")

    return W.detach()


class _DummyModel(nn.Module):
    """Stand-in `model` for BaseAttack. AdaptedNettackAttack only queries the
    surrogate W during its greedy loop, so the victim model is never used inside
    the attack class -- we just need an object with `.eval()`.
    """

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise RuntimeError("DummyModel should not be invoked.")


class AdaptedNettackTemporalAttack:
    """Per-slice Adapted-NETTACK runner that shares one pre-trained surrogate W
    across all temporal slices.

    The class is stateless across slices: each `attack_slice(...)` call builds a
    fresh `AdaptedNettackAttack` over the slice's `(x, edge_index, y)` using the
    cached `W`, runs the greedy attack, and returns the perturbed `(x, edge_index)`.
    The caller (driver) is responsible for routing those into the victim model
    and managing any sequence state.
    """

    def __init__(
        self,
        W: torch.Tensor,
        device,
        attack_dim: Optional[int] = None,
        clamp: tuple[float, float] | None = None,
        d_min: int = 2,
        chi2_tau: float = 0.004,
        enforce_degree_constraint: bool = True,
        verbose: bool = False,
        progress_every: int = 100,
    ):
        self.W = W.to(device).detach()
        self.device = device
        self.attack_dim = None if attack_dim is None else int(attack_dim)
        self.clamp = clamp
        self.d_min = int(d_min)
        self.chi2_tau = float(chi2_tau)
        self.enforce_degree_constraint = bool(enforce_degree_constraint)
        self.verbose = bool(verbose)
        self.progress_every = int(progress_every)
        self._dummy_model = _DummyModel().to(device).eval()

    def attack_slice(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        target_idx: torch.Tensor,
        n_struct: int = 2,
        eps_feat: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Run Adapted-NETTACK on a single temporal slice.

        Returns `(x_adv, edge_index_adv, info)` where `info` reports
        `n_unique_edges_added`, `n_targets_with_edge_added`, and
        `n_directed_edges_added` for downstream metrics.
        """
        x = x.to(self.device).float()
        edge_index = edge_index.to(self.device).long()
        y = y.to(self.device).long()
        target_idx = target_idx.to(self.device).long().view(-1)

        data = SimpleNamespace(x=x, y=y, edge_index=edge_index)
        atk = AdaptedNettackAttack(
            self._dummy_model,
            data,
            self.device,
            clamp=self.clamp,
            attack_dim=self.attack_dim,
            rebuild_fn=None,
            d_min=self.d_min,
            chi2_tau=self.chi2_tau,
            enforce_degree_constraint=self.enforce_degree_constraint,
            pretrained_W=self.W,
            verbose=self.verbose,
            progress_every=self.progress_every,
        )

        x_adv, edge_index_adv = atk.attack(
            target_idx, n_struct=int(n_struct), eps_feat=float(eps_feat),
        )

        added = list(atk._added_edges)
        info = {
            "n_unique_edges_added": int(len(added)),
            "n_directed_edges_added": int(edge_index_adv.size(1) - edge_index.size(1)),
            "n_targets_with_edge_added": int(len({v0 for (v0, _) in added})),
            "perturbable_dim": int(atk.attack_dim),
        }
        return x_adv.detach(), edge_index_adv.detach(), info

    @staticmethod
    def edge_index_from_sparse_adj(adj_sparse: torch.Tensor) -> torch.Tensor:
        """Convert a torch sparse-COO adjacency to a (2, |E|) edge_index.

        Used by EvolveGCN/CoSemiGNN drivers whose datasets store adjacencies as
        sparse tensors. The adjacency is assumed to encode a directed-pair list
        (NETTACK will symmetrize internally).
        """
        if not adj_sparse.is_sparse:
            raise ValueError("edge_index_from_sparse_adj expects a sparse tensor.")
        return adj_sparse.coalesce().indices().long()

    @staticmethod
    def sparse_adj_from_edge_index(
        edge_index: torch.Tensor,
        num_nodes: int,
        values: Optional[torch.Tensor] = None,
        device=None,
    ) -> torch.Tensor:
        """Inverse of `edge_index_from_sparse_adj`: build an unweighted (or weighted)
        sparse adjacency from an edge_index. Used to feed NETTACK-perturbed edges
        back into models that consume sparse adjacencies (CoSemiGNN raw).
        """
        device = device or edge_index.device
        if values is None:
            values = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            edge_index.long(), values.float(), size=(num_nodes, num_nodes)
        ).coalesce()
