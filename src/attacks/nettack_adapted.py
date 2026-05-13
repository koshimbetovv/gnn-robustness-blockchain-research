"""Adapted NETTACK for binary illicit -> licit attacks with continuous features.

Adaptations from Zuegner et al. (KDD 2018):
1. Targets only clean-correct illicit nodes (y == 1) selected by the driver.
2. Targeted: surrogate loss = [A_hat^2 X W]_{v0, licit} - [A_hat^2 X W]_{v0, illicit},
   maximized so the licit logit overtakes the illicit one.
3. Continuous features -> closed-form L2 perturbation step using the surrogate's
   gradient w.r.t. X[v0] (the surrogate is linear in X, so the optimal direction
   under an L2 budget is g / ||g|| with gain eps_feat * ||g||).
4. Evasion: model parameters are static; the victim is queried only after the attack.
5. Direct attack (A = {v_0}); only edges (v_0, u) and X[v_0, :] may be perturbed.
6. Edge ADDITIONS only (no deletions).
7. Power-law chi^2 degree-distribution unnoticeability constraint (Eq. 6-9).

The greedy loop matches the paper's per-iteration choice between a structure or
feature step. Once the L2 feature budget is committed (single closed-form step),
remaining iterations are edge-only.

Per-target candidate edge scoring is fully vectorized using closed-form expressions
for Z'[v0] after a single edge addition, derived from the linearized GCN surrogate
(equivalent to the incremental update in Theorem 5.1 specialized to additions).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack


def _make_undirected_no_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return deduplicated undirected edges without self-loops, shape (2, |E|)."""
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    row, col = row[mask], col[mask]
    full_row = torch.cat([row, col]).long()
    full_col = torch.cat([col, row]).long()
    keys = full_row * num_nodes + full_col
    unique_keys = torch.unique(keys)
    new_row = unique_keys // num_nodes
    new_col = unique_keys % num_nodes
    return torch.stack([new_row, new_col], dim=0)


def _build_A_hat(
    edges_no_sl: torch.Tensor, num_nodes: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2} as a sparse COO tensor.

    Returns (A_hat, d_tilde) where d_tilde[i] = deg(i) + 1 (with self-loop).
    """
    N = num_nodes
    row = edges_no_sl[0].long()
    col = edges_no_sl[1].long()
    sl = torch.arange(N, device=device, dtype=torch.long)
    full_row = torch.cat([row, sl])
    full_col = torch.cat([col, sl])
    deg_tilde = torch.zeros(N, device=device, dtype=torch.float)
    deg_tilde.scatter_add_(0, full_row, torch.ones_like(full_row, dtype=torch.float))
    inv_sqrt = deg_tilde.clamp(min=1.0).pow(-0.5)
    values = inv_sqrt[full_row] * inv_sqrt[full_col]
    indices = torch.stack([full_row, full_col], dim=0)
    A_hat = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    return A_hat, deg_tilde


class AdaptedNettackAttack(BaseAttack):
    """Adapted NETTACK driver matching the FGSM / PGD scaffolding.

    Threat-model parameters (mirror FGSM/PGD):
      attack_dim  : leading feature columns the attacker may perturb (raw slice for
                    composite features).
      rebuild_fn  : optional callable rebuilding derived feature columns after the
                    raw slice is perturbed (e.g. ChronoWaveGNN wavelet features).

    Attack-specific parameters:
      n_struct                  : max number of edge ADDITIONS per target.
      eps_feat                  : per-target L2 budget for the closed-form feature step.
      d_min, chi2_tau           : power-law chi^2 test parameters (paper defaults 2, 0.004).
      enforce_degree_constraint : if False, skip the chi^2 filter on edge candidates.
    """

    def __init__(
        self,
        model,
        data,
        device,
        clamp: tuple[float, float] | None = None,
        attack_dim: Optional[int] = None,
        rebuild_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        d_min: int = 2,
        chi2_tau: float = 0.004,
        enforce_degree_constraint: bool = True,
        surrogate_epochs: int = 200,
        surrogate_lr: float = 0.01,
        surrogate_weight_decay: float = 5e-4,
        pretrained_W: Optional[torch.Tensor] = None,
        verbose: bool = False,
        progress_every: int = 100,
    ):
        super().__init__(model, data, device)
        self.x = data.x.to(device).detach().clone()
        self.y = data.y.to(device).long()
        self.edge_index_orig = data.edge_index.to(device).clone()
        ts = getattr(data, "time_step", None)
        self.time_step = ts.to(device) if torch.is_tensor(ts) else None
        self.clamp = clamp
        self.attack_dim = int(attack_dim) if attack_dim is not None else int(self.x.size(1))
        if not (1 <= self.attack_dim <= self.x.size(1)):
            raise ValueError(f"attack_dim must be in [1, {self.x.size(1)}], got {self.attack_dim}")
        self.rebuild_fn = rebuild_fn
        self.d_min = int(d_min)
        self.chi2_tau = float(chi2_tau)
        self.enforce_degree_constraint = bool(enforce_degree_constraint)
        self.N = int(self.x.size(0))
        self.K = 2
        self.verbose = bool(verbose)
        self.progress_every = int(progress_every)

        # Symmetrized clean adjacency used by the surrogate (NETTACK assumes undirected).
        self.edges_no_sl_orig = _make_undirected_no_self_loops(self.edge_index_orig, self.N)
        deg_orig = torch.zeros(self.N, device=device, dtype=torch.long)
        deg_orig.scatter_add_(
            0, self.edges_no_sl_orig[0],
            torch.ones_like(self.edges_no_sl_orig[0], dtype=torch.long),
        )
        self.deg_orig = deg_orig

        # chi^2 stats for the original (unperturbed) graph -- baseline for Lambda.
        self._n_orig, self._R_orig, self._alpha_orig, self._l_orig = self._powerlaw_stats(self.deg_orig)

        # Train the linearized 2-layer GCN surrogate W (D x K) -- or accept a
        # pretrained W (e.g. trained on the union of train timesteps for the
        # temporal driver, then reused per test slice).
        if pretrained_W is not None:
            W = pretrained_W.to(device).detach()
            if W.shape != (self.x.size(1), self.K):
                raise ValueError(
                    f"pretrained_W shape {tuple(W.shape)} does not match "
                    f"({self.x.size(1)}, {self.K})."
                )
            self.W = W
        else:
            self.W = self._train_surrogate(
                self.edges_no_sl_orig,
                epochs=int(surrogate_epochs),
                lr=float(surrogate_lr),
                weight_decay=float(surrogate_weight_decay),
            )

    # ---------- surrogate ----------
    def _train_surrogate(self, edges_no_sl, *, epochs, lr, weight_decay):
        """Train W in R^{D x K} for Z = softmax(A_hat^2 X W) on the clean graph."""
        device = self.device
        A_hat, _ = _build_A_hat(edges_no_sl, self.N, device)
        AX = torch.sparse.mm(A_hat, self.x)
        AAX = torch.sparse.mm(A_hat, AX)
        D = self.x.size(1)
        W = torch.empty(D, self.K, device=device)
        torch.nn.init.xavier_uniform_(W)
        W.requires_grad_(True)

        train_mask = getattr(self.data, "train_mask", None)
        if train_mask is None:
            train_mask = (self.y != -1)
        train_mask = train_mask.to(device).bool() & (self.y != -1)

        opt = torch.optim.Adam([W], lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            logits = AAX @ W
            loss = F.cross_entropy(logits[train_mask], self.y[train_mask])
            loss.backward()
            opt.step()
        return W.detach()

    # ---------- power-law chi^2 test (Eq. 6-9) ----------
    def _powerlaw_stats(self, deg: torch.Tensor):
        """Return (n, R, alpha, l) for the multiset of degrees >= d_min.
        n: count, R: sum log d_i, alpha and l per Eq. 6, 7.
        """
        d_min = self.d_min
        d = deg[deg >= d_min].float()
        n = float(d.numel())
        if n <= 0:
            return 0.0, 0.0, float("nan"), 0.0
        sum_log_d = float(torch.log(d).sum().item())
        R_tilde = sum_log_d - n * math.log(d_min - 0.5)
        if R_tilde <= 0:
            return n, sum_log_d, float("nan"), 0.0
        alpha = 1.0 + n / R_tilde
        l = n * math.log(alpha) + n * alpha * math.log(d_min) - (alpha + 1) * sum_log_d
        return n, sum_log_d, alpha, l

    def _vec_l_from_n_R(self, n: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Vectorized log-likelihood l(D) given tensor n and R = sum log d_i."""
        d_min = self.d_min
        log_dmin_half = math.log(d_min - 0.5)
        log_dmin = math.log(d_min)
        R_tilde = R - n * log_dmin_half
        valid = (n > 0) & (R_tilde > 0)
        l = torch.zeros_like(n)
        if valid.any():
            n_v = n[valid]
            R_v = R[valid]
            R_tilde_v = R_tilde[valid]
            alpha_v = 1.0 + n_v / R_tilde_v
            l[valid] = (
                n_v * torch.log(alpha_v)
                + n_v * alpha_v * log_dmin
                - (alpha_v + 1.0) * R_v
            )
        return l

    def _chi2_filter_additions(
        self, v0: int, candidates: torch.Tensor, deg_curr: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized: for each candidate u, decide if adding (v0, u) keeps Lambda < chi2_tau.

        Lambda is computed against the ORIGINAL graph (deg_orig), not the running state,
        so the constraint is on cumulative degree-distribution drift, matching the paper.
        """
        device = self.device
        d_min = self.d_min

        # State BEFORE this candidate: stats of the running perturbed graph.
        n_curr, R_curr, _, _ = self._powerlaw_stats(deg_curr)

        d_v0 = float(deg_curr[v0].item())
        d_u_vec = deg_curr[candidates].float()

        # Update from incrementing v0 (constant across candidates).
        if d_v0 >= d_min:
            dn_v0 = 0.0
            dR_v0 = math.log(d_v0 + 1.0) - math.log(d_v0)
        elif d_v0 == d_min - 1:
            dn_v0 = 1.0
            dR_v0 = math.log(d_v0 + 1.0)
        else:
            dn_v0 = 0.0
            dR_v0 = 0.0

        # Vectorized update from incrementing u.
        is_above = d_u_vec >= d_min
        is_at = d_u_vec == (d_min - 1)
        dn_u = is_at.float()
        dR_u = torch.where(
            is_above,
            torch.log(d_u_vec + 1.0) - torch.log(d_u_vec.clamp(min=1.0)),
            torch.where(is_at, torch.log(d_u_vec + 1.0), torch.zeros_like(d_u_vec)),
        )

        n_new = n_curr + dn_v0 + dn_u
        R_new = R_curr + dR_v0 + dR_u

        l_new = self._vec_l_from_n_R(n_new, R_new)
        n_comb = self._n_orig + n_new
        R_comb = self._R_orig + R_new
        l_comb = self._vec_l_from_n_R(n_comb, R_comb)

        Lambda = -2.0 * l_comb + 2.0 * (self._l_orig + l_new)
        return Lambda < self.chi2_tau

    # ---------- attack ----------
    def attack(
        self,
        target_nodes: torch.Tensor,
        n_struct: int = 2,
        eps_feat: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run adapted NETTACK on each target. Returns (x_adv, edge_index_adv).

        edge_index_adv contains the original edge_index with new edges appended
        in BOTH directions (NETTACK is undirected -- per-edge add inserts (v0, u)
        and (u, v0)). Original edges are preserved verbatim.
        """
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)
        target_nodes = torch.unique(target_nodes)

        # Defense in depth -- driver should already pass clean-correct illicit nodes.
        labeled = self.y[target_nodes] != -1
        target_nodes = target_nodes[labeled]
        illicit = self.y[target_nodes] == 1
        target_nodes = target_nodes[illicit]
        if target_nodes.numel() == 0:
            return self.x.clone(), self.edge_index_orig.clone()

        # Working state (cumulative across targets).
        self._x_curr = self.x.clone()
        self._edges_curr = self.edges_no_sl_orig.clone()
        self._deg_curr = self.deg_orig.clone()
        self._added_edges: list[tuple[int, int]] = []

        n_targets = int(target_nodes.numel())
        for i, v0 in enumerate(target_nodes.tolist()):
            self._attack_single_target(int(v0), int(n_struct), float(eps_feat))
            if self.verbose and (i + 1) % self.progress_every == 0:
                print(
                    f"  [adapted-nettack] {i + 1}/{n_targets} targets done; "
                    f"edges added so far={len(self._added_edges)}"
                )

        # Build output edge_index: original + symmetric additions.
        edge_index_adv = self._build_output_edge_index()
        x_adv = self._x_curr
        if self.clamp is not None:
            x_adv = torch.clamp(x_adv, min=self.clamp[0], max=self.clamp[1])
        return x_adv, edge_index_adv

    def _build_output_edge_index(self) -> torch.Tensor:
        """Original edge_index plus all new edges added during the attack (both directions)."""
        device = self.device
        if not self._added_edges:
            return self.edge_index_orig.clone()
        added = torch.tensor(self._added_edges, device=device, dtype=self.edge_index_orig.dtype)
        # added has shape (M, 2) with entries (v0, u). Add both directions.
        forward = added.t()  # (2, M)
        backward = torch.stack([added[:, 1], added[:, 0]], dim=0)
        return torch.cat([self.edge_index_orig, forward, backward], dim=1)

    def _attack_single_target(self, v0: int, n_struct: int, eps_feat: float):
        """Mutates self._x_curr, self._edges_curr, self._deg_curr, self._added_edges."""
        device = self.device
        N = self.N

        feat_used = False
        edges_used = 0

        while edges_used < n_struct or not feat_used:
            # Recompute surrogate quantities on the running state.
            A_hat, d_tilde = _build_A_hat(self._edges_curr, N, device)
            C = self._x_curr @ self.W
            M = torch.sparse.mm(A_hat, C)
            Z = torch.sparse.mm(A_hat, M)
            L_clean = float((Z[v0, 0] - Z[v0, 1]).item())

            # ----- best edge candidate -----
            best_edge_score = -float("inf")
            best_edge_u: Optional[int] = None

            row = self._edges_curr[0]
            col = self._edges_curr[1]
            v0_mask = row == v0
            nbrs_v0 = col[v0_mask].long()
            in_N_v0 = torch.zeros(N, device=device, dtype=torch.bool)
            in_N_v0[nbrs_v0] = True
            cand_mask = ~in_N_v0
            cand_mask[v0] = False

            if edges_used < n_struct:
                candidates = torch.where(cand_mask)[0]
                if candidates.numel() > 0:
                    scores = self._score_edge_additions(
                        v0, candidates, d_tilde, C, M, L_clean, nbrs_v0, row, col
                    )
                    if self.enforce_degree_constraint:
                        valid = self._chi2_filter_additions(v0, candidates, self._deg_curr)
                        scores = torch.where(
                            valid, scores, torch.full_like(scores, -float("inf"))
                        )
                    if torch.isfinite(scores).any():
                        idx = torch.argmax(scores)
                        best_edge_score = float(scores[idx].item())
                        best_edge_u = int(candidates[idx].item())

            # ----- feature option (closed-form L2) -----
            feat_score = -float("inf")
            feat_g = None
            if not feat_used:
                d_t_v0 = float(d_tilde[v0].item())
                if d_t_v0 > 0:
                    if nbrs_v0.numel() > 0:
                        T1 = float((1.0 / d_tilde[nbrs_v0]).sum().item())
                    else:
                        T1 = 0.0
                    A2_v0_v0 = (1.0 / d_t_v0) * ((1.0 / d_t_v0) + T1)
                    w_diff = self.W[: self.attack_dim, 0] - self.W[: self.attack_dim, 1]
                    feat_g = A2_v0_v0 * w_diff
                    g_norm = float(feat_g.norm().item())
                    if g_norm > 0:
                        feat_score = eps_feat * g_norm

            # ----- pick the better action -----
            if (not feat_used) and feat_score > best_edge_score and feat_score > 0.0:
                g_norm = float(feat_g.norm().item())
                step = (eps_feat / max(g_norm, 1e-12)) * feat_g
                self._x_curr[v0, : self.attack_dim] = (
                    self._x_curr[v0, : self.attack_dim] + step
                )
                if self.rebuild_fn is not None:
                    self._x_curr = self.rebuild_fn(self._x_curr)
                feat_used = True
            elif edges_used < n_struct and best_edge_u is not None and best_edge_score > -float("inf"):
                u = best_edge_u
                new_edges = torch.tensor(
                    [[v0, u], [u, v0]], device=device, dtype=self._edges_curr.dtype
                )
                self._edges_curr = torch.cat([self._edges_curr, new_edges], dim=1)
                self._deg_curr[v0] = self._deg_curr[v0] + 1
                self._deg_curr[u] = self._deg_curr[u] + 1
                self._added_edges.append((v0, u))
                edges_used += 1
            else:
                # No improving action available -- stop attacking this target.
                break

    def _score_edge_additions(
        self,
        v0: int,
        candidates: torch.Tensor,
        d_tilde: torch.Tensor,
        C: torch.Tensor,
        M: torch.Tensor,
        L_clean: float,
        nbrs_v0: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized closed-form score for all candidate edge additions (v0, u).

        Uses the linearized GCN surrogate Z = A_hat^2 X W. After adding (v0, u)
        (with u not in N(v0), additions only), the new Z'[v0] is a linear
        combination of pre-computed quantities:
            Z'[v0] = (1/sqrt(d~'_v0)) * [
                (1/sqrt(d~'_v0)) * M'_v0
              + (1/sqrt(d~'_u))  * M'_u
              + S_v0_M
              + delta_v0 * C_v0 * T1
              + delta_u  * C_u  * T2(u)
            ]
        where:
          - M'_v0, M'_u absorb the new edge (formulas in module docstring),
          - delta_v0 = 1/sqrt(d~_v0+1) - 1/sqrt(d~_v0)        (constant in u),
          - delta_u(u) = 1/sqrt(d~_u+1) - 1/sqrt(d~_u),
          - T1 = sum_{k in N(v0)} 1/d~_k                      (constant in u),
          - T2(u) = sum_{k in N(v0) cap N(u)} 1/d~_k.

        Returns scores of shape (M,) = L_s(A + e_u, X) - L_s(A, X) where
        L_s = Z[v0, licit] - Z[v0, illicit].
        """
        device = self.device
        N = self.N
        M_K = self.K

        # v0-side scalars (independent of u).
        d_t_v0 = d_tilde[v0]
        d_t_v0_new = d_t_v0 + 1.0
        inv_sqrt_d_v0 = 1.0 / d_t_v0.sqrt()
        inv_sqrt_d_v0_new = 1.0 / d_t_v0_new.sqrt()
        delta_v0 = inv_sqrt_d_v0_new - inv_sqrt_d_v0

        if nbrs_v0.numel() > 0:
            inv_sqrt_d_nbrs = 1.0 / d_tilde[nbrs_v0].sqrt()
            S_v0_C = (inv_sqrt_d_nbrs.unsqueeze(1) * C[nbrs_v0]).sum(dim=0)
            S_v0_M = (inv_sqrt_d_nbrs.unsqueeze(1) * M[nbrs_v0]).sum(dim=0)
            T1 = (1.0 / d_tilde[nbrs_v0]).sum()
        else:
            S_v0_C = torch.zeros(M_K, device=device)
            S_v0_M = torch.zeros(M_K, device=device)
            T1 = torch.tensor(0.0, device=device)

        # T2(u) for ALL u -> one scatter over current edges.
        in_N_v0 = torch.zeros(N, device=device, dtype=torch.bool)
        in_N_v0[nbrs_v0] = True
        edge_mask = in_N_v0[row]
        contrib_dst = col[edge_mask]
        contrib_val = 1.0 / d_tilde[row[edge_mask]]
        T2_all = torch.zeros(N, device=device, dtype=torch.float)
        T2_all.scatter_add_(0, contrib_dst, contrib_val)
        T2_cand = T2_all[candidates]

        # u-side per-candidate.
        d_t_u = d_tilde[candidates]
        d_t_u_new = d_t_u + 1.0
        inv_sqrt_d_u = 1.0 / d_t_u.sqrt()
        inv_sqrt_d_u_new = 1.0 / d_t_u_new.sqrt()
        delta_u = inv_sqrt_d_u_new - inv_sqrt_d_u

        C_v0 = C[v0]
        C_u = C[candidates]
        M_u = M[candidates]
        # S_u_C = sum_{k in N(u)} (1/sqrt(d~_k)) * C_k
        #       = sqrt(d~_u) * M_u - (1/sqrt(d~_u)) * C_u
        S_u_C = d_t_u.sqrt().unsqueeze(1) * M_u - inv_sqrt_d_u.unsqueeze(1) * C_u

        # M'_v0 (per candidate)
        M_prime_v0 = inv_sqrt_d_v0_new * (
            inv_sqrt_d_v0_new * C_v0.unsqueeze(0)
            + inv_sqrt_d_u_new.unsqueeze(1) * C_u
            + S_v0_C.unsqueeze(0)
        )

        # M'_u (per candidate)
        M_prime_u = inv_sqrt_d_u_new.unsqueeze(1) * (
            inv_sqrt_d_u_new.unsqueeze(1) * C_u
            + inv_sqrt_d_v0_new * C_v0.unsqueeze(0)
            + S_u_C
        )

        bracket = (
            inv_sqrt_d_v0_new * M_prime_v0
            + inv_sqrt_d_u_new.unsqueeze(1) * M_prime_u
            + S_v0_M.unsqueeze(0)
            + delta_v0 * C_v0.unsqueeze(0) * T1
            + delta_u.unsqueeze(1) * C_u * T2_cand.unsqueeze(1)
        )
        z_prime_v0 = inv_sqrt_d_v0_new * bracket  # (M, K)

        L_new = z_prime_v0[:, 0] - z_prime_v0[:, 1]
        return L_new - L_clean
