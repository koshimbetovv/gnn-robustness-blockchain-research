from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class RecGNNPGDAttack:
    """L_inf PGD for RecGNN, operating one timestep at a time within a sequence.

    RecGNN maintains recurrent state across timesteps (m-LSTM hidden/cell state
    + evolving GCN weights inside the cell). The attacker can only modify
    features at a single timestep; the sequence state carried from previous
    timesteps is treated as fixed context (i.e. we do not unroll BPTT).

    Threat-model defaults for Elliptic:
      attack_dim = 93 (the local standardized features).
      The trailing 2 ANF columns are counts of labeled antecedent neighbours;
      they are derived from neighbour labels and are not controllable by
      perturbing the target node's own feature vector, so we treat them as
      fixed (analogous to CoSemiGNN's 6 semi-supervised columns).

    Protocol per test timestep:
      1. Save pre-timestep state.
      2. Clean forward → clean log-probs; save post-timestep state.
      3. PGD inner loop: for each step, restore pre-state, run gradient
         forward with the current `delta`, update `delta` and project to the
         L_inf eps ball (and into the optional clamp box).
      4. Restore pre-state again; run final adv forward → adv log-probs.
      5. Restore post-state so the next timestep advances along the clean
         trajectory (prevents one timestep's attack from bleeding into the
         next's evaluation).
    """

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

    def _apply_delta(self, x: torch.Tensor, target_idx: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.attack_dim == x.size(1):
            row_vals = x[target_idx] + delta
        else:
            raw = x[target_idx, : self.attack_dim] + delta
            other = x[target_idx, self.attack_dim :]
            row_vals = torch.cat([raw, other], dim=1)
        return x.index_copy(0, target_idx, row_vals)

    def _project_clamp(self, x, target_idx, delta, eps):
        """Project resulting features into the clamp box on the perturbable
        slice and re-clip delta to the L_inf eps ball."""
        if self.clamp is None:
            return delta.detach()
        x_tmp = self._apply_delta(x, target_idx, delta)
        x_tmp = torch.clamp(x_tmp, min=self.clamp[0], max=self.clamp[1])
        new_delta = x_tmp[target_idx, : self.attack_dim] - x[target_idx, : self.attack_dim]
        return new_delta.clamp(min=-float(eps), max=float(eps)).detach()

    @torch.no_grad()
    def prime(self, graphs) -> None:
        """Run the model forward over `graphs` (no grad) to build sequence state."""
        self.model.reset_sequence_state(self.device)
        for g in graphs:
            g = g.to(self.device)
            _ = self._forward(g.x.float(), g.edge_index.long())
            self.model.detach_sequence_state()

    def attack_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target_idx: torch.Tensor,
        labels_true: torch.Tensor,
        eps: float = 0.01,
        alpha: float = 0.002,
        steps: int = 10,
        random_start: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run multi-step L_inf PGD on the current timestep.

        Requires the model's sequence state to be positioned at the start of
        this timestep (i.e. previous timesteps already consumed). Advances the
        state along the clean trajectory before returning, so the caller can
        immediately call `attack_step` again on the next timestep.

        Returns `(x_adv, log_probs_adv, log_probs_clean)` — all detached, all
        over the full node set of the current timestep.
        """
        target_idx = target_idx.to(self.device).long().view(-1)
        target_idx = torch.unique(target_idx)

        snap_pre = self._save_state()

        with torch.no_grad():
            log_probs_clean = self._forward(x, edge_index).detach()
            self.model.detach_sequence_state()
        snap_post = self._save_state()

        if target_idx.numel() == 0:
            # No targets — state is already on clean trajectory.
            return x.clone(), log_probs_clean.clone(), log_probs_clean

        direction = +1.0

        if random_start:
            delta = (2 * torch.rand((target_idx.numel(), self.attack_dim),
                                    device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((target_idx.numel(), self.attack_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(x, target_idx, delta, eps)

        for _ in range(int(steps)):
            self._restore_state(snap_pre)
            delta.requires_grad_(True)
            x_grad = self._apply_delta(x, target_idx, delta)
            log_probs_grad = self._forward(x_grad, edge_index)
            loss = F.nll_loss(log_probs_grad[target_idx], labels_true)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(x, target_idx, delta, eps)

        self._restore_state(snap_pre)
        x_out = self._apply_delta(x, target_idx, delta)
        if self.clamp is not None:
            x_out = torch.clamp(x_out, min=self.clamp[0], max=self.clamp[1])

        with torch.no_grad():
            log_probs_adv = self._forward(x_out, edge_index).detach()
            self.model.detach_sequence_state()

        # Advance state along the CLEAN trajectory for the next timestep.
        self._restore_state(snap_post)

        return x_out.detach(), log_probs_adv, log_probs_clean
