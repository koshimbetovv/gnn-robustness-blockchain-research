from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RecGNNNodeInjectionResult:
    x_adv: torch.Tensor              # (n_existing + n_inject, F)
    edge_index_adv: torch.Tensor     # (2, E_existing + E_injected)
    log_probs_adv: torch.Tensor      # (n_existing, num_classes) — sliced to original nodes
    log_probs_clean: torch.Tensor    # (n_existing, num_classes)
    x_injected_base: torch.Tensor    # (n_inject, F) before PGD on injected rows
    injected_node_ids: list[int]     # global node ids assigned to injected rows (within this timestep)
    injected_edges: list[tuple[int, int]]


class RecGNNNodeInjectionAttack:
    """L_inf node-injection evasion attack for RecGNN, one test timestep at a time.

    Threat model (per spec):
      - Injected nodes exist ONLY at the attack timestep. The next timestep
        advances along the clean (non-augmented) trajectory, so injected nodes
        do not persist into the model's recurrent state.
      - Injected rows have zero prior m-LSTM hidden/cell state (cold start).
        We accomplish this by restoring the pre-timestep state (computed before
        the injected rows existed) and zeroing the slots reserved for injected
        rows before each adversarial forward.
      - Only the injected nodes' features participate in `delta`; existing
        nodes are untouched.

    Threat-model parameters mirror `RecGNNPGDAttack`:
      attack_dim : number of leading feature columns the attacker can perturb on
                   injected nodes. The trailing 2 ANF columns are derived from
                   neighbour labels and are not attacker-controllable, so we
                   exclude them from `delta` (defaults to 93 for Elliptic).

    Capacity constraint: m-LSTM's `state_rows` upper-bounds the per-timestep
    node count. We require `n_existing + n_inject <= state_rows`.

    Protocol per test timestep:
      1. Save pre-timestep state.
      2. Clean forward → clean log-probs; save post-timestep state.
      3. Build injected nodes (features init + edges) and expand x / edge_index.
      4. PGD inner loop: for each step, restore pre-state, zero injected state
         rows, run gradient forward with current `delta`, update + project.
      5. Final adv forward (restore pre-state, zero injected rows) → adv log-probs.
      6. Restore post-state so the next timestep advances along the clean
         trajectory (injected rows do not persist).
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
        """Zero the m-LSTM hidden/cell rows reserved for injected nodes.

        m-LSTM keeps a fixed `state_rows` tensor; the rows in
        `[n_existing : n_existing + n_inject]` may carry residual values from
        prior timesteps' padding cycles. Zeroing them enforces the cold-start
        threat model for injected rows.
        """
        ml = self.model.m_lstm
        if ml._h_state is not None:
            ml._h_state[n_existing : n_existing + n_inject].zero_()
        if ml._c_state is not None:
            ml._c_state[n_existing : n_existing + n_inject].zero_()

    def _init_injected_features(
        self,
        x: torch.Tensor,
        n_inject: int,
        init: str = "mean",
        reference_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if reference_nodes is None or reference_nodes.numel() == 0:
            ref = x
        else:
            ref = x[reference_nodes]
        if init == "mean":
            return ref.mean(dim=0, keepdim=True).repeat(n_inject, 1).detach()
        if init == "randn":
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True).clamp_min(1e-6)
            return (mu + torch.randn((n_inject, x.size(1)), device=self.device) * std).detach()
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

    def _apply_delta(self, x: torch.Tensor, base_inj: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.attack_dim == x.size(1):
            inj = base_inj + delta
        else:
            raw = base_inj[:, : self.attack_dim] + delta
            other = base_inj[:, self.attack_dim :]
            inj = torch.cat([raw, other], dim=1)
        return torch.cat([x, inj], dim=0)

    def _project_clamp(self, base_inj: torch.Tensor, delta: torch.Tensor, eps: float) -> torch.Tensor:
        if self.clamp is None:
            return delta.detach()
        raw_base = base_inj[:, : self.attack_dim]
        raw_adv = raw_base + delta
        raw_adv = torch.clamp(raw_adv, min=self.clamp[0], max=self.clamp[1])
        new_delta = raw_adv - raw_base
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
    ) -> RecGNNNodeInjectionResult:
        """Run multi-step L_inf node-injection PGD on the current timestep.

        Returns a `RecGNNNodeInjectionResult` whose `log_probs_adv` is sliced to
        the original node set so callers can compute metrics over the same node
        index space as `log_probs_clean`.
        """
        target_nodes = target_nodes.to(self.device).long().view(-1)
        labels_true = labels_true.to(self.device).long().view(-1)
        target_nodes, labels_true = self._dedupe_targets_with_labels(target_nodes, labels_true)
        n_existing = int(x.size(0))

        snap_pre = self._save_state()

        with torch.no_grad():
            log_probs_clean = self._forward(x, edge_index).detach()
            self.model.detach_sequence_state()
        snap_post = self._save_state()

        if target_nodes.numel() == 0:
            return RecGNNNodeInjectionResult(
                x_adv=x.clone(),
                edge_index_adv=edge_index.clone(),
                log_probs_adv=log_probs_clean.clone(),
                log_probs_clean=log_probs_clean,
                x_injected_base=x.new_empty((0, x.size(1))),
                injected_node_ids=[],
                injected_edges=[],
            )

        n_inject = int(n_inject)
        state_rows = int(self.model.m_lstm.state_rows)
        if n_existing + n_inject > state_rows:
            raise ValueError(
                f"n_existing + n_inject = {n_existing + n_inject} exceeds m-LSTM "
                f"state_rows = {state_rows}. Reduce n_inject or use a model with larger state_rows."
            )

        injected_ids = torch.arange(n_existing, n_existing + n_inject, device=self.device, dtype=torch.long)
        base_inj = self._init_injected_features(x, n_inject, init=init, reference_nodes=reference_nodes)

        new_edges = self._build_injection_edges(
            injected_ids, target_nodes, int(edges_per_injected), connect_strategy
        )
        if new_edges:
            add_ei = torch.tensor(new_edges, dtype=torch.long, device=self.device).t().contiguous()
            edge_index_adv = torch.cat([edge_index, add_ei], dim=1)
        else:
            edge_index_adv = edge_index.clone()

        direction = +1.0

        if random_start:
            delta = (2 * torch.rand((n_inject, self.attack_dim), device=self.device) - 1.0) * float(eps)
        else:
            delta = torch.zeros((n_inject, self.attack_dim), device=self.device)
        delta = delta.clamp(min=-float(eps), max=float(eps)).detach()
        delta = self._project_clamp(base_inj, delta, eps)

        for _ in range(int(steps)):
            self._restore_state(snap_pre)
            self._zero_inject_rows(n_existing, n_inject)
            delta.requires_grad_(True)
            x_grad = self._apply_delta(x, base_inj, delta)
            log_probs_grad = self._forward(x_grad, edge_index_adv)
            loss = F.nll_loss(log_probs_grad[target_nodes], labels_true)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = (delta + direction * float(alpha) * grad.sign()).detach()
            delta = delta.clamp(min=-float(eps), max=float(eps))
            delta = self._project_clamp(base_inj, delta, eps)

        self._restore_state(snap_pre)
        self._zero_inject_rows(n_existing, n_inject)
        x_out = self._apply_delta(x, base_inj, delta)

        with torch.no_grad():
            log_probs_adv_full = self._forward(x_out, edge_index_adv).detach()
            self.model.detach_sequence_state()

        # Advance state along the CLEAN trajectory for the next timestep.
        self._restore_state(snap_post)

        return RecGNNNodeInjectionResult(
            x_adv=x_out.detach(),
            edge_index_adv=edge_index_adv.detach(),
            log_probs_adv=log_probs_adv_full[:n_existing],
            log_probs_clean=log_probs_clean,
            x_injected_base=base_inj.detach(),
            injected_node_ids=injected_ids.detach().cpu().tolist(),
            injected_edges=new_edges,
        )
