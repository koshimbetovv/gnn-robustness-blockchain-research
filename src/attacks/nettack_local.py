"""Nettack-style local greedy structural attack (baseline)."""

import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.attacks.base_attack import BaseAttack


class NettackLocalAttack(BaseAttack):
    def __init__(
        self,
        model,
        data,
        device,
        adj_list=None,
        undirected: bool = False,
        allow_removals: bool = False,
        attack_incoming: bool = True,   # <<< NEW (important for directed graphs)
        seed: int | None = 0,
    ):
        super().__init__(model, data, device)
        self.x = data.x.to(device)
        self.y = data.y.to(device)
        self.undirected = bool(undirected)
        self.allow_removals = bool(allow_removals)
        self.attack_incoming = bool(attack_incoming)
        self.adj_list = adj_list
        self.rng = random.Random(seed)

    def _build_adj_list(self, edge_index: torch.Tensor):
        """
        For undirected: adj[v] = neighbors.
        For directed:
          - if attack_incoming: adj[v] = incoming sources u with u->v
          - else:              adj[u] = outgoing destinations v with u->v
        """
        adj = defaultdict(set)
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        for u, v in zip(u_list, v_list):
            if self.undirected:
                adj[u].add(v)
                adj[v].add(u)
            else:
                if self.attack_incoming:
                    adj[v].add(u)   # incoming neighbors of v
                else:
                    adj[u].add(v)   # outgoing neighbors of u
        return adj

    def _edge_for_candidate(self, target: int, u: int) -> tuple[int, int]:
        if self.undirected:
            return (target, u)  # reverse will be added too
        return (u, target) if self.attack_incoming else (target, u)

    def _add_edge_set(self, edge_set: set[tuple[int, int]], e: tuple[int, int]):
        edge_set.add(e)
        if self.undirected:
            edge_set.add((e[1], e[0]))

    def _remove_edge_set(self, edge_set: set[tuple[int, int]], e: tuple[int, int]):
        edge_set.discard(e)
        if self.undirected:
            edge_set.discard((e[1], e[0]))

    def _update_adj_add(self, adj: dict[int, set[int]], e: tuple[int, int]):
        src, dst = e
        if self.undirected:
            adj[src].add(dst); adj[dst].add(src)
        else:
            if self.attack_incoming:
                adj[dst].add(src)
            else:
                adj[src].add(dst)

    def _update_adj_remove(self, adj: dict[int, set[int]], e: tuple[int, int]):
        src, dst = e
        if self.undirected:
            adj[src].discard(dst); adj[dst].discard(src)
        else:
            if self.attack_incoming:
                adj[dst].discard(src)
            else:
                adj[src].discard(dst)

    @torch.no_grad()
    def _loss_and_pred_on_target(self, edge_index: torch.Tensor, target_node: int) -> tuple[float, int]:
        logits = self.model(self.x, edge_index)
        if int(self.y[target_node].item()) == -1:
            return float("-inf"), -1
        loss = F.cross_entropy(logits[target_node].unsqueeze(0), self.y[target_node].unsqueeze(0))
        pred = int(logits[target_node].argmax().item())
        return float(loss.item()), pred

    def attack(
        self,
        target_node: int,
        edge_index: torch.Tensor,
        n_perturbations: int = 5,
        sample_size: int = 200,
        include_neighbors: bool = True,
        early_stop: bool = True,   # <<< NEW
        show_progress: bool = False
    ) -> torch.Tensor:
        edge_set: set[tuple[int, int]] = set(map(tuple, edge_index.t().tolist()))
        adj = self.adj_list if self.adj_list is not None else self._build_adj_list(edge_index)
        if target_node not in adj:
            adj[target_node] = set()

        all_nodes = set(range(self.data.num_nodes))
        y_t = int(self.y[target_node].item())

        itr = range(int(n_perturbations))
        if show_progress:
            itr = tqdm(itr, desc=f"Perturbations t={target_node}", leave=False)

        for step in itr:
            neighbors = set(adj[target_node])
            non_neighbors = list(all_nodes - neighbors - {target_node})
            sampled_non = self.rng.sample(non_neighbors, k=min(sample_size, len(non_neighbors)))

            candidates: list[tuple[str, int]] = []
            for u in sampled_non:
                e = self._edge_for_candidate(target_node, u)
                if e not in edge_set:
                    candidates.append(("add", u))

            if include_neighbors and self.allow_removals:
                for u in neighbors:
                    e = self._edge_for_candidate(target_node, u)
                    if e in edge_set:
                        candidates.append(("remove", u))

            if show_progress:itr.set_postfix(neigh=len(neighbors), cand=len(candidates))

            if not candidates:
                if show_progress:
                    itr.set_postfix(neigh=len(neighbors), cand=0, reason="no_candidates")
                    # finish bar visually
                    itr.n = itr.total
                    itr.refresh()
                break

            best_loss = float("-inf")
            best_op = None
            best_pred = None

            for op, u in candidates:
                tmp_edges = edge_set.copy()
                e = self._edge_for_candidate(target_node, u)

                if op == "add":
                    self._add_edge_set(tmp_edges, e)
                else:
                    self._remove_edge_set(tmp_edges, e)

                new_edge_index = torch.tensor(list(tmp_edges), dtype=torch.long).t().contiguous().to(self.device)
                loss_val, pred_val = self._loss_and_pred_on_target(new_edge_index, target_node)

                if loss_val > best_loss:
                    best_loss = loss_val
                    best_op = (op, u)
                    best_pred = pred_val

            if best_op is None:
                if show_progress:
                    itr.set_postfix(reason="no_best_op")
                    itr.n = itr.total
                    itr.refresh()
                break

            op, u = best_op
            e = self._edge_for_candidate(target_node, u)
            if op == "add":
                self._add_edge_set(edge_set, e)
                self._update_adj_add(adj, e)
            else:
                self._remove_edge_set(edge_set, e)
                self._update_adj_remove(adj, e)

            if show_progress: itr.update(1)

            if early_stop and best_pred is not None and best_pred != y_t:
                if show_progress:
                    itr.set_postfix(neigh=len(neighbors), cand=len(candidates), flipped=1, step=step+1)
                    itr.n = itr.total
                    itr.refresh()
                break

        if show_progress: itr.close()

        return torch.tensor(list(edge_set), dtype=torch.long).t().contiguous().to(self.device)
