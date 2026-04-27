import random
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Literal

import torch
import torch.nn.functional as F

from src.attacks.base_attack import BaseAttack


@dataclass
class InjectionResult:
    x_adv: torch.Tensor
    edge_index_adv: torch.Tensor
    y_adv: torch.Tensor
    injected_node_ids: list[int]
    injected_edges: list[tuple[int, int]]  # directed (src, dst)


def _cw_targeted_loss(logits: torch.Tensor, desired_label: int) -> torch.Tensor:
    """
    Targeted C&W-style margin loss:
        relu(max_{c != y*} logit_c - logit_{y*})
    For binary: relu(logit_other - logit_desired).

    MonTi uses a C&W loss to push fraud nodes to be predicted benign (class 0). :contentReference[oaicite:4]{index=4}
    """
    num_classes = logits.size(1)
    y_star = int(desired_label)

    other = torch.arange(num_classes, device=logits.device)
    other = other[other != y_star]

    max_other = logits[:, other].max(dim=1).values
    return F.relu(max_other - logits[:, y_star])


class MonTiOneTimeInjectionAttack(BaseAttack):
    """
    MonTi-style Multi-target One-time Injection (practical adaptation for your repo):
      - Candidate selection from K-hop neighbors of targets (subset if too large). :contentReference[oaicite:5]{index=5}
      - Inject all attack nodes at once (one-time).
      - Allocate edges globally under an edge budget, guaranteeing >=1 edge per injected node to targets. :contentReference[oaicite:6]{index=6}
      - Optimize injected node features by PGD to minimize targeted C&W margin loss. :contentReference[oaicite:7]{index=7}

    This does NOT implement MonTi's trained transformer generator / STE end-to-end.
    """

    def __init__(
        self,
        model,
        data,
        device,
        *,
        undirected: bool = False,
        attack_incoming: bool = True,
        clamp: Optional[tuple[float, float]] = None,
        seed: int = 0,
    ):
        super().__init__(model, data, device)
        self.x = data.x.to(device).detach()
        self.y = data.y.to(device)
        self.edge_index = data.edge_index.to(device)
        self.undirected = bool(undirected)
        self.attack_incoming = bool(attack_incoming)
        self.clamp = clamp
        self.rng = random.Random(seed)

    # ---------- graph utilities ----------
    def _build_adj(self, edge_index: torch.Tensor, mode: Literal["undirected", "incoming", "outgoing"]) -> dict[int, set[int]]:
        adj = defaultdict(set)
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if mode == "undirected":
                adj[u].add(v); adj[v].add(u)
            elif mode == "incoming":
                adj[v].add(u)   # incoming neighbors of v
            else:  # "outgoing"
                adj[u].add(v)   # outgoing neighbors of u
        return adj

    def _k_hop_neighbors(
        self,
        targets: list[int],
        edge_index: torch.Tensor,
        K: int,
        mode: Literal["undirected", "incoming", "outgoing"] = "undirected",
    ) -> set[int]:
        adj = self._build_adj(edge_index, mode=mode)
        visited = set(targets)
        frontier = set(targets)
        for _ in range(int(K)):
            nxt = set()
            for v in frontier:
                nxt |= adj.get(v, set())
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        return visited - set(targets)

    @torch.no_grad()
    def _degree(self, edge_index: torch.Tensor) -> torch.Tensor:
        n = self.x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        deg = torch.bincount(src, minlength=n) + torch.bincount(dst, minlength=n)
        return deg

    # ---------- candidate selection ----------
    @torch.no_grad()
    def select_candidates(
        self,
        target_nodes: torch.Tensor,
        *,
        K: int = 2,
        alpha: int = 500,
        neighbor_mode: Literal["undirected", "incoming", "outgoing"] = "undirected",
        score_mode: Literal["benign_prob", "degree"] = "benign_prob",
        logits_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MonTi narrows candidates among K-hop neighbors of targets and (if too many) selects a subset. :contentReference[oaicite:8]{index=8}
        We approximate MonTi's learnable scoring J with:
          - benign_prob: p(y=0|v) from surrogate
          - degree: degree-based heuristic
        """
        t = target_nodes.detach().cpu().tolist()
        Nk = self._k_hop_neighbors(t, self.edge_index, K=K, mode=neighbor_mode)
        if len(Nk) == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        Nk_list = torch.tensor(list(Nk), dtype=torch.long, device=self.device)
        if Nk_list.numel() <= int(alpha):
            return Nk_list

        if score_mode == "degree":
            deg = self._degree(self.edge_index)
            scores = deg[Nk_list].float()
        else:
            if logits_clean is None:
                logits_clean = self.model(self.x, self.edge_index)
            p0 = F.softmax(logits_clean[Nk_list], dim=1)[:, 0]  # benign prob
            scores = p0

        topk = min(int(alpha), int(Nk_list.numel()))
        idx = torch.topk(scores, k=topk, largest=True).indices
        return Nk_list[idx]

    # ---------- edge generation (one-time, global budget) ----------
    def _edge_dir(self, src: int, dst: int) -> tuple[int, int]:
        # Always injected -> target so the target aggregates from the injected node.
        # attack_incoming controls candidate traversal direction, not edge direction.
        return (src, dst)

    def _add_edge_list(self, edges: list[tuple[int, int]], e: tuple[int, int]):
        edges.append(e)
        if self.undirected:
            edges.append((e[1], e[0]))

    def _select_edges_one_time(
        self,
        target_nodes: torch.Tensor,
        candidate_nodes: torch.Tensor,
        x_inj: torch.Tensor,
        injected_ids: torch.Tensor,
        edge_budget: int,
        *,
        allow_attack_attack_edges: bool = True,
    ) -> list[tuple[int, int]]:
        """
        Mirrors MonTi's "generate all edges at once": score edges between injected nodes and (targets + candidates + injected),
        ensure 1 edge per injected to a target, then select remaining edges globally. :contentReference[oaicite:9]{index=9}
        """
        m = int(target_nodes.numel())
        if m == 0:
            return []

        # pool = targets || candidates || injected
        pool = torch.cat([target_nodes, candidate_nodes, injected_ids], dim=0)
        pool_x = torch.cat([self.x[target_nodes], self.x[candidate_nodes], x_inj], dim=0)

        # normalize for cosine similarity
        inj_n = F.normalize(x_inj, dim=1)
        pool_n = F.normalize(pool_x, dim=1)
        scores = inj_n @ pool_n.t()  # [Delta, M]

        Delta = int(injected_ids.numel())
        M = int(pool.numel())

        # mask self loops inj->its own node (in the injected segment)
        # injected segment starts at m + |C|
        inj_offset = m + int(candidate_nodes.numel())
        for i in range(Delta):
            scores[i, inj_offset + i] = -1e9

        # optional: disallow inj->inj edges
        if not allow_attack_attack_edges:
            scores[:, inj_offset:] = -1e9

        # edge_budget is directed edges count (undirected will double in edge_index)
        edge_budget = int(edge_budget)
        edge_budget = max(edge_budget, Delta)  # must allow 1 per injected

        chosen: set[tuple[int, int]] = set()
        edges: list[tuple[int, int]] = []

        # (A) guarantee one edge per injected node to a target (Top-1 over target columns)
        for i in range(Delta):
            j = int(scores[i, :m].argmax().item())
            src = int(injected_ids[i].item())
            dst = int(pool[j].item())
            e = self._edge_dir(src, dst)
            if e not in chosen:
                chosen.add(e)
                self._add_edge_list(edges, e)

        remaining = edge_budget - Delta
        if remaining <= 0:
            return edges

        # (B) choose remaining edges globally by top-k over all scores (flatten)
        # mask already chosen target-edges
        mask = torch.ones_like(scores, dtype=torch.bool)
        for i in range(Delta):
            src = int(injected_ids[i].item())
            # mark already selected edge destination in pool (if any)
            for j in range(M):
                dst = int(pool[j].item())
                e = self._edge_dir(src, dst)
                if e in chosen:
                    mask[i, j] = False

        flat_scores = scores.masked_fill(~mask, -1e9).view(-1)
        k = min(int(remaining), int((flat_scores > -1e8).sum().item()))
        if k <= 0:
            return edges

        top_idx = torch.topk(flat_scores, k=k, largest=True).indices
        for idx in top_idx.tolist():
            i = idx // M
            j = idx % M
            src = int(injected_ids[i].item())
            dst = int(pool[j].item())
            e = self._edge_dir(src, dst)
            if e in chosen:
                continue
            chosen.add(e)
            self._add_edge_list(edges, e)

        return edges

    # ---------- main attack ----------
    def attack(
        self,
        target_nodes: torch.Tensor,
        *,
        n_inject: int = 5,          # Delta
        edge_budget: int = 50,      # eta (directed)
        K: int = 2,
        alpha: int = 500,
        neighbor_mode: Literal["undirected", "incoming", "outgoing"] = "undirected",
        candidate_score_mode: Literal["benign_prob", "degree"] = "benign_prob",
        # feature optimization (PGD on injected features)
        eps: float = 0.05,
        alpha_step: float = 0.01,
        inner_steps: int = 30,
        outer_rounds: int = 3,      # reselection rounds (approx joint edge/attr generation)
        random_start: bool = True,
        init: Literal["mean_benign", "randn_benign"] = "mean_benign",
        desired_label: int = 0,     # benign
        allow_attack_attack_edges: bool = True,
        early_stop: bool = True,
    ) -> InjectionResult:
        if not torch.is_tensor(target_nodes):
            target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        target_nodes = target_nodes.to(self.device).long().view(-1)

        # keep labeled targets
        target_nodes = target_nodes[self.y[target_nodes] != -1]
        if target_nodes.numel() == 0:
            return InjectionResult(
                x_adv=self.x.clone(),
                edge_index_adv=self.edge_index.clone(),
                y_adv=self.y.clone(),
                injected_node_ids=[],
                injected_edges=[],
            )

        with torch.no_grad():
            logits_clean = self.model(self.x, self.edge_index)

        # candidates from K-hop neighbors
        cand = self.select_candidates(
            target_nodes,
            K=K,
            alpha=alpha,
            neighbor_mode=neighbor_mode,
            score_mode=candidate_score_mode,
            logits_clean=logits_clean,
        )

        # initialize injected features close to benign candidates (camouflage)
        if cand.numel() == 0:
            ref = self.x
        else:
            ref = self.x[cand]

        if init == "mean_benign":
            x0 = ref.mean(dim=0, keepdim=True).repeat(int(n_inject), 1)
        else:
            mu = ref.mean(dim=0, keepdim=True)
            std = ref.std(dim=0, keepdim=True).clamp_min(1e-6)
            x0 = mu + torch.randn((int(n_inject), self.x.size(1)), device=self.device) * std

        if random_start:
            delta = (2 * torch.rand_like(x0) - 1.0) * float(eps)
        else:
            delta = torch.zeros_like(x0)

        delta = delta.clamp(-float(eps), float(eps)).detach()

        n0 = int(self.x.size(0))
        injected_ids = torch.arange(n0, n0 + int(n_inject), device=self.device, dtype=torch.long)

        injected_edges_final: list[tuple[int, int]] = []
        edge_index_adv = self.edge_index

        for _round in range(int(outer_rounds)):
            # current injected features
            x_inj = x0 + delta
            if self.clamp is not None:
                x_inj = torch.clamp(x_inj, min=self.clamp[0], max=self.clamp[1])

            # one-time edge selection under global budget
            injected_edges = self._select_edges_one_time(
                target_nodes=target_nodes,
                candidate_nodes=cand,
                x_inj=x_inj.detach(),
                injected_ids=injected_ids,
                edge_budget=int(edge_budget),
                allow_attack_attack_edges=allow_attack_attack_edges,
            )
            injected_edges_final = injected_edges

            add_ei = torch.tensor(injected_edges, dtype=torch.long, device=self.device).t().contiguous()
            edge_index_adv = torch.cat([self.edge_index, add_ei], dim=1)

            # inner PGD to optimize injected features for fixed edges
            for _ in range(int(inner_steps)):
                delta.requires_grad_(True)
                x_inj = x0 + delta
                x_adv = torch.cat([self.x, x_inj], dim=0)

                if self.clamp is not None:
                    x_adv = torch.clamp(x_adv, min=self.clamp[0], max=self.clamp[1])

                logits = self.model(x_adv, edge_index_adv)
                loss = _cw_targeted_loss(logits[target_nodes], desired_label=int(desired_label)).mean()

                grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
                # minimize loss => subtract sign(grad)
                delta = (delta - float(alpha_step) * grad.sign()).detach()
                delta = delta.clamp(-float(eps), float(eps))

                if early_stop:
                    with torch.no_grad():
                        pred = logits[target_nodes].argmax(dim=1)
                        if (pred == int(desired_label)).all():
                            break

        # build final perturbed graph
        x_inj = x0 + delta
        x_adv = torch.cat([self.x, x_inj], dim=0)
        if self.clamp is not None:
            x_adv = torch.clamp(x_adv, min=self.clamp[0], max=self.clamp[1])

        y_adv = torch.cat(
            [self.y, torch.full((int(n_inject),), -1, device=self.device, dtype=self.y.dtype)],
            dim=0
        )

        return InjectionResult(
            x_adv=x_adv.detach(),
            edge_index_adv=edge_index_adv.detach(),
            y_adv=y_adv.detach(),
            injected_node_ids=injected_ids.detach().cpu().tolist(),
            injected_edges=injected_edges_final,
        )
