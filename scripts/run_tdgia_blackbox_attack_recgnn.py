import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch_geometric.nn import GCNConv

from src.datasets.recgnn_elliptic import RecGNNEllipticConfig, RecGNNEllipticDataset
from src.datasets.recgnn_ellipticpp_actors import (
    RecGNNEllipticPPActorsConfig, RecGNNEllipticPPActorsDataset,
)
from src.attacks.tdgia_recgnn import RecGNNTDGIAAttack
from src.utils.model_loader import load_model
from src.utils.seed import set_seed
from src.utils.tdgia_schedule import (
    degree_from_edge_index,
    score_based_injection_schedule,
    slot_scores_from_node_scores,
    tdgia_defective_scores_from_probability,
)
from src.training.metrics import (
    binary_classification_metrics, roc_auc_binary,
    mean_confidence_drop,
)
from src.utils.tdgia_metrics import (
    aggregate_attacked_target_outcomes,
    budget_efficiency_metrics,
    attacked_target_ids_from_edges,
    attacked_target_outcome,
    coverage_metrics,
    mask_from_target_ids,
)

# ---------- victim / surrogate parameters ----------
MODEL_NAME = "recgnn"
MODEL_DIR = "models/Elliptic"
RUN_ID = None

SURROGATE_MODEL_NAME = "temporal_gcn"
# SURROGATE_MODEL_NAME = "recgnn"
SURROGATE_MODEL_DIR = "models/Elliptic"
SURROGATE_RUN_ID = None

# Train a paper-style slice GCN surrogate on RecGNN train graphs instead of
# loading another recurrent checkpoint. Use SURROGATE_MODEL_NAME = "temporal_gcn"
# to enable.
TEMPORAL_GCN_HIDDEN_DIMS = (256, 128, 64)
TEMPORAL_GCN_DROPOUT = 0.5
TEMPORAL_GCN_EPOCHS = 150
TEMPORAL_GCN_LR = 0.005
TEMPORAL_GCN_WEIGHT_DECAY = 5e-4
TEMPORAL_GCN_LOG_EVERY = 50

# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (93 local + 2 ANF = 95 features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 local + 2 ANF = 57 features)
DATASET = "elliptic"

# ---------- TDGIA hyperparameters ----------
N_INJECT = 20
DEGREE_LIMIT = 3
BATCH_SIZE = 1
EPS_FEATURE = 0.05
STEPS = 30
LR = 0.05
SMOOTH_R = 0.7
ALPHA_MU = 0.5
K1 = 1.0
K2 = 1.0
INIT = "randn"
SIGMA_SCALE = 1.0
CLAMP = None
# Number of local (controllable) feature columns. The trailing 2 ANF columns
# are antecedent-neighbor-label counts derived from the graph and are not
# attacker-controllable for an injected node either, so they are excluded
# from `delta`. `None` selects the dataset-appropriate default.
ATTACK_DIM = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = False
TARGET_SELECTION_MODEL = "surrogate"  # "surrogate" matches black-box crafting; "victim" is eval-oracle mode
SEED = 0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(model_name: str, surrogate_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_tdgia_blackbox_{surrogate_name}_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_targets_graph(
    y: torch.Tensor,
    pred_clean: torch.Tensor,
    only_illicit: bool,
    only_clean_correct: bool,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"ATTACK_FRACTION must be in (0, 1], got {fraction}")
    idx = torch.arange(y.numel(), device=y.device)
    idx = idx[y[idx] != -1]
    if only_illicit:
        idx = idx[y[idx] == 1]
    if only_clean_correct and idx.numel() > 0:
        idx = idx[pred_clean[idx] == y[idx]]
    if idx.numel() == 0:
        return idx
    n = max(1, min(int(round(float(fraction) * float(idx.numel()))), int(idx.numel())))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=g)
    return idx[perm[:n].to(idx.device)]


def sample_injection_schedule(
    eligible_timesteps: list[int],
    timestep_capacity: dict[int, int],
    timestep_slot_scores: dict[int, list[float]],
    n_inject: int,
):
    """Allocate a feasible sequence-wide node budget to highest TDGIA-scored timesteps."""
    if n_inject <= 0 or not eligible_timesteps:
        return [], {}

    total_capacity = int(
        sum(
            int(timestep_capacity.get(int(t), 0))
            for t in eligible_timesteps
            if int(timestep_capacity.get(int(t), 0)) > 0
        )
    )
    if total_capacity < int(n_inject):
        raise ValueError(
            f"Global injection budget N_INJECT={int(n_inject)} is infeasible for RecGNN: "
            f"total available timestep capacity is {total_capacity}. Reduce N_INJECT "
            f"or use a model/checkpoint with larger m-LSTM state_rows."
        )

    return score_based_injection_schedule(
        eligible_timesteps,
        timestep_slot_scores,
        n_inject,
        timestep_capacity=timestep_capacity,
    )


class PaperStyleSliceGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dims=(256, 128, 64), dropout: float = 0.5):
        super().__init__()
        dims = [int(in_dim)] + [int(h) for h in hidden_dims] + [int(out_dim)]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = float(dropout)
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.norms.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def fit_train_graph_feature_transform(sequence, attack_dim: int, device):
    xs = []
    for graph in sequence.train_graphs:
        g = graph.to(device)
        if g.x.numel() > 0:
            xs.append(g.x.float()[:, :int(attack_dim)].detach())
    if not xs:
        raise RuntimeError("Cannot fit temporal GCN surrogate transform: no non-empty train graphs.")
    x_train = torch.cat(xs, dim=0)
    mean = x_train.mean(dim=0).to(device)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-12).to(device)
    return mean, scale


def apply_feature_transform(features: torch.Tensor, transform):
    if transform is None:
        return features
    mean, scale = transform
    return (features - mean.to(features.device)) / scale.to(features.device)


class _DummyEvolveLinear:
    def __init__(self):
        self._row_h = None
        self._row_c = None
        self._current_weight = None


class _DummyCell:
    def __init__(self):
        self.evolve_linear = _DummyEvolveLinear()


class _DummyMLSTM:
    def __init__(self, state_rows: int):
        self.state_rows = int(state_rows)
        self._h_state = None
        self._c_state = None
        self.cell = _DummyCell()


class RecTemporalGCNSurrogate(nn.Module):
    def __init__(self, gcn: PaperStyleSliceGCN, transform, attack_dim: int, state_rows: int):
        super().__init__()
        self.gcn = gcn
        self.transform = transform
        self.attack_dim = int(attack_dim)
        self.m_lstm = _DummyMLSTM(state_rows)

    def reset_sequence_state(self, device=None):
        del device
        self.m_lstm._h_state = None
        self.m_lstm._c_state = None
        self.m_lstm.cell.evolve_linear._row_h = None
        self.m_lstm.cell.evolve_linear._row_c = None
        self.m_lstm.cell.evolve_linear._current_weight = None

    def detach_sequence_state(self):
        return None

    def forward(self, x, edge_index):
        x_sur = apply_feature_transform(x[:, : self.attack_dim], self.transform)
        logits = self.gcn(x_sur, edge_index)
        return F.log_softmax(logits, dim=1)


def train_temporal_gcn_surrogate(
    sequence,
    attack_dim: int,
    state_rows: int,
    device,
    *,
    hidden_dims,
    dropout,
    epochs,
    lr,
    weight_decay,
    log_every,
):
    transform = fit_train_graph_feature_transform(sequence, attack_dim, device)
    model = PaperStyleSliceGCN(attack_dim, 2, hidden_dims=hidden_dims, dropout=dropout).to(device)

    labels_all = []
    for graph in sequence.train_graphs:
        g = graph.to(device)
        if g.y is not None and g.y.numel() > 0:
            labels_all.append(g.y[g.y != -1].detach())
    if not labels_all:
        raise RuntimeError("Cannot train temporal GCN surrogate: no labeled train nodes.")
    y_all = torch.cat(labels_all).long()
    counts = torch.bincount(y_all, minlength=2).float().to(device).clamp_min(1.0)
    class_weight = counts.sum() / (2.0 * counts)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    model.train()
    for ep in range(int(epochs)):
        loss_total = 0.0
        n_terms = 0
        for graph in sequence.train_graphs:
            g = graph.to(device)
            x = g.x.float()
            y = g.y
            mask = y != -1
            if int(mask.sum().item()) == 0:
                continue
            x_sur = apply_feature_transform(x[:, :int(attack_dim)], transform)
            logits = model(x_sur, g.edge_index.long())
            loss = F.cross_entropy(logits[mask], y[mask].long(), weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += float(loss.item())
            n_terms += 1

        if log_every and ((ep + 1) % int(log_every) == 0 or ep == 0 or ep + 1 == int(epochs)):
            avg = loss_total / max(n_terms, 1)
            print(f"  [temporal_gcn_surrogate] epoch {ep + 1}/{int(epochs)} loss={avg:.4f}")

    model.eval()
    wrapper = RecTemporalGCNSurrogate(model, transform, attack_dim, state_rows).to(device)
    wrapper.eval()
    return wrapper, {
        "type": "PaperStyleSliceGCN",
        "input_dim": int(attack_dim),
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "feature_transform": "train_graph_standard_scaler",
        "attack_dim": int(attack_dim),
        "state_rows": int(state_rows),
    }


def eval_recgnn_transfer(victim_atk, x, edge_index, x_adv, edge_index_adv, n_existing: int, n_injected: int):
    snap_pre = victim_atk._save_state()
    with torch.no_grad():
        log_probs_clean = victim_atk._forward(x, edge_index).detach()
        victim_atk.model.detach_sequence_state()
    snap_post = victim_atk._save_state()

    victim_atk._restore_state(snap_pre)
    victim_atk._zero_inject_rows(n_existing, n_injected)
    with torch.no_grad():
        log_probs_adv = victim_atk._forward(x_adv, edge_index_adv).detach()[:n_existing]
        victim_atk.model.detach_sequence_state()

    victim_atk._restore_state(snap_post)
    return log_probs_clean, log_probs_adv


def main():
    device = get_device()
    set_seed(SEED, deterministic=True, benchmark=False)

    from src.utils.model_loader import resolve_checkpoint
    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    train_temporal_gcn_surrogate_flag = SURROGATE_MODEL_NAME.lower() == "temporal_gcn"
    surrogate_ckpt_path = None
    surrogate_ckpt_cfg = {"model": {}, "data": {}}
    if not train_temporal_gcn_surrogate_flag:
        surrogate_ckpt_path, surrogate_ckpt_run_dir = resolve_checkpoint(
            SURROGATE_MODEL_NAME, model_dir=SURROGATE_MODEL_DIR, run_id=SURROGATE_RUN_ID
        )
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)
    if not train_temporal_gcn_surrogate_flag:
        with open(os.path.join(surrogate_ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
            surrogate_ckpt_cfg = json.load(f)

    if DATASET == "elliptic":
        data_cfg = RecGNNEllipticConfig(**ckpt_cfg["data"])
        print(f"Loading RecGNN Elliptic sequence (filter_unknown={data_cfg.filter_unknown}) ...")
        sequence = RecGNNEllipticDataset(data_cfg).get_sequence()
        default_attack_dim = 93
    elif DATASET == "ellipticpp_actors":
        data_cfg = RecGNNEllipticPPActorsConfig(**ckpt_cfg["data"])
        print(f"Loading RecGNN Elliptic++ Actors sequence (filter_unknown={data_cfg.filter_unknown}) ...")
        sequence = RecGNNEllipticPPActorsDataset(data_cfg).get_sequence()
        default_attack_dim = sequence.num_features - 2
    else:
        raise ValueError(
            f"Unknown DATASET={DATASET!r}. Supported: 'elliptic', 'ellipticpp_actors'."
        )

    attack_dim = default_attack_dim if ATTACK_DIM is None else int(ATTACK_DIM)

    model = load_model(MODEL_NAME, sequence.num_features, 2, device=device, model_dir=MODEL_DIR, run_id=RUN_ID)
    if train_temporal_gcn_surrogate_flag:
        print("Training temporal paper-style GCN surrogate on RecGNN train graphs ...")
        surrogate_model, surrogate_model_cfg = train_temporal_gcn_surrogate(
            sequence,
            attack_dim,
            int(model.m_lstm.state_rows),
            device,
            hidden_dims=TEMPORAL_GCN_HIDDEN_DIMS,
            dropout=TEMPORAL_GCN_DROPOUT,
            epochs=TEMPORAL_GCN_EPOCHS,
            lr=TEMPORAL_GCN_LR,
            weight_decay=TEMPORAL_GCN_WEIGHT_DECAY,
            log_every=TEMPORAL_GCN_LOG_EVERY,
        )
    else:
        surrogate_model = load_model(
            SURROGATE_MODEL_NAME, sequence.num_features, 2,
            device=device, model_dir=SURROGATE_MODEL_DIR, run_id=SURROGATE_RUN_ID,
        )
        surrogate_model_cfg = surrogate_ckpt_cfg["model"]
    print(
        f"Train graphs: {len(sequence.train_graphs)} | Test graphs: {len(sequence.test_graphs)} | "
        f"num_features={sequence.num_features} | attack_dim={attack_dim}"
    )

    victim_atk = RecGNNTDGIAAttack(model, device, attack_dim=attack_dim, clamp=CLAMP)
    surrogate_atk = RecGNNTDGIAAttack(surrogate_model, device, attack_dim=attack_dim, clamp=CLAMP)
    print("Priming sequence state over train graphs ...")
    victim_atk.prime(sequence.train_graphs)
    surrogate_atk.prime(sequence.train_graphs)

    print("Planning clean test trajectory and sampling global injection schedule ...")
    planned_targets: dict[int, torch.Tensor] = {}
    eligible_timesteps: list[int] = []
    timestep_capacity: dict[int, int] = {}
    timestep_slot_scores: dict[int, list[float]] = {}
    state_rows = min(int(model.m_lstm.state_rows), int(surrogate_model.m_lstm.state_rows))
    for graph in sequence.test_graphs:
        g = graph.to(device)
        x = g.x.float()
        edge_index = g.edge_index.long()
        y = g.y
        t = int(g.graph_timestep)
        timestep_capacity[t] = max(0, state_rows - int(x.size(0)))

        preview_atk = surrogate_atk if TARGET_SELECTION_MODEL == "surrogate" else victim_atk
        preview_model = surrogate_model if TARGET_SELECTION_MODEL == "surrogate" else model
        if TARGET_SELECTION_MODEL not in ("surrogate", "victim"):
            raise ValueError(f"TARGET_SELECTION_MODEL must be 'surrogate' or 'victim', got {TARGET_SELECTION_MODEL!r}.")
        with torch.no_grad():
            log_probs_preview = preview_atk._forward(x, edge_index).detach()
            preview_model.detach_sequence_state()
        if TARGET_SELECTION_MODEL == "surrogate":
            with torch.no_grad():
                _ = victim_atk._forward(x, edge_index).detach()
                model.detach_sequence_state()
        else:
            with torch.no_grad():
                _ = surrogate_atk._forward(x, edge_index).detach()
                surrogate_model.detach_sequence_state()
        pred_preview = log_probs_preview.argmax(dim=1)

        labeled_mask = y != -1
        if int(labeled_mask.sum().item()) == 0:
            planned_targets[t] = torch.empty(0, dtype=torch.long)
            timestep_slot_scores[t] = []
            continue

        targets = pick_targets_graph(
            y, pred_preview,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )
        planned_targets[t] = targets.detach().cpu()
        if targets.numel() > 0 and timestep_capacity[t] > 0:
            attack_labels = y[targets].long()
            log_pv = log_probs_preview[targets].gather(1, attack_labels.view(-1, 1)).view(-1)
            pv = log_pv.exp().clamp_min(1e-12)
            degree = degree_from_edge_index(int(x.size(0)), edge_index)[targets]
            mu = tdgia_defective_scores_from_probability(
                pv, degree, DEGREE_LIMIT, ALPHA_MU, K1, K2
            )
            timestep_slot_scores[t] = slot_scores_from_node_scores(
                mu, N_INJECT, DEGREE_LIMIT, max_slots=timestep_capacity[t]
            )
            eligible_timesteps.append(t)
        else:
            timestep_slot_scores[t] = []

    sampled_injection_timesteps, injection_allocation = sample_injection_schedule(
        eligible_timesteps, timestep_capacity, timestep_slot_scores, N_INJECT
    )
    print(
        f"Eligible attack timesteps: {len(eligible_timesteps)} | "
        f"score-selected injections: {len(sampled_injection_timesteps)} | "
        f"total capacity: {sum(timestep_capacity.get(t, 0) for t in eligible_timesteps)}"
    )

    print("Replaying test trajectory with the sampled global injection schedule ...")
    victim_atk.prime(sequence.train_graphs)
    surrogate_atk.prime(sequence.train_graphs)

    per_timestep = []
    pooled_y_true = []
    pooled_pred_clean = []
    pooled_pred_adv = []
    pooled_logits_clean = []
    pooled_logits_adv = []
    pooled_attack_mask = []
    pooled_injected_l2 = []
    pooled_injected_abs = []
    pooled_injected_signed = []
    total_injected_nodes = 0
    total_edges_added = 0
    attack_time_seconds = 0.0
    attacked_target_outcomes = []

    for graph in sequence.test_graphs:
        g = graph.to(device)
        x = g.x.float()
        edge_index = g.edge_index.long()
        y = g.y
        t = int(g.graph_timestep)
        assigned_n_inject = int(injection_allocation.get(t, 0))

        labeled_mask = y != -1
        if int(labeled_mask.sum().item()) == 0:
            surrogate_atk.attack_step(
                x, edge_index,
                torch.empty(0, dtype=torch.long, device=device),
                n_inject=0, degree_limit=DEGREE_LIMIT,
                batch_size=BATCH_SIZE, steps=STEPS, lr=LR, smooth_r=SMOOTH_R,
                alpha_mu=ALPHA_MU, k1=K1, k2=K2,
                init=INIT, sigma_scale=SIGMA_SCALE, eps_feature=EPS_FEATURE,
            )
            victim_log_probs_clean, victim_log_probs_adv = eval_recgnn_transfer(
                victim_atk, x, edge_index, x, edge_index, int(x.size(0)), 0
            )
            per_timestep.append({
                "t": t,
                "n_labeled": 0,
                "n_targets_available": 0,
                "n_budgeted_selected_targets": 0,
                "n_attacked_targets": 0,
                "n_injected_nodes_budgeted": assigned_n_inject,
                "n_injected_nodes": 0,
                "skipped": True,
            })
            continue

        targets = planned_targets.get(t, torch.empty(0, dtype=torch.long)).to(device)

        if targets.numel() == 0 or assigned_n_inject == 0:
            surrogate_atk.attack_step(
                x, edge_index,
                torch.empty(0, dtype=torch.long, device=device),
                n_inject=0, degree_limit=DEGREE_LIMIT,
                batch_size=BATCH_SIZE, steps=STEPS, lr=LR, smooth_r=SMOOTH_R,
                alpha_mu=ALPHA_MU, k1=K1, k2=K2,
                init=INIT, sigma_scale=SIGMA_SCALE, eps_feature=EPS_FEATURE,
            )
            victim_log_probs_clean, victim_log_probs_adv = eval_recgnn_transfer(
                victim_atk, x, edge_index, x, edge_index, int(x.size(0)), 0
            )
            pred_clean = victim_log_probs_clean.argmax(dim=1)
            per_timestep.append({
                "t": t,
                "n_labeled": int(labeled_mask.sum().item()),
                "n_targets_available": int(targets.numel()),
                "n_budgeted_selected_targets": 0,
                "n_attacked_targets": 0,
                "n_injected_nodes_budgeted": assigned_n_inject,
                "n_injected_nodes": 0,
                "skipped": True,
            })
            pooled_y_true.append(y[labeled_mask].detach().cpu())
            pooled_pred_clean.append(pred_clean[labeled_mask].detach().cpu())
            pooled_pred_adv.append(pred_clean[labeled_mask].detach().cpu())
            pooled_logits_clean.append(victim_log_probs_clean[labeled_mask].detach().cpu())
            pooled_logits_adv.append(victim_log_probs_adv[labeled_mask].detach().cpu())
            pooled_attack_mask.append(torch.zeros(int(labeled_mask.sum().item()), dtype=torch.bool))
            continue

        # Init reference: licit labeled nodes in this snapshot, excluding targets.
        init_ref_mask = (y == 0)
        init_ref_mask[targets] = False
        init_reference = init_ref_mask.nonzero(as_tuple=False).view(-1)
        if init_reference.numel() == 0:
            raise RuntimeError(
                f"No licit non-target nodes available at timestep {t} for feature init."
            )

        t0 = time.perf_counter()
        res = surrogate_atk.attack_step(
            x, edge_index, targets,
            n_inject=assigned_n_inject, degree_limit=DEGREE_LIMIT,
            batch_size=BATCH_SIZE, steps=STEPS, lr=LR, smooth_r=SMOOTH_R,
            alpha_mu=ALPHA_MU, k1=K1, k2=K2,
            init=INIT, reference_nodes=init_reference, sigma_scale=SIGMA_SCALE,
            eps_feature=EPS_FEATURE,
            attack_labels=y[targets].long(),
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        total_injected_nodes += len(res.injected_node_ids)
        total_edges_added += len(res.injected_edges)

        victim_log_probs_clean, victim_log_probs_adv = eval_recgnn_transfer(
            victim_atk, x, edge_index, res.x_adv, res.edge_index_adv,
            int(x.size(0)), len(res.injected_node_ids),
        )
        pred_clean = victim_log_probs_clean.argmax(dim=1)
        pred_adv = victim_log_probs_adv.argmax(dim=1)
        surrogate_pred_clean = res.log_probs_clean.argmax(dim=1)
        surrogate_pred_adv = res.log_probs_adv.argmax(dim=1)

        y_lab = y[labeled_mask]
        pred_clean_lab = pred_clean[labeled_mask]
        pred_adv_lab = pred_adv[labeled_mask]
        logits_clean_lab = victim_log_probs_clean[labeled_mask]
        logits_adv_lab = victim_log_probs_adv[labeled_mask]
        surrogate_pred_clean_lab = surrogate_pred_clean[labeled_mask]
        surrogate_pred_adv_lab = surrogate_pred_adv[labeled_mask]

        attacked_target_ids = attacked_target_ids_from_edges(res.injected_edges, targets)
        attacked_mask_full = mask_from_target_ids(y.numel(), attacked_target_ids, device)
        attacked_mask_lab = attacked_mask_full[labeled_mask]
        target_outcome = attacked_target_outcome(
            y_true=y_lab,
            victim_pred_clean=pred_clean_lab,
            victim_pred_adv=pred_adv_lab,
            surrogate_pred_clean=surrogate_pred_clean_lab,
            surrogate_pred_adv=surrogate_pred_adv_lab,
            attacked_mask=attacked_mask_lab,
            target_unit="timestep_node",
        )
        attacked_target_outcomes.append(target_outcome)

        full_mask_lab = torch.ones(y_lab.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean_lab, y_lab, full_mask_lab)
        roc_adv = roc_auc_binary(logits_adv_lab, y_lab, full_mask_lab)
        clean_m = binary_classification_metrics(y_lab, pred_clean_lab)
        adv_m = binary_classification_metrics(y_lab, pred_adv_lab)

        conf_drop, n_used = mean_confidence_drop(
            y_lab, logits_clean_lab, logits_adv_lab, attacked_mask_lab, only_clean_correct=True
        )

        if len(res.injected_node_ids) > 0:
            clean_inj = res.x_injected_base[:, :surrogate_atk.attack_dim]
            adv_inj = res.x_adv[res.injected_node_ids, :surrogate_atk.attack_dim]
            delta_inj = (adv_inj - clean_inj).float()
            per_node_l2 = torch.linalg.vector_norm(delta_inj, ord=2, dim=1)
            pert_l2_mean = float(per_node_l2.mean().item())
            pert_l2_n = int(per_node_l2.numel())
            avg_perturbation = float(delta_inj.abs().mean().item())
            avg_perturbation_signed = float(delta_inj.mean().item())
            pooled_injected_l2.append(per_node_l2.detach().cpu())
            pooled_injected_abs.append(delta_inj.abs().reshape(-1).detach().cpu())
            pooled_injected_signed.append(delta_inj.reshape(-1).detach().cpu())
        else:
            pert_l2_mean, pert_l2_n = 0.0, 0
            avg_perturbation = 0.0
            avg_perturbation_signed = 0.0

        f1_pos_drop = float(clean_m["f1_pos"] - adv_m["f1_pos"])
        recall_pos_drop = float(clean_m["recall_pos"] - adv_m["recall_pos"])

        per_timestep.append({
            "t": t,
            "n_labeled": int(y_lab.numel()),
            "n_targets_available": int(targets.numel()),
            "n_budgeted_selected_targets": int(targets.numel()),
            "n_attacked_targets": int(target_outcome["n_attacked_targets"]),
            "n_injected_nodes_budgeted": assigned_n_inject,
            "n_injected_nodes": len(res.injected_node_ids),
            "edges_added": len(res.injected_edges),
            "attacked_target_ids": attacked_target_ids,
            "target_outcome": target_outcome,
            "f1_pos_clean": clean_m["f1_pos"], "f1_pos_adv": adv_m["f1_pos"], "f1_pos_drop": f1_pos_drop,
            "f1_macro_clean": clean_m["f1_macro"], "f1_macro_adv": adv_m["f1_macro"],
            "recall_pos_clean": clean_m["recall_pos"], "recall_pos_adv": adv_m["recall_pos"], "recall_pos_drop": recall_pos_drop,
            "roc_auc_clean": roc_clean, "roc_auc_adv": roc_adv,
            "mean_confidence_drop": conf_drop, "conf_drop_n": n_used,
            "perturbation_l2_on_injected_nodes": pert_l2_mean,
            "perturbation_l2_n_injected_nodes": pert_l2_n,
            "avg_perturbation": avg_perturbation,
            "avg_perturbation_signed": avg_perturbation_signed,
        })

        pooled_y_true.append(y_lab.detach().cpu())
        pooled_pred_clean.append(pred_clean_lab.detach().cpu())
        pooled_pred_adv.append(pred_adv_lab.detach().cpu())
        pooled_logits_clean.append(logits_clean_lab.detach().cpu())
        pooled_logits_adv.append(logits_adv_lab.detach().cpu())
        pooled_attack_mask.append(attacked_mask_lab.detach().cpu())

        roc_c_show = roc_clean if roc_clean == roc_clean else float("nan")
        roc_a_show = roc_adv if roc_adv == roc_adv else float("nan")
        asr_obj = target_outcome["asr"]
        asr_value = asr_obj["value"]
        asr_text = "nan" if asr_value is None else f"{asr_value:.4f}"
        print(
            f"t={t:2d}  n_labeled={y_lab.numel():5d}  n_targets={targets.numel():4d}  "
            f"ASR={asr_text} ({asr_obj['success']}/{asr_obj['attempted_clean_correct']})  "
            f"F1_pos {clean_m['f1_pos']:.3f}->{adv_m['f1_pos']:.3f}  "
            f"ROC {roc_c_show:.3f}->{roc_a_show:.3f}"
        )

    # ---------- aggregation ----------
    concat_classification = None
    concat_attack = None
    if pooled_y_true:
        y_true = torch.cat(pooled_y_true)
        pred_clean_c = torch.cat(pooled_pred_clean)
        pred_adv_c = torch.cat(pooled_pred_adv)
        logits_clean_c = torch.cat(pooled_logits_clean)
        logits_adv_c = torch.cat(pooled_logits_adv)
        attack_mask_c = torch.cat(pooled_attack_mask)

        clean_c = binary_classification_metrics(y_true, pred_clean_c)
        adv_c = binary_classification_metrics(y_true, pred_adv_c)
        full_mask = torch.ones(y_true.numel(), dtype=torch.bool)
        roc_c_clean = roc_auc_binary(logits_clean_c, y_true, full_mask)
        roc_c_adv = roc_auc_binary(logits_adv_c, y_true, full_mask)

        conf_drop_c, conf_drop_n_c = mean_confidence_drop(
            y_true, logits_clean_c, logits_adv_c, attack_mask_c, only_clean_correct=True
        )

        if pooled_injected_l2:
            pert_l2_c = float(torch.cat(pooled_injected_l2).mean().item())
            pert_l2_n_c = int(torch.cat(pooled_injected_l2).numel())
            avg_perturbation_c = float(torch.cat(pooled_injected_abs).mean().item())
            avg_perturbation_signed_c = float(torch.cat(pooled_injected_signed).mean().item())
        else:
            pert_l2_c, pert_l2_n_c = 0.0, 0
            avg_perturbation_c, avg_perturbation_signed_c = 0.0, 0.0

        f1_pos_drop_c = float(clean_c["f1_pos"] - adv_c["f1_pos"])
        recall_pos_drop_c = float(clean_c["recall_pos"] - adv_c["recall_pos"])

        concat_classification = {
            "n_total": int(y_true.numel()),
            "f1_pos_clean": clean_c["f1_pos"], "f1_pos_adv": adv_c["f1_pos"], "f1_pos_drop": f1_pos_drop_c,
            "f1_macro_clean": clean_c["f1_macro"], "f1_macro_adv": adv_c["f1_macro"],
            "recall_pos_clean": clean_c["recall_pos"], "recall_pos_adv": adv_c["recall_pos"], "recall_pos_drop": recall_pos_drop_c,
            "roc_auc_clean": roc_c_clean, "roc_auc_adv": roc_c_adv,
        }
        n_selected_targets_total = int(sum(item.get("n_targets_available", 0) for item in per_timestep))
        n_budgeted_selected_targets_total = int(
            sum(item.get("n_budgeted_selected_targets", 0) for item in per_timestep)
        )
        n_eligible_timesteps = int(sum(1 for item in per_timestep if item.get("n_targets_available", 0) > 0))
        n_budgeted_timesteps = int(sum(1 for item in per_timestep if item.get("n_injected_nodes_budgeted", 0) > 0))
        target_outcome_c = aggregate_attacked_target_outcomes(
            attacked_target_outcomes,
            target_unit="timestep_node",
        )
        coverage_c = coverage_metrics(
            n_selected_targets=n_selected_targets_total,
            n_budgeted_selected_targets=n_budgeted_selected_targets_total,
            n_attacked_targets=int(target_outcome_c["n_attacked_targets"]),
            target_unit="timestep_node",
        )
        budget_efficiency_c = budget_efficiency_metrics(
            n_attacked_targets=int(target_outcome_c["n_attacked_targets"]),
            n_success=int(target_outcome_c["asr"]["success"]),
            n_injected_nodes=total_injected_nodes,
            n_logical_injected_edges=total_edges_added,
        )
        concat_attack = {
            "target_outcome": target_outcome_c,
            "coverage": coverage_c,
            "budget_efficiency": budget_efficiency_c,
            "n_eligible_timesteps": n_eligible_timesteps,
            "n_budgeted_timesteps": n_budgeted_timesteps,
            "mean_confidence_drop": {
                "scope": "attacked_targets",
                "value": conf_drop_c,
                "n_clean_correct": conf_drop_n_c,
            },
            "perturbation_l2_on_injected_nodes": pert_l2_c,
            "perturbation_l2_n_injected_nodes": pert_l2_n_c,
            "avg_perturbation": avg_perturbation_c,
            "avg_perturbation_signed": avg_perturbation_signed_c,
        }

    print()
    print(f"TDGIA RecGNN | total injected nodes={total_injected_nodes}, edges={total_edges_added}")
    if concat_classification is not None and concat_attack is not None:
        asr_obj = concat_attack["target_outcome"]["asr"]
        asr_value = asr_obj["value"]
        asr_text = "nan" if asr_value is None else f"{asr_value:.4f}"
        coverage_value = concat_attack["coverage"]["attacked_target_coverage"]
        coverage_text = "nan" if coverage_value is None else f"{coverage_value:.4f}"
        print(f"[concatenated across all test timesteps, n={concat_classification['n_total']}]")
        print(
            f"  Attacked-target ASR: {asr_text} "
            f"({asr_obj['success']}/{asr_obj['attempted_clean_correct']})"
        )
        print(f"  Attacked-target coverage: {coverage_text}")
        print(f"  F1_pos: {concat_classification['f1_pos_clean']:.4f} -> {concat_classification['f1_pos_adv']:.4f}")
    print(f"Attack time (total over test timesteps): {attack_time_seconds:.4f} s")

    run_dir, ts = make_run_dir(MODEL_NAME, SURROGATE_MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "TDGIA-BlackBox",
        "threat_model": {
            "access": "black_box_transfer",
            "attack_stage": "evasion",
            "victim_access_during_attack": "none" if TARGET_SELECTION_MODEL == "surrogate" else "target_selection_only",
            "surrogate": "trained_temporal_gcn" if train_temporal_gcn_surrogate_flag else "separate_temporal_model",
        },
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "run_id": RUN_ID,
        "surrogate_model_name": SURROGATE_MODEL_NAME,
        "surrogate_model_dir": SURROGATE_MODEL_DIR,
        "surrogate_run_id": SURROGATE_RUN_ID,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "surrogate_checkpoint_path": surrogate_ckpt_path,
        "device": str(device),
        "attack_params": {
            "n_inject": N_INJECT, "degree_limit": DEGREE_LIMIT, "batch_size": BATCH_SIZE,
            "eps_feature": EPS_FEATURE, "steps": STEPS, "lr": LR, "smooth_r": SMOOTH_R,
            "alpha_mu": ALPHA_MU, "k1": K1, "k2": K2,
            "init": INIT, "sigma_scale": SIGMA_SCALE, "clamp": CLAMP,
            "attack_dim": attack_dim,
            "budget_mode": "global_sequence",
            "persistence": "non_persistent",
            "feature_bounds": "manual_clamp" if CLAMP is not None else "per_feature_data_minmax",
            "static_surrogate_feature_transform": (
                "train_graph_standard_scaler"
                if train_temporal_gcn_surrogate_flag
                else "none"
            ),
            "crafting_model": "surrogate_model",
            "evaluation_model": "victim_model",
            "attack_label_source": "true_target_labels",
        },
        "target_selection": {
            "selection_logits": f"{TARGET_SELECTION_MODEL}_clean_logits",
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
        },
        "data": ckpt_cfg["data"],
        "model_hparams": ckpt_cfg["model"],
        "surrogate_data": surrogate_ckpt_cfg["data"],
        "surrogate_model_hparams": surrogate_model_cfg,
        "injection_schedule": {
            "strategy": "top_tdgia_defective_score",
            "eligible_timesteps": eligible_timesteps,
            "sampled_timesteps": sampled_injection_timesteps,
            "selected_timesteps": sampled_injection_timesteps,
            "timestep_capacity": [
                {"t": int(t), "capacity": int(timestep_capacity[t])}
                for t in sorted(timestep_capacity)
            ],
            "timestep_defective_scores": [
                {
                    "t": int(t),
                    "top_slot_score": float(timestep_slot_scores.get(int(t), [0.0])[0])
                    if timestep_slot_scores.get(int(t), [])
                    else 0.0,
                    "slot_scores": [float(s) for s in timestep_slot_scores.get(int(t), [])],
                }
                for t in sorted(timestep_slot_scores)
            ],
            "allocation_by_timestep": [
                {"t": int(t), "n_inject": int(injection_allocation[t])}
                for t in sorted(injection_allocation)
            ],
        },
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "TDGIA-BlackBox",
        "model_name": MODEL_NAME,
        "surrogate_model_name": SURROGATE_MODEL_NAME,
        "dataset": DATASET,
        "classification": {
            "scope": "temporal",
            "aggregate_concat": concat_classification,
        },
        "attack_effect": {
            "attack_time_seconds": attack_time_seconds,
            "target_outcome": concat_attack["target_outcome"] if concat_attack else None,
            "coverage": concat_attack["coverage"] if concat_attack else None,
            "budget_efficiency": concat_attack["budget_efficiency"] if concat_attack else None,
            "timesteps": {
                "n_eligible_timesteps": concat_attack["n_eligible_timesteps"] if concat_attack else 0,
                "n_budgeted_timesteps": concat_attack["n_budgeted_timesteps"] if concat_attack else 0,
            },
            "mean_confidence_drop": concat_attack["mean_confidence_drop"] if concat_attack else None,
            "perturbation": {
                "l2_on_injected_nodes": concat_attack["perturbation_l2_on_injected_nodes"] if concat_attack else 0.0,
                "n_injected_nodes": concat_attack["perturbation_l2_n_injected_nodes"] if concat_attack else 0,
                "avg_abs": concat_attack["avg_perturbation"] if concat_attack else 0.0,
                "avg_signed": concat_attack["avg_perturbation_signed"] if concat_attack else 0.0,
            },
        },
        "injection": {
            "total_injected_nodes": total_injected_nodes,
            "total_edges_added": total_edges_added,
            "total_logical_injected_edges": total_edges_added,
            "budget_mode": "global_sequence",
            "schedule_strategy": "top_tdgia_defective_score",
            "n_inject_total_budget": int(N_INJECT),
            "n_selected_targets_total": int(sum(item.get("n_targets_available", 0) for item in per_timestep)),
            "n_budgeted_selected_targets_total": int(sum(item.get("n_budgeted_selected_targets", 0) for item in per_timestep)),
            "n_attacked_targets_total": (
                int(concat_attack["target_outcome"]["n_attacked_targets"])
                if concat_attack
                else 0
            ),
            "eligible_timesteps": eligible_timesteps,
            "sampled_timesteps": sampled_injection_timesteps,
            "selected_timesteps": sampled_injection_timesteps,
            "timestep_capacity": [
                {"t": int(t), "capacity": int(timestep_capacity[t])}
                for t in sorted(timestep_capacity)
            ],
            "timestep_defective_scores": [
                {
                    "t": int(t),
                    "top_slot_score": float(timestep_slot_scores.get(int(t), [0.0])[0])
                    if timestep_slot_scores.get(int(t), [])
                    else 0.0,
                    "slot_scores": [float(s) for s in timestep_slot_scores.get(int(t), [])],
                }
                for t in sorted(timestep_slot_scores)
            ],
            "allocation_by_timestep": [
                {"t": int(t), "n_inject": int(injection_allocation[t])}
                for t in sorted(injection_allocation)
            ],
        },
        "per_timestep": per_timestep,
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
