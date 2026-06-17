import os
import sys
import time
import torch
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset, EllipticConfig
from src.datasets.ellipticpp_actors import EllipticPPActorsDataset, EllipticPPActorsConfig
from src.attacks.nettack_adapted import (
    AdaptedNettackAttack, _build_A_hat, _make_undirected_no_self_loops,
)
from src.attacks.model_forward import forward_logits, STATIC_MODELS
from src.utils.model_loader import load_model
from src.utils.seed import set_seed
from src.training.metrics import (
    get_split_mask, evaluate_logits_on_split, attack_success_rate,
    roc_auc_binary, mean_confidence_drop, asr_pos_neg,
    mean_perturbation_l2_on_success,
)
from src.utils.attack_targets import pick_target_nodes

# ---------- attack parameters ----------
MODEL_NAME = "gcn"          # "gcn", "graphsage", "gat", "chronowave_gnn"
MODEL_DIR = "models/Elliptic"  # "models/Elliptic" or "models/Elliptic++"
# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (165 tx features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 wallet features)
DATASET = "elliptic"
SPLIT = "test"

# Adapted-NETTACK threat model:
#   N_STRUCT  : maximum number of edge ADDITIONS per target (no deletions).
#   EPS_FEAT  : per-target L2 budget for the closed-form continuous feature step.
#   CLAMP     : optional [lo, hi] clip applied to the final x_adv (e.g. (-3.0, 3.0)).
N_STRUCT = 5
EPS_FEAT = 0.05
CLAMP = None

# Power-law chi^2 unnoticeability test (Eqs. 6-9 in the paper).
D_MIN = 2
CHI2_TAU = 0.004
ENFORCE_DEGREE_CONSTRAINT = True

# Surrogate (linearized 2-layer GCN) training params.
SURROGATE_EPOCHS = 200
SURROGATE_LR = 0.01
SURROGATE_WEIGHT_DECAY = 5e-4

# ---------- target selection controls ----------
# Adapted NETTACK is binary illicit -> licit, so always attack only illicit nodes.
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = False
SEED = 0

# Progress logging during the per-target greedy loop.
VERBOSE = True
PROGRESS_EVERY = 50


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    #if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(model_name: str):
    """Create attacks/model_adapted_nettack_YYYYMMDD_HHMMSS/ and return (run_dir, timestamp)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_adapted_nettack_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def surrogate_logits(x: torch.Tensor, edge_index: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    num_nodes = int(x.size(0))
    edges_no_sl = _make_undirected_no_self_loops(edge_index, num_nodes)
    A_hat, _ = _build_A_hat(edges_no_sl, num_nodes, x.device)
    AX = torch.sparse.mm(A_hat, x)
    AAX = torch.sparse.mm(A_hat, AX)
    return AAX @ W


def main():
    if MODEL_NAME not in STATIC_MODELS:
        raise NotImplementedError(
            f"{MODEL_NAME!r} is not a static-feature model supported by this Adapted-NETTACK driver. "
            f"Temporal/sequence models (e.g. recgnn, evolvegcn_o) require a different threat "
            f"model. Supported here: {STATIC_MODELS}."
        )

    device = get_device()
    set_seed(SEED, deterministic=True, benchmark=False)
    print(f"Random seed set to {SEED} (deterministic=True)")
    if DATASET == "elliptic":
        data = EllipticDataset(EllipticConfig(filter_unknown=False)).get_data()
    elif DATASET == "ellipticpp_actors":
        data = EllipticPPActorsDataset(EllipticPPActorsConfig(filter_unknown=False)).get_data()
    else:
        raise ValueError(
            f"Unknown DATASET={DATASET!r}. Supported: 'elliptic', 'ellipticpp_actors'."
        )

    # Model-specific feature preprocessing (mirrors FGSM/PGD drivers). ChronoWaveGNN
    # was trained on [standardized_raw || standardized_Haar-level2(raw)]; rebuild_fn
    # keeps the derived slice consistent after the raw slice is perturbed.
    attack_dim = None
    rebuild_fn = None
    if MODEL_NAME == "chronowave_gnn":
        from src.datasets.chronowave_features import build_paper_features, make_consistent_rebuild
        build_paper_features(data)

    data = data.to(device)

    if MODEL_NAME == "chronowave_gnn":
        attack_dim = int(data.raw_feature_dim)
        rebuild_fn = make_consistent_rebuild(data)

    model = load_model(MODEL_NAME, data.num_features, 2, device=device, model_dir=MODEL_DIR)

    time_step = getattr(data, "time_step", None)

    split_mask = get_split_mask(data, SPLIT).to(device)
    with torch.no_grad():
        logits_clean = forward_logits(model, data.x, data.edge_index, time_step=time_step)

    atk = AdaptedNettackAttack(
        model, data, device,
        clamp=CLAMP,
        attack_dim=attack_dim,
        rebuild_fn=rebuild_fn,
        d_min=D_MIN,
        chi2_tau=CHI2_TAU,
        enforce_degree_constraint=ENFORCE_DEGREE_CONSTRAINT,
        surrogate_epochs=SURROGATE_EPOCHS,
        surrogate_lr=SURROGATE_LR,
        surrogate_weight_decay=SURROGATE_WEIGHT_DECAY,
        verbose=VERBOSE,
        progress_every=PROGRESS_EVERY,
    )
    with torch.no_grad():
        logits_surrogate_clean = surrogate_logits(data.x.float(), data.edge_index.long(), atk.W)

    targets = pick_target_nodes(
        data, logits_surrogate_clean, split_mask,
        only_illicit=ATTACK_ONLY_ILLICIT,
        fraction=ATTACK_FRACTION,
        only_clean_correct=ONLY_CLEAN_CORRECT,
        seed=SEED,
        device=device,
    )
    if targets.numel() == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    t_start = time.perf_counter()
    x_adv, edge_index_adv = atk.attack(targets, n_struct=N_STRUCT, eps_feat=EPS_FEAT)
    attack_time_seconds = float(time.perf_counter() - t_start)

    # NOTE (vs. FGSM/PGD): the victim is queried on the perturbed edge_index,
    # because Adapted-NETTACK also adds graph edges, not just feature deltas.
    with torch.no_grad():
        logits_adv = forward_logits(model, x_adv, edge_index_adv, time_step=time_step)
        logits_surrogate_adv = surrogate_logits(x_adv.float(), edge_index_adv.long(), atk.W)

    attack_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    attack_mask[targets] = True
    attack_mask = attack_mask & split_mask

    # Split-wide metrics (include message-passing collateral on non-attacked nodes).
    roc_clean_split = roc_auc_binary(logits_clean, data.y, split_mask)
    roc_adv_split = roc_auc_binary(logits_adv, data.y, split_mask)
    clean_m_split = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    adv_m_split = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)

    conf_drop, n_used = mean_confidence_drop(
        data.y, logits_clean, logits_adv, attack_mask, only_clean_correct=True
    )

    asr, ns, na = attack_success_rate(data.y, logits_clean.argmax(1), logits_adv.argmax(1), attack_mask)
    asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(data.y, logits_clean, logits_adv, attack_mask)
    surrogate_asr, surrogate_ns, surrogate_na = attack_success_rate(
        data.y,
        logits_surrogate_clean.argmax(1),
        logits_surrogate_adv.argmax(1),
        attack_mask,
    )
    surrogate_asr_p, surrogate_sp, surrogate_ap, surrogate_asr_n, surrogate_sn, surrogate_an = asr_pos_neg(
        data.y, logits_surrogate_clean, logits_surrogate_adv, attack_mask,
    )

    # Mean L2 perturbation over successful label flips on the perturbable raw slice.
    pert_dim = int(atk.attack_dim)
    clean_rows = data.x[targets, :pert_dim]
    adv_rows = x_adv[targets, :pert_dim]
    pred_clean_targets = logits_clean[targets].argmax(1)
    pred_adv_targets = logits_adv[targets].argmax(1)
    y_targets = data.y[targets].long()
    pert_l2_mean, pert_l2_n = mean_perturbation_l2_on_success(
        clean_rows, adv_rows, pred_clean_targets, pred_adv_targets, y_targets,
    )

    f1_pos_drop_split = float(clean_m_split.f1_pos - adv_m_split.f1_pos)
    recall_pos_drop_split = float(clean_m_split.recall_pos - adv_m_split.recall_pos)
    surrogate_clean_m_split = evaluate_logits_on_split(
        logits_surrogate_clean, data.y, split_mask, SPLIT,
    )
    surrogate_adv_m_split = evaluate_logits_on_split(
        logits_surrogate_adv, data.y, split_mask, SPLIT,
    )
    surrogate_f1_pos_drop_split = float(
        surrogate_clean_m_split.f1_pos - surrogate_adv_m_split.f1_pos
    )

    # Edge-addition accounting (Adapted-NETTACK specific).
    n_edges_orig = int(data.edge_index.size(1))
    n_edges_adv = int(edge_index_adv.size(1))
    n_unique_added = len(atk._added_edges)  # (v0, u) pairs, before symmetrization
    n_directed_added = n_edges_adv - n_edges_orig
    n_targets_with_edge = len({v0 for (v0, _) in atk._added_edges})

    print()
    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "AdaptedNETTACK",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "split": SPLIT,
        "device": str(device),
        "attack_params": {
            "n_struct": N_STRUCT,
            "eps_feat": EPS_FEAT,
            "clamp": CLAMP,
            "d_min": D_MIN,
            "chi2_tau": CHI2_TAU,
            "enforce_degree_constraint": ENFORCE_DEGREE_CONSTRAINT,
            "surrogate_epochs": SURROGATE_EPOCHS,
            "surrogate_lr": SURROGATE_LR,
            "surrogate_weight_decay": SURROGATE_WEIGHT_DECAY,
        },
        "target_selection": {
            "model": "surrogate_linearized_gcn",
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
            "n_targets": int(targets.numel()),
        },
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "AdaptedNETTACK",
        "model_name": MODEL_NAME,
        "dataset": DATASET,
        "classification": {
            "scope": "static",
            "aggregate": {
                "split": SPLIT,
                "n": clean_m_split.n_labeled,
                "f1_pos": {
                    "clean": clean_m_split.f1_pos,
                    "adv": adv_m_split.f1_pos,
                    "drop": f1_pos_drop_split,
                },
                "recall_pos": {
                    "clean": clean_m_split.recall_pos,
                    "adv": adv_m_split.recall_pos,
                    "drop": recall_pos_drop_split,
                },
                "f1_macro": {
                    "clean": clean_m_split.f1_macro,
                    "adv": adv_m_split.f1_macro,
                    "drop": float(clean_m_split.f1_macro - adv_m_split.f1_macro),
                },
                "roc_auc": {"clean": roc_clean_split, "adv": roc_adv_split},
                "clean_metrics": vars(clean_m_split),
                "adv_metrics": vars(adv_m_split),
            },
        },
        "attack_effect": {
            "n_targets": int(targets.numel()),
            "attack_mask_n": int(attack_mask.sum().item()),
            "attack_time_seconds": attack_time_seconds,
            "asr": {"value": asr, "success": ns, "attempted": na},
            "asr_pos_neg": {
                "asr_pos": asr_p, "succ_pos": sp, "attempted_pos": ap,
                "asr_neg": asr_n, "succ_neg": sn, "attempted_neg": an,
            },
            "mean_confidence_drop": {"value": conf_drop, "n": n_used},
            "perturbation_l2_on_success": {"value": pert_l2_mean, "n_flipped": pert_l2_n},
            "structural": {
                "n_edges_orig": n_edges_orig,
                "n_edges_adv": n_edges_adv,
                "n_directed_edges_added": n_directed_added,
                "n_unique_edges_added": n_unique_added,
                "n_targets_with_edge_added": n_targets_with_edge,
                "mean_edges_per_target": (
                    float(n_unique_added) / float(targets.numel())
                    if targets.numel() > 0 else 0.0
                ),
            },
        },
        "surrogate": {
            "classification": {
                "split": SPLIT,
                "n": surrogate_clean_m_split.n_labeled,
                "f1_pos": {
                    "clean": surrogate_clean_m_split.f1_pos,
                    "adv": surrogate_adv_m_split.f1_pos,
                    "drop": surrogate_f1_pos_drop_split,
                },
                "clean_metrics": vars(surrogate_clean_m_split),
                "adv_metrics": vars(surrogate_adv_m_split),
            },
            "attack_effect": {
                "asr": {
                    "value": surrogate_asr,
                    "success": surrogate_ns,
                    "attempted": surrogate_na,
                },
                "asr_pos_neg": {
                    "asr_pos": surrogate_asr_p,
                    "succ_pos": surrogate_sp,
                    "attempted_pos": surrogate_ap,
                    "asr_neg": surrogate_asr_n,
                    "succ_neg": surrogate_sn,
                    "attempted_neg": surrogate_an,
                },
            },
        },
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    print()
    print(f"Saved to attacks/")


if __name__ == "__main__":
    main()
