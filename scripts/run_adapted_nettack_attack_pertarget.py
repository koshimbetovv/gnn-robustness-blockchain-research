import os
import sys
import time
import torch
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.adapted_nettack_pertarget_utils import (
    diagnostics_summary,
    perturbation_summary,
    summarize_per_target_entries,
    target_outcome_branch,
    transfer_summary,
)
from src.datasets.elliptic import EllipticDataset, EllipticConfig
from src.datasets.ellipticpp_actors import EllipticPPActorsDataset, EllipticPPActorsConfig
from src.attacks.nettack_adapted import (
    AdaptedNettackAttack,
    _build_A_hat,
    _build_A_hat_directed,
    _make_directed_no_self_loops,
    _make_undirected_no_self_loops,
)
from src.attacks.model_forward import forward_logits, STATIC_MODELS
from src.utils.model_loader import load_model
from src.utils.seed import set_seed
from src.training.metrics import (
    get_split_mask, evaluate_logits_on_split,
    roc_auc_binary, mean_confidence_drop,
)
from src.utils.attack_targets import pick_target_nodes

# ---------- attack parameters ----------
MODEL_NAME = "graphsage"          # "gcn", "graphsage", "gat", "chronowave_gnn"
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
N_STRUCT = 1
EPS_FEAT = 0.05
CLAMP = None

# Static Elliptic/Elliptic++ edge_index is directed. Directed mode adds one
# incoming edge u -> target per structural perturbation. Use "undirected" to
# recover the original Nettack-style symmetrized threat model.
STRUCTURE_MODE = "directed"  # "directed" or "undirected"
DIRECTED_EDGE_DIRECTION = "incoming_to_target"

# Power-law chi^2 unnoticeability test (Eqs. 6-9 in the paper).
D_MIN = 2
CHI2_TAU = 0.04
ENFORCE_DEGREE_CONSTRAINT = True

# Surrogate (linearized 2-layer GCN) training params.
SURROGATE_EPOCHS = 200
SURROGATE_LR = 0.01
SURROGATE_WEIGHT_DECAY = 5e-4

# ---------- target selection controls ----------
# Adapted NETTACK is binary illicit -> licit, so always attack only illicit nodes.
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 0.2
ONLY_CLEAN_CORRECT = False
SEED = 0

# Progress logging across independent per-target runs.
VERBOSE = True
PROGRESS_EVERY = 50
SAVE_PER_TARGET_DETAILS = False


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    # if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(model_name: str):
    """Create attacks/model_adapted_nettack_pertarget_YYYYMMDD_HHMMSS/."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_adapted_nettack_pertarget_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def surrogate_logits(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    W: torch.Tensor,
    structure_mode: str,
) -> torch.Tensor:
    num_nodes = int(x.size(0))
    if structure_mode == "directed":
        edges_no_sl = _make_directed_no_self_loops(edge_index, num_nodes)
        A_hat, _ = _build_A_hat_directed(edges_no_sl, num_nodes, x.device)
    elif structure_mode == "undirected":
        edges_no_sl = _make_undirected_no_self_loops(edge_index, num_nodes)
        A_hat, _ = _build_A_hat(edges_no_sl, num_nodes, x.device)
    else:
        raise ValueError(f"Unknown structure_mode={structure_mode!r}.")
    AX = torch.sparse.mm(A_hat, x)
    AAX = torch.sparse.mm(A_hat, AX)
    return AAX @ W


def _nanmean(values):
    vals = [float(v) for v in values if float(v) == float(v)]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _target_prob(logits: torch.Tensor, node: int, label: int) -> float:
    probs = torch.softmax(logits[int(node)], dim=0)
    return float(probs[int(label)].item())


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
        structure_mode=STRUCTURE_MODE,
        directed_edge_direction=DIRECTED_EDGE_DIRECTION,
        verbose=False,
        progress_every=PROGRESS_EVERY,
    )
    with torch.no_grad():
        logits_surrogate_clean = surrogate_logits(
            data.x.float(), data.edge_index.long(), atk.W, STRUCTURE_MODE,
        )

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

    clean_labeled_m = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    roc_clean_labeled = roc_auc_binary(logits_clean, data.y, split_mask)
    surrogate_clean_labeled_m = evaluate_logits_on_split(
        logits_surrogate_clean, data.y, split_mask, SPLIT,
    )
    surrogate_roc_clean_labeled = roc_auc_binary(logits_surrogate_clean, data.y, split_mask)

    print(f"Selected {int(targets.numel())} targets for independent per-target NETTACK runs.")

    per_target = []
    target_y = []
    target_pred_clean = []
    target_pred_adv = []
    target_logits_clean = []
    target_logits_adv = []
    target_pred_surrogate_clean = []
    target_pred_surrogate_adv = []
    target_logits_surrogate_clean = []
    target_logits_surrogate_adv = []
    split_adv_f1_pos = []
    split_adv_recall_pos = []
    split_adv_f1_macro = []
    split_adv_roc_auc = []

    n_edges_orig = int(data.edge_index.size(1))
    n_directed_added_total = 0
    n_targets_with_edge_total = 0
    attack_time_seconds = 0.0
    pert_dim = int(atk.attack_dim)

    for i, target in enumerate(targets.tolist()):
        target_tensor = torch.tensor([int(target)], dtype=torch.long, device=device)

        atk._added_edges = []
        t_start = time.perf_counter()
        x_adv, edge_index_adv = atk.attack(target_tensor, n_struct=N_STRUCT, eps_feat=EPS_FEAT)
        attack_time_seconds += float(time.perf_counter() - t_start)

        with torch.no_grad():
            logits_adv = forward_logits(model, x_adv, edge_index_adv, time_step=time_step)
            logits_surrogate_adv = surrogate_logits(
                x_adv.float(), edge_index_adv.long(), atk.W, STRUCTURE_MODE,
            )

        adv_m_split = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)
        roc_adv_split = roc_auc_binary(logits_adv, data.y, split_mask)
        split_adv_f1_pos.append(adv_m_split.f1_pos)
        split_adv_recall_pos.append(adv_m_split.recall_pos)
        split_adv_f1_macro.append(adv_m_split.f1_macro)
        split_adv_roc_auc.append(roc_adv_split)

        n_edges_adv = int(edge_index_adv.size(1))
        n_directed_added = n_edges_adv - n_edges_orig
        n_targets_with_edge = len(set(getattr(atk, "_added_edge_target_nodes", [])))
        n_directed_added_total += n_directed_added
        n_targets_with_edge_total += n_targets_with_edge

        y_val = int(data.y[target].item())
        pred_clean = int(logits_clean[target].argmax().item())
        pred_adv = int(logits_adv[target].argmax().item())
        pred_surrogate_clean = int(logits_surrogate_clean[target].argmax().item())
        pred_surrogate_adv = int(logits_surrogate_adv[target].argmax().item())
        clean_correct = pred_clean == y_val
        surrogate_clean_correct = pred_surrogate_clean == y_val
        success = clean_correct and pred_adv != y_val
        surrogate_success = surrogate_clean_correct and pred_surrogate_adv != y_val

        clean_row = data.x[target, :pert_dim].detach()
        adv_row = x_adv[target, :pert_dim].detach()
        feature_l2 = float(torch.linalg.vector_norm((adv_row - clean_row).float(), ord=2).item())
        prob_clean = _target_prob(logits_clean, target, y_val)
        prob_adv = _target_prob(logits_adv, target, y_val)
        surrogate_prob_clean = _target_prob(logits_surrogate_clean, target, y_val)
        surrogate_prob_adv = _target_prob(logits_surrogate_adv, target, y_val)

        target_y.append(y_val)
        target_pred_clean.append(pred_clean)
        target_pred_adv.append(pred_adv)
        target_logits_clean.append(logits_clean[target].detach().cpu())
        target_logits_adv.append(logits_adv[target].detach().cpu())
        target_pred_surrogate_clean.append(pred_surrogate_clean)
        target_pred_surrogate_adv.append(pred_surrogate_adv)
        target_logits_surrogate_clean.append(logits_surrogate_clean[target].detach().cpu())
        target_logits_surrogate_adv.append(logits_surrogate_adv[target].detach().cpu())

        per_target.append({
            "target": int(target),
            "y": y_val,
            "victim": {
                "pred_clean": pred_clean,
                "pred_adv": pred_adv,
                "clean_correct": bool(clean_correct),
                "success": bool(success),
                "true_prob_clean": prob_clean,
                "true_prob_adv": prob_adv,
                "confidence_drop": float(prob_clean - prob_adv),
            },
            "surrogate": {
                "pred_clean": pred_surrogate_clean,
                "pred_adv": pred_surrogate_adv,
                "clean_correct": bool(surrogate_clean_correct),
                "success": bool(surrogate_success),
                "true_prob_clean": surrogate_prob_clean,
                "true_prob_adv": surrogate_prob_adv,
                "confidence_drop": float(surrogate_prob_clean - surrogate_prob_adv),
            },
            "perturbation": {
                "feature_l2": feature_l2,
                "n_edges_orig": n_edges_orig,
                "n_edges_adv": n_edges_adv,
                "n_directed_edges_added": n_directed_added,
                "n_targets_with_edge_added": n_targets_with_edge,
            },
            "split_adv_metrics": {
                "f1_pos": adv_m_split.f1_pos,
                "recall_pos": adv_m_split.recall_pos,
                "f1_macro": adv_m_split.f1_macro,
                "roc_auc": roc_adv_split,
            },
        })

        if VERBOSE and ((i + 1) == 1 or (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == targets.numel()):
            print(
                f"  [adapted-nettack-pertarget] {i + 1}/{int(targets.numel())} targets done; "
                f"last_target={int(target)} success={bool(success)} edges+={n_directed_added}"
            )

    y_t = torch.tensor(target_y, dtype=torch.long)
    pred_clean_t = torch.tensor(target_pred_clean, dtype=torch.long)
    pred_adv_t = torch.tensor(target_pred_adv, dtype=torch.long)
    logits_clean_t = torch.stack(target_logits_clean, dim=0)
    logits_adv_t = torch.stack(target_logits_adv, dim=0)
    pred_surrogate_clean_t = torch.tensor(target_pred_surrogate_clean, dtype=torch.long)
    pred_surrogate_adv_t = torch.tensor(target_pred_surrogate_adv, dtype=torch.long)
    logits_surrogate_clean_t = torch.stack(target_logits_surrogate_clean, dim=0)
    logits_surrogate_adv_t = torch.stack(target_logits_surrogate_adv, dim=0)
    target_mask = torch.ones(y_t.numel(), dtype=torch.bool)
    per_target_summary = summarize_per_target_entries(per_target)

    conf_drop, n_used = mean_confidence_drop(
        y_t, logits_clean_t, logits_adv_t, target_mask, only_clean_correct=True
    )

    surrogate_conf_drop, surrogate_n_used = mean_confidence_drop(
        y_t, logits_surrogate_clean_t, logits_surrogate_adv_t, target_mask, only_clean_correct=True
    )

    victim_outcome = target_outcome_branch(
        y=y_t,
        pred_clean=pred_clean_t,
        pred_adv=pred_adv_t,
        logits_clean=logits_clean_t,
        logits_adv=logits_adv_t,
        summary=per_target_summary["victim"],
        confidence_drop_value=conf_drop,
        confidence_drop_n=n_used,
    )
    surrogate_outcome = target_outcome_branch(
        y=y_t,
        pred_clean=pred_surrogate_clean_t,
        pred_adv=pred_surrogate_adv_t,
        logits_clean=logits_surrogate_clean_t,
        logits_adv=logits_surrogate_adv_t,
        summary=per_target_summary["surrogate"],
        confidence_drop_value=surrogate_conf_drop,
        confidence_drop_n=surrogate_n_used,
    )
    target_outcome = {
        "n_targets": int(targets.numel()),
        "victim": victim_outcome,
        "surrogate": surrogate_outcome,
        "transfer": transfer_summary(
            y=y_t,
            victim_pred_clean=pred_clean_t,
            victim_pred_adv=pred_adv_t,
            surrogate_pred_clean=pred_surrogate_clean_t,
            surrogate_pred_adv=pred_surrogate_adv_t,
            victim_asr=victim_outcome["asr"],
            surrogate_asr=surrogate_outcome["asr"],
        ),
    }
    perturbation = perturbation_summary(
        per_target_summary=per_target_summary,
        structure={
            "mode": "independent_per_target_sum",
            "n_edges_orig_per_run": n_edges_orig,
            "n_directed_edges_added_total": n_directed_added_total,
            "n_targets_with_edge_added_total": n_targets_with_edge_total,
            "mean_directed_edges_per_target": (
                float(n_directed_added_total) / float(targets.numel())
                if targets.numel() > 0 else 0.0
            ),
        },
    )
    diagnostics = diagnostics_summary(per_target_summary)

    print()
    run_dir, ts = make_run_dir(MODEL_NAME)
    clean_labeled_summary = {
        **vars(clean_labeled_m),
        "roc_auc": roc_clean_labeled,
    }
    surrogate_clean_labeled_summary = {
        **vars(surrogate_clean_labeled_m),
        "roc_auc": surrogate_roc_clean_labeled,
    }
    config = {
        "timestamp": ts,
        "attack": "AdaptedNETTACKPerTarget",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "split": SPLIT,
        "device": str(device),
        "execution": {
            "mode": "independent_per_target",
            "description": (
                "Each selected target is attacked from the clean graph independently; "
                "there is no single cumulative adversarial graph."
            ),
        },
        "outputs": {
            "metrics": "metrics.json",
            "config": "config.json",
            "save_per_target_details": SAVE_PER_TARGET_DETAILS,
            "per_target_details": (
                "per_target_details.json" if SAVE_PER_TARGET_DETAILS else None
            ),
        },
        "attack_params": {
            "n_struct": N_STRUCT,
            "eps_feat": EPS_FEAT,
            "clamp": CLAMP,
            "d_min": D_MIN,
            "chi2_tau": CHI2_TAU,
            "enforce_degree_constraint": ENFORCE_DEGREE_CONSTRAINT,
            "structure_mode": STRUCTURE_MODE,
            "directed_edge_direction": (
                DIRECTED_EDGE_DIRECTION if STRUCTURE_MODE == "directed" else None
            ),
            "degree_constraint_scope": (
                "in_degree" if STRUCTURE_MODE == "directed" else "undirected_degree"
            ),
            "same_timestep_edge_candidates": bool(atk.time_step is not None),
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
        "attack": "AdaptedNETTACKPerTarget",
        "model_name": MODEL_NAME,
        "dataset": DATASET,
        "classification": {
            "scope": "static_independent_per_target",
            "victim": {
                "labeled_clean": clean_labeled_summary,
                "labeled_adv_independent_mean": {
                    "n_runs": int(targets.numel()),
                    "f1_pos": _nanmean(split_adv_f1_pos),
                    "recall_pos": _nanmean(split_adv_recall_pos),
                    "f1_macro": _nanmean(split_adv_f1_macro),
                    "roc_auc": _nanmean(split_adv_roc_auc),
                },
            },
            "surrogate": {
                "labeled_clean": surrogate_clean_labeled_summary,
            },
        },
        "target_outcome": target_outcome,
        "perturbation": perturbation,
        "diagnostics": diagnostics,
        "runtime": {"attack_time_seconds": attack_time_seconds},
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    if SAVE_PER_TARGET_DETAILS:
        write_json(os.path.join(run_dir, "per_target_details.json"), per_target)

    asr_obj = victim_outcome["asr"]
    asr_text = "nan" if asr_obj["value"] is None else f"{asr_obj['value']:.4f}"
    print(
        f"Per-target ASR={asr_text} "
        f"({asr_obj['success']}/{asr_obj['attempted_clean_correct']})  "
        f"targets={int(targets.numel())}  "
        f"mean_edges/target={metrics['perturbation']['structure']['directed_edges_added']['mean_per_target']:.2f}"
    )
    print(f"Saved to {os.path.relpath(run_dir, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))}")


if __name__ == "__main__":
    main()
