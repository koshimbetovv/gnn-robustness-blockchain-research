import os
import sys
import time
import torch
import json
import math
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset, EllipticConfig
from src.datasets.ellipticpp_actors import EllipticPPActorsDataset, EllipticPPActorsConfig
from src.attacks.tdgia import TDGIAAttack
from src.attacks.model_forward import forward_logits, STATIC_MODELS
from src.utils.model_loader import load_model
from src.training.metrics import (
    get_split_mask, evaluate_logits_on_split,
    roc_auc_binary, mean_confidence_drop,
)
from src.utils.attack_targets import pick_target_nodes
from src.utils.seed import set_seed
from src.utils.tdgia_metrics import (
    budget_efficiency_metrics,
    attacked_target_ids_from_edges,
    attacked_target_outcome,
    coverage_metrics,
    mask_from_target_ids,
)

# ---------- victim / surrogate parameters ----------
# The victim is never used while crafting the injection. TDGIA is crafted on
# SURROGATE_MODEL_NAME and then transferred to MODEL_NAME, matching the original
# paper's black-box GIA setup.
MODEL_NAME = "gcn"          # victim: "gcn", "graphsage", "gat", "chronowave_gnn"
MODEL_DIR = "models/Elliptic"
RUN_ID = None

SURROGATE_MODEL_NAME = "gcn"      # surrogate used for attack construction
SURROGATE_MODEL_DIR = "models/Elliptic"
# Pin a normal repo GCN checkpoint. Leaving this as None may pick
# gcn_tdgia_paper_surrogate, whose architecture is handled by attack2.py.
SURROGATE_RUN_ID = "seed46_20260330_142208"

# Must match the dataset the checkpoints were trained on. Options:
#   "elliptic"           -> Elliptic (165 tx features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 wallet features)
DATASET = "elliptic"
SPLIT = "test"

# ---------- TDGIA hyperparameters ----------
EPS_FEATURE = 0.05     # None -> no local feature budget; else constrain injected features to base +/- eps
N_INJECT = 1
DEGREE_LIMIT = 545
BATCH_SIZE = 1
STEPS = 30
LR = 0.05
SMOOTH_R = 0.5
ALPHA_MU = 0.5
K1 = 1.0
K2 = 1.0
INIT = "randn"         # "zeros", "mean", "randn"
SIGMA_SCALE = 1.0
CLAMP = None           # None -> infer feature range from data; else e.g. (-3.0, 3.0)

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = False
TARGET_SELECTION_MODEL = "surrogate"  # "surrogate" matches black-box crafting; "victim" is eval-oracle mode
SEED = 0

# ---------- budget-aware target controls ----------
# "per_injection_degree": cap targets to degree_limit * multiplier. This is the
# most focused option: each injected node can directly connect to every selected
# target when multiplier=1.
# "total_edge_capacity": cap targets to n_inject * degree_limit * multiplier.
# "none": keep the original target count.
BUDGET_AWARE_TARGET_CAP_MODE = "per_injection_degree"
BUDGET_AWARE_TARGET_CAP_MULTIPLIER = 1.0
BUDGET_AWARE_CLEAN_CORRECT_FIRST = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(model_name: str, surrogate_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_tdgia_blackbox_budget_aware_{surrogate_name}_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _validate_static_model(name: str, role: str) -> str:
    name = name.lower()
    if name not in STATIC_MODELS:
        raise NotImplementedError(
            f"{role} model {name!r} is not a static-feature model supported by this TDGIA driver. "
            f"Temporal/sequence models require a different TDGIA threat model. "
            f"Supported here: {STATIC_MODELS}."
        )
    return name


def _budget_target_cap(n_candidates: int) -> int:
    n_candidates = int(n_candidates)
    if n_candidates <= 0:
        return 0

    mode = str(BUDGET_AWARE_TARGET_CAP_MODE).lower()
    if mode == "none":
        return n_candidates
    if mode == "per_injection_degree":
        base = int(DEGREE_LIMIT)
    elif mode == "total_edge_capacity":
        base = int(N_INJECT) * int(DEGREE_LIMIT)
    else:
        raise ValueError(
            "BUDGET_AWARE_TARGET_CAP_MODE must be one of "
            "'per_injection_degree', 'total_edge_capacity', or 'none'."
        )

    cap = int(math.ceil(float(BUDGET_AWARE_TARGET_CAP_MULTIPLIER) * float(max(base, 1))))
    return max(1, min(n_candidates, cap))


def _rank_targets_for_tdgia(
    atk: TDGIAAttack,
    target_nodes: torch.Tensor,
    attack_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_nodes.numel() == 0:
        return target_nodes, attack_labels, target_nodes.new_empty((0,), dtype=torch.float32)

    mu, _ = atk._defective_scores(
        x_curr=atk.x,
        edge_index_curr=atk.edge_index,
        time_step_curr=atk.time_step,
        target_nodes=target_nodes,
        surrogate_labels=attack_labels,
        degree_limit=DEGREE_LIMIT,
        alpha_mu=ALPHA_MU,
        k1=K1,
        k2=K2,
    )
    order = torch.argsort(mu, descending=True)
    return target_nodes[order], attack_labels[order], mu[order]


def select_budget_aware_targets(
    *,
    atk: TDGIAAttack,
    data,
    candidate_targets: torch.Tensor,
    selection_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    candidate_targets = candidate_targets.to(atk.device).long().view(-1)
    cap = _budget_target_cap(int(candidate_targets.numel()))
    candidate_labels = data.y[candidate_targets].long()

    pred_clean = selection_logits.argmax(dim=1)
    clean_correct_mask = pred_clean[candidate_targets] == candidate_labels
    n_clean_correct_candidates = int(clean_correct_mask.sum().item())

    if not BUDGET_AWARE_CLEAN_CORRECT_FIRST:
        ranked_targets, ranked_labels, ranked_mu = _rank_targets_for_tdgia(atk, candidate_targets, candidate_labels)
        selected_targets = ranked_targets[:cap]
        selected_mu = ranked_mu[:cap]
    else:
        primary_targets = candidate_targets[clean_correct_mask]
        primary_labels = candidate_labels[clean_correct_mask]
        secondary_targets = candidate_targets[~clean_correct_mask]
        secondary_labels = candidate_labels[~clean_correct_mask]

        primary_targets, _, primary_mu = _rank_targets_for_tdgia(atk, primary_targets, primary_labels)
        secondary_targets, _, secondary_mu = _rank_targets_for_tdgia(atk, secondary_targets, secondary_labels)

        selected_parts = []
        selected_mu_parts = []
        if primary_targets.numel() > 0:
            take_primary = min(cap, int(primary_targets.numel()))
            selected_parts.append(primary_targets[:take_primary])
            selected_mu_parts.append(primary_mu[:take_primary])
        else:
            take_primary = 0

        remaining = cap - take_primary
        if remaining > 0 and secondary_targets.numel() > 0:
            take_secondary = min(remaining, int(secondary_targets.numel()))
            selected_parts.append(secondary_targets[:take_secondary])
            selected_mu_parts.append(secondary_mu[:take_secondary])

        selected_targets = torch.cat(selected_parts) if selected_parts else candidate_targets[:0]
        selected_mu = torch.cat(selected_mu_parts) if selected_mu_parts else candidate_targets.new_empty((0,), dtype=torch.float32)

    if selected_targets.numel() > 0:
        n_clean_correct_selected = int((pred_clean[selected_targets] == data.y[selected_targets]).sum().item())
    else:
        n_clean_correct_selected = 0

    info = {
        "enabled": True,
        "cap_mode": BUDGET_AWARE_TARGET_CAP_MODE,
        "cap_multiplier": BUDGET_AWARE_TARGET_CAP_MULTIPLIER,
        "clean_correct_first": BUDGET_AWARE_CLEAN_CORRECT_FIRST,
        "n_candidate_targets": int(candidate_targets.numel()),
        "n_selected_targets": int(selected_targets.numel()),
        "n_targets_dropped": int(candidate_targets.numel() - selected_targets.numel()),
        "target_cap": int(cap),
        "selection_clean_correct_candidates": n_clean_correct_candidates,
        "selection_clean_correct_selected": n_clean_correct_selected,
        "selected_defective_score_mean": float(selected_mu.mean().item()) if selected_mu.numel() > 0 else 0.0,
        "selected_defective_score_min": float(selected_mu.min().item()) if selected_mu.numel() > 0 else 0.0,
        "selected_defective_score_max": float(selected_mu.max().item()) if selected_mu.numel() > 0 else 0.0,
    }
    return selected_targets, info


def main():
    victim_name = _validate_static_model(MODEL_NAME, "Victim")
    surrogate_name = _validate_static_model(SURROGATE_MODEL_NAME, "Surrogate")

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

    attack_dim = None
    rebuild_fn = None
    if victim_name == "chronowave_gnn" or surrogate_name == "chronowave_gnn":
        from src.datasets.chronowave_features import build_paper_features, make_consistent_rebuild
        build_paper_features(data)

    data = data.to(device)

    if victim_name == "chronowave_gnn" or surrogate_name == "chronowave_gnn":
        attack_dim = int(data.raw_feature_dim)
        rebuild_fn = make_consistent_rebuild(data)

    victim_model = load_model(
        victim_name,
        data.num_features,
        2,
        device=device,
        model_dir=MODEL_DIR,
        run_id=RUN_ID,
    )
    surrogate_model = load_model(
        surrogate_name,
        data.num_features,
        2,
        device=device,
        model_dir=SURROGATE_MODEL_DIR,
        run_id=SURROGATE_RUN_ID,
    )

    time_step = getattr(data, "time_step", None)

    split_mask = get_split_mask(data, SPLIT).to(device)
    with torch.no_grad():
        surrogate_logits_clean = forward_logits(surrogate_model, data.x, data.edge_index, time_step=time_step)

    if TARGET_SELECTION_MODEL not in ("surrogate", "victim"):
        raise ValueError(
            f"TARGET_SELECTION_MODEL must be 'surrogate' or 'victim', got {TARGET_SELECTION_MODEL!r}."
        )

    victim_logits_clean = None
    selection_logits = surrogate_logits_clean
    if TARGET_SELECTION_MODEL == "victim":
        with torch.no_grad():
            victim_logits_clean = forward_logits(victim_model, data.x, data.edge_index, time_step=time_step)
        selection_logits = victim_logits_clean

    # Target selection is an evaluation protocol choice. The attack construction
    # below only sees the selected node ids and the surrogate model. Keep the
    # default on surrogate logits to avoid target-model queries before transfer.
    candidate_targets = pick_target_nodes(
        data, selection_logits, split_mask,
        only_illicit=ATTACK_ONLY_ILLICIT,
        fraction=ATTACK_FRACTION,
        only_clean_correct=ONLY_CLEAN_CORRECT,
        seed=SEED,
        device=device,
    )
    if candidate_targets.numel() == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    atk = TDGIAAttack(
        surrogate_model,
        data,
        device,
        clamp=CLAMP,
        attack_dim=attack_dim,
        rebuild_fn=rebuild_fn,
    )

    targets, budget_info = select_budget_aware_targets(
        atk=atk,
        data=data,
        candidate_targets=candidate_targets,
        selection_logits=selection_logits,
    )
    if targets.numel() == 0:
        print("No budget-aware target nodes remain after capping.")
        return

    print(
        "Budget-aware target cap: "
        f"{budget_info['n_selected_targets']}/{budget_info['n_candidate_targets']} selected "
        f"(mode={budget_info['cap_mode']}, cap={budget_info['target_cap']})"
    )

    init_ref_mask = split_mask & (data.y == 0)
    init_ref_mask[targets] = False
    init_reference = init_ref_mask.nonzero(as_tuple=False).view(-1)
    if init_reference.numel() == 0:
        init_reference = None

    t_start = time.perf_counter()
    res = atk.attack(
        target_nodes=targets,
        attack_labels=data.y[targets].long(),
        n_inject=N_INJECT,
        degree_limit=DEGREE_LIMIT,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        lr=LR,
        smooth_r=SMOOTH_R,
        alpha_mu=ALPHA_MU,
        k1=K1,
        k2=K2,
        init=INIT,
        reference_nodes=init_reference,
        sigma_scale=SIGMA_SCALE,
        eps_feature=EPS_FEATURE,
    )
    attack_time_seconds = float(time.perf_counter() - t_start)

    with torch.no_grad():
        if victim_logits_clean is None:
            victim_logits_clean = forward_logits(victim_model, data.x, data.edge_index, time_step=time_step)
        victim_logits_adv_full = forward_logits(
            victim_model, res.x_adv, res.edge_index_adv, time_step=res.time_step_adv
        )
        surrogate_logits_adv_full = forward_logits(
            surrogate_model, res.x_adv, res.edge_index_adv, time_step=res.time_step_adv
        )
    victim_logits_adv = victim_logits_adv_full[: data.num_nodes]
    surrogate_logits_adv = surrogate_logits_adv_full[: data.num_nodes]

    attacked_target_ids = attacked_target_ids_from_edges(res.injected_edges, targets)
    attacked_mask = mask_from_target_ids(data.num_nodes, attacked_target_ids, device) & split_mask
    target_outcome = attacked_target_outcome(
        y_true=data.y,
        victim_pred_clean=victim_logits_clean.argmax(1),
        victim_pred_adv=victim_logits_adv.argmax(1),
        surrogate_pred_clean=surrogate_logits_clean.argmax(1),
        surrogate_pred_adv=surrogate_logits_adv.argmax(1),
        attacked_mask=attacked_mask,
        target_unit="node",
    )

    roc_clean_split = roc_auc_binary(victim_logits_clean, data.y, split_mask)
    roc_adv_split = roc_auc_binary(victim_logits_adv, data.y, split_mask)
    clean_m_split = evaluate_logits_on_split(victim_logits_clean, data.y, split_mask, SPLIT)
    adv_m_split = evaluate_logits_on_split(victim_logits_adv, data.y, split_mask, SPLIT)

    surrogate_clean_m_split = evaluate_logits_on_split(surrogate_logits_clean, data.y, split_mask, SPLIT)
    surrogate_adv_m_split = evaluate_logits_on_split(surrogate_logits_adv, data.y, split_mask, SPLIT)

    conf_drop, n_used = mean_confidence_drop(
        data.y, victim_logits_clean, victim_logits_adv, attacked_mask, only_clean_correct=True
    )

    if len(res.injected_node_ids) > 0:
        pert_dim = int(atk.attack_dim)
        clean_inj = res.x_injected_base[:, :pert_dim]
        adv_inj = res.x_adv[res.injected_node_ids, :pert_dim]
        delta_inj = (adv_inj - clean_inj).float()
        per_node_l2 = torch.linalg.vector_norm(delta_inj, ord=2, dim=1)
        pert_l2_mean = float(per_node_l2.mean().item())
        pert_l2_n = int(per_node_l2.numel())
        avg_perturbation = float(delta_inj.abs().mean().item())
        avg_perturbation_signed = float(delta_inj.mean().item())
    else:
        pert_l2_mean, pert_l2_n = 0.0, 0
        avg_perturbation = 0.0
        avg_perturbation_signed = 0.0

    f1_pos_drop_split = float(clean_m_split.f1_pos - adv_m_split.f1_pos)
    recall_pos_drop_split = float(clean_m_split.recall_pos - adv_m_split.recall_pos)

    edges_added = int(res.edge_index_adv.size(1) - data.edge_index.size(1))
    n_injected_nodes = int(res.x_adv.size(0) - data.x.size(0))
    logical_edges_added = int(len(res.injected_edges))
    coverage = coverage_metrics(
        n_selected_targets=int(candidate_targets.numel()),
        n_budgeted_selected_targets=int(targets.numel()),
        n_attacked_targets=int(target_outcome["n_attacked_targets"]),
        target_unit="node",
    )
    budget_efficiency = budget_efficiency_metrics(
        n_attacked_targets=int(target_outcome["n_attacked_targets"]),
        n_success=int(target_outcome["asr"]["success"]),
        n_injected_nodes=n_injected_nodes,
        n_logical_injected_edges=logical_edges_added,
    )

    run_dir, ts = make_run_dir(victim_name, surrogate_name)
    config = {
        "timestamp": ts,
        "attack": "TDGIA-BlackBox-BudgetAware",
        "threat_model": {
            "access": "black_box_transfer",
            "attack_stage": "evasion",
            "victim_access_during_attack": "none",
            "surrogate": "separate_model",
        },
        "model_name": victim_name,
        "model_dir": MODEL_DIR,
        "run_id": RUN_ID,
        "surrogate_model_name": surrogate_name,
        "surrogate_model_dir": SURROGATE_MODEL_DIR,
        "surrogate_run_id": SURROGATE_RUN_ID,
        "dataset": DATASET,
        "split": SPLIT,
        "device": str(device),
        "attack_params": {
            "n_inject": N_INJECT,
            "degree_limit": DEGREE_LIMIT,
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "lr": LR,
            "smooth_r": SMOOTH_R,
            "alpha_mu": ALPHA_MU,
            "k1": K1,
            "k2": K2,
            "init": INIT,
            "sigma_scale": SIGMA_SCALE,
            "clamp": CLAMP,
            "eps_feature": EPS_FEATURE,
            "feature_bounds": "manual_clamp" if CLAMP is not None else "per_feature_data_minmax",
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
            "n_candidate_targets": int(candidate_targets.numel()),
            "n_targets": int(targets.numel()),
            "budget_aware": budget_info,
        },
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "TDGIA-BlackBox-BudgetAware",
        "model_name": victim_name,
        "surrogate_model_name": surrogate_name,
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
            "surrogate_transfer_source": {
                "split": SPLIT,
                "clean_metrics": vars(surrogate_clean_m_split),
                "adv_metrics": vars(surrogate_adv_m_split),
            },
        },
        "attack_effect": {
            "attack_time_seconds": attack_time_seconds,
            "budget_aware": budget_info,
            "target_outcome": target_outcome,
            "coverage": coverage,
            "budget_efficiency": budget_efficiency,
            "mean_confidence_drop": {
                "scope": "attacked_targets",
                "value": conf_drop,
                "n_clean_correct": n_used,
            },
            "perturbation": {
                "l2_on_injected_nodes": pert_l2_mean,
                "n_injected_nodes": pert_l2_n,
                "avg_abs": avg_perturbation,
                "avg_signed": avg_perturbation_signed,
                "scope": "injected_nodes",
                "aggregation": "mean_over_perturbable_feature_values",
            },
        },
        "injection": {
            "n_injected_nodes": n_injected_nodes,
            "edges_added": edges_added,
            "logical_injected_edges": logical_edges_added,
            "attacked_target_ids": attacked_target_ids,
            "injected_node_ids": res.injected_node_ids,
            "injected_edges": res.injected_edges,
        },
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    print()
    print(f"Saved black-box TDGIA run to {run_dir}")


if __name__ == "__main__":
    main()
