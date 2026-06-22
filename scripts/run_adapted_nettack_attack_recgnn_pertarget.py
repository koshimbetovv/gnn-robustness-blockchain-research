import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.adapted_nettack_pertarget_utils import (
    PerTargetResults,
    make_pertarget_run_dir,
    write_json,
)
from src.datasets.recgnn_elliptic import RecGNNEllipticConfig, RecGNNEllipticDataset
from src.datasets.recgnn_ellipticpp_actors import (
    RecGNNEllipticPPActorsConfig,
    RecGNNEllipticPPActorsDataset,
)
from src.attacks.nettack_adapted_temporal import (
    AdaptedNettackTemporalAttack,
    linearized_surrogate_logits,
    train_surrogate_on_train_slices,
)
from src.utils.model_loader import load_model, resolve_checkpoint
from src.utils.seed import set_seed
from src.training.metrics import binary_classification_metrics, roc_auc_binary

# ---------- attack parameters ----------
MODEL_NAME = "recgnn"
MODEL_DIR = "models/Elliptic"  # "models/Elliptic" or "models/Elliptic++"
# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (165 tx features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 wallet features)
DATASET = "elliptic"  # "elliptic" or "ellipticpp_actors"
RUN_ID = None

# Adapted-NETTACK threat model:
#   N_STRUCT  : maximum number of edge ADDITIONS per target (no deletions).
#   EPS_FEAT  : per-target L2 budget for the closed-form continuous feature step.
#   CLAMP     : optional [lo, hi] clip applied to the final x_adv (e.g. (-3.0, 3.0)).
N_STRUCT = 1
EPS_FEAT = 0.05
CLAMP = None

# RecGNN edge_index is directed source -> target. Directed mode adds one
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

# Number of controllable feature columns. RecGNN appends 2 ANF columns derived
# from the graph; NETTACK only touches the leading raw slice.
ATTACK_DIM = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = False
SEED = 0

VERBOSE = True
PROGRESS_EVERY = 100


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_targets_graph(y, pred_clean, only_illicit, only_clean_correct, fraction, seed):
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


def _save_state(model):
    ml = model.m_lstm
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


def _restore_state(model, snap):
    ml = model.m_lstm
    ev = ml.cell.evolve_linear

    def _c(t):
        return t.detach().clone() if t is not None else None

    ml._h_state = _c(snap["h"])
    ml._c_state = _c(snap["c"])
    ev._row_h = _c(snap["ev_h"])
    ev._row_c = _c(snap["ev_c"])
    ev._current_weight = _c(snap["ev_w"])


def _entries_asr(entries, branch: str):
    attempted = sum(1 for item in entries if item[branch]["clean_correct"])
    success = sum(1 for item in entries if item[branch]["success"])
    return (float(success) / float(attempted) if attempted else 0.0, success, attempted)


def main():
    device = get_device()
    set_seed(SEED, deterministic=True, benchmark=False)
    print(f"Random seed set to {SEED} (deterministic=True)")

    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    if DATASET == "elliptic":
        data_cfg = RecGNNEllipticConfig(**ckpt_cfg["data"])
        sequence = RecGNNEllipticDataset(data_cfg).get_sequence()
        default_attack_dim = 93
    elif DATASET == "ellipticpp_actors":
        data_cfg = RecGNNEllipticPPActorsConfig(**ckpt_cfg["data"])
        sequence = RecGNNEllipticPPActorsDataset(data_cfg).get_sequence()
        default_attack_dim = sequence.num_features - 2
    else:
        raise ValueError(f"Unknown DATASET={DATASET!r}.")

    attack_dim = default_attack_dim if ATTACK_DIM is None else int(ATTACK_DIM)

    model = load_model(
        MODEL_NAME,
        sequence.num_features,
        2,
        device=device,
        model_dir=MODEL_DIR,
        run_id=RUN_ID,
    )
    print(
        f"Train graphs: {len(sequence.train_graphs)}  Test graphs: {len(sequence.test_graphs)}  "
        f"num_features={sequence.num_features}  attack_dim={attack_dim}"
    )

    print("Training surrogate (linearized GCN) on union of train timesteps ...")
    train_slices = []
    for g in sequence.train_graphs:
        g = g.to(device)
        train_mask = g.y != -1
        if int(train_mask.sum().item()) == 0:
            continue
        train_slices.append((g.x.float(), g.edge_index.long(), g.y.long(), train_mask))
    W = train_surrogate_on_train_slices(
        train_slices,
        num_features=sequence.num_features,
        num_classes=2,
        device=device,
        epochs=SURROGATE_EPOCHS,
        lr=SURROGATE_LR,
        weight_decay=SURROGATE_WEIGHT_DECAY,
        structure_mode=STRUCTURE_MODE,
    )

    atk = AdaptedNettackTemporalAttack(
        W=W,
        device=device,
        attack_dim=attack_dim,
        clamp=CLAMP,
        d_min=D_MIN,
        chi2_tau=CHI2_TAU,
        enforce_degree_constraint=ENFORCE_DEGREE_CONSTRAINT,
        structure_mode=STRUCTURE_MODE,
        directed_edge_direction=DIRECTED_EDGE_DIRECTION,
        verbose=False,
        progress_every=PROGRESS_EVERY,
    )

    print("Priming sequence state over train graphs ...")
    model.reset_sequence_state(device)
    with torch.no_grad():
        for g in sequence.train_graphs:
            g = g.to(device)
            _ = model(g.x.float(), g.edge_index.long())
            model.detach_sequence_state()

    results = PerTargetResults()
    per_timestep = []
    attack_time_seconds = 0.0

    for graph in sequence.test_graphs:
        g = graph.to(device)
        x = g.x.float()
        edge_index = g.edge_index.long()
        y = g.y
        t = int(g.graph_timestep)
        labeled_mask = y != -1

        if int(labeled_mask.sum().item()) == 0:
            with torch.no_grad():
                _ = model(x, edge_index)
                model.detach_sequence_state()
            per_timestep.append({"t": t, "n_labeled": 0, "n_targets": 0, "skipped": True})
            continue

        snap_pre = _save_state(model)
        with torch.no_grad():
            log_probs_clean = model(x, edge_index).detach()
            model.detach_sequence_state()
        snap_post = _save_state(model)

        pred_clean = log_probs_clean.argmax(dim=1)
        with torch.no_grad():
            logits_surrogate_clean = linearized_surrogate_logits(
                x, edge_index, W, structure_mode=STRUCTURE_MODE
            )
        pred_surrogate_clean = logits_surrogate_clean.argmax(dim=1)

        targets = pick_targets_graph(
            y,
            pred_surrogate_clean,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )

        if targets.numel() == 0:
            _restore_state(model, snap_post)
            per_timestep.append(
                {
                    "t": t,
                    "n_labeled": int(labeled_mask.sum().item()),
                    "n_targets": 0,
                    "skipped": True,
                }
            )
            continue

        n_edges_orig = int(edge_index.size(1))
        timestep_start = len(results)
        edge_unique_t = 0
        edge_directed_t = 0
        edge_targets_t = 0

        for target in targets.tolist():
            target = int(target)
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)

            t0 = time.perf_counter()
            x_adv, edge_index_adv, info = atk.attack_slice(
                x,
                edge_index,
                y,
                target_tensor,
                n_struct=N_STRUCT,
                eps_feat=EPS_FEAT,
                time_step=torch.zeros(x.size(0), dtype=torch.long, device=device),
            )
            attack_time_seconds += float(time.perf_counter() - t0)

            _restore_state(model, snap_pre)
            with torch.no_grad():
                log_probs_adv = model(x_adv, edge_index_adv).detach()
                model.detach_sequence_state()
                logits_surrogate_adv = linearized_surrogate_logits(
                    x_adv, edge_index_adv, W, structure_mode=STRUCTURE_MODE
                )

            pred_adv = log_probs_adv.argmax(dim=1)
            y_lab = y[labeled_mask]
            pred_adv_lab = pred_adv[labeled_mask]
            logits_adv_lab = log_probs_adv[labeled_mask]
            adv_m = binary_classification_metrics(y_lab, pred_adv_lab)
            roc_adv = roc_auc_binary(
                logits_adv_lab,
                y_lab,
                torch.ones(y_lab.numel(), dtype=torch.bool, device=device),
            )

            n_edges_adv = int(edge_index_adv.size(1))
            n_unique_added = int(info["n_unique_edges_added"])
            n_directed_added = int(info["n_directed_edges_added"])
            n_targets_with_edge = int(info["n_targets_with_edge_added"])
            edge_unique_t += n_unique_added
            edge_directed_t += n_directed_added
            edge_targets_t += n_targets_with_edge

            node_id = None
            if hasattr(g, "node_ids") and target < len(g.node_ids):
                node_id = g.node_ids[target]

            results.add(
                context={"t": t, "target_local": target, "node_id": node_id},
                target=target,
                y_val=int(y[target].item()),
                logits_clean_target=log_probs_clean[target],
                logits_adv_target=log_probs_adv[target],
                logits_surrogate_clean_target=logits_surrogate_clean[target],
                logits_surrogate_adv_target=logits_surrogate_adv[target],
                clean_row=x[target, : int(info["perturbable_dim"])],
                adv_row=x_adv[target, : int(info["perturbable_dim"])],
                n_edges_orig=n_edges_orig,
                n_edges_adv=n_edges_adv,
                n_directed_added=n_directed_added,
                n_unique_added=n_unique_added,
                n_targets_with_edge=n_targets_with_edge,
                adv_metrics={
                    "f1_pos": adv_m["f1_pos"],
                    "recall_pos": adv_m["recall_pos"],
                    "f1_macro": adv_m["f1_macro"],
                    "roc_auc": roc_adv,
                },
            )

            if VERBOSE and (
                len(results) == 1
                or len(results) % PROGRESS_EVERY == 0
            ):
                last = results.per_target[-1]
                print(
                    f"  [adapted-nettack-recgnn-pertarget] targets done={len(results)} "
                    f"t={t} target={target} success={last['victim']['success']} "
                    f"edges+={n_unique_added}"
                )

        _restore_state(model, snap_post)

        entries = results.per_target[timestep_start:]
        asr, ns, na = _entries_asr(entries, "victim")
        surrogate_asr, surrogate_ns, surrogate_na = _entries_asr(entries, "surrogate")
        per_timestep.append(
            {
                "t": t,
                "n_labeled": int(labeled_mask.sum().item()),
                "n_targets": int(targets.numel()),
                "n_unique_edges_added": edge_unique_t,
                "n_directed_edges_added": edge_directed_t,
                "n_targets_with_edge_added": edge_targets_t,
                "asr": asr,
                "asr_success": ns,
                "asr_attempted": na,
                "surrogate_asr": surrogate_asr,
                "surrogate_asr_success": surrogate_ns,
                "surrogate_asr_attempted": surrogate_na,
            }
        )
        print(
            f"t={t:2d}  n_labeled={int(labeled_mask.sum().item()):5d}  "
            f"n_targets={targets.numel():4d}  independent ASR={asr:.4f} ({ns}/{na})"
        )

    if len(results) == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    summary = results.summarize()
    run_dir, ts = make_pertarget_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "AdaptedNETTACKPerTarget",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "device": str(device),
        "execution": {
            "mode": "temporal_independent_per_target",
            "description": (
                "Each selected target is attacked from the clean timestep graph "
                "independently; there is no cumulative adversarial graph per timestep."
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
            "same_timestep_edge_candidates": True,
            "edge_candidate_scope": "current_timestep_nodes",
            "surrogate_epochs": SURROGATE_EPOCHS,
            "surrogate_lr": SURROGATE_LR,
            "surrogate_weight_decay": SURROGATE_WEIGHT_DECAY,
            "attack_dim": attack_dim,
        },
        "target_selection": {
            "model": "surrogate_linearized_gcn",
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
            "n_targets": len(results),
        },
        "data": ckpt_cfg["data"],
        "model_hparams": ckpt_cfg["model"],
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "AdaptedNETTACKPerTarget",
        "model_name": MODEL_NAME,
        "dataset": DATASET,
        "classification": summary["classification"],
        "attack_effect": {
            "attack_time_seconds": attack_time_seconds,
            **summary["attack_effect"],
        },
        "per_timestep": per_timestep,
        "surrogate": summary["surrogate"],
        "per_target": summary["per_target"],
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    asr_obj = summary["attack_effect"]["asr"]
    structural = summary["attack_effect"]["structural"]
    print(
        f"Per-target ASR={asr_obj['value']:.4f} "
        f"({asr_obj['success']}/{asr_obj['attempted']})  "
        f"targets={len(results)}  "
        f"mean_edges/target={structural['mean_unique_edges_per_target']:.2f}"
    )
    print(
        f"Saved to {os.path.relpath(run_dir, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))}"
    )


if __name__ == "__main__":
    main()
