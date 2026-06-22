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
from scripts.run_adapted_nettack_attack_evolvegcn import (
    _active_slice_marker,
    _build_y_full,
    edge_index_from_normalized_adj,
    normalize_adj_evolvegcn,
)
from scripts.training.train_evolvegcn_utils import build_evolvegcn_model, move_sample
from src.datasets.evolvegcn_elliptic import EvolveGCNEllipticConfig, EvolveGCNEllipticDataset
from src.datasets.evolvegcn_ellipticpp_actors import (
    EvolveGCNActorsConfig,
    EvolveGCNEllipticPPActorsDataset,
)
from src.attacks.nettack_adapted_temporal import (
    AdaptedNettackTemporalAttack,
    linearized_surrogate_logits,
    train_surrogate_on_train_slices,
)
from src.utils.model_loader import resolve_checkpoint
from src.utils.seed import set_seed
from src.training.metrics import binary_classification_metrics, roc_auc_binary

# ---------- attack parameters ----------
MODEL_NAME = "evolvegcn_o"
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

# EvolveGCN consumes sparse row-aggregation matrices, but the attack works in
# source -> target edge_index convention. Directed mode adds one incoming edge
# u -> target per structural perturbation.
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

# Column 0 of the EvolveGCN Elliptic feature matrix is the IBM time_step
# metadata. Protect it from perturbation. Elliptic++ actors has no metadata
# column. `None` -> dataset-appropriate default.
ATTACK_START_COL = None

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
    return torch.device("cpu")


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
        data_cfg = EvolveGCNEllipticConfig(**ckpt_cfg["data"])
        sequence = EvolveGCNEllipticDataset(data_cfg).get_sequence()
        default_attack_start_col = 1
    elif DATASET == "ellipticpp_actors":
        data_cfg = EvolveGCNActorsConfig(**ckpt_cfg["data"])
        sequence = EvolveGCNEllipticPPActorsDataset(data_cfg).get_sequence()
        default_attack_start_col = 0
    else:
        raise ValueError(f"Unknown DATASET={DATASET!r}.")

    attack_start_col = (
        default_attack_start_col
        if ATTACK_START_COL is None
        else int(ATTACK_START_COL)
    )
    num_nodes = int(sequence.num_nodes)
    num_features = int(sequence.num_features)
    attack_dim = num_features - attack_start_col

    def _features_view(x_full: torch.Tensor) -> torch.Tensor:
        if attack_start_col == 0:
            return x_full
        return torch.cat([x_full[:, attack_start_col:], x_full[:, :attack_start_col]], dim=1)

    def _features_unview(x_view: torch.Tensor) -> torch.Tensor:
        if attack_start_col == 0:
            return x_view
        perturbable = x_view[:, :attack_dim]
        protected = x_view[:, attack_dim:]
        return torch.cat([protected, perturbable], dim=1)

    model = build_evolvegcn_model(num_features, ckpt_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Loaded {MODEL_NAME.upper()} from {ckpt_path}")
    print(
        f"Train windows: {len(sequence.train_samples)}  Test windows: {len(sequence.test_samples)}  "
        f"num_nodes={num_nodes}  num_features={num_features}  attack_start_col={attack_start_col}"
    )

    print("Training surrogate (linearized GCN) on union of train timesteps ...")
    train_slices = []
    for sample in sequence.train_samples:
        hist_adj_list, hist_ndFeats_list, _, label_idx, label_vals = move_sample(sample, device)
        x_view = _features_view(hist_ndFeats_list[-1].float())
        edge_index = edge_index_from_normalized_adj(hist_adj_list[-1], STRUCTURE_MODE)
        y_full = _build_y_full(num_nodes, label_idx, label_vals, device)
        train_mask = y_full != -1
        if int(train_mask.sum().item()) == 0:
            continue
        train_slices.append((x_view, edge_index, y_full, train_mask))
    W = train_surrogate_on_train_slices(
        train_slices,
        num_features=num_features,
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

    results = PerTargetResults()
    per_window = []
    attack_time_seconds = 0.0

    for sample in sequence.test_samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        t = int(sample.current_time)

        with torch.no_grad():
            logits_clean = model(
                hist_adj_list,
                hist_ndFeats_list,
                node_mask_list,
                label_idx,
            ).detach()
        pred_clean = logits_clean.argmax(dim=1)

        x_view = _features_view(hist_ndFeats_list[-1].float())
        edge_index_curr = edge_index_from_normalized_adj(hist_adj_list[-1], STRUCTURE_MODE)
        with torch.no_grad():
            logits_surrogate_clean_full = linearized_surrogate_logits(
                x_view, edge_index_curr, W, structure_mode=STRUCTURE_MODE
            )
        logits_surrogate_clean = logits_surrogate_clean_full[label_idx.long()]
        pred_surrogate_clean = logits_surrogate_clean.argmax(dim=1)

        pos = torch.arange(label_vals.numel(), device=device)
        pos = pos[label_vals != -1]
        if ATTACK_ONLY_ILLICIT:
            pos = pos[label_vals[pos] == 1]
        if ONLY_CLEAN_CORRECT and pos.numel() > 0:
            pos = pos[pred_surrogate_clean[pos] == label_vals[pos]]

        if pos.numel() == 0:
            per_window.append({"t": t, "n_labeled": int(label_idx.numel()), "n_targets": 0, "skipped": True})
            continue

        if 0.0 < float(ATTACK_FRACTION) < 1.0:
            n = max(1, min(int(round(ATTACK_FRACTION * float(pos.numel()))), int(pos.numel())))
            g = torch.Generator(device="cpu")
            g.manual_seed(int(SEED + t))
            perm = torch.randperm(pos.numel(), generator=g)
            pos = pos[perm[:n].to(pos.device)]

        target_global = label_idx[pos].long()
        y_full = _build_y_full(num_nodes, label_idx, label_vals, device)
        active_marker = _active_slice_marker(node_mask_list[-1], device)
        n_edges_orig = int(edge_index_curr.size(1))
        window_start = len(results)
        edge_unique_t = 0
        edge_directed_t = 0
        edge_targets_t = 0

        for target_pos, target in zip(pos.tolist(), target_global.tolist()):
            target_pos = int(target_pos)
            target = int(target)
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)

            t0 = time.perf_counter()
            x_view_adv, edge_index_adv, info = atk.attack_slice(
                x_view,
                edge_index_curr,
                y_full,
                target_tensor,
                n_struct=N_STRUCT,
                eps_feat=EPS_FEAT,
                time_step=active_marker,
            )
            attack_time_seconds += float(time.perf_counter() - t0)

            x_last_adv = _features_unview(x_view_adv)
            adj_last_adv = normalize_adj_evolvegcn(
                edge_index_adv, num_nodes, device, STRUCTURE_MODE
            )
            hist_ndFeats_adv = list(hist_ndFeats_list[:-1]) + [x_last_adv]
            hist_adj_adv = list(hist_adj_list[:-1]) + [adj_last_adv]

            with torch.no_grad():
                logits_adv = model(
                    hist_adj_adv,
                    hist_ndFeats_adv,
                    node_mask_list,
                    label_idx,
                ).detach()
                logits_surrogate_adv_full = linearized_surrogate_logits(
                    x_view_adv,
                    edge_index_adv,
                    W,
                    structure_mode=STRUCTURE_MODE,
                )
            logits_surrogate_adv = logits_surrogate_adv_full[label_idx.long()]
            pred_adv = logits_adv.argmax(dim=1)

            adv_m = binary_classification_metrics(label_vals, pred_adv)
            roc_adv = roc_auc_binary(
                logits_adv,
                label_vals,
                torch.ones(label_idx.numel(), dtype=torch.bool, device=device),
            )

            n_edges_adv = int(edge_index_adv.size(1))
            n_unique_added = int(info["n_unique_edges_added"])
            n_directed_added = int(info["n_directed_edges_added"])
            n_targets_with_edge = int(info["n_targets_with_edge_added"])
            edge_unique_t += n_unique_added
            edge_directed_t += n_directed_added
            edge_targets_t += n_targets_with_edge

            results.add(
                context={"t": t, "target_pos": target_pos, "target_global": target},
                target=target,
                y_val=int(label_vals[target_pos].item()),
                logits_clean_target=logits_clean[target_pos],
                logits_adv_target=logits_adv[target_pos],
                logits_surrogate_clean_target=logits_surrogate_clean[target_pos],
                logits_surrogate_adv_target=logits_surrogate_adv[target_pos],
                clean_row=hist_ndFeats_list[-1][
                    target,
                    attack_start_col : attack_start_col + attack_dim,
                ],
                adv_row=x_last_adv[
                    target,
                    attack_start_col : attack_start_col + attack_dim,
                ],
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
                    f"  [adapted-nettack-evolvegcn-pertarget] targets done={len(results)} "
                    f"t={t} target={target} success={last['victim']['success']} "
                    f"edges+={n_unique_added}"
                )

        entries = results.per_target[window_start:]
        asr, ns, na = _entries_asr(entries, "victim")
        surrogate_asr, surrogate_ns, surrogate_na = _entries_asr(entries, "surrogate")
        per_window.append(
            {
                "t": t,
                "n_labeled": int(label_idx.numel()),
                "n_targets": int(target_global.numel()),
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
            f"t={t:2d}  n_labeled={label_idx.numel():5d}  "
            f"n_targets={target_global.numel():4d}  independent ASR={asr:.4f} ({ns}/{na})"
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
                "Each selected target is attacked from the clean current-window graph "
                "independently; there is no cumulative adversarial graph per window."
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
            "edge_candidate_scope": "current_active_window_nodes",
            "surrogate_epochs": SURROGATE_EPOCHS,
            "surrogate_lr": SURROGATE_LR,
            "surrogate_weight_decay": SURROGATE_WEIGHT_DECAY,
            "attack_start_col": attack_start_col,
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
        "per_window": per_window,
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
