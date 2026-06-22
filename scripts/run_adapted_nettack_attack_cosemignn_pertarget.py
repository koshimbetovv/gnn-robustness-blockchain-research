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
from src.datasets.cosemignn_elliptic import load_cosemignn_elliptic
from src.datasets.cosemignn_ellipticpp_addraddr import load_cosemignn_ellipticpp_addraddr
from src.attacks.nettack_adapted_temporal import (
    AdaptedNettackTemporalAttack,
    linearized_surrogate_logits,
    train_surrogate_on_train_slices,
)
from src.utils.model_loader import resolve_checkpoint
from src.utils.seed import set_seed
from src.training.metrics import binary_classification_metrics, roc_auc_binary

try:
    from src.models.cosemignn import CoSemiGNN
except ModuleNotFoundError as exc:
    CoSemiGNN = None
    COSEMI_IMPORT_ERROR = exc

# ---------- attack parameters ----------
MODEL_NAME = "cosemignn"
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

# CoSemiGNN consumes PyG-style edge_index. Directed mode adds one incoming edge
# u -> target per structural perturbation. Use "undirected" to recover the
# original Nettack-style symmetrized threat model.
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


def cosemi_forward_logits(model, features, edge_index, ca_weights):
    """Lift CoSemiGNN's BCE-style `(N,)` output to `(N, 2)` `[0, s]`."""
    out_line, _ = model(features, edge_index, ca_weights)
    return torch.stack([torch.zeros_like(out_line), out_line], dim=1)


def _entries_asr(entries, branch: str):
    attempted = sum(1 for item in entries if item[branch]["clean_correct"])
    success = sum(1 for item in entries if item[branch]["success"])
    return (float(success) / float(attempted) if attempted else 0.0, success, attempted)


def main():
    if CoSemiGNN is None:
        raise ModuleNotFoundError(
            "CoSemiGNN requires torch_geometric_temporal. Install that dependency "
            "before running run_adapted_nettack_attack_cosemignn_pertarget.py."
        ) from COSEMI_IMPORT_ERROR

    device = get_device()
    set_seed(SEED, deterministic=True, benchmark=False)
    print(f"Random seed set to {SEED} (deterministic=True)")

    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    data_cfg = ckpt_cfg["data"]
    time_cfg = ckpt_cfg["time"]
    train_start = int(time_cfg.get("train_start", 1))
    train_end = int(time_cfg.get("train_end", time_cfg.get("predict_start", 1)))
    predict_start = int(time_cfg["predict_start"])
    predict_end = int(time_cfg["predict_end"])
    semi_cache_dir = data_cfg["semi_cache_dir"]

    print(f"Loading CoSemiGNN {DATASET} slices (cache: {semi_cache_dir}) ...")
    if DATASET == "elliptic":
        data = load_cosemignn_elliptic(
            feature_path=data_cfg["feature_path"],
            class_path=data_cfg["class_path"],
            edge_path=data_cfg["edge_path"],
            semi_cache_dir=semi_cache_dir,
            device=device,
            rebuild_semi=bool(data_cfg.get("rebuild_semi", False)),
        )
    elif DATASET == "ellipticpp_actors":
        data = load_cosemignn_ellipticpp_addraddr(
            feature_path=data_cfg["feature_path"],
            class_path=data_cfg["class_path"],
            edge_path=data_cfg["edge_path"],
            semi_cache_dir=semi_cache_dir,
            device=device,
            rebuild_semi=bool(data_cfg.get("rebuild_semi", False)),
        )
    else:
        raise ValueError(f"Unknown DATASET={DATASET!r}.")

    feature_list, adj_list, label_list, _ca_matrix_list, ca_weights_list, *_ = data

    feature_in = None
    for ft in feature_list[1:]:
        if ft is not None and ft.numel() > 0:
            feature_in = ft.size(1)
            break
    if feature_in is None:
        raise RuntimeError("No non-empty feature slices found in dataset.")
    raw_feature_dim = int(feature_in - 6)
    if raw_feature_dim <= 0:
        raise ValueError(
            f"Expected CoSemiGNN feature_in to be raw + 6 semi columns, got feature_in={feature_in}."
        )

    model_cfg = ckpt_cfg.get("cosemignn", {})
    model = CoSemiGNN(
        feature_in=feature_in,
        dim=int(model_cfg.get("dim", 128)),
        dim2=int(model_cfg.get("dim2", 256)),
        dim3=int(model_cfg.get("dim3", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Loaded {MODEL_NAME.upper()} from {ckpt_path}")

    print("Training surrogate (linearized GCN) on union of train timesteps ...")
    train_slices = []
    for t in range(train_start, train_end):
        if t >= len(feature_list):
            break
        ft = feature_list[t]
        adj = adj_list[t]
        lbl = label_list[t]
        if ft is None or lbl is None or ft.numel() == 0 or lbl.numel() == 0:
            continue
        train_mask = lbl != -1
        if int(train_mask.sum().item()) == 0:
            continue
        train_slices.append((ft[:, :raw_feature_dim].float(), adj.long(), lbl.long(), train_mask))
    if not train_slices:
        raise RuntimeError("No CoSemiGNN train slices found for surrogate training.")

    W = train_surrogate_on_train_slices(
        train_slices,
        num_features=raw_feature_dim,
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
        attack_dim=raw_feature_dim,
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
    per_slice = []
    attack_time_seconds = 0.0

    for t in range(predict_start, predict_end):
        if t >= len(feature_list):
            per_slice.append({"t": t, "n_labeled": 0, "n_targets": 0, "skipped": True})
            continue
        features = feature_list[t]
        adj = adj_list[t]
        labels = label_list[t]
        ca_weights = ca_weights_list[t]

        if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
            per_slice.append({"t": t, "n_labeled": 0, "n_targets": 0, "skipped": True})
            continue

        with torch.no_grad():
            logits_clean = cosemi_forward_logits(model, features, adj, ca_weights).detach()
            logits_surrogate_clean = linearized_surrogate_logits(
                features[:, :raw_feature_dim].float(),
                adj.long(),
                W,
                structure_mode=STRUCTURE_MODE,
            )
        pred_surrogate_clean = logits_surrogate_clean.argmax(dim=1)

        idx = torch.arange(labels.numel(), device=device)
        idx = idx[labels[idx] != -1]
        if ATTACK_ONLY_ILLICIT:
            idx = idx[labels[idx] == 1]
        if ONLY_CLEAN_CORRECT and idx.numel() > 0:
            idx = idx[pred_surrogate_clean[idx] == labels[idx]]

        if idx.numel() == 0:
            per_slice.append(
                {"t": t, "n_labeled": int(labels.numel()), "n_targets": 0, "skipped": True}
            )
            continue

        if 0.0 < float(ATTACK_FRACTION) < 1.0:
            n = max(1, min(int(round(ATTACK_FRACTION * float(idx.numel()))), int(idx.numel())))
            g = torch.Generator(device="cpu")
            g.manual_seed(int(SEED + t))
            perm = torch.randperm(idx.numel(), generator=g)
            idx = idx[perm[:n].to(idx.device)]

        n_edges_orig = int(adj.size(1))
        slice_start = len(results)
        edge_unique_t = 0
        edge_directed_t = 0
        edge_targets_t = 0

        for target in idx.tolist():
            target = int(target)
            target_tensor = torch.tensor([target], dtype=torch.long, device=device)

            t0 = time.perf_counter()
            x_raw_adv, edge_index_adv, info = atk.attack_slice(
                features[:, :raw_feature_dim].float(),
                adj.long(),
                labels.long(),
                target_tensor,
                n_struct=N_STRUCT,
                eps_feat=EPS_FEAT,
                time_step=torch.zeros(features.size(0), dtype=torch.long, device=device),
            )
            attack_time_seconds += float(time.perf_counter() - t0)

            x_adv = features.clone()
            x_adv[:, :raw_feature_dim] = x_raw_adv

            with torch.no_grad():
                logits_adv = cosemi_forward_logits(model, x_adv, edge_index_adv, ca_weights).detach()
                logits_surrogate_adv = linearized_surrogate_logits(
                    x_raw_adv,
                    edge_index_adv,
                    W,
                    structure_mode=STRUCTURE_MODE,
                )
            pred_adv = logits_adv.argmax(dim=1)
            adv_m = binary_classification_metrics(labels, pred_adv)
            roc_adv = roc_auc_binary(
                logits_adv,
                labels,
                torch.ones(labels.numel(), dtype=torch.bool, device=device),
            )

            n_edges_adv = int(edge_index_adv.size(1))
            n_unique_added = int(info["n_unique_edges_added"])
            n_directed_added = int(info["n_directed_edges_added"])
            n_targets_with_edge = int(info["n_targets_with_edge_added"])
            edge_unique_t += n_unique_added
            edge_directed_t += n_directed_added
            edge_targets_t += n_targets_with_edge

            results.add(
                context={"t": t},
                target=target,
                y_val=int(labels[target].item()),
                logits_clean_target=logits_clean[target],
                logits_adv_target=logits_adv[target],
                logits_surrogate_clean_target=logits_surrogate_clean[target],
                logits_surrogate_adv_target=logits_surrogate_adv[target],
                clean_row=features[target, :raw_feature_dim],
                adv_row=x_adv[target, :raw_feature_dim],
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
                    f"  [adapted-nettack-cosemignn-pertarget] targets done={len(results)} "
                    f"t={t} target={target} success={last['victim']['success']} "
                    f"edges+={n_unique_added}"
                )

        entries = results.per_target[slice_start:]
        asr, ns, na = _entries_asr(entries, "victim")
        surrogate_asr, surrogate_ns, surrogate_na = _entries_asr(entries, "surrogate")
        per_slice.append(
            {
                "t": t,
                "n_labeled": int(labels.numel()),
                "n_targets": int(idx.numel()),
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
            f"t={t:2d}  n_labeled={labels.numel():5d}  "
            f"n_targets={idx.numel():4d}  independent ASR={asr:.4f} ({ns}/{na})"
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
                "independently; there is no cumulative adversarial graph per slice."
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
            "raw_feature_dim": raw_feature_dim,
            "surrogate_feature_scope": "raw_only",
            "surrogate_train_range": {
                "start": train_start,
                "end_exclusive": train_end,
            },
        },
        "timesteps": {"predict_start": predict_start, "predict_end": predict_end},
        "target_selection": {
            "model": "surrogate_linearized_gcn",
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
            "n_targets": len(results),
        },
        "data": data_cfg,
        "model_hparams": model_cfg,
        "semi_cache_dir": semi_cache_dir,
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
        "per_timestep": per_slice,
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
