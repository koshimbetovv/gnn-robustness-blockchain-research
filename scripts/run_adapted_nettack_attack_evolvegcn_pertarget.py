import os
import sys
import json
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.adapted_nettack_pertarget_utils import (
    PerTargetResults,
    labeled_logits_summary,
    make_pertarget_run_dir,
    write_json,
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
ATTACK_FRACTION = 0.2
ONLY_CLEAN_CORRECT = False
SEED = 0

VERBOSE = True
PROGRESS_EVERY = 100
SAVE_PER_TARGET_DETAILS = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    #if torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def _entries_asr(entries, branch: str):
    attempted = sum(1 for item in entries if item[branch]["clean_correct"])
    success = sum(1 for item in entries if item[branch]["success"])
    return (float(success) / float(attempted) if attempted else None, success, attempted)

def edge_index_from_normalized_adj(
    adj_sparse: torch.Tensor,
    structure_mode: str,
) -> torch.Tensor:
    """Recover attack-facing edge_index from EvolveGCN's normalized adjacency.

    The victim stores a row-aggregation matrix and computes `Ahat.matmul(XW)`.
    In directed mode we expose this to NETTACK as source -> target by mapping
    sparse `(row=target, col=source)` entries to `(source, target)`. In
    undirected mode we keep the historical orientation; symmetric graphs are
    unaffected either way.
    """
    idx = adj_sparse.coalesce().indices().long()
    mask = idx[0] != idx[1]
    idx = idx[:, mask]
    if structure_mode == "directed":
        return torch.stack([idx[1], idx[0]], dim=0).contiguous()
    if structure_mode == "undirected":
        return idx
    raise ValueError(f"Unknown structure_mode={structure_mode!r}.")


def normalize_adj_evolvegcn(
    edge_index: torch.Tensor,
    num_nodes: int,
    device,
    structure_mode: str,
) -> torch.Tensor:
    """Mirror EvolveGCN's `_normalize_adj` so a NETTACK-perturbed edge_index can
    be fed back into the victim. Adds self-loops and applies symmetric
    `D~^{-1/2} (A + I) D~^{-1/2}` normalization, returning a coalesced sparse
    COO tensor of size (N, N). In directed mode, `edge_index` is source -> target
    and the sparse matrix is written as row=target, col=source to match the
    victim's row aggregation.
    """
    src, dst = edge_index[0].long(), edge_index[1].long()
    mask = src != dst
    src, dst = src[mask], dst[mask]
    if structure_mode == "directed":
        row, col = dst, src
    elif structure_mode == "undirected":
        row, col = src, dst
    else:
        raise ValueError(f"Unknown structure_mode={structure_mode!r}.")
    sl = torch.arange(num_nodes, device=device, dtype=torch.long)
    full_row = torch.cat([row, sl])
    full_col = torch.cat([col, sl])
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.scatter_add_(0, full_row, torch.ones_like(full_row, dtype=torch.float32))
    inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    vals = inv_sqrt[full_row] * inv_sqrt[full_col]
    indices = torch.stack([full_row, full_col], dim=0)
    return torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes)).coalesce()


def _build_y_full(num_nodes: int, label_idx: torch.Tensor, label_vals: torch.Tensor, device) -> torch.Tensor:
    """Construct an `(N,)` label tensor from `(label_idx, label_vals)`,
    -1 elsewhere, used both by the surrogate trainer and the per-slice attack.
    """
    y = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    y[label_idx.long()] = label_vals.long()
    return y


def _active_slice_marker(node_mask: torch.Tensor, device) -> torch.Tensor:
    """Return a candidate-scope marker compatible with AdaptedNettackAttack.

    EvolveGCN windows use the full global node set, with inactive rows masked by
    `-inf`. Mark current active nodes as 0 and inactive nodes as 1 so the static
    same-timestep candidate filter keeps additions inside the active slice.
    """
    values = node_mask.to(device).view(-1)
    active = torch.isfinite(values) & (values >= 0)
    marker = torch.ones(values.numel(), dtype=torch.long, device=device)
    marker[active] = 0
    return marker


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
    clean_y_parts = []
    clean_logits_parts = []
    surrogate_clean_logits_parts = []
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
        labeled_pos = label_vals != -1
        if int(labeled_pos.sum().item()) > 0:
            clean_y_parts.append(label_vals[labeled_pos].detach().cpu())
            clean_logits_parts.append(logits_clean[labeled_pos].detach().cpu())
            surrogate_clean_logits_parts.append(
                logits_surrogate_clean[labeled_pos].detach().cpu()
            )

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
            n_directed_added = int(info["n_directed_edges_added"])
            n_targets_with_edge = int(info["n_targets_with_edge_added"])
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
                    f"edges+={n_directed_added}"
                )

        entries = results.per_target[window_start:]
        asr, ns, na = _entries_asr(entries, "victim")
        surrogate_asr, surrogate_ns, surrogate_na = _entries_asr(entries, "surrogate")
        per_window.append(
            {
                "t": t,
                "n_labeled": int(label_idx.numel()),
                "n_targets": int(target_global.numel()),
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
        asr_text = "nan" if asr is None else f"{asr:.4f}"
        print(
            f"t={t:2d}  n_labeled={label_idx.numel():5d}  "
            f"n_targets={target_global.numel():4d}  independent ASR={asr_text} ({ns}/{na})"
        )

    if len(results) == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    summary = results.summarize()
    clean_labeled = labeled_logits_summary(clean_logits_parts, clean_y_parts)
    surrogate_clean_labeled = labeled_logits_summary(
        surrogate_clean_logits_parts,
        clean_y_parts,
    )
    classification = {
        "scope": summary["classification"]["scope"],
        "victim": {
            "labeled_clean": clean_labeled,
            **summary["classification"]["victim"],
        },
        "surrogate": {
            "labeled_clean": surrogate_clean_labeled,
        },
    }
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
        "classification": classification,
        "target_outcome": summary["target_outcome"],
        "perturbation": summary["perturbation"],
        "diagnostics": {
            **summary["diagnostics"],
            "per_window": per_window,
        },
        "runtime": {"attack_time_seconds": attack_time_seconds},
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    if SAVE_PER_TARGET_DETAILS:
        write_json(os.path.join(run_dir, "per_target_details.json"), results.per_target)

    asr_obj = summary["target_outcome"]["victim"]["asr"]
    asr_text = "nan" if asr_obj["value"] is None else f"{asr_obj['value']:.4f}"
    edge_mean = summary["perturbation"]["structure"]["directed_edges_added"]["mean_per_target"]
    print(
        f"Per-target ASR={asr_text} "
        f"({asr_obj['success']}/{asr_obj['attempted_clean_correct']})  "
        f"targets={len(results)}  "
        f"mean_edges/target={edge_mean:.2f}"
    )
    print(
        f"Saved to {os.path.relpath(run_dir, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))}"
    )


if __name__ == "__main__":
    main()
