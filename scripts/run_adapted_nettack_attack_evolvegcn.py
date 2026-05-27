import os
import sys
import json
import time
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.training.train_evolvegcn_utils import build_evolvegcn_model, move_sample
from src.datasets.evolvegcn_elliptic import EvolveGCNEllipticConfig, EvolveGCNEllipticDataset
from src.datasets.evolvegcn_ellipticpp_actors import (
    EvolveGCNActorsConfig, EvolveGCNEllipticPPActorsDataset,
)
from src.attacks.nettack_adapted_temporal import (
    AdaptedNettackTemporalAttack, linearized_surrogate_logits,
    train_surrogate_on_train_slices,
)
from src.utils.model_loader import resolve_checkpoint
from src.utils.seed import set_seed
from src.training.metrics import (
    binary_classification_metrics, attack_success_rate, roc_auc_binary,
    mean_confidence_drop, asr_pos_neg, mean_perturbation_l2_on_success,
)

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
N_STRUCT = 2
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

# Column 0 of the EvolveGCN Elliptic feature matrix is the IBM time_step
# metadata. Protect it from perturbation. Elliptic++ actors has no metadata
# column. `None` -> dataset-appropriate default.
ATTACK_START_COL = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = False
SEED = 0

VERBOSE = False
PROGRESS_EVERY = 100


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_run_dir(model_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_adapted_nettack_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def edge_index_from_normalized_adj(adj_sparse: torch.Tensor) -> torch.Tensor:
    """Recover an undirected edge_index from EvolveGCN's normalized adjacency.

    The dataset's `_normalize_adj` adds self-loops then symmetric-normalizes; we
    drop self-loops to get the bare graph edges (still symmetric, since the
    EvolveGCN loaders insert both (u,v) and (v,u)).
    """
    idx = adj_sparse.coalesce().indices().long()
    mask = idx[0] != idx[1]
    return idx[:, mask]


def normalize_adj_evolvegcn(edge_index: torch.Tensor, num_nodes: int, device) -> torch.Tensor:
    """Mirror EvolveGCN's `_normalize_adj` so a NETTACK-perturbed edge_index can
    be fed back into the victim. Adds self-loops and applies symmetric
    `D~^{-1/2} (A + I) D~^{-1/2}` normalization, returning a coalesced sparse
    COO tensor of size (N, N).
    """
    src, dst = edge_index[0].long(), edge_index[1].long()
    mask = src != dst
    src, dst = src[mask], dst[mask]
    sl = torch.arange(num_nodes, device=device, dtype=torch.long)
    full_src = torch.cat([src, sl])
    full_dst = torch.cat([dst, sl])
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.scatter_add_(0, full_src, torch.ones_like(full_src, dtype=torch.float32))
    inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    vals = inv_sqrt[full_src] * inv_sqrt[full_dst]
    indices = torch.stack([full_src, full_dst], dim=0)
    return torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes)).coalesce()


def _build_y_full(num_nodes: int, label_idx: torch.Tensor, label_vals: torch.Tensor, device) -> torch.Tensor:
    """Construct an `(N,)` label tensor from `(label_idx, label_vals)`,
    -1 elsewhere, used both by the surrogate trainer and the per-slice attack.
    """
    y = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    y[label_idx.long()] = label_vals.long()
    return y


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

    attack_start_col = default_attack_start_col if ATTACK_START_COL is None else int(ATTACK_START_COL)
    num_nodes = int(sequence.num_nodes)
    num_features = int(sequence.num_features)

    model = build_evolvegcn_model(num_features, ckpt_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Loaded {MODEL_NAME.upper()} from {ckpt_path}")
    print(
        f"Train windows: {len(sequence.train_samples)}  Test windows: {len(sequence.test_samples)}  "
        f"num_nodes={num_nodes}  num_features={num_features}  attack_start_col={attack_start_col}"
    )

    # ----- train surrogate on union of train timesteps -----
    # NETTACK perturbs only the **leading** raw slice (`attack_dim` columns
    # starting at column 0). EvolveGCN protects column 0 (IBM time_step), so we
    # build a "raw view" feature matrix where col 0 is dropped, train the
    # surrogate on that view, then map back when applying the attack: NETTACK
    # output's first `attack_dim` columns get written into the victim's columns
    # `[attack_start_col : attack_start_col + attack_dim]`.
    attack_dim = num_features - attack_start_col

    def _features_view(x_full: torch.Tensor) -> torch.Tensor:
        if attack_start_col == 0:
            return x_full
        return torch.cat([x_full[:, attack_start_col:], x_full[:, :attack_start_col]], dim=1)

    def _features_unview(x_view: torch.Tensor) -> torch.Tensor:
        # Inverse of `_features_view`: restore the original column order.
        if attack_start_col == 0:
            return x_view
        perturbable = x_view[:, :attack_dim]
        protected = x_view[:, attack_dim:]
        return torch.cat([protected, perturbable], dim=1)

    print("Training surrogate (linearized GCN) on union of train timesteps ...")
    train_slices = []
    for sample in sequence.train_samples:
        hist_adj_list, hist_ndFeats_list, _, label_idx, label_vals = move_sample(sample, device)
        x_view = _features_view(hist_ndFeats_list[-1].float())
        edge_index = edge_index_from_normalized_adj(hist_adj_list[-1])
        y_full = _build_y_full(num_nodes, label_idx, label_vals, device)
        train_mask = (y_full != -1)
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
    )

    atk = AdaptedNettackTemporalAttack(
        W=W, device=device,
        attack_dim=attack_dim,
        clamp=CLAMP,
        d_min=D_MIN, chi2_tau=CHI2_TAU,
        enforce_degree_constraint=ENFORCE_DEGREE_CONSTRAINT,
        verbose=VERBOSE, progress_every=PROGRESS_EVERY,
    )

    # ----- per-window attack -----
    per_window = []
    pooled_y_true = []
    pooled_pred_clean = []
    pooled_pred_adv = []
    pooled_logits_clean = []
    pooled_logits_adv = []
    pooled_attack_mask = []
    pooled_surrogate_pred_clean = []
    pooled_surrogate_pred_adv = []
    pooled_surrogate_logits_clean = []
    pooled_surrogate_logits_adv = []
    pooled_delta_success = []
    n_unique_added_total = 0
    n_targets_with_edge_total = 0
    attack_time_seconds = 0.0

    for sample in sequence.test_samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        t = int(sample.current_time)

        # Clean forward over full window.
        with torch.no_grad():
            logits_clean = model(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
        pred_clean = logits_clean.argmax(dim=1)
        x_view = _features_view(hist_ndFeats_list[-1].float())
        edge_index_curr = edge_index_from_normalized_adj(hist_adj_list[-1])
        with torch.no_grad():
            logits_surrogate_clean_full = linearized_surrogate_logits(x_view, edge_index_curr, W)
        logits_surrogate_clean = logits_surrogate_clean_full[label_idx.long()]
        pred_surrogate_clean = logits_surrogate_clean.argmax(dim=1)

        # Pick targets among labeled positions in this window.
        pos = torch.arange(label_vals.numel(), device=device)
        pos = pos[label_vals != -1]
        if ATTACK_ONLY_ILLICIT:
            pos = pos[label_vals[pos] == 1]
        if ONLY_CLEAN_CORRECT and pos.numel() > 0:
            pos = pos[pred_surrogate_clean[pos] == label_vals[pos]]

        if pos.numel() == 0:
            per_window.append({"t": t, "n_labeled": int(label_idx.numel()),
                               "n_targets": 0, "skipped": True})
            pooled_y_true.append(label_vals.detach().cpu())
            pooled_pred_clean.append(pred_clean.detach().cpu())
            pooled_pred_adv.append(pred_clean.detach().cpu())
            pooled_logits_clean.append(logits_clean.detach().cpu())
            pooled_logits_adv.append(logits_clean.detach().cpu())
            pooled_attack_mask.append(torch.zeros(label_idx.numel(), dtype=torch.bool))
            pooled_surrogate_pred_clean.append(pred_surrogate_clean.detach().cpu())
            pooled_surrogate_pred_adv.append(pred_surrogate_clean.detach().cpu())
            pooled_surrogate_logits_clean.append(logits_surrogate_clean.detach().cpu())
            pooled_surrogate_logits_adv.append(logits_surrogate_clean.detach().cpu())
            continue

        if 0.0 < float(ATTACK_FRACTION) < 1.0:
            n = max(1, min(int(round(ATTACK_FRACTION * float(pos.numel()))), int(pos.numel())))
            g = torch.Generator(device="cpu")
            g.manual_seed(int(SEED + t))
            perm = torch.randperm(pos.numel(), generator=g)
            pos = pos[perm[:n].to(pos.device)]

        target_global = label_idx[pos].long()
        target_pos = pos

        # Build per-slice tensors for NETTACK.
        y_full = _build_y_full(num_nodes, label_idx, label_vals, device)

        t0 = time.perf_counter()
        x_view_adv, edge_index_adv, info = atk.attack_slice(
            x_view, edge_index_curr, y_full, target_global,
            n_struct=N_STRUCT, eps_feat=EPS_FEAT,
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        n_unique_added_total += int(info["n_unique_edges_added"])
        n_targets_with_edge_total += int(info["n_targets_with_edge_added"])

        # Reassemble the perturbed window: only the LAST step is modified.
        x_last_adv = _features_unview(x_view_adv)
        adj_last_adv = normalize_adj_evolvegcn(edge_index_adv, num_nodes, device)
        hist_ndFeats_adv = list(hist_ndFeats_list[:-1]) + [x_last_adv]
        hist_adj_adv = list(hist_adj_list[:-1]) + [adj_last_adv]

        with torch.no_grad():
            logits_adv = model(hist_adj_adv, hist_ndFeats_adv, node_mask_list, label_idx).detach()
            logits_surrogate_adv_full = linearized_surrogate_logits(x_view_adv, edge_index_adv, W)
        logits_surrogate_adv = logits_surrogate_adv_full[label_idx.long()]
        pred_adv = logits_adv.argmax(dim=1)
        pred_surrogate_adv = logits_surrogate_adv.argmax(dim=1)

        attack_mask = torch.zeros(label_idx.numel(), dtype=torch.bool, device=device)
        attack_mask[target_pos] = True

        asr, ns, na = attack_success_rate(label_vals, pred_clean, pred_adv, attack_mask)
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(label_vals, logits_clean, logits_adv, attack_mask)
        surrogate_asr, surrogate_ns, surrogate_na = attack_success_rate(
            label_vals, pred_surrogate_clean, pred_surrogate_adv, attack_mask,
        )
        surrogate_asr_p, surrogate_sp, surrogate_ap, surrogate_asr_n, surrogate_sn, surrogate_an = asr_pos_neg(
            label_vals, logits_surrogate_clean, logits_surrogate_adv, attack_mask,
        )

        full_mask = torch.ones(label_idx.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean, label_vals, full_mask)
        roc_adv = roc_auc_binary(logits_adv, label_vals, full_mask)
        clean_m = binary_classification_metrics(label_vals, pred_clean)
        adv_m = binary_classification_metrics(label_vals, pred_adv)
        surrogate_clean_m = binary_classification_metrics(label_vals, pred_surrogate_clean)
        surrogate_adv_m = binary_classification_metrics(label_vals, pred_surrogate_adv)

        conf_drop, n_used = mean_confidence_drop(
            label_vals, logits_clean, logits_adv, attack_mask, only_clean_correct=True,
        )

        # L2 perturbation on successful flips, on the perturbable column slice.
        clean_rows = hist_ndFeats_list[-1][target_global, attack_start_col: attack_start_col + attack_dim]
        adv_rows = x_last_adv[target_global, attack_start_col: attack_start_col + attack_dim]
        pred_clean_targets = pred_clean[target_pos]
        pred_adv_targets = pred_adv[target_pos]
        y_targets = label_vals[target_pos].long()
        pert_l2_mean, pert_l2_n = mean_perturbation_l2_on_success(
            clean_rows, adv_rows, pred_clean_targets, pred_adv_targets, y_targets,
        )
        success_mask_targets = (pred_clean_targets == y_targets) & (pred_adv_targets != y_targets)
        if int(success_mask_targets.sum().item()) > 0:
            pooled_delta_success.append(
                (adv_rows[success_mask_targets].float() - clean_rows[success_mask_targets].float()).detach().cpu()
            )

        f1_pos_drop = float(clean_m["f1_pos"] - adv_m["f1_pos"])
        recall_pos_drop = float(clean_m["recall_pos"] - adv_m["recall_pos"])
        surrogate_f1_pos_drop = float(surrogate_clean_m["f1_pos"] - surrogate_adv_m["f1_pos"])

        per_window.append({
            "t": t,
            "n_labeled": int(label_idx.numel()),
            "n_targets": int(target_global.numel()),
            "n_unique_edges_added": int(info["n_unique_edges_added"]),
            "n_directed_edges_added": int(info["n_directed_edges_added"]),
            "asr": asr, "asr_success": ns, "asr_attempted": na,
            "asr_pos": asr_p, "asr_pos_success": sp, "asr_pos_attempted": ap,
            "asr_neg": asr_n, "asr_neg_success": sn, "asr_neg_attempted": an,
            "f1_pos_clean": clean_m["f1_pos"], "f1_pos_adv": adv_m["f1_pos"], "f1_pos_drop": f1_pos_drop,
            "f1_macro_clean": clean_m["f1_macro"], "f1_macro_adv": adv_m["f1_macro"],
            "recall_pos_clean": clean_m["recall_pos"], "recall_pos_adv": adv_m["recall_pos"], "recall_pos_drop": recall_pos_drop,
            "roc_auc_clean": roc_clean, "roc_auc_adv": roc_adv,
            "mean_confidence_drop": conf_drop, "conf_drop_n": n_used,
            "pert_l2_success": pert_l2_mean, "pert_l2_n": pert_l2_n,
            "surrogate_asr": surrogate_asr,
            "surrogate_asr_success": surrogate_ns,
            "surrogate_asr_attempted": surrogate_na,
            "surrogate_asr_pos": surrogate_asr_p,
            "surrogate_asr_pos_success": surrogate_sp,
            "surrogate_asr_pos_attempted": surrogate_ap,
            "surrogate_asr_neg": surrogate_asr_n,
            "surrogate_asr_neg_success": surrogate_sn,
            "surrogate_asr_neg_attempted": surrogate_an,
            "surrogate_f1_pos_clean": surrogate_clean_m["f1_pos"],
            "surrogate_f1_pos_adv": surrogate_adv_m["f1_pos"],
            "surrogate_f1_pos_drop": surrogate_f1_pos_drop,
        })

        pooled_y_true.append(label_vals.detach().cpu())
        pooled_pred_clean.append(pred_clean.detach().cpu())
        pooled_pred_adv.append(pred_adv.detach().cpu())
        pooled_logits_clean.append(logits_clean.detach().cpu())
        pooled_logits_adv.append(logits_adv.detach().cpu())
        pooled_attack_mask.append(attack_mask.detach().cpu())
        pooled_surrogate_pred_clean.append(pred_surrogate_clean.detach().cpu())
        pooled_surrogate_pred_adv.append(pred_surrogate_adv.detach().cpu())
        pooled_surrogate_logits_clean.append(logits_surrogate_clean.detach().cpu())
        pooled_surrogate_logits_adv.append(logits_surrogate_adv.detach().cpu())

        roc_c_show = roc_clean if roc_clean == roc_clean else float("nan")
        roc_a_show = roc_adv if roc_adv == roc_adv else float("nan")
        print(
            f"t={t:2d}  n_labeled={label_idx.numel():5d}  n_targets={target_global.numel():4d}  "
            f"edges+={info['n_unique_edges_added']:3d}  ASR={asr:.4f} ({ns}/{na})  "
            f"F1_pos {clean_m['f1_pos']:.3f}->{adv_m['f1_pos']:.3f}  "
            f"ROC {roc_c_show:.3f}->{roc_a_show:.3f}"
        )

    # ---------- aggregation ----------
    concat_classification = None
    concat_attack = None
    concat_surrogate_classification = None
    concat_surrogate_attack = None
    if pooled_y_true:
        y_true = torch.cat(pooled_y_true)
        pred_clean_c = torch.cat(pooled_pred_clean)
        pred_adv_c = torch.cat(pooled_pred_adv)
        logits_clean_c = torch.cat(pooled_logits_clean)
        logits_adv_c = torch.cat(pooled_logits_adv)
        attack_mask_c = torch.cat(pooled_attack_mask)
        pred_surrogate_clean_c = torch.cat(pooled_surrogate_pred_clean)
        pred_surrogate_adv_c = torch.cat(pooled_surrogate_pred_adv)
        logits_surrogate_clean_c = torch.cat(pooled_surrogate_logits_clean)
        logits_surrogate_adv_c = torch.cat(pooled_surrogate_logits_adv)

        clean_c = binary_classification_metrics(y_true, pred_clean_c)
        adv_c = binary_classification_metrics(y_true, pred_adv_c)
        full_mask = torch.ones(y_true.numel(), dtype=torch.bool)
        roc_c_clean = roc_auc_binary(logits_clean_c, y_true, full_mask)
        roc_c_adv = roc_auc_binary(logits_adv_c, y_true, full_mask)

        asr_c, ns_c, na_c = attack_success_rate(y_true, pred_clean_c, pred_adv_c, attack_mask_c)
        asr_cp, sp_c, ap_c, asr_cn, sn_c, an_c = asr_pos_neg(y_true, logits_clean_c, logits_adv_c, attack_mask_c)
        surrogate_asr_c, surrogate_ns_c, surrogate_na_c = attack_success_rate(
            y_true, pred_surrogate_clean_c, pred_surrogate_adv_c, attack_mask_c,
        )
        surrogate_asr_cp, surrogate_sp_c, surrogate_ap_c, surrogate_asr_cn, surrogate_sn_c, surrogate_an_c = asr_pos_neg(
            y_true, logits_surrogate_clean_c, logits_surrogate_adv_c, attack_mask_c,
        )
        conf_drop_c, conf_drop_n_c = mean_confidence_drop(
            y_true, logits_clean_c, logits_adv_c, attack_mask_c, only_clean_correct=True,
        )

        if pooled_delta_success:
            delta_all = torch.cat(pooled_delta_success, dim=0)
            pert_l2_c = float(torch.linalg.vector_norm(delta_all, ord=2, dim=1).mean().item())
            pert_l2_n_c = int(delta_all.size(0))
        else:
            pert_l2_c, pert_l2_n_c = 0.0, 0

        concat_classification = {
            "n_total": int(y_true.numel()),
            "f1_pos_clean": clean_c["f1_pos"], "f1_pos_adv": adv_c["f1_pos"],
            "f1_pos_drop": float(clean_c["f1_pos"] - adv_c["f1_pos"]),
            "f1_macro_clean": clean_c["f1_macro"], "f1_macro_adv": adv_c["f1_macro"],
            "recall_pos_clean": clean_c["recall_pos"], "recall_pos_adv": adv_c["recall_pos"],
            "recall_pos_drop": float(clean_c["recall_pos"] - adv_c["recall_pos"]),
            "roc_auc_clean": roc_c_clean, "roc_auc_adv": roc_c_adv,
        }
        concat_attack = {
            "n_targets_total": int(attack_mask_c.sum().item()),
            "asr": asr_c, "asr_success": ns_c, "asr_attempted": na_c,
            "asr_pos": asr_cp, "asr_pos_success": sp_c, "asr_pos_attempted": ap_c,
            "asr_neg": asr_cn, "asr_neg_success": sn_c, "asr_neg_attempted": an_c,
            "mean_confidence_drop": conf_drop_c, "conf_drop_n": conf_drop_n_c,
            "pert_l2_success": pert_l2_c, "pert_l2_n": pert_l2_n_c,
            "n_unique_edges_added_total": n_unique_added_total,
            "n_targets_with_edge_added_total": n_targets_with_edge_total,
        }
        surrogate_clean_c = binary_classification_metrics(y_true, pred_surrogate_clean_c)
        surrogate_adv_c = binary_classification_metrics(y_true, pred_surrogate_adv_c)
        concat_surrogate_classification = {
            "n_total": int(y_true.numel()),
            "f1_pos_clean": surrogate_clean_c["f1_pos"],
            "f1_pos_adv": surrogate_adv_c["f1_pos"],
            "f1_pos_drop": float(surrogate_clean_c["f1_pos"] - surrogate_adv_c["f1_pos"]),
            "f1_macro_clean": surrogate_clean_c["f1_macro"],
            "f1_macro_adv": surrogate_adv_c["f1_macro"],
            "recall_pos_clean": surrogate_clean_c["recall_pos"],
            "recall_pos_adv": surrogate_adv_c["recall_pos"],
            "recall_pos_drop": float(surrogate_clean_c["recall_pos"] - surrogate_adv_c["recall_pos"]),
        }
        concat_surrogate_attack = {
            "n_targets_total": int(attack_mask_c.sum().item()),
            "asr": surrogate_asr_c,
            "asr_success": surrogate_ns_c,
            "asr_attempted": surrogate_na_c,
            "asr_pos": surrogate_asr_cp,
            "asr_pos_success": surrogate_sp_c,
            "asr_pos_attempted": surrogate_ap_c,
            "asr_neg": surrogate_asr_cn,
            "asr_neg_success": surrogate_sn_c,
            "asr_neg_attempted": surrogate_an_c,
        }

    print()
    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "AdaptedNETTACK",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "device": str(device),
        "attack_params": {
            "n_struct": N_STRUCT, "eps_feat": EPS_FEAT, "clamp": CLAMP,
            "d_min": D_MIN, "chi2_tau": CHI2_TAU,
            "enforce_degree_constraint": ENFORCE_DEGREE_CONSTRAINT,
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
        },
        "data": ckpt_cfg["data"],
        "model_hparams": ckpt_cfg["model"],
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "AdaptedNETTACK",
        "model_name": MODEL_NAME,
        "dataset": DATASET,
        "classification": {
            "scope": "temporal",
            "aggregate_concat": concat_classification,
        },
        "attack_effect": {
            "attack_time_seconds": attack_time_seconds,
            "aggregate_concat": concat_attack,
        },
        "per_timestep": per_window,
        "surrogate": {
            "classification": {
                "scope": "temporal",
                "aggregate_concat": concat_surrogate_classification,
            },
            "attack_effect": {
                "aggregate_concat": concat_surrogate_attack,
            },
        },
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
