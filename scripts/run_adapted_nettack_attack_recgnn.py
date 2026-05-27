import os
import sys
import json
import time
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.recgnn_elliptic import RecGNNEllipticConfig, RecGNNEllipticDataset
from src.datasets.recgnn_ellipticpp_actors import (
    RecGNNEllipticPPActorsConfig, RecGNNEllipticPPActorsDataset,
)
from src.attacks.nettack_adapted_temporal import (
    AdaptedNettackTemporalAttack, linearized_surrogate_logits,
    train_surrogate_on_train_slices,
)
from src.utils.model_loader import load_model, resolve_checkpoint
from src.utils.seed import set_seed
from src.training.metrics import (
    binary_classification_metrics, attack_success_rate, roc_auc_binary,
    mean_confidence_drop, asr_pos_neg, mean_perturbation_l2_on_success,
)

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

# Number of controllable feature columns. RecGNN appends 2 ANF (antecedent
# neighbour-label count) columns derived from the graph; they're not directly
# controllable by perturbing the target's own feature row, so NETTACK only
# touches the leading raw slice. `None` -> dataset-appropriate default.
ATTACK_DIM = None

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
    if torch.backends.mps.is_available():
        return torch.device("mps")
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
    def _c(t): return t.detach().clone() if t is not None else None
    return {
        "h": _c(ml._h_state), "c": _c(ml._c_state),
        "ev_h": _c(ev._row_h), "ev_c": _c(ev._row_c),
        "ev_w": _c(ev._current_weight),
    }


def _restore_state(model, snap):
    ml = model.m_lstm
    ev = ml.cell.evolve_linear
    def _c(t): return t.detach().clone() if t is not None else None
    ml._h_state = _c(snap["h"]); ml._c_state = _c(snap["c"])
    ev._row_h = _c(snap["ev_h"]); ev._row_c = _c(snap["ev_c"])
    ev._current_weight = _c(snap["ev_w"])


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

    model = load_model(MODEL_NAME, sequence.num_features, 2, device=device,
                       model_dir=MODEL_DIR, run_id=RUN_ID)
    print(f"Train graphs: {len(sequence.train_graphs)}  Test graphs: {len(sequence.test_graphs)}  "
          f"num_features={sequence.num_features}  attack_dim={attack_dim}")

    # ----- train surrogate on union of train timesteps -----
    print("Training surrogate (linearized GCN) on union of train timesteps ...")
    train_slices = []
    for g in sequence.train_graphs:
        g = g.to(device)
        # Use ALL labeled nodes in the train slice as surrogate-training data.
        train_mask = (g.y != -1)
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
    )

    atk = AdaptedNettackTemporalAttack(
        W=W, device=device,
        attack_dim=attack_dim,
        clamp=CLAMP,
        d_min=D_MIN, chi2_tau=CHI2_TAU,
        enforce_degree_constraint=ENFORCE_DEGREE_CONSTRAINT,
        verbose=VERBOSE, progress_every=PROGRESS_EVERY,
    )

    # ----- prime sequence state on clean train graphs -----
    print("Priming sequence state over train graphs ...")
    model.reset_sequence_state(device)
    with torch.no_grad():
        for g in sequence.train_graphs:
            g = g.to(device)
            _ = model(g.x.float(), g.edge_index.long())
            model.detach_sequence_state()

    # ----- per-timestep attack -----
    per_timestep = []
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

    for graph in sequence.test_graphs:
        g = graph.to(device)
        x = g.x.float()
        edge_index = g.edge_index.long()
        y = g.y
        t = int(g.graph_timestep)

        labeled_mask = y != -1
        if int(labeled_mask.sum().item()) == 0:
            # Advance state along clean trajectory and skip.
            with torch.no_grad():
                _ = model(x, edge_index)
                model.detach_sequence_state()
            per_timestep.append({"t": t, "n_labeled": 0, "n_targets": 0, "skipped": True})
            continue

        # Save pre-state, run clean forward, save post-state.
        snap_pre = _save_state(model)
        with torch.no_grad():
            log_probs_clean = model(x, edge_index).detach()
            model.detach_sequence_state()
        snap_post = _save_state(model)
        pred_clean = log_probs_clean.argmax(dim=1)
        with torch.no_grad():
            logits_surrogate_clean = linearized_surrogate_logits(x, edge_index, W)
        pred_surrogate_clean = logits_surrogate_clean.argmax(dim=1)

        targets = pick_targets_graph(
            y, pred_surrogate_clean,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )

        if targets.numel() == 0:
            # No attackable targets: keep clean trajectory, pool clean as adv.
            _restore_state(model, snap_post)
            per_timestep.append({"t": t, "n_labeled": int(labeled_mask.sum().item()),
                                 "n_targets": 0, "skipped": True})
            pooled_y_true.append(y[labeled_mask].detach().cpu())
            pooled_pred_clean.append(pred_clean[labeled_mask].detach().cpu())
            pooled_pred_adv.append(pred_clean[labeled_mask].detach().cpu())
            pooled_logits_clean.append(log_probs_clean[labeled_mask].detach().cpu())
            pooled_logits_adv.append(log_probs_clean[labeled_mask].detach().cpu())
            pooled_attack_mask.append(torch.zeros(int(labeled_mask.sum().item()), dtype=torch.bool))
            pooled_surrogate_pred_clean.append(pred_surrogate_clean[labeled_mask].detach().cpu())
            pooled_surrogate_pred_adv.append(pred_surrogate_clean[labeled_mask].detach().cpu())
            pooled_surrogate_logits_clean.append(logits_surrogate_clean[labeled_mask].detach().cpu())
            pooled_surrogate_logits_adv.append(logits_surrogate_clean[labeled_mask].detach().cpu())
            continue

        # NETTACK queries only the (stateless) surrogate -- no victim state touched.
        t0 = time.perf_counter()
        x_adv, edge_index_adv, info = atk.attack_slice(
            x, edge_index, y, targets, n_struct=N_STRUCT, eps_feat=EPS_FEAT,
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        n_unique_added_total += int(info["n_unique_edges_added"])
        n_targets_with_edge_total += int(info["n_targets_with_edge_added"])

        # Restore pre-state, run adv forward.
        _restore_state(model, snap_pre)
        with torch.no_grad():
            log_probs_adv = model(x_adv, edge_index_adv).detach()
            model.detach_sequence_state()
            logits_surrogate_adv = linearized_surrogate_logits(x_adv, edge_index_adv, W)
        # Restore post-state so the next test timestep continues on the clean trajectory.
        _restore_state(model, snap_post)

        pred_adv = log_probs_adv.argmax(dim=1)
        pred_surrogate_adv = logits_surrogate_adv.argmax(dim=1)
        y_lab = y[labeled_mask]
        pred_clean_lab = pred_clean[labeled_mask]
        pred_adv_lab = pred_adv[labeled_mask]
        logits_clean_lab = log_probs_clean[labeled_mask]
        logits_adv_lab = log_probs_adv[labeled_mask]
        pred_surrogate_clean_lab = pred_surrogate_clean[labeled_mask]
        pred_surrogate_adv_lab = pred_surrogate_adv[labeled_mask]
        logits_surrogate_clean_lab = logits_surrogate_clean[labeled_mask]
        logits_surrogate_adv_lab = logits_surrogate_adv[labeled_mask]

        attack_mask_full = torch.zeros(y.numel(), dtype=torch.bool, device=device)
        attack_mask_full[targets] = True
        attack_mask_lab = attack_mask_full[labeled_mask]

        asr, ns, na = attack_success_rate(y_lab, pred_clean_lab, pred_adv_lab, attack_mask_lab)
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(y_lab, logits_clean_lab, logits_adv_lab, attack_mask_lab)
        surrogate_asr, surrogate_ns, surrogate_na = attack_success_rate(
            y_lab, pred_surrogate_clean_lab, pred_surrogate_adv_lab, attack_mask_lab,
        )
        surrogate_asr_p, surrogate_sp, surrogate_ap, surrogate_asr_n, surrogate_sn, surrogate_an = asr_pos_neg(
            y_lab, logits_surrogate_clean_lab, logits_surrogate_adv_lab, attack_mask_lab,
        )

        full_mask_lab = torch.ones(y_lab.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean_lab, y_lab, full_mask_lab)
        roc_adv = roc_auc_binary(logits_adv_lab, y_lab, full_mask_lab)
        clean_m = binary_classification_metrics(y_lab, pred_clean_lab)
        adv_m = binary_classification_metrics(y_lab, pred_adv_lab)
        surrogate_clean_m = binary_classification_metrics(y_lab, pred_surrogate_clean_lab)
        surrogate_adv_m = binary_classification_metrics(y_lab, pred_surrogate_adv_lab)

        conf_drop, n_used = mean_confidence_drop(
            y_lab, logits_clean_lab, logits_adv_lab, attack_mask_lab, only_clean_correct=True,
        )

        pert_dim = int(info["perturbable_dim"])
        clean_rows = x[targets, :pert_dim]
        adv_rows = x_adv[targets, :pert_dim]
        pred_clean_targets = pred_clean[targets]
        pred_adv_targets = pred_adv[targets]
        y_targets = y[targets].long()
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

        per_timestep.append({
            "t": t,
            "n_labeled": int(y_lab.numel()),
            "n_targets": int(targets.numel()),
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

        pooled_y_true.append(y_lab.detach().cpu())
        pooled_pred_clean.append(pred_clean_lab.detach().cpu())
        pooled_pred_adv.append(pred_adv_lab.detach().cpu())
        pooled_logits_clean.append(logits_clean_lab.detach().cpu())
        pooled_logits_adv.append(logits_adv_lab.detach().cpu())
        pooled_attack_mask.append(attack_mask_lab.detach().cpu())
        pooled_surrogate_pred_clean.append(pred_surrogate_clean_lab.detach().cpu())
        pooled_surrogate_pred_adv.append(pred_surrogate_adv_lab.detach().cpu())
        pooled_surrogate_logits_clean.append(logits_surrogate_clean_lab.detach().cpu())
        pooled_surrogate_logits_adv.append(logits_surrogate_adv_lab.detach().cpu())

        roc_c_show = roc_clean if roc_clean == roc_clean else float("nan")
        roc_a_show = roc_adv if roc_adv == roc_adv else float("nan")
        print(
            f"t={t:2d}  n_labeled={y_lab.numel():5d}  n_targets={targets.numel():4d}  "
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
        "per_timestep": per_timestep,
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
