import os
import sys
import json
import time
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.cosemignn_elliptic import load_cosemignn_elliptic
from src.datasets.cosemignn_ellipticpp_addraddr import load_cosemignn_ellipticpp_addraddr
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
from src.models.cosemignn import CoSemiGNN

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
CHI2_TAU = 0.004
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


def cosemi_forward_logits(model, features, edge_index, ca_weights):
    """Lift CoSemiGNN's BCE-style `(N,)` output to `(N, 2)` `[0, s]`, matching
    the convention used by `pgd_cosemignn.py`. Argmax and CE on `[0, s]` agree
    with BCE-with-logits on `s`.
    """
    out_line, _ = model(features, edge_index, ca_weights)
    return torch.stack([torch.zeros_like(out_line), out_line], dim=1)


def main():
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

    # Determine feature dimensions from the first non-empty slice. CoSemiGNN
    # features are `[raw (D_raw) || semi (6)]`; only the leading raw_dim columns
    # are perturbable (the 6 semi columns come from a non-differentiable cached
    # auxiliary classifier, just like in PGD-CoSemi).
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

    # ----- train surrogate on union of train timesteps -----
    print("Training surrogate (linearized GCN) on union of train timesteps ...")
    train_slices = []
    for t in range(train_start, train_end):
        if t >= len(feature_list):
            break
        ft = feature_list[t]; adj = adj_list[t]; lbl = label_list[t]
        if ft is None or lbl is None or ft.numel() == 0 or lbl.numel() == 0:
            continue
        # CoSemiGNN labels are 0/1 (no -1), but be defensive.
        train_mask = (lbl != -1)
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

    # NETTACK perturbs the leading raw_feature_dim columns; the trailing 6 semi
    # columns are frozen oracle outputs (matching PGD-CoSemi convention). The
    # surrogate is trained and queried on raw features only to avoid using the
    # label-derived semi-supervised teacher columns as attack-side information.
    atk = AdaptedNettackTemporalAttack(
        W=W, device=device,
        attack_dim=raw_feature_dim,
        clamp=CLAMP,
        d_min=D_MIN, chi2_tau=CHI2_TAU,
        enforce_degree_constraint=ENFORCE_DEGREE_CONSTRAINT,
        structure_mode=STRUCTURE_MODE,
        directed_edge_direction=DIRECTED_EDGE_DIRECTION,
        verbose=VERBOSE, progress_every=PROGRESS_EVERY,
    )

    # ----- per-timestep attack -----
    # NOTE: CoSemiGNN's CMOS LSTM advances state on every forward; we follow the
    # same pattern as `run_pgd_attack_cosemignn.py` and do not save/restore
    # state. This means the adv forward starts from a slightly different state
    # than the clean forward did, but it matches the existing benchmark.
    per_slice = []
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
        pred_clean = logits_clean.argmax(dim=1)
        pred_surrogate_clean = logits_surrogate_clean.argmax(dim=1)

        # Pick targets among labeled positions.
        idx = torch.arange(labels.numel(), device=device)
        idx = idx[labels[idx] != -1]
        if ATTACK_ONLY_ILLICIT:
            idx = idx[labels[idx] == 1]
        if ONLY_CLEAN_CORRECT and idx.numel() > 0:
            idx = idx[pred_surrogate_clean[idx] == labels[idx]]

        if idx.numel() == 0:
            per_slice.append({"t": t, "n_labeled": int(labels.numel()), "n_targets": 0, "skipped": True})
            pooled_y_true.append(labels.detach().cpu())
            pooled_pred_clean.append(pred_clean.detach().cpu())
            pooled_pred_adv.append(pred_clean.detach().cpu())
            pooled_logits_clean.append(logits_clean.detach().cpu())
            pooled_logits_adv.append(logits_clean.detach().cpu())
            pooled_attack_mask.append(torch.zeros(labels.numel(), dtype=torch.bool))
            pooled_surrogate_pred_clean.append(pred_surrogate_clean.detach().cpu())
            pooled_surrogate_pred_adv.append(pred_surrogate_clean.detach().cpu())
            pooled_surrogate_logits_clean.append(logits_surrogate_clean.detach().cpu())
            pooled_surrogate_logits_adv.append(logits_surrogate_clean.detach().cpu())
            continue

        if 0.0 < float(ATTACK_FRACTION) < 1.0:
            n = max(1, min(int(round(ATTACK_FRACTION * float(idx.numel()))), int(idx.numel())))
            g = torch.Generator(device="cpu")
            g.manual_seed(int(SEED + t))
            perm = torch.randperm(idx.numel(), generator=g)
            idx = idx[perm[:n].to(idx.device)]

        targets = idx

        t0 = time.perf_counter()
        x_raw_adv, edge_index_adv, info = atk.attack_slice(
            features[:, :raw_feature_dim].float(), adj.long(), labels.long(), targets,
            n_struct=N_STRUCT,
            eps_feat=EPS_FEAT,
            time_step=torch.zeros(features.size(0), dtype=torch.long, device=device),
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        n_unique_added_total += int(info["n_unique_edges_added"])
        n_targets_with_edge_total += int(info["n_targets_with_edge_added"])

        x_adv = features.clone()
        x_adv[:, :raw_feature_dim] = x_raw_adv

        with torch.no_grad():
            logits_adv = cosemi_forward_logits(model, x_adv, edge_index_adv, ca_weights).detach()
            logits_surrogate_adv = linearized_surrogate_logits(
                x_raw_adv, edge_index_adv, W, structure_mode=STRUCTURE_MODE,
            )
        pred_adv = logits_adv.argmax(dim=1)
        pred_surrogate_adv = logits_surrogate_adv.argmax(dim=1)

        attack_mask = torch.zeros(labels.numel(), dtype=torch.bool, device=device)
        attack_mask[targets] = True

        asr, ns, na = attack_success_rate(labels, pred_clean, pred_adv, attack_mask)
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(labels, logits_clean, logits_adv, attack_mask)
        surrogate_asr, surrogate_ns, surrogate_na = attack_success_rate(
            labels, pred_surrogate_clean, pred_surrogate_adv, attack_mask,
        )
        surrogate_asr_p, surrogate_sp, surrogate_ap, surrogate_asr_n, surrogate_sn, surrogate_an = asr_pos_neg(
            labels, logits_surrogate_clean, logits_surrogate_adv, attack_mask,
        )

        slice_mask = torch.ones(labels.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean, labels, slice_mask)
        roc_adv = roc_auc_binary(logits_adv, labels, slice_mask)
        clean_m = binary_classification_metrics(labels, pred_clean)
        adv_m = binary_classification_metrics(labels, pred_adv)
        surrogate_clean_m = binary_classification_metrics(labels, pred_surrogate_clean)
        surrogate_adv_m = binary_classification_metrics(labels, pred_surrogate_adv)

        conf_drop, n_used = mean_confidence_drop(
            labels, logits_clean, logits_adv, attack_mask, only_clean_correct=True,
        )

        clean_rows = features[targets, :raw_feature_dim]
        adv_rows = x_adv[targets, :raw_feature_dim]
        pred_clean_targets = pred_clean[targets]
        pred_adv_targets = pred_adv[targets]
        y_targets = labels[targets].long()
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

        per_slice.append({
            "t": t,
            "n_labeled": int(labels.numel()),
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

        pooled_y_true.append(labels.detach().cpu())
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
            f"t={t:2d}  n_labeled={labels.numel():5d}  n_targets={targets.numel():4d}  "
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
        },
        "data": data_cfg,
        "model_hparams": model_cfg,
        "semi_cache_dir": semi_cache_dir,
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
        "per_timestep": per_slice,
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
