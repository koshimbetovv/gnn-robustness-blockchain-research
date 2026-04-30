import os
import sys
import json
import time
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.cosemignn_elliptic import load_cosemignn_elliptic
from src.datasets.cosemignn_ellipticpp_addraddr import load_cosemignn_ellipticpp_addraddr
from src.attacks.fgsm_cosemignn import CoSemiFGSMAttack
from src.utils.model_loader import resolve_checkpoint
from src.training.metrics import (
    binary_classification_metrics, attack_success_rate, roc_auc_binary,
    mean_confidence_drop, asr_pos_neg, mean_perturbation_l2_on_success,
)
from src.models.cosemignn import CoSemiGNN

# ---------- attack parameters ----------
MODEL_NAME = "cosemignn"
MODEL_DIR = "models/Elliptic++"
# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (165 raw + 6 semi = 171 features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 raw + 6 semi = 61 features)
DATASET = "ellipticpp_actors"
RUN_ID = None
EPS = 0.05
CLAMP = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = True
SEED = 0

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(model_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_fgsm_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_targets_slice(labels, pred_clean, only_illicit, only_clean_correct, fraction, seed):
    """Pick target indices within a single slice."""
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"ATTACK_FRACTION must be in (0, 1], got {fraction}")
    idx = torch.arange(labels.numel(), device=labels.device)
    # labels -1 are already filtered by cosemi loader, but be safe
    idx = idx[labels != -1]
    if only_illicit:
        idx = idx[labels[idx] == 1]
    if only_clean_correct and idx.numel() > 0:
        idx = idx[pred_clean[idx] == labels[idx]]
    if idx.numel() == 0:
        return idx
    n = max(1, min(int(round(float(fraction) * float(idx.numel()))), int(idx.numel())))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(idx.numel(), generator=g)
    return idx[perm[:n].to(idx.device)]


def main():
    device = get_device()

    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    data_cfg = ckpt_cfg["data"]
    time_cfg = ckpt_cfg["time"]
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
        raise ValueError(
            f"Unknown DATASET={DATASET!r}. Supported: 'elliptic', 'ellipticpp_actors'."
        )
    feature_list, adj_list, label_list, _ca_matrix_list, ca_weights_list, *_ = data

    # CoSemiGNN feature_in = raw + 6 semi. Elliptic=171, Elliptic++=61.
    # Find the first non-empty slice to read feature_in (loaders place None at idx 0
    # and may have None for empty slices).
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
    atk = CoSemiFGSMAttack(model, device, raw_feature_dim=raw_feature_dim, clamp=CLAMP)

    # Per-timestep results + pooled buffers for concatenated metrics
    per_slice = []
    pooled_y_true = []
    pooled_pred_clean = []
    pooled_pred_adv = []
    pooled_logits_clean = []
    pooled_logits_adv = []
    pooled_attack_mask = []
    pooled_delta_success = []
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
            logits_clean = atk.forward_logits(features, adj, ca_weights)
        pred_clean = logits_clean.argmax(dim=1)

        targets = pick_targets_slice(
            labels, pred_clean,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,  # vary seed per slice so random subsamples differ
        )
        if targets.numel() == 0:
            # Keep zero-target slices in the concatenated metrics so the pooled
            # evaluation covers the full test horizon, matching RecGNN and
            # EvolveGCN. Since no perturbation is applied, adv == clean here.
            per_slice.append({"t": t, "n_labeled": int(labels.numel()), "n_targets": 0, "skipped": True})
            pooled_y_true.append(labels.detach().cpu())
            pooled_pred_clean.append(pred_clean.detach().cpu())
            pooled_pred_adv.append(pred_clean.detach().cpu())
            pooled_logits_clean.append(logits_clean.detach().cpu())
            pooled_logits_adv.append(logits_clean.detach().cpu())
            pooled_attack_mask.append(torch.zeros(labels.numel(), dtype=torch.bool))
            continue

        labels_true = labels[targets].long()

        t0 = time.perf_counter()
        x_adv, logits_adv = atk.attack_slice(
            features, adj, targets, labels_true,
            eps=EPS, ca_weights=ca_weights,
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        pred_adv = logits_adv.argmax(dim=1)

        # Attack mask over the slice
        attack_mask = torch.zeros(labels.numel(), dtype=torch.bool, device=device)
        attack_mask[targets] = True

        asr, ns, na = attack_success_rate(labels, pred_clean, pred_adv, attack_mask)
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(labels, logits_clean, logits_adv, attack_mask)

        # Slice-level F1/ROC over ALL slice nodes (captures message-passing collateral)
        slice_mask = torch.ones(labels.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean, labels, slice_mask)
        roc_adv = roc_auc_binary(logits_adv, labels, slice_mask)
        clean_m = binary_classification_metrics(labels, pred_clean)
        adv_m = binary_classification_metrics(labels, pred_adv)

        conf_drop, n_used = mean_confidence_drop(
            labels, logits_clean, logits_adv, attack_mask, only_clean_correct=True
        )

        # L2 perturbation on successful flips (restrict to perturbable raw slice).
        clean_rows = features[targets, :atk.raw_dim]
        adv_rows = x_adv[targets, :atk.raw_dim]
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

        per_slice.append({
            "t": t,
            "n_labeled": int(labels.numel()),
            "n_targets": int(targets.numel()),
            "asr": asr, "asr_success": ns, "asr_attempted": na,
            "asr_pos": asr_p, "asr_pos_success": sp, "asr_pos_attempted": ap,
            "asr_neg": asr_n, "asr_neg_success": sn, "asr_neg_attempted": an,
            "f1_pos_clean": clean_m["f1_pos"], "f1_pos_adv": adv_m["f1_pos"], "f1_pos_drop": f1_pos_drop,
            "f1_macro_clean": clean_m["f1_macro"], "f1_macro_adv": adv_m["f1_macro"],
            "recall_pos_clean": clean_m["recall_pos"], "recall_pos_adv": adv_m["recall_pos"], "recall_pos_drop": recall_pos_drop,
            "roc_auc_clean": roc_clean, "roc_auc_adv": roc_adv,
            "mean_confidence_drop": conf_drop, "conf_drop_n": n_used,
            "pert_l2_success": pert_l2_mean, "pert_l2_n": pert_l2_n,
        })

        pooled_y_true.append(labels.detach().cpu())
        pooled_pred_clean.append(pred_clean.detach().cpu())
        pooled_pred_adv.append(pred_adv.detach().cpu())
        pooled_logits_clean.append(logits_clean.detach().cpu())
        pooled_logits_adv.append(logits_adv.detach().cpu())
        pooled_attack_mask.append(attack_mask.detach().cpu())

        print(
            f"t={t:2d}  n_labeled={labels.numel():5d}  n_targets={targets.numel():4d}  "
            f"ASR={asr:.4f} ({ns}/{na})  F1_pos {clean_m['f1_pos']:.3f}->{adv_m['f1_pos']:.3f}  "
            f"ROC {roc_clean if roc_clean==roc_clean else float('nan'):.3f}->{roc_adv if roc_adv==roc_adv else float('nan'):.3f}"
        )

    # ---------- aggregation ----------
    # Concatenate-over-timesteps: pool all slice rows and recompute metrics
    concat_classification = None
    concat_attack = None
    if pooled_y_true:
        y_true = torch.cat(pooled_y_true)
        pred_clean_c = torch.cat(pooled_pred_clean)
        pred_adv_c = torch.cat(pooled_pred_adv)
        logits_clean_c = torch.cat(pooled_logits_clean)
        logits_adv_c = torch.cat(pooled_logits_adv)

        clean_c = binary_classification_metrics(y_true, pred_clean_c)
        adv_c = binary_classification_metrics(y_true, pred_adv_c)
        full_mask = torch.ones(y_true.numel(), dtype=torch.bool)
        roc_c_clean = roc_auc_binary(logits_clean_c, y_true, full_mask)
        roc_c_adv = roc_auc_binary(logits_adv_c, y_true, full_mask)

        attack_mask_c = torch.cat(pooled_attack_mask)
        asr_c, ns_c, na_c = attack_success_rate(y_true, pred_clean_c, pred_adv_c, attack_mask_c)
        asr_cp, sp_c, ap_c, asr_cn, sn_c, an_c = asr_pos_neg(y_true, logits_clean_c, logits_adv_c, attack_mask_c)

        conf_drop_c, conf_drop_n_c = mean_confidence_drop(
            y_true, logits_clean_c, logits_adv_c, attack_mask_c, only_clean_correct=True
        )

        if pooled_delta_success:
            delta_all = torch.cat(pooled_delta_success, dim=0)
            pert_l2_c = float(torch.linalg.vector_norm(delta_all, ord=2, dim=1).mean().item())
            pert_l2_n_c = int(delta_all.size(0))
        else:
            pert_l2_c, pert_l2_n_c = 0.0, 0

        f1_pos_drop_c = float(clean_c["f1_pos"] - adv_c["f1_pos"])
        recall_pos_drop_c = float(clean_c["recall_pos"] - adv_c["recall_pos"])

        concat_classification = {
            "n_total": int(y_true.numel()),
            "f1_pos_clean": clean_c["f1_pos"], "f1_pos_adv": adv_c["f1_pos"], "f1_pos_drop": f1_pos_drop_c,
            "f1_macro_clean": clean_c["f1_macro"], "f1_macro_adv": adv_c["f1_macro"],
            "recall_pos_clean": clean_c["recall_pos"], "recall_pos_adv": adv_c["recall_pos"], "recall_pos_drop": recall_pos_drop_c,
            "roc_auc_clean": roc_c_clean, "roc_auc_adv": roc_c_adv,
        }
        concat_attack = {
            "n_targets_total": int(attack_mask_c.sum().item()),
            "asr": asr_c, "asr_success": ns_c, "asr_attempted": na_c,
            "asr_pos": asr_cp, "asr_pos_success": sp_c, "asr_pos_attempted": ap_c,
            "asr_neg": asr_cn, "asr_neg_success": sn_c, "asr_neg_attempted": an_c,
            "mean_confidence_drop": conf_drop_c, "conf_drop_n": conf_drop_n_c,
            "pert_l2_success": pert_l2_c, "pert_l2_n": pert_l2_n_c,
        }

    print()
    # print(f"FGSM CoSemiGNN | eps={EPS}")
    # if concat_classification is not None and concat_attack is not None:
    #     print(f"[concatenated across timesteps, n={concat_classification['n_total']}]")
    #     print(
    #         f"  ASR     : {concat_attack['asr']:.4f} "
    #         f"({concat_attack['asr_success']}/{concat_attack['asr_attempted']})"
    #     )
    #     print(
    #         f"  ASR_pos : {concat_attack['asr_pos']:.4f} "
    #         f"({concat_attack['asr_pos_success']}/{concat_attack['asr_pos_attempted']})  "
    #         f"ASR_neg : {concat_attack['asr_neg']:.4f} "
    #         f"({concat_attack['asr_neg_success']}/{concat_attack['asr_neg_attempted']})"
    #     )
    #     print(f"  F1_pos     : {concat_classification['f1_pos_clean']:.4f} -> {concat_classification['f1_pos_adv']:.4f}  (drop {concat_classification['f1_pos_drop']:.4f})")
    #     print(f"  Recall_pos : {concat_classification['recall_pos_clean']:.4f} -> {concat_classification['recall_pos_adv']:.4f}  (drop {concat_classification['recall_pos_drop']:.4f})")
    #     print(f"  F1_macro   : {concat_classification['f1_macro_clean']:.4f} -> {concat_classification['f1_macro_adv']:.4f}")
    #     print(f"  ROC-AUC    : {concat_classification['roc_auc_clean']:.4f} -> {concat_classification['roc_auc_adv']:.4f}")
    #     print(f"  Mean conf drop (clean-correct): {concat_attack['mean_confidence_drop']:.4f} over n={concat_attack['conf_drop_n']}")
    #     print(f"  Mean L2 pert (on flips)       : {concat_attack['pert_l2_success']:.4f} over n={concat_attack['pert_l2_n']}")
    # print(f"Attack time (total over test timesteps): {attack_time_seconds:.4f} s")

    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "FGSM",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "device": str(device),
        "attack_params": {
            "eps": EPS, "clamp": CLAMP,
            "raw_feature_dim": raw_feature_dim,
        },
        "timesteps": {"predict_start": predict_start, "predict_end": predict_end},
        "target_selection": {
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
        "attack": "FGSM",
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
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
