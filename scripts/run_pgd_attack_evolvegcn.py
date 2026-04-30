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
from src.attacks.pgd_evolvegcn import EvolveGCNPGDAttack
from src.utils.model_loader import resolve_checkpoint
from src.training.metrics import (
    binary_classification_metrics, attack_success_rate, roc_auc_binary,
    mean_confidence_drop, asr_pos_neg, mean_perturbation_l2_on_success,
)

# ---------- attack parameters ----------
MODEL_NAME = "evolvegcn_o"
MODEL_DIR = "models/Elliptic++"
# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (166 features incl. time_step at col 0)
#   "ellipticpp_actors"  -> Elliptic++ actors (pure wallet features, no metadata col)
DATASET = "ellipticpp_actors"
RUN_ID = None  # e.g. "seed43_20260331_133701" for a specific checkpoint; None picks latest
EPS = 0.05
ALPHA = 0.002
STEPS = 10
RANDOM_START = True
CLAMP = None
# Column 0 of the EvolveGCN Elliptic feature matrix is the IBM time_step metadata;
# protect it from perturbation (rewriting timestamps is not a plausible feature-space
# attack). Elliptic++ actors excludes the time_step column from features, so no
# metadata column to protect there. `None` selects the dataset-appropriate default.
ATTACK_START_COL = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = True
SEED = 0


def get_device():
    # EvolveGCN uses nn.RReLU, which MPS does not implement; training uses CPU
    # fallback (allow_mps=False). We match that here.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_run_dir(model_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_pgd_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_targets_window(
    label_idx: torch.Tensor,
    label_vals: torch.Tensor,
    pred_clean: torch.Tensor,
    only_illicit: bool,
    only_clean_correct: bool,
    fraction: float,
    seed: int,
):
    """Pick targets from the labeled nodes of a window.

    Inputs are 1-D tensors of length L (the window's labeled count):
      - `label_idx`  : global node ids (0..num_nodes-1)
      - `label_vals` : 0/1 labels at those positions
      - `pred_clean` : clean predictions at those positions

    Returns `(target_global, target_pos)` — global node ids and their positions
    within the labeled order (used to build an attack mask over `label_vals`).
    """
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"ATTACK_FRACTION must be in (0, 1], got {fraction}")
    pos = torch.arange(label_vals.numel(), device=label_vals.device)
    pos = pos[label_vals[pos] != -1]
    if only_illicit:
        pos = pos[label_vals[pos] == 1]
    if only_clean_correct and pos.numel() > 0:
        pos = pos[pred_clean[pos] == label_vals[pos]]
    if pos.numel() == 0:
        return pos, pos
    n = max(1, min(int(round(float(fraction) * float(pos.numel()))), int(pos.numel())))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(pos.numel(), generator=g)
    sel = pos[perm[:n].to(pos.device)]
    return label_idx[sel], sel


def main():
    device = get_device()

    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    if DATASET == "elliptic":
        data_cfg = EvolveGCNEllipticConfig(**ckpt_cfg["data"])
        print(f"Loading EvolveGCN Elliptic sequence (filter_unknown={data_cfg.filter_unknown}) ...")
        sequence = EvolveGCNEllipticDataset(data_cfg).get_sequence()
        default_attack_start_col = 1  # col 0 = IBM time_step metadata
    elif DATASET == "ellipticpp_actors":
        data_cfg = EvolveGCNActorsConfig(**ckpt_cfg["data"])
        print(f"Loading EvolveGCN Elliptic++ Actors sequence (filter_unknown={data_cfg.filter_unknown}) ...")
        sequence = EvolveGCNEllipticPPActorsDataset(data_cfg).get_sequence()
        default_attack_start_col = 0  # no metadata col; time_step excluded from features
    else:
        raise ValueError(
            f"Unknown DATASET={DATASET!r}. Supported: 'elliptic', 'ellipticpp_actors'."
        )

    attack_start_col = default_attack_start_col if ATTACK_START_COL is None else int(ATTACK_START_COL)

    model = build_evolvegcn_model(sequence.num_features, ckpt_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"✓ Loaded {MODEL_NAME.upper()} from {ckpt_path}")
    print(
        f"Test windows: {len(sequence.test_samples)} | num_nodes={sequence.num_nodes} | "
        f"num_features={sequence.num_features} | attack_start_col={attack_start_col}"
    )

    atk = EvolveGCNPGDAttack(
        model, device,
        attack_start_col=attack_start_col,
        clamp=CLAMP,
    )

    per_window = []
    pooled_y_true = []
    pooled_pred_clean = []
    pooled_pred_adv = []
    pooled_logits_clean = []
    pooled_logits_adv = []
    pooled_attack_mask = []
    pooled_delta_success = []  # rows of applied perturbation on successful flips
    attack_time_seconds = 0.0

    for sample in sequence.test_samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        t = int(sample.current_time)

        with torch.no_grad():
            logits_clean = atk.forward_labels(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
        pred_clean = logits_clean.argmax(dim=1)

        target_global, target_pos = pick_targets_window(
            label_idx, label_vals, pred_clean,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )
        if target_global.numel() == 0:
            # No attackable targets in this window — still pool clean logits so
            # the concatenated baseline covers the full test split; adv==clean
            # here since no perturbation was applied.
            per_window.append({"t": t, "n_labeled": int(label_idx.numel()), "n_targets": 0, "skipped": True})
            pooled_y_true.append(label_vals.detach().cpu())
            pooled_pred_clean.append(pred_clean.detach().cpu())
            pooled_pred_adv.append(pred_clean.detach().cpu())
            pooled_logits_clean.append(logits_clean.detach().cpu())
            pooled_logits_adv.append(logits_clean.detach().cpu())
            pooled_attack_mask.append(torch.zeros(label_idx.numel(), dtype=torch.bool))
            continue

        labels_true = label_vals[target_pos].long()

        t0 = time.perf_counter()
        hist_adv, logits_adv = atk.attack_window(
            hist_adj_list, hist_ndFeats_list, node_mask_list,
            label_idx, target_global, labels_true,
            eps=EPS, alpha=ALPHA, steps=STEPS, random_start=RANDOM_START,
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        pred_adv = logits_adv.argmax(dim=1)

        attack_mask = torch.zeros(label_idx.numel(), dtype=torch.bool, device=device)
        attack_mask[target_pos] = True

        asr, ns, na = attack_success_rate(label_vals, pred_clean, pred_adv, attack_mask)
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(label_vals, logits_clean, logits_adv, attack_mask)

        full_mask = torch.ones(label_idx.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean, label_vals, full_mask)
        roc_adv = roc_auc_binary(logits_adv, label_vals, full_mask)
        clean_m = binary_classification_metrics(label_vals, pred_clean)
        adv_m = binary_classification_metrics(label_vals, pred_adv)

        conf_drop, n_used = mean_confidence_drop(
            label_vals, logits_clean, logits_adv, attack_mask, only_clean_correct=True
        )

        # Per-target L2 perturbation on successful clean-correct flips. Only
        # columns [attack_start_col:] are actually perturbed by PGD, so we
        # restrict the diff to that slice.
        clean_rows = hist_ndFeats_list[-1][target_global, attack_start_col:]
        adv_rows = hist_adv[-1][target_global, attack_start_col:]
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

        per_window.append({
            "t": t,
            "n_labeled": int(label_idx.numel()),
            "n_targets": int(target_global.numel()),
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

        pooled_y_true.append(label_vals.detach().cpu())
        pooled_pred_clean.append(pred_clean.detach().cpu())
        pooled_pred_adv.append(pred_adv.detach().cpu())
        pooled_logits_clean.append(logits_clean.detach().cpu())
        pooled_logits_adv.append(logits_adv.detach().cpu())
        pooled_attack_mask.append(attack_mask.detach().cpu())

        roc_c_show = roc_clean if roc_clean == roc_clean else float("nan")
        roc_a_show = roc_adv if roc_adv == roc_adv else float("nan")
        print(
            f"t={t:2d}  n_labeled={label_idx.numel():5d}  n_targets={target_global.numel():4d}  "
            f"ASR={asr:.4f} ({ns}/{na})  F1_pos {clean_m['f1_pos']:.3f}->{adv_m['f1_pos']:.3f}  "
            f"ROC {roc_c_show:.3f}->{roc_a_show:.3f}"
        )

    # ---------- aggregation ----------
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
    # print(f"PGD EvolveGCN-O | eps={EPS} alpha={ALPHA} steps={STEPS} random_start={RANDOM_START}")
    # if concat_classification is not None and concat_attack is not None:
    #     print(f"[concatenated across all test windows, n={concat_classification['n_total']}]")
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
    # print(f"Attack time (total over test windows): {attack_time_seconds:.4f} s")

    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "PGD",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "device": str(device),
        "attack_params": {
            "eps": EPS, "alpha": ALPHA, "steps": STEPS, "random_start": RANDOM_START,
            "clamp": CLAMP,
            "attack_start_col": attack_start_col,
        },
        "target_selection": {
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
        "attack": "PGD",
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
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
