import os
import sys
import time
import torch
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset, EllipticConfig
from src.datasets.ellipticpp_actors import EllipticPPActorsDataset, EllipticPPActorsConfig
from src.attacks.fgsm import FGSMAttack
from src.attacks.model_forward import forward_logits, STATIC_MODELS
from src.utils.model_loader import load_model
from src.training.metrics import (
    get_split_mask, evaluate_logits_on_split, attack_success_rate,
    roc_auc_binary, mean_confidence_drop, asr_pos_neg,
    mean_perturbation_l2_on_success,
)
from src.utils.attack_targets import pick_target_nodes

# ---------- attack parameters ----------
MODEL_NAME = "graphsage"
MODEL_DIR = "models/Elliptic++"  # "models/Elliptic" or "models/Elliptic++"
# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (165 tx features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 wallet features)
DATASET = "ellipticpp_actors"
SPLIT = "test"
EPS = 0.05
CLAMP = None  # e.g. (-3.0, 3.0)

# ---------- target selection controls ----------
# If True, attack only illicit nodes (y==1) in the chosen split.
ATTACK_ONLY_ILLICIT = True
# What fraction of eligible nodes in the split to attack. 
ATTACK_FRACTION = 1.0
# If True, restrict targets to nodes the model classifies correctly on clean inputs.
ONLY_CLEAN_CORRECT = True
SEED = 0

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def make_run_dir(model_name: str):
    """Create attacks/model_YYYYMMDD_HHMMSS/ under repo root and return (run_dir, timestamp)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_fgsm_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts

def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def main():
    if MODEL_NAME not in STATIC_MODELS:
        raise NotImplementedError(
            f"{MODEL_NAME!r} is not a static-feature model supported by this FGSM driver. "
            f"Temporal/sequence models (e.g. recgnn, evolvegcn_o) require a different threat "
            f"model; use a temporal attack script. Supported here: {STATIC_MODELS}."
        )

    device = get_device()
    if DATASET == "elliptic":
        data = EllipticDataset(EllipticConfig(filter_unknown=False)).get_data()
    elif DATASET == "ellipticpp_actors":
        data = EllipticPPActorsDataset(EllipticPPActorsConfig(filter_unknown=False)).get_data()
    else:
        raise ValueError(
            f"Unknown DATASET={DATASET!r}. Supported: 'elliptic', 'ellipticpp_actors'."
        )

    # Model-specific feature preprocessing. ChronoWaveGNN was trained on
    # [standardized_raw || standardized_Haar-level2(raw)]; we must reproduce that
    # layout before moving to device and before loading the model.
    attack_dim = None
    rebuild_fn = None
    if MODEL_NAME == "chronowave_gnn":
        from src.datasets.chronowave_features import build_paper_features, make_consistent_rebuild
        build_paper_features(data)

    data = data.to(device)

    if MODEL_NAME == "chronowave_gnn":
        attack_dim = int(data.raw_feature_dim)
        rebuild_fn = make_consistent_rebuild(data)

    model = load_model(MODEL_NAME, data.num_features, 2, device=device, model_dir=MODEL_DIR)

    time_step = getattr(data, "time_step", None)

    split_mask = get_split_mask(data, SPLIT).to(device)
    with torch.no_grad():
        logits_clean = forward_logits(model, data.x, data.edge_index, time_step=time_step)

    targets = pick_target_nodes(
        data, logits_clean, split_mask,
        only_illicit=ATTACK_ONLY_ILLICIT,
        fraction=ATTACK_FRACTION,
        only_clean_correct=ONLY_CLEAN_CORRECT,
        seed=SEED,
        device=device,
    )
    if targets.numel() == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    atk = FGSMAttack(model, data, device, clamp=CLAMP, attack_dim=attack_dim, rebuild_fn=rebuild_fn)
    t_start = time.perf_counter()
    x_adv = atk.attack(targets, eps=EPS)
    attack_time_seconds = float(time.perf_counter() - t_start)

    with torch.no_grad():
        logits_adv = forward_logits(model, x_adv, data.edge_index, time_step=time_step)

    # Optional: evaluate only on attacked nodes (recommended, otherwise it gets diluted)
    attack_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    attack_mask[targets] = True
    attack_mask = attack_mask & split_mask  # keep inside split

    # Split-wide metrics (include message-passing collateral on non-attacked nodes)
    roc_clean_split = roc_auc_binary(logits_clean, data.y, split_mask)
    roc_adv_split = roc_auc_binary(logits_adv, data.y, split_mask)
    clean_m_split = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    adv_m_split = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)

    conf_drop, n_used = mean_confidence_drop(
        data.y, logits_clean, logits_adv, attack_mask, only_clean_correct=True
    )

    # ASR over attacked subset (recommended when ATTACK_FRACTION < 1)
    asr, ns, na = attack_success_rate(data.y, logits_clean.argmax(1), logits_adv.argmax(1), attack_mask)

    asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(data.y, logits_clean, logits_adv, attack_mask)

    # Mean L2 perturbation over successful label flips (clean-correct → wrong).
    # Restrict to the perturbable slice [:, :atk.attack_dim] since the FGSM
    # driver only touches those columns.
    pert_dim = int(atk.attack_dim)
    clean_rows = data.x[targets, :pert_dim]
    adv_rows = x_adv[targets, :pert_dim]
    pred_clean_targets = logits_clean[targets].argmax(1)
    pred_adv_targets = logits_adv[targets].argmax(1)
    y_targets = data.y[targets].long()
    pert_l2_mean, pert_l2_n = mean_perturbation_l2_on_success(
        clean_rows, adv_rows, pred_clean_targets, pred_adv_targets, y_targets,
    )

    # Drop metrics (paper-ready: expose directly instead of requiring subtraction).
    f1_pos_drop_split = float(clean_m_split.f1_pos - adv_m_split.f1_pos)
    recall_pos_drop_split = float(clean_m_split.recall_pos - adv_m_split.recall_pos)

    print()
    # print(f"FGSM | eps={EPS}")

    # print(f"ASR={asr:.6f} ({ns}/{na})")
    # print(f"ASR_pos (illicit flips) = {asr_p:.6f} ({sp}/{ap}), ASR_neg (licit flips) = {asr_n:.6f} ({sn}/{an})")

    # print()
    # print(f"[split={SPLIT}, n={clean_m_split.n_labeled}]")
    # print(f"  F1_pos     : {clean_m_split.f1_pos:.4f} -> {adv_m_split.f1_pos:.4f}  (drop {f1_pos_drop_split:.4f})")
    # print(f"  Recall_pos : {clean_m_split.recall_pos:.4f} -> {adv_m_split.recall_pos:.4f}  (drop {recall_pos_drop_split:.4f})")
    # print(f"  F1_macro   : {clean_m_split.f1_macro:.4f} -> {adv_m_split.f1_macro:.4f} (drop {clean_m_split.f1_macro-adv_m_split.f1_macro:.4f})")
    # print(f"  ROC-AUC    : {roc_clean_split:.6f} -> {roc_adv_split:.6f}")

    # print(f"Mean confidence drop (attacked, clean-correct): {conf_drop:.6f} over n={n_used}")
    # print(f"Mean L2 perturbation on successful flips: {pert_l2_mean:.6f} over n={pert_l2_n}")
    # print(f"Attack time: {attack_time_seconds:.4f} s")


    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "FGSM",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "split": SPLIT,
        "device": str(device),
        "attack_params": {
            "eps": EPS,
            "clamp": CLAMP,
        },
        "target_selection": {
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
            "n_targets": int(targets.numel()),
        },
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "FGSM",
        "model_name": MODEL_NAME,
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
        },
        "attack_effect": {
            "n_targets": int(targets.numel()),
            "attack_mask_n": int(attack_mask.sum().item()),
            "attack_time_seconds": attack_time_seconds,
            "asr": {"value": asr, "success": ns, "attempted": na},
            "asr_pos_neg": {
                "asr_pos": asr_p, "succ_pos": sp, "attempted_pos": ap,
                "asr_neg": asr_n, "succ_neg": sn, "attempted_neg": an,
            },
            "mean_confidence_drop": {"value": conf_drop, "n": n_used},
            "perturbation_l2_on_success": {"value": pert_l2_mean, "n_flipped": pert_l2_n},
        },
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    print()
    print(f"Saved to attacks/")



if __name__ == "__main__":
    main()
