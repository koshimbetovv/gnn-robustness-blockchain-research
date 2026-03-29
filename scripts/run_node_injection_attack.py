import os
import sys
import json
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.utils.model_loader import load_model
from src.attacks.node_injection_evasion import NodeInjectionEvasionAttack
from src.training.metrics import (
    get_split_mask,
    evaluate_logits_on_split,
    attack_success_rate,
    asr_pos_neg,
    roc_auc_binary,
    mean_confidence_drop,
)

# ---------- model / split ----------
MODEL_NAME = "gcn"
SPLIT = "test"

# ---------- node-injection attack hyperparams ----------
N_INJECT = 1
EDGES_PER_INJECTED = 5
EPS = 0.05
ALPHA = 0.01
STEPS = 30
RANDOM_START = True
INIT = "mean"          # "mean" or "randn"
CLAMP = None           # e.g. (-3.0, 3.0)
TARGETED = False
TARGET_LABEL = 0
EARLY_STOP = True
CONNECT_STRATEGY = "round_robin"  # "round_robin" or "all_to_all"

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True        # if True, only y==1 nodes
ATTACK_FRACTION = 0.1            # fraction of eligible split nodes to attack
ONLY_CLEAN_CORRECT = True         # recommended for ASR meaning
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
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_node_injection_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_target_nodes(
    data,
    logits_clean: torch.Tensor,
    split_mask: torch.Tensor,
    *,
    only_illicit: bool,
    fraction: float,
    only_clean_correct: bool,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"ATTACK_FRACTION must be in (0, 1], got {fraction}")

    mask = split_mask.bool() & (data.y != -1)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return idx.to(device)

    if only_illicit:
        idx = idx[data.y[idx] == 1]

    if only_clean_correct and idx.numel() > 0:
        pred_clean = logits_clean.argmax(dim=1)
        idx = idx[pred_clean[idx] == data.y[idx]]

    if idx.numel() == 0:
        return idx.to(device)

    n = int(round(float(fraction) * float(idx.numel())))
    n = max(1, min(n, int(idx.numel())))

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    idx_cpu = idx.detach().cpu()
    perm = torch.randperm(idx_cpu.numel(), generator=g)

    return idx_cpu[perm[:n]].to(device)


def main():
    device = get_device()
    torch.manual_seed(SEED)

    dataset = EllipticDataset()
    data = dataset.get_data().to(device)

    model = load_model(MODEL_NAME, data.num_features, 2, device=device)

    split_mask = get_split_mask(data, SPLIT).to(device)

    with torch.no_grad():
        logits_clean = model(data.x, data.edge_index)

    targets = pick_target_nodes(
        data,
        logits_clean,
        split_mask,
        only_illicit=ATTACK_ONLY_ILLICIT,
        fraction=ATTACK_FRACTION,
        only_clean_correct=ONLY_CLEAN_CORRECT,
        seed=SEED,
        device=device,
    )

    if targets.numel() == 0:
        print("No eligible target nodes found for the chosen settings.")
        return

    # For attacked-only ASR/metrics
    attack_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    attack_mask[targets] = True
    attack_mask = attack_mask & split_mask


    # --- run attack ---
    atk = NodeInjectionEvasionAttack(model, data, device, clamp=CLAMP, seed=SEED)
    res = atk.attack(
        target_nodes=targets,
        n_inject=N_INJECT,
        edges_per_injected=EDGES_PER_INJECTED,
        eps=EPS,
        alpha=ALPHA,
        steps=STEPS,
        random_start=RANDOM_START,
        init=INIT,
        reference_nodes=targets,  # good default: initialize near attacked subset
        targeted=TARGETED,
        target_label=TARGET_LABEL,
        early_stop=EARLY_STOP,
        connect_strategy=CONNECT_STRATEGY,
    )

    with torch.no_grad():
        logits_adv_full = model(res.x_adv, res.edge_index_adv)

    # IMPORTANT: evaluate only on original nodes (injected nodes are appended)
    logits_adv = logits_adv_full[: data.num_nodes]

    # --- metrics ---
    clean_m = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    adv_m = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)

    # attacked-only ASR (recommended)
    asr_attacked, ns_attacked, na_attacked = attack_success_rate(
        data.y, logits_clean.argmax(1), logits_adv.argmax(1), attack_mask
    )

    # split-wide ASR (optional; can be diluted if fraction < 1)
    asr_split, ns_split, na_split = attack_success_rate(
        data.y, logits_clean.argmax(1), logits_adv.argmax(1), split_mask
    )

    asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(data.y, logits_clean, logits_adv, attack_mask)

    roc_clean = roc_auc_binary(logits_clean, data.y, split_mask)
    roc_adv = roc_auc_binary(logits_adv, data.y, split_mask)

    conf_drop, n_used = mean_confidence_drop(
        data.y, logits_clean, logits_adv, attack_mask, only_clean_correct=True
    )

    edges_added = int(res.edge_index_adv.size(1) - data.edge_index.size(1))
    n_injected_nodes = int(res.x_adv.size(0) - data.x.size(0))

    print()
    print(f"[NodeInjection] split={SPLIT} n_targets={int(targets.numel())} injected_nodes={n_injected_nodes} edges_added={edges_added}")
    print(f"F1_pos : {clean_m.f1_pos:.4f} -> {adv_m.f1_pos:.4f}")
    print(f"F1_macro: {clean_m.f1_macro:.4f} -> {adv_m.f1_macro:.4f}")
    print(f"ROC-AUC: {roc_clean:.6f} -> {roc_adv:.6f}")
    print(f"ASR(attacked): {asr_attacked:.6f} ({ns_attacked}/{na_attacked}) | ASR_pos={asr_p:.6f} ({sp}/{ap}) ASR_neg={asr_n:.6f} ({sn}/{an})")
    print(f"ASR(split):    {asr_split:.6f} ({ns_split}/{na_split})")
    print(f"Mean confidence drop (attacked, clean-correct): {conf_drop:.6f} over n={n_used}")

    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "NodeInjectionEvasion",
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "device": str(device),
        "attack_params": {
            "n_inject": N_INJECT,
            "edges_per_injected": EDGES_PER_INJECTED,
            "eps": EPS,
            "alpha": ALPHA,
            "steps": STEPS,
            "random_start": RANDOM_START,
            "init": INIT,
            "clamp": CLAMP,
            "targeted": TARGETED,
            "target_label": TARGET_LABEL,
            "early_stop": EARLY_STOP,
            "connect_strategy": CONNECT_STRATEGY,
            "seed": SEED,
        },
        "target_selection": {
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "n_targets": int(targets.numel()),
        },
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "NodeInjectionEvasion",
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "n_targets": int(targets.numel()),
        "injection": {
            "n_injected_nodes": n_injected_nodes,
            "edges_added": edges_added,
            "injected_node_ids": res.injected_node_ids,
            "injected_edges": res.injected_edges,
        },
        "f1": {
            "pos_clean": clean_m.f1_pos,
            "pos_adv": adv_m.f1_pos,
            "macro_clean": clean_m.f1_macro,
            "macro_adv": adv_m.f1_macro,
        },
        "roc_auc": {"clean": roc_clean, "adv": roc_adv},
        "asr": {
            "attacked": {"value": asr_attacked, "success": ns_attacked, "attempted": na_attacked},
            "split": {"value": asr_split, "success": ns_split, "attempted": na_split},
            "pos_neg_on_attacked": {
                "asr_pos": asr_p, "succ_pos": sp, "attempted_pos": ap,
                "asr_neg": asr_n, "succ_neg": sn, "attempted_neg": an,
            },
        },
        "mean_confidence_drop": {"value": conf_drop, "n": n_used},
        "clean_split_metrics": vars(clean_m),
        "adv_split_metrics": vars(adv_m),
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)

    print()
    print(f"Saved to attacks/")


if __name__ == "__main__":
    main()
