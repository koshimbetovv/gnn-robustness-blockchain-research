import os
import sys
import json
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.elliptic import EllipticDataset
from src.utils.model_loader import load_model
from src.attacks.monti_attack import MonTiOneTimeInjectionAttack
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

# ---------- targets selection ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 0.1
ONLY_CLEAN_CORRECT = True
SEED = 0

# ---------- MonTi-style budgets (Delta, eta) ----------
N_INJECT = 10
EDGE_BUDGET = 50  # directed edges; if undirected=True, edge_index will contain ~2x

# ---------- candidates ----------
K_HOP = 2
ALPHA_CAND = 500
NEIGHBOR_MODE = "undirected"   # "undirected" recommended for candidates

# ---------- feature optimization ----------
EPS = 0.05
ALPHA_STEP = 0.01
INNER_STEPS = 30
OUTER_ROUNDS = 3
DESIRED_LABEL = 0  # benign
UNDIRECTED = False
ATTACK_INCOMING = True  
CLAMP = None  # e.g. (-3.0, 3.0)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(model_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_monti_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_targets(data, logits_clean, split_mask, device):
    mask = split_mask.bool() & (data.y != -1)
    candidates = torch.where(mask)[0]

    y_pred = logits_clean.argmax(dim=1)

    if ATTACK_ONLY_ILLICIT:
        candidates = candidates[data.y[candidates] == 1]

    if ONLY_CLEAN_CORRECT and candidates.numel() > 0:
        candidates = candidates[y_pred[candidates] == data.y[candidates]]

    if candidates.numel() == 0:
        return candidates

    n = int(round(float(ATTACK_FRACTION) * float(candidates.numel())))
    n = max(1, min(n, int(candidates.numel())))

    g = torch.Generator(device="cpu")
    g.manual_seed(int(SEED))
    perm = torch.randperm(candidates.numel(), generator=g)

    return candidates.detach().cpu()[perm[:n]].to(device)


def main():
    device = get_device()
    torch.manual_seed(SEED)

    data = EllipticDataset().get_data().to(device)
    model = load_model(MODEL_NAME, data.num_features, 2, device=device)

    split_mask = get_split_mask(data, SPLIT).to(device)

    with torch.no_grad():
        logits_clean = model(data.x, data.edge_index)

    targets = pick_targets(data, logits_clean, split_mask, device)
    if targets.numel() == 0:
        print("No eligible targets for these settings.")
        return

    # attacked-only mask for ASR and confidence drop
    attacked_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    attacked_mask[targets] = True
    attacked_mask = attacked_mask & split_mask

    atk = MonTiOneTimeInjectionAttack(
        model, data, device,
        undirected=UNDIRECTED,
        attack_incoming=ATTACK_INCOMING,
        clamp=CLAMP,
        seed=SEED,
    )

    res = atk.attack(
        target_nodes=targets,
        n_inject=N_INJECT,
        edge_budget=EDGE_BUDGET,
        K=K_HOP,
        alpha=ALPHA_CAND,
        neighbor_mode=NEIGHBOR_MODE,
        eps=EPS,
        alpha_step=ALPHA_STEP,
        inner_steps=INNER_STEPS,
        outer_rounds=OUTER_ROUNDS,
        desired_label=DESIRED_LABEL,
        early_stop=True,
    )

    with torch.no_grad():
        logits_adv_full = model(res.x_adv, res.edge_index_adv)

    # evaluate only original nodes (injected nodes appended)
    logits_adv = logits_adv_full[: data.num_nodes]

    clean_m = evaluate_logits_on_split(logits_clean, data.y, split_mask, SPLIT)
    adv_m = evaluate_logits_on_split(logits_adv, data.y, split_mask, SPLIT)

    asr_att, succ_att, att_att = attack_success_rate(
        data.y, logits_clean.argmax(1), logits_adv.argmax(1), attacked_mask
    )
    asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(data.y, logits_clean, logits_adv, attacked_mask)

    roc_clean = roc_auc_binary(logits_clean, data.y, split_mask)
    roc_adv = roc_auc_binary(logits_adv, data.y, split_mask)

    conf_drop, n_used = mean_confidence_drop(
        data.y, logits_clean, logits_adv, attacked_mask, only_clean_correct=True
    )

    edges_added = int(res.edge_index_adv.size(1) - data.edge_index.size(1))
    nodes_injected = int(res.x_adv.size(0) - data.x.size(0))

    print()
    print(f"[MonTi-style] split={SPLIT} n_targets={int(targets.numel())} injected_nodes={nodes_injected} edges_added={edges_added}")
    print(f"F1_pos : {clean_m.f1_pos:.4f} -> {adv_m.f1_pos:.4f}")
    print(f"F1_macro: {clean_m.f1_macro:.4f} -> {adv_m.f1_macro:.4f}")
    print(f"ROC-AUC: {roc_clean:.6f} -> {roc_adv:.6f}")
    print(f"ASR(attacked): {asr_att:.6f} ({succ_att}/{att_att}) | ASR_pos={asr_p:.6f} ({sp}/{ap}) ASR_neg={asr_n:.6f} ({sn}/{an})")
    print(f"Mean confidence drop (attacked, clean-correct): {conf_drop:.6f} over n={n_used}")

    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "MonTiOneTimeInjectionAttack (practical adaptation)",
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "device": str(device),
        "target_selection": {
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "n_targets": int(targets.numel()),
        },
        "budgets": {"n_inject": N_INJECT, "edge_budget": EDGE_BUDGET},
        "candidates": {"K_hop": K_HOP, "alpha": ALPHA_CAND, "neighbor_mode": NEIGHBOR_MODE},
        "opt": {"eps": EPS, "alpha_step": ALPHA_STEP, "inner_steps": INNER_STEPS, "outer_rounds": OUTER_ROUNDS},
        "graph": {"undirected": UNDIRECTED, "attack_incoming": ATTACK_INCOMING},
        "desired_label": DESIRED_LABEL,
        "clamp": CLAMP,
        "seed": SEED,
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "MonTiOneTimeInjectionAttack (practical adaptation)",
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "n_targets": int(targets.numel()),
        "injection": {
            "n_injected_nodes": nodes_injected,
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
            "attacked": {"value": asr_att, "success": succ_att, "attempted": att_att},
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
