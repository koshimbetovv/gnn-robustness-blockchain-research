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
from src.attacks.node_injection_recgnn import RecGNNNodeInjectionAttack
from src.utils.model_loader import load_model
from src.training.metrics import (
    binary_classification_metrics, attack_success_rate, roc_auc_binary,
    mean_confidence_drop, asr_pos_neg,
)

# ---------- attack parameters ----------
MODEL_NAME = "recgnn"
MODEL_DIR = "models/Elliptic"
# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (93 local + 2 ANF = 95 features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 local + 2 ANF = 57 features)
DATASET = "elliptic"
RUN_ID = None

# ---------- node-injection attack hyperparams ----------
N_INJECT = 5
EDGES_PER_INJECTED = 20
EPS = 0.05
ALPHA = 0.01
STEPS = 30
RANDOM_START = True
INIT = "mean"
CLAMP = None
CONNECT_STRATEGY = "round_robin"
# Number of local (controllable) feature columns. The trailing 2 ANF columns
# are antecedent-neighbor-label counts derived from the graph and are not
# attacker-controllable for an injected node either, so they are excluded
# from `delta`. `None` selects the dataset-appropriate default.
ATTACK_DIM = None

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
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_node_injection_{ts}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, ts


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def pick_targets_graph(
    y: torch.Tensor,
    pred_clean: torch.Tensor,
    only_illicit: bool,
    only_clean_correct: bool,
    fraction: float,
    seed: int,
) -> torch.Tensor:
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


def main():
    device = get_device()

    from src.utils.model_loader import resolve_checkpoint
    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    if DATASET == "elliptic":
        data_cfg = RecGNNEllipticConfig(**ckpt_cfg["data"])
        print(f"Loading RecGNN Elliptic sequence (filter_unknown={data_cfg.filter_unknown}) ...")
        sequence = RecGNNEllipticDataset(data_cfg).get_sequence()
        default_attack_dim = 93
    elif DATASET == "ellipticpp_actors":
        data_cfg = RecGNNEllipticPPActorsConfig(**ckpt_cfg["data"])
        print(f"Loading RecGNN Elliptic++ Actors sequence (filter_unknown={data_cfg.filter_unknown}) ...")
        sequence = RecGNNEllipticPPActorsDataset(data_cfg).get_sequence()
        default_attack_dim = sequence.num_features - 2
    else:
        raise ValueError(
            f"Unknown DATASET={DATASET!r}. Supported: 'elliptic', 'ellipticpp_actors'."
        )

    attack_dim = default_attack_dim if ATTACK_DIM is None else int(ATTACK_DIM)

    model = load_model(MODEL_NAME, sequence.num_features, 2, device=device, model_dir=MODEL_DIR, run_id=RUN_ID)
    print(
        f"Train graphs: {len(sequence.train_graphs)} | Test graphs: {len(sequence.test_graphs)} | "
        f"num_features={sequence.num_features} | attack_dim={attack_dim}"
    )

    atk = RecGNNNodeInjectionAttack(model, device, attack_dim=attack_dim, clamp=CLAMP)
    print("Priming sequence state over train graphs ...")
    atk.prime(sequence.train_graphs)

    per_timestep = []
    pooled_y_true = []
    pooled_pred_clean = []
    pooled_pred_adv = []
    pooled_logits_clean = []
    pooled_logits_adv = []
    pooled_attack_mask = []
    total_injected_nodes = 0
    total_edges_added = 0
    attack_time_seconds = 0.0

    for graph in sequence.test_graphs:
        g = graph.to(device)
        x = g.x.float()
        edge_index = g.edge_index.long()
        y = g.y
        t = int(g.graph_timestep)

        labeled_mask = y != -1
        if int(labeled_mask.sum().item()) == 0:
            atk.attack_step(
                x, edge_index,
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                n_inject=N_INJECT, edges_per_injected=EDGES_PER_INJECTED,
                eps=EPS, alpha=ALPHA, steps=STEPS, random_start=RANDOM_START,
                init=INIT, connect_strategy=CONNECT_STRATEGY,
            )
            per_timestep.append({"t": t, "n_labeled": 0, "n_targets": 0, "skipped": True})
            continue

        # Clean preview to pick targets without disturbing sequence state.
        snap = atk._save_state()
        with torch.no_grad():
            log_probs_preview = atk._forward(x, edge_index).detach()
            model.detach_sequence_state()
        atk._restore_state(snap)
        pred_preview = log_probs_preview.argmax(dim=1)

        targets = pick_targets_graph(
            y, pred_preview,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )

        if targets.numel() == 0:
            res = atk.attack_step(
                x, edge_index,
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                n_inject=N_INJECT, edges_per_injected=EDGES_PER_INJECTED,
                eps=EPS, alpha=ALPHA, steps=STEPS, random_start=RANDOM_START,
                init=INIT, connect_strategy=CONNECT_STRATEGY,
            )
            pred_clean = res.log_probs_clean.argmax(dim=1)
            per_timestep.append({
                "t": t,
                "n_labeled": int(labeled_mask.sum().item()),
                "n_targets": 0,
                "skipped": True,
            })
            pooled_y_true.append(y[labeled_mask].detach().cpu())
            pooled_pred_clean.append(pred_clean[labeled_mask].detach().cpu())
            pooled_pred_adv.append(pred_clean[labeled_mask].detach().cpu())
            pooled_logits_clean.append(res.log_probs_clean[labeled_mask].detach().cpu())
            pooled_logits_adv.append(res.log_probs_clean[labeled_mask].detach().cpu())
            pooled_attack_mask.append(torch.zeros(int(labeled_mask.sum().item()), dtype=torch.bool))
            continue

        labels_true = y[targets].long()

        # Init reference: licit labeled nodes in this snapshot, excluding targets.
        init_ref_mask = (y == 0)
        init_ref_mask[targets] = False
        init_reference = init_ref_mask.nonzero(as_tuple=False).view(-1)
        if init_reference.numel() == 0:
            raise RuntimeError(
                f"No licit non-target nodes available at timestep {t} for feature init."
            )

        t0 = time.perf_counter()
        res = atk.attack_step(
            x, edge_index, targets, labels_true,
            n_inject=N_INJECT, edges_per_injected=EDGES_PER_INJECTED,
            eps=EPS, alpha=ALPHA, steps=STEPS, random_start=RANDOM_START,
            init=INIT, reference_nodes=init_reference, connect_strategy=CONNECT_STRATEGY,
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        total_injected_nodes += len(res.injected_node_ids)
        total_edges_added += len(res.injected_edges)

        pred_clean = res.log_probs_clean.argmax(dim=1)
        pred_adv = res.log_probs_adv.argmax(dim=1)

        y_lab = y[labeled_mask]
        pred_clean_lab = pred_clean[labeled_mask]
        pred_adv_lab = pred_adv[labeled_mask]
        logits_clean_lab = res.log_probs_clean[labeled_mask]
        logits_adv_lab = res.log_probs_adv[labeled_mask]

        attack_mask_full = torch.zeros(y.numel(), dtype=torch.bool, device=device)
        attack_mask_full[targets] = True
        attack_mask_lab = attack_mask_full[labeled_mask]

        asr, ns, na = attack_success_rate(y_lab, pred_clean_lab, pred_adv_lab, attack_mask_lab)
        asr_p, sp, ap, asr_n, sn, an = asr_pos_neg(y_lab, logits_clean_lab, logits_adv_lab, attack_mask_lab)

        full_mask_lab = torch.ones(y_lab.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean_lab, y_lab, full_mask_lab)
        roc_adv = roc_auc_binary(logits_adv_lab, y_lab, full_mask_lab)
        clean_m = binary_classification_metrics(y_lab, pred_clean_lab)
        adv_m = binary_classification_metrics(y_lab, pred_adv_lab)

        conf_drop, n_used = mean_confidence_drop(
            y_lab, logits_clean_lab, logits_adv_lab, attack_mask_lab, only_clean_correct=True
        )

        f1_pos_drop = float(clean_m["f1_pos"] - adv_m["f1_pos"])
        recall_pos_drop = float(clean_m["recall_pos"] - adv_m["recall_pos"])

        per_timestep.append({
            "t": t,
            "n_labeled": int(y_lab.numel()),
            "n_targets": int(targets.numel()),
            "n_injected_nodes": len(res.injected_node_ids),
            "edges_added": len(res.injected_edges),
            "asr": asr, "asr_success": ns, "asr_attempted": na,
            "asr_pos": asr_p, "asr_pos_success": sp, "asr_pos_attempted": ap,
            "asr_neg": asr_n, "asr_neg_success": sn, "asr_neg_attempted": an,
            "f1_pos_clean": clean_m["f1_pos"], "f1_pos_adv": adv_m["f1_pos"], "f1_pos_drop": f1_pos_drop,
            "f1_macro_clean": clean_m["f1_macro"], "f1_macro_adv": adv_m["f1_macro"],
            "recall_pos_clean": clean_m["recall_pos"], "recall_pos_adv": adv_m["recall_pos"], "recall_pos_drop": recall_pos_drop,
            "roc_auc_clean": roc_clean, "roc_auc_adv": roc_adv,
            "mean_confidence_drop": conf_drop, "conf_drop_n": n_used,
        })

        pooled_y_true.append(y_lab.detach().cpu())
        pooled_pred_clean.append(pred_clean_lab.detach().cpu())
        pooled_pred_adv.append(pred_adv_lab.detach().cpu())
        pooled_logits_clean.append(logits_clean_lab.detach().cpu())
        pooled_logits_adv.append(logits_adv_lab.detach().cpu())
        pooled_attack_mask.append(attack_mask_lab.detach().cpu())

        roc_c_show = roc_clean if roc_clean == roc_clean else float("nan")
        roc_a_show = roc_adv if roc_adv == roc_adv else float("nan")
        print(
            f"t={t:2d}  n_labeled={y_lab.numel():5d}  n_targets={targets.numel():4d}  "
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
        attack_mask_c = torch.cat(pooled_attack_mask)

        clean_c = binary_classification_metrics(y_true, pred_clean_c)
        adv_c = binary_classification_metrics(y_true, pred_adv_c)
        full_mask = torch.ones(y_true.numel(), dtype=torch.bool)
        roc_c_clean = roc_auc_binary(logits_clean_c, y_true, full_mask)
        roc_c_adv = roc_auc_binary(logits_adv_c, y_true, full_mask)

        asr_c, ns_c, na_c = attack_success_rate(y_true, pred_clean_c, pred_adv_c, attack_mask_c)
        asr_cp, sp_c, ap_c, asr_cn, sn_c, an_c = asr_pos_neg(y_true, logits_clean_c, logits_adv_c, attack_mask_c)

        conf_drop_c, conf_drop_n_c = mean_confidence_drop(
            y_true, logits_clean_c, logits_adv_c, attack_mask_c, only_clean_correct=True
        )

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
        }

    print()
    # print(
    #     f"NodeInjection RecGNN | n_inject={N_INJECT} edges_per_injected={EDGES_PER_INJECTED} "
    #     f"eps={EPS} alpha={ALPHA} steps={STEPS} random_start={RANDOM_START} connect={CONNECT_STRATEGY}"
    # )
    # print(f"Total injected nodes: {total_injected_nodes}, edges added: {total_edges_added}")
    # if concat_classification is not None and concat_attack is not None:
    #     print(f"[concatenated across all test timesteps, n={concat_classification['n_total']}]")
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
    # print(f"Attack time (total over test timesteps): {attack_time_seconds:.4f} s")

    run_dir, ts = make_run_dir(MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "NodeInjectionEvasion",
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "device": str(device),
        "attack_params": {
            "n_inject": N_INJECT, "edges_per_injected": EDGES_PER_INJECTED,
            "eps": EPS, "alpha": ALPHA, "steps": STEPS, "random_start": RANDOM_START,
            "init": INIT, "clamp": CLAMP, "connect_strategy": CONNECT_STRATEGY,
            "attack_dim": attack_dim,
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
        "attack": "NodeInjectionEvasion",
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
        "injection": {
            "total_injected_nodes": total_injected_nodes,
            "total_edges_added": total_edges_added,
        },
        "per_timestep": per_timestep,
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
