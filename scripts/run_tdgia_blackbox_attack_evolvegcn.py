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
from src.attacks.tdgia_evolvegcn import EvolveGCNTDGIAAttack
from src.utils.model_loader import resolve_checkpoint
from src.utils.seed import set_seed
from src.training.metrics import (
    binary_classification_metrics, attack_success_rate, roc_auc_binary,
    mean_confidence_drop, asr_pos_neg,
)

# ---------- victim / surrogate parameters ----------
MODEL_NAME = "evolvegcn_o"
MODEL_DIR = "models/Elliptic"
RUN_ID = None  # e.g. "seed43_20260331_133701" for a specific checkpoint; None picks latest

SURROGATE_MODEL_NAME = "evolvegcn_o"
SURROGATE_MODEL_DIR = "models/Elliptic"
SURROGATE_RUN_ID = None

# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (166 features incl. time_step at col 0)
#   "ellipticpp_actors"  -> Elliptic++ actors (pure wallet features, no metadata col)
DATASET = "elliptic"

# ---------- TDGIA hyperparameters ----------
N_INJECT = 5
DEGREE_LIMIT = 20
BATCH_SIZE = 1
EPS_FEATURE = 0.05
STEPS = 30
LR = 0.05
SMOOTH_R = 0.5
ALPHA_MU = 0.5
K1 = 1.0
K2 = 1.0
INIT = "randn"          # "mean" or "randn"
SIGMA_SCALE = 1.0
CLAMP = None           # e.g. (-3.0, 3.0)
# Column 0 of the EvolveGCN Elliptic feature matrix is the IBM time_step metadata;
# protect it from perturbation on injected nodes (rewriting timestamps is not a
# plausible feature-space attack). Elliptic++ actors excludes the time_step column
# from features. `None` selects the dataset-appropriate default.
ATTACK_START_COL = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = True
TARGET_SELECTION_MODEL = "surrogate"  # "surrogate" matches black-box crafting; "victim" is eval-oracle mode
SEED = 0


def get_device():
    # EvolveGCN uses nn.RReLU, which MPS does not implement; training uses CPU
    # fallback (allow_mps=False). We match that here.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_run_dir(model_name: str, surrogate_name: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(repo_root, "attacks", f"{model_name}_tdgia_blackbox_{surrogate_name}_{ts}")
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


def sample_injection_schedule(eligible_timesteps: list[int], n_inject: int, seed: int):
    """Sample a sequence-wide node budget allocation over eligible timesteps."""
    if n_inject <= 0 or not eligible_timesteps:
        return [], {}
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    draw = torch.randint(len(eligible_timesteps), (int(n_inject),), generator=g)
    sampled_timesteps = [int(eligible_timesteps[i]) for i in draw.tolist()]
    allocation = {int(t): 0 for t in eligible_timesteps}
    for t in sampled_timesteps:
        allocation[int(t)] += 1
    return sampled_timesteps, allocation


def main():
    device = get_device()
    set_seed(SEED, deterministic=True, benchmark=False)

    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    surrogate_ckpt_path, surrogate_ckpt_run_dir = resolve_checkpoint(
        SURROGATE_MODEL_NAME, model_dir=SURROGATE_MODEL_DIR, run_id=SURROGATE_RUN_ID
    )
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)
    with open(os.path.join(surrogate_ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        surrogate_ckpt_cfg = json.load(f)

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
    surrogate_model = build_evolvegcn_model(sequence.num_features, surrogate_ckpt_cfg).to(device)
    surrogate_state = torch.load(surrogate_ckpt_path, map_location=device)
    surrogate_model.load_state_dict(surrogate_state)
    surrogate_model.eval()
    print(f"✓ Loaded {MODEL_NAME.upper()} from {ckpt_path}")
    print(f"✓ Loaded surrogate {SURROGATE_MODEL_NAME.upper()} from {surrogate_ckpt_path}")
    print(
        f"Test windows: {len(sequence.test_samples)} | num_nodes={sequence.num_nodes} | "
        f"num_features={sequence.num_features} | attack_start_col={attack_start_col}"
    )

    victim_atk = EvolveGCNTDGIAAttack(
        model, device,
        attack_start_col=attack_start_col,
        clamp=CLAMP,
    )
    surrogate_atk = EvolveGCNTDGIAAttack(
        surrogate_model, device,
        attack_start_col=attack_start_col,
        clamp=CLAMP,
    )

    planning_targets: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    eligible_timesteps: list[int] = []
    for sample in sequence.test_samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        t = int(sample.current_time)

        if TARGET_SELECTION_MODEL not in ("surrogate", "victim"):
            raise ValueError(f"TARGET_SELECTION_MODEL must be 'surrogate' or 'victim', got {TARGET_SELECTION_MODEL!r}.")
        preview_atk = surrogate_atk if TARGET_SELECTION_MODEL == "surrogate" else victim_atk
        with torch.no_grad():
            logits_clean = preview_atk.forward_labels(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
        pred_clean = logits_clean.argmax(dim=1)

        target_global, target_pos = pick_targets_window(
            label_idx, label_vals, pred_clean,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )
        planning_targets[t] = (target_global.detach().cpu(), target_pos.detach().cpu())
        if target_global.numel() > 0:
            eligible_timesteps.append(t)

    sampled_injection_timesteps, injection_allocation = sample_injection_schedule(
        eligible_timesteps, N_INJECT, SEED
    )
    print(
        f"Eligible attack windows: {len(eligible_timesteps)} | "
        f"sampled injections: {len(sampled_injection_timesteps)}"
    )

    per_window = []
    pooled_y_true = []
    pooled_pred_clean = []
    pooled_pred_adv = []
    pooled_logits_clean = []
    pooled_logits_adv = []
    pooled_attack_mask = []
    pooled_injected_l2 = []
    pooled_injected_abs = []
    pooled_injected_signed = []
    total_injected_nodes = 0
    total_edges_added = 0
    attack_time_seconds = 0.0

    for sample in sequence.test_samples:
        hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
        t = int(sample.current_time)
        assigned_n_inject = int(injection_allocation.get(t, 0))

        with torch.no_grad():
            logits_clean = victim_atk.forward_labels(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx).detach()
        pred_clean = logits_clean.argmax(dim=1)

        target_global_cpu, target_pos_cpu = planning_targets.get(
            t, (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        )
        target_global = target_global_cpu.to(device)
        target_pos = target_pos_cpu.to(device)

        if target_global.numel() == 0 or assigned_n_inject == 0:
            per_window.append({
                "t": t,
                "n_labeled": int(label_idx.numel()),
                "n_targets_available": int(target_global.numel()),
                "n_targets_attacked": 0,
                "n_injected_nodes_budgeted": assigned_n_inject,
                "n_injected_nodes": 0,
                "skipped": True,
            })
            pooled_y_true.append(label_vals.detach().cpu())
            pooled_pred_clean.append(pred_clean.detach().cpu())
            pooled_pred_adv.append(pred_clean.detach().cpu())
            pooled_logits_clean.append(logits_clean.detach().cpu())
            pooled_logits_adv.append(logits_clean.detach().cpu())
            pooled_attack_mask.append(torch.zeros(label_idx.numel(), dtype=torch.bool))
            continue

        # Init reference: licit labeled nodes in this window, excluding targets.
        target_set = set(target_global.detach().cpu().tolist())
        licit_global = label_idx[label_vals == 0]
        init_reference = torch.tensor(
            [int(i) for i in licit_global.detach().cpu().tolist() if int(i) not in target_set],
            dtype=torch.long, device=device,
        )
        if init_reference.numel() == 0:
            raise RuntimeError(
                f"No licit non-target nodes available at window t={t} for feature init."
            )

        t0 = time.perf_counter()
        res = surrogate_atk.attack_window(
            hist_adj_list, hist_ndFeats_list, node_mask_list,
            label_idx, target_global,
            n_inject=assigned_n_inject, degree_limit=DEGREE_LIMIT,
            batch_size=BATCH_SIZE, steps=STEPS, lr=LR, smooth_r=SMOOTH_R,
            alpha_mu=ALPHA_MU, k1=K1, k2=K2,
            init=INIT, reference_nodes=init_reference, sigma_scale=SIGMA_SCALE,
            eps_feature=EPS_FEATURE,
        )
        attack_time_seconds += float(time.perf_counter() - t0)
        total_injected_nodes += len(res.injected_node_ids)
        total_edges_added += len(res.injected_edges)
        with torch.no_grad():
            logits_adv = victim_atk.forward_labels(
                res.hist_adj_list, res.hist_ndFeats_list, res.node_mask_list, res.label_idx
            ).detach()
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

        if len(res.injected_node_ids) > 0:
            clean_inj = res.x_injected_base[:, attack_start_col:]
            adv_inj = res.hist_ndFeats_list[-1][res.injected_node_ids, attack_start_col:]
            delta_inj = (adv_inj - clean_inj).float()
            per_node_l2 = torch.linalg.vector_norm(delta_inj, ord=2, dim=1)
            pert_l2_mean = float(per_node_l2.mean().item())
            pert_l2_n = int(per_node_l2.numel())
            avg_perturbation = float(delta_inj.abs().mean().item())
            avg_perturbation_signed = float(delta_inj.mean().item())
            pooled_injected_l2.append(per_node_l2.detach().cpu())
            pooled_injected_abs.append(delta_inj.abs().reshape(-1).detach().cpu())
            pooled_injected_signed.append(delta_inj.reshape(-1).detach().cpu())
        else:
            pert_l2_mean, pert_l2_n = 0.0, 0
            avg_perturbation = 0.0
            avg_perturbation_signed = 0.0

        f1_pos_drop = float(clean_m["f1_pos"] - adv_m["f1_pos"])
        recall_pos_drop = float(clean_m["recall_pos"] - adv_m["recall_pos"])

        per_window.append({
            "t": t,
            "n_labeled": int(label_idx.numel()),
            "n_targets_available": int(target_global.numel()),
            "n_targets_attacked": int(target_global.numel()),
            "n_injected_nodes_budgeted": assigned_n_inject,
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
            "perturbation_l2_on_injected_nodes": pert_l2_mean,
            "perturbation_l2_n_injected_nodes": pert_l2_n,
            "avg_perturbation": avg_perturbation,
            "avg_perturbation_signed": avg_perturbation_signed,
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

        if pooled_injected_l2:
            pert_l2_c = float(torch.cat(pooled_injected_l2).mean().item())
            pert_l2_n_c = int(torch.cat(pooled_injected_l2).numel())
            avg_perturbation_c = float(torch.cat(pooled_injected_abs).mean().item())
            avg_perturbation_signed_c = float(torch.cat(pooled_injected_signed).mean().item())
        else:
            pert_l2_c, pert_l2_n_c = 0.0, 0
            avg_perturbation_c, avg_perturbation_signed_c = 0.0, 0.0

        f1_pos_drop_c = float(clean_c["f1_pos"] - adv_c["f1_pos"])
        recall_pos_drop_c = float(clean_c["recall_pos"] - adv_c["recall_pos"])

        concat_classification = {
            "n_total": int(y_true.numel()),
            "f1_pos_clean": clean_c["f1_pos"], "f1_pos_adv": adv_c["f1_pos"], "f1_pos_drop": f1_pos_drop_c,
            "f1_macro_clean": clean_c["f1_macro"], "f1_macro_adv": adv_c["f1_macro"],
            "recall_pos_clean": clean_c["recall_pos"], "recall_pos_adv": adv_c["recall_pos"], "recall_pos_drop": recall_pos_drop_c,
            "roc_auc_clean": roc_c_clean, "roc_auc_adv": roc_c_adv,
        }
        n_targets_available_total = int(sum(item.get("n_targets_available", 0) for item in per_window))
        n_targets_attacked_total = int(attack_mask_c.sum().item())
        n_eligible_timesteps = int(sum(1 for item in per_window if item.get("n_targets_available", 0) > 0))
        n_budgeted_timesteps = int(sum(1 for item in per_window if item.get("n_injected_nodes_budgeted", 0) > 0))
        concat_attack = {
            "n_targets_available_total": n_targets_available_total,
            "n_targets_attacked_total": n_targets_attacked_total,
            "n_targets_total": n_targets_attacked_total,
            "n_eligible_timesteps": n_eligible_timesteps,
            "n_budgeted_timesteps": n_budgeted_timesteps,
            "asr": asr_c, "asr_success": ns_c, "asr_attempted": na_c,
            "asr_pos": asr_cp, "asr_pos_success": sp_c, "asr_pos_attempted": ap_c,
            "asr_neg": asr_cn, "asr_neg_success": sn_c, "asr_neg_attempted": an_c,
            "mean_confidence_drop": conf_drop_c, "conf_drop_n": conf_drop_n_c,
            "perturbation_l2_on_injected_nodes": pert_l2_c,
            "perturbation_l2_n_injected_nodes": pert_l2_n_c,
            "avg_perturbation": avg_perturbation_c,
            "avg_perturbation_signed": avg_perturbation_signed_c,
        }

    print()
    # print(
    #     f"TDGIA EvolveGCN-O | n_inject={N_INJECT} degree_limit={DEGREE_LIMIT} "
    #     f"eps_feature={EPS_FEATURE} lr={LR} steps={STEPS}"
    # )
    # print(f"Total injected nodes: {total_injected_nodes}, edges added: {total_edges_added}")
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
    # print(f"Attack time (total over test windows): {attack_time_seconds:.4f} s")

    run_dir, ts = make_run_dir(MODEL_NAME, SURROGATE_MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "TDGIA-BlackBox",
        "threat_model": {
            "access": "black_box_transfer",
            "attack_stage": "evasion",
            "victim_access_during_attack": "none" if TARGET_SELECTION_MODEL == "surrogate" else "target_selection_only",
            "surrogate": "separate_temporal_model",
        },
        "model_name": MODEL_NAME,
        "model_dir": MODEL_DIR,
        "run_id": RUN_ID,
        "surrogate_model_name": SURROGATE_MODEL_NAME,
        "surrogate_model_dir": SURROGATE_MODEL_DIR,
        "surrogate_run_id": SURROGATE_RUN_ID,
        "dataset": DATASET,
        "checkpoint_path": ckpt_path,
        "surrogate_checkpoint_path": surrogate_ckpt_path,
        "device": str(device),
        "attack_params": {
            "n_inject": N_INJECT, "degree_limit": DEGREE_LIMIT, "batch_size": BATCH_SIZE,
            "eps_feature": EPS_FEATURE, "steps": STEPS, "lr": LR, "smooth_r": SMOOTH_R,
            "alpha_mu": ALPHA_MU, "k1": K1, "k2": K2,
            "init": INIT, "sigma_scale": SIGMA_SCALE, "clamp": CLAMP,
            "attack_start_col": attack_start_col,
            "budget_mode": "global_sequence",
            "persistence": "non_persistent",
            "feature_bounds": "manual_clamp" if CLAMP is not None else "per_feature_data_minmax",
            "crafting_model": "surrogate_model",
            "evaluation_model": "victim_model",
        },
        "target_selection": {
            "selection_logits": f"{TARGET_SELECTION_MODEL}_clean_logits",
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
        },
        "data": ckpt_cfg["data"],
        "model_hparams": ckpt_cfg["model"],
        "surrogate_data": surrogate_ckpt_cfg["data"],
        "surrogate_model_hparams": surrogate_ckpt_cfg["model"],
        "injection_schedule": {
            "eligible_timesteps": eligible_timesteps,
            "sampled_timesteps": sampled_injection_timesteps,
            "allocation_by_timestep": [
                {"t": int(t), "n_inject": int(injection_allocation[t])}
                for t in sorted(injection_allocation)
            ],
        },
    }
    write_json(os.path.join(run_dir, "config.json"), config)

    metrics = {
        "attack": "TDGIA-BlackBox",
        "model_name": MODEL_NAME,
        "surrogate_model_name": SURROGATE_MODEL_NAME,
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
            "budget_mode": "global_sequence",
            "n_inject_total_budget": int(N_INJECT),
            "n_targets_available_total": int(sum(item.get("n_targets_available", 0) for item in per_window)),
            "n_targets_attacked_total": int(sum(item.get("n_targets_attacked", 0) for item in per_window)),
            "eligible_timesteps": eligible_timesteps,
            "sampled_timesteps": sampled_injection_timesteps,
            "allocation_by_timestep": [
                {"t": int(t), "n_inject": int(injection_allocation[t])}
                for t in sorted(injection_allocation)
            ],
        },
        "per_timestep": per_window,
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
