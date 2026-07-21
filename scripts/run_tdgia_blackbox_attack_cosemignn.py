import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from types import SimpleNamespace

from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.cosemignn_elliptic import load_cosemignn_elliptic
from src.datasets.cosemignn_ellipticpp_addraddr import load_cosemignn_ellipticpp_addraddr
from src.attacks.tdgia_cosemignn import CoSemiGNNTDGIAAttack
from src.attacks.tdgia import TDGIAAttack
from src.attacks.model_forward import STATIC_MODELS, forward_logits
from src.utils.model_loader import resolve_checkpoint
from src.utils.model_loader import load_model
from src.utils.seed import set_seed
from src.utils.tdgia_schedule import (
    degree_from_edge_index,
    score_based_injection_schedule,
    slot_scores_from_node_scores,
    tdgia_defective_scores_from_probability,
)
from src.training.metrics import (
    binary_classification_metrics, roc_auc_binary,
    mean_confidence_drop,
)
from src.models.cosemignn import CoSemiGNN
from src.utils.tdgia_metrics import (
    aggregate_attacked_target_outcomes,
    budget_efficiency_metrics,
    attacked_target_ids_from_edges,
    attacked_target_outcome,
    coverage_metrics,
    mask_from_target_ids,
)

# ---------- victim / surrogate parameters ----------
MODEL_NAME = "cosemignn"
MODEL_DIR = "models/Elliptic"
RUN_ID = None

SURROGATE_MODEL_NAME = "temporal_gcn"
#SURROGATE_MODEL_NAME = "cosemignn"
SURROGATE_MODEL_DIR = "models/Elliptic"
SURROGATE_RUN_ID = None

# Train a slice-wise GCN surrogate on raw CoSemi train-slice features instead
# of loading a checkpoint. Use SURROGATE_MODEL_NAME = "temporal_gcn" to enable.
TEMPORAL_GCN_HIDDEN_DIMS = (256, 128, 64)
TEMPORAL_GCN_DROPOUT = 0.5
TEMPORAL_GCN_EPOCHS = 150
TEMPORAL_GCN_LR = 0.005
TEMPORAL_GCN_WEIGHT_DECAY = 5e-4
TEMPORAL_GCN_LOG_EVERY = 50

# Must match the dataset the checkpoint was trained on. Options:
#   "elliptic"           -> Elliptic (165 raw + 6 semi = 171 features)
#   "ellipticpp_actors"  -> Elliptic++ actors (55 raw + 6 semi = 61 features)
DATASET = "elliptic"

# ---------- TDGIA hyperparameters ----------
N_INJECT = 20
DEGREE_LIMIT = 3
BATCH_SIZE = 1
EPS_FEATURE = 0.2
STEPS = 30
LR = 0.05
SMOOTH_R = 0.7
ALPHA_MU = 0.5
K1 = 1.0
K2 = 1.0
INIT = "randn"
SIGMA_SCALE = 1.0
CLAMP = None

# ---------- target selection controls ----------
ATTACK_ONLY_ILLICIT = True
ATTACK_FRACTION = 1.0
ONLY_CLEAN_CORRECT = False
TARGET_SELECTION_MODEL = "surrogate"  # "surrogate" matches black-box crafting; "victim" is eval-oracle mode
SEED = 0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    #if torch.backends.mps.is_available():
    #    return torch.device("mps")
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


def pick_targets_slice(labels, pred_clean, only_illicit, only_clean_correct, fraction, seed):
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(f"ATTACK_FRACTION must be in (0, 1], got {fraction}")
    idx = torch.arange(labels.numel(), device=labels.device)
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


def sample_injection_schedule(
    eligible_timesteps: list[int],
    timestep_slot_scores: dict[int, list[float]],
    n_inject: int,
):
    """Allocate the sequence-wide node budget to the highest TDGIA-scored slices."""
    return score_based_injection_schedule(eligible_timesteps, timestep_slot_scores, n_inject)


class PaperStyleSliceGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dims=(256, 128, 64), dropout: float = 0.5):
        super().__init__()
        dims = [int(in_dim)] + [int(h) for h in hidden_dims] + [int(out_dim)]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = float(dropout)
        for i in range(len(dims) - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.norms.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def fit_cosemi_train_slice_feature_transform(feature_list, train_times, feature_dim, device):
    xs = []
    for t in train_times:
        if t >= len(feature_list):
            continue
        features = feature_list[t]
        if features is not None and features.numel() > 0:
            xs.append(features[:, :feature_dim].detach())
    if not xs:
        raise RuntimeError("Cannot fit temporal GCN surrogate transform: no non-empty train slices.")
    x_train = torch.cat(xs, dim=0)
    mean = x_train.mean(dim=0).to(device)
    scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-12).to(device)
    return mean, scale


def train_temporal_gcn_surrogate(
    feature_list,
    adj_list,
    label_list,
    train_times,
    feature_dim,
    device,
    *,
    hidden_dims,
    dropout,
    epochs,
    lr,
    weight_decay,
    log_every,
):
    transform = fit_cosemi_train_slice_feature_transform(feature_list, train_times, feature_dim, device)
    model = PaperStyleSliceGCN(feature_dim, 2, hidden_dims=hidden_dims, dropout=dropout).to(device)

    labels_all = []
    for t in train_times:
        if t >= len(label_list):
            continue
        labels = label_list[t]
        if labels is not None and labels.numel() > 0:
            labels_all.append(labels[labels != -1].detach())
    if not labels_all:
        raise RuntimeError("Cannot train temporal GCN surrogate: no labeled train nodes.")
    y_all = torch.cat(labels_all).long()
    counts = torch.bincount(y_all, minlength=2).float().to(device).clamp_min(1.0)
    class_weight = counts.sum() / (2.0 * counts)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    model.train()
    for ep in range(int(epochs)):
        loss_total = 0.0
        n_terms = 0
        for t in train_times:
            if t >= len(feature_list):
                continue
            features = feature_list[t]
            adj = adj_list[t]
            labels = label_list[t]
            if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
                continue
            mask = labels != -1
            if int(mask.sum().item()) == 0:
                continue

            x = apply_static_surrogate_feature_transform(features[:, :feature_dim], transform)
            logits = model(x, adj)
            loss = F.cross_entropy(logits[mask], labels[mask].long(), weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += float(loss.item())
            n_terms += 1

        if log_every and ((ep + 1) % int(log_every) == 0 or ep == 0 or ep + 1 == int(epochs)):
            avg = loss_total / max(n_terms, 1)
            print(f"  [temporal_gcn_surrogate] epoch {ep + 1}/{int(epochs)} loss={avg:.4f}")

    model.eval()
    return model, transform


def load_cosemignn_from_checkpoint(path: str, cfg: dict, feature_in: int, device: torch.device):
    model_cfg = cfg.get("cosemignn", {})
    model = CoSemiGNN(
        feature_in=feature_in,
        dim=int(model_cfg.get("dim", 128)),
        dim2=int(model_cfg.get("dim2", 256)),
        dim3=int(model_cfg.get("dim3", 128)),
        num_heads=int(model_cfg.get("num_heads", 4)),
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, model_cfg


def make_static_surrogate_data(features, labels, adj, raw_feature_dim, feature_transform):
    x_sur = apply_static_surrogate_feature_transform(features[:, :raw_feature_dim], feature_transform)
    return SimpleNamespace(
        x=x_sur,
        y=labels,
        edge_index=adj,
        num_nodes=int(features.size(0)),
    )


def fit_static_surrogate_raw_transform(dataset: str, static_cfg: dict, raw_feature_dim: int, device: torch.device):
    """Match the raw-feature preprocessing used by static surrogate checkpoints."""
    data_cfg = static_cfg.get("data", {})
    if dataset != "elliptic":
        return None

    feature_path = data_cfg.get("feature_path", "data/raw/elliptic/elliptic_txs_features.csv")
    class_path = data_cfg.get("class_path", "data/raw/elliptic/elliptic_txs_classes.csv")
    train_start = int(data_cfg.get("train_start", 1))
    train_end = int(data_cfg.get("train_end", 34))
    filter_unknown = bool(data_cfg.get("filter_unknown", False))

    features = pd.read_csv(feature_path, header=None).sort_values(by=0).reset_index(drop=True)
    classes = pd.read_csv(class_path)
    classes["txId"] = classes["txId"].astype(str)
    raw_cls = classes["class"].astype(str).str.strip().str.lower()
    y_map = raw_cls.map({"1": 1, "2": 0, "unknown": -1, "3": -1, "nan": -1}).fillna(-1).astype(int)
    y_dict = dict(zip(classes["txId"], y_map))

    tx_ids = features.iloc[:, 0].astype(str).tolist()
    time_step = features.iloc[:, 1].astype(int)
    x = features.iloc[:, 2 : 2 + raw_feature_dim].astype("float32")
    y = pd.Series([y_dict.get(tx, -1) for tx in tx_ids])

    if filter_unknown:
        keep = y != -1
        x = x.loc[keep]
        time_step = time_step.loc[keep]

    train_mask = (time_step >= train_start) & (time_step <= train_end)
    if int(train_mask.sum()) == 0:
        raise ValueError("Cannot fit static surrogate feature transform: train split is empty.")

    scaler = StandardScaler()
    scaler.fit(x.loc[train_mask].to_numpy(dtype="float32"))
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).clamp_min(1e-12)
    return mean, scale


def apply_static_surrogate_feature_transform(features: torch.Tensor, transform):
    if transform is None:
        return features
    mean, scale = transform
    return (features - mean.to(features.device)) / scale.to(features.device)


def invert_static_surrogate_feature_transform(features: torch.Tensor, transform):
    if transform is None:
        return features
    mean, scale = transform
    return features * scale.to(features.device) + mean.to(features.device)


def sample_reference_semi_features(features, raw_feature_dim, reference_nodes, n_inj):
    semi_existing = features[:, raw_feature_dim:]
    semi_dim = int(semi_existing.size(1))
    if int(n_inj) <= 0 or semi_dim <= 0:
        return features.new_empty((int(n_inj), max(semi_dim, 0)))

    if reference_nodes is None or reference_nodes.numel() == 0:
        ref_semi = semi_existing
    else:
        ref_semi = semi_existing[reference_nodes.to(features.device).long()]

    sample_idx = torch.randint(
        int(ref_semi.size(0)),
        (int(n_inj),),
        device=features.device,
    )
    return ref_semi[sample_idx].detach().to(features.dtype)


def transfer_static_surrogate_result_to_cosemi(
    res,
    features,
    raw_feature_dim,
    feature_transform,
    reference_nodes,
):
    n_existing = int(features.size(0))
    x_adv_raw = invert_static_surrogate_feature_transform(res.x_adv, feature_transform)
    semi_existing = features[:, raw_feature_dim:]
    n_inj = len(res.injected_node_ids)
    raw_existing = x_adv_raw[:n_existing]
    raw_inj = x_adv_raw[n_existing:]
    if n_inj > 0:
        semi_inj = sample_reference_semi_features(
            features,
            raw_feature_dim,
            reference_nodes,
            n_inj,
        )
        x_adv = torch.cat(
            [
                torch.cat([raw_existing, semi_existing], dim=1),
                torch.cat([raw_inj, semi_inj], dim=1),
            ],
            dim=0,
        )
    else:
        x_adv = features.clone()
    return x_adv


def main():
    device = get_device()
    set_seed(SEED, deterministic=True, benchmark=False)

    ckpt_path, ckpt_run_dir = resolve_checkpoint(MODEL_NAME, model_dir=MODEL_DIR, run_id=RUN_ID)
    train_temporal_gcn_surrogate_flag = SURROGATE_MODEL_NAME.lower() == "temporal_gcn"
    surrogate_ckpt_path = None
    surrogate_ckpt_cfg = {"model": {}, "data": {}}
    if not train_temporal_gcn_surrogate_flag:
        surrogate_ckpt_path, surrogate_ckpt_run_dir = resolve_checkpoint(
            SURROGATE_MODEL_NAME, model_dir=SURROGATE_MODEL_DIR, run_id=SURROGATE_RUN_ID
        )
    with open(os.path.join(ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)
    if not train_temporal_gcn_surrogate_flag:
        with open(os.path.join(surrogate_ckpt_run_dir, "config.json"), "r", encoding="utf-8") as f:
            surrogate_ckpt_cfg = json.load(f)

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

    model, model_cfg = load_cosemignn_from_checkpoint(ckpt_path, ckpt_cfg, feature_in, device)
    train_times = [i for i in range(1, int(time_cfg["train_end"]))]
    surrogate_is_static = SURROGATE_MODEL_NAME.lower() in STATIC_MODELS or train_temporal_gcn_surrogate_flag
    static_surrogate_feature_transform = None
    if train_temporal_gcn_surrogate_flag:
        print("Training temporal paper-style GCN surrogate on raw CoSemi train-slice features ...")
        surrogate_model, static_surrogate_feature_transform = train_temporal_gcn_surrogate(
            feature_list, adj_list, label_list, train_times, raw_feature_dim, device,
            hidden_dims=TEMPORAL_GCN_HIDDEN_DIMS,
            dropout=TEMPORAL_GCN_DROPOUT,
            epochs=TEMPORAL_GCN_EPOCHS,
            lr=TEMPORAL_GCN_LR,
            weight_decay=TEMPORAL_GCN_WEIGHT_DECAY,
            log_every=TEMPORAL_GCN_LOG_EVERY,
        )
        surrogate_model_cfg = {
            "type": "PaperStyleSliceGCN",
            "input_dim": int(raw_feature_dim),
            "feature_scope": "raw_features",
            "hidden_dims": list(TEMPORAL_GCN_HIDDEN_DIMS),
            "dropout": TEMPORAL_GCN_DROPOUT,
            "epochs": TEMPORAL_GCN_EPOCHS,
            "lr": TEMPORAL_GCN_LR,
            "weight_decay": TEMPORAL_GCN_WEIGHT_DECAY,
            "train_times": train_times,
        }
    elif surrogate_is_static:
        surrogate_model = load_model(
            SURROGATE_MODEL_NAME,
            raw_feature_dim,
            2,
            device=device,
            model_dir=SURROGATE_MODEL_DIR,
            run_id=SURROGATE_RUN_ID,
        )
        static_surrogate_feature_transform = fit_static_surrogate_raw_transform(
            DATASET, surrogate_ckpt_cfg, raw_feature_dim, device
        )
        surrogate_model_cfg = surrogate_ckpt_cfg.get("model", {})
    else:
        if SURROGATE_MODEL_NAME.lower() != "cosemignn":
            raise ValueError(
                "CoSemiGNN black-box TDGIA supports either a CoSemiGNN surrogate or a static "
                f"surrogate in {STATIC_MODELS}; got {SURROGATE_MODEL_NAME!r}."
            )
        surrogate_model, surrogate_model_cfg = load_cosemignn_from_checkpoint(
            surrogate_ckpt_path, surrogate_ckpt_cfg, feature_in, device
    )
    print(f"✓ Loaded {MODEL_NAME.upper()} from {ckpt_path}")
    if train_temporal_gcn_surrogate_flag:
        print("✓ Trained surrogate TEMPORAL_GCN")
    else:
        print(f"✓ Loaded surrogate {SURROGATE_MODEL_NAME.upper()} from {surrogate_ckpt_path}")

    victim_atk = CoSemiGNNTDGIAAttack(model, device, raw_feature_dim=raw_feature_dim, clamp=CLAMP)
    surrogate_atk = (
        None if surrogate_is_static
        else CoSemiGNNTDGIAAttack(surrogate_model, device, raw_feature_dim=raw_feature_dim, clamp=CLAMP)
    )

    planning_targets: dict[int, torch.Tensor] = {}
    eligible_timesteps: list[int] = []
    timestep_slot_scores: dict[int, list[float]] = {}
    for t in range(predict_start, predict_end):
        if t >= len(feature_list):
            planning_targets[t] = torch.empty(0, dtype=torch.long)
            timestep_slot_scores[t] = []
            continue
        features = feature_list[t]
        adj = adj_list[t]
        labels = label_list[t]
        ca_weights = ca_weights_list[t]

        if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
            planning_targets[t] = torch.empty(0, dtype=torch.long)
            timestep_slot_scores[t] = []
            continue

        if TARGET_SELECTION_MODEL not in ("surrogate", "victim"):
            raise ValueError(f"TARGET_SELECTION_MODEL must be 'surrogate' or 'victim', got {TARGET_SELECTION_MODEL!r}.")
        with torch.no_grad():
            if TARGET_SELECTION_MODEL == "surrogate" and surrogate_is_static:
                x_static = apply_static_surrogate_feature_transform(
                    features[:, :raw_feature_dim], static_surrogate_feature_transform
                )
                logits_clean = forward_logits(surrogate_model, x_static, adj)
            else:
                preview_atk = surrogate_atk if TARGET_SELECTION_MODEL == "surrogate" else victim_atk
                logits_clean = preview_atk.forward_logits(features, adj, ca_weights)
        pred_clean = logits_clean.argmax(dim=1)

        targets = pick_targets_slice(
            labels, pred_clean,
            only_illicit=ATTACK_ONLY_ILLICIT,
            only_clean_correct=ONLY_CLEAN_CORRECT,
            fraction=ATTACK_FRACTION,
            seed=SEED + t,
        )
        planning_targets[t] = targets.detach().cpu()
        if targets.numel() > 0:
            attack_labels = labels[targets].long()
            probs = F.softmax(logits_clean[targets], dim=1)
            pv = probs.gather(1, attack_labels.view(-1, 1)).view(-1).clamp_min(1e-12)
            degree = degree_from_edge_index(int(features.size(0)), adj)[targets]
            mu = tdgia_defective_scores_from_probability(
                pv, degree, DEGREE_LIMIT, ALPHA_MU, K1, K2
            )
            timestep_slot_scores[t] = slot_scores_from_node_scores(
                mu, N_INJECT, DEGREE_LIMIT
            )
            eligible_timesteps.append(t)
        else:
            timestep_slot_scores[t] = []

    sampled_injection_timesteps, injection_allocation = sample_injection_schedule(
        eligible_timesteps, timestep_slot_scores, N_INJECT
    )
    print(
        f"Eligible attack slices: {len(eligible_timesteps)} | "
        f"score-selected injections: {len(sampled_injection_timesteps)}"
    )

    per_slice = []
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
    attacked_target_outcomes = []

    for t in range(predict_start, predict_end):
        assigned_n_inject = int(injection_allocation.get(t, 0))
        if t >= len(feature_list):
            per_slice.append({
                "t": t,
                "n_labeled": 0,
                "n_targets_available": 0,
                "n_budgeted_selected_targets": 0,
                "n_attacked_targets": 0,
                "n_injected_nodes_budgeted": assigned_n_inject,
                "n_injected_nodes": 0,
                "skipped": True,
            })
            continue
        features = feature_list[t]
        adj = adj_list[t]
        labels = label_list[t]
        ca_weights = ca_weights_list[t]

        if features is None or labels is None or features.numel() == 0 or labels.numel() == 0:
            per_slice.append({
                "t": t,
                "n_labeled": 0,
                "n_targets_available": 0,
                "n_budgeted_selected_targets": 0,
                "n_attacked_targets": 0,
                "n_injected_nodes_budgeted": assigned_n_inject,
                "n_injected_nodes": 0,
                "skipped": True,
            })
            continue

        with torch.no_grad():
            logits_clean = victim_atk.forward_logits(features, adj, ca_weights)
        pred_clean = logits_clean.argmax(dim=1)
        with torch.no_grad():
            if surrogate_is_static:
                x_sur_clean = apply_static_surrogate_feature_transform(
                    features[:, :raw_feature_dim], static_surrogate_feature_transform
                )
                surrogate_logits_clean = forward_logits(surrogate_model, x_sur_clean, adj)
            else:
                surrogate_logits_clean = surrogate_atk.forward_logits(features, adj, ca_weights)
        surrogate_pred_clean = surrogate_logits_clean.argmax(dim=1)

        targets = planning_targets.get(t, torch.empty(0, dtype=torch.long)).to(device)
        if targets.numel() == 0 or assigned_n_inject == 0:
            per_slice.append({
                "t": t,
                "n_labeled": int(labels.numel()),
                "n_targets_available": int(targets.numel()),
                "n_budgeted_selected_targets": 0,
                "n_attacked_targets": 0,
                "n_injected_nodes_budgeted": assigned_n_inject,
                "n_injected_nodes": 0,
                "skipped": True,
            })
            pooled_y_true.append(labels.detach().cpu())
            pooled_pred_clean.append(pred_clean.detach().cpu())
            pooled_pred_adv.append(pred_clean.detach().cpu())
            pooled_logits_clean.append(logits_clean.detach().cpu())
            pooled_logits_adv.append(logits_clean.detach().cpu())
            pooled_attack_mask.append(torch.zeros(labels.numel(), dtype=torch.bool))
            continue

        # Init reference: licit labeled nodes in this slice, excluding targets.
        init_ref_mask = (labels == 0)
        init_ref_mask[targets] = False
        init_reference = init_ref_mask.nonzero(as_tuple=False).view(-1)
        if init_reference.numel() == 0:
            raise RuntimeError(
                f"No licit non-target nodes available at slice t={t} for feature init."
            )

        t0 = time.perf_counter()
        if surrogate_is_static:
            surrogate_data = make_static_surrogate_data(
                features, labels, adj, raw_feature_dim, static_surrogate_feature_transform
            )
            static_atk = TDGIAAttack(
                surrogate_model,
                surrogate_data,
                device,
                clamp=CLAMP,
                attack_dim=raw_feature_dim,
            )
            res = static_atk.attack(
                target_nodes=targets,
                attack_labels=labels[targets].long(),
                n_inject=assigned_n_inject, degree_limit=DEGREE_LIMIT,
                batch_size=BATCH_SIZE, steps=STEPS, lr=LR, smooth_r=SMOOTH_R,
                alpha_mu=ALPHA_MU, k1=K1, k2=K2,
                init=INIT, reference_nodes=init_reference, sigma_scale=SIGMA_SCALE,
                eps_feature=EPS_FEATURE,
            )
            x_adv_transfer = transfer_static_surrogate_result_to_cosemi(
                res,
                features,
                raw_feature_dim,
                static_surrogate_feature_transform,
                init_reference,
            )
            edge_index_adv_transfer = res.edge_index_adv
        else:
            res = surrogate_atk.attack_slice(
                features, adj, targets,
                n_inject=assigned_n_inject, degree_limit=DEGREE_LIMIT,
                batch_size=BATCH_SIZE, steps=STEPS, lr=LR, smooth_r=SMOOTH_R,
                alpha_mu=ALPHA_MU, k1=K1, k2=K2,
                init=INIT, reference_nodes=init_reference, sigma_scale=SIGMA_SCALE,
                eps_feature=EPS_FEATURE,
                ca_weights=ca_weights,
                attack_labels=labels[targets].long(),
            )
            x_adv_transfer = res.x_adv
            edge_index_adv_transfer = res.edge_index_adv
        attack_time_seconds += float(time.perf_counter() - t0)
        total_injected_nodes += len(res.injected_node_ids)
        total_edges_added += len(res.injected_edges)
        with torch.no_grad():
            logits_adv = victim_atk.forward_logits(
                x_adv_transfer, edge_index_adv_transfer, ca_weights
            )[: labels.numel()]
            if surrogate_is_static:
                surrogate_logits_adv_full = forward_logits(surrogate_model, res.x_adv, res.edge_index_adv)
                surrogate_logits_adv = surrogate_logits_adv_full[: labels.numel()]
            else:
                surrogate_logits_adv = res.logits_adv
        pred_adv = logits_adv.argmax(dim=1)
        surrogate_pred_adv = surrogate_logits_adv.argmax(dim=1)

        attacked_target_ids = attacked_target_ids_from_edges(res.injected_edges, targets)
        attacked_mask = mask_from_target_ids(
            labels.numel(), attacked_target_ids, device
        )
        target_outcome = attacked_target_outcome(
            y_true=labels,
            victim_pred_clean=pred_clean,
            victim_pred_adv=pred_adv,
            surrogate_pred_clean=surrogate_pred_clean,
            surrogate_pred_adv=surrogate_pred_adv,
            attacked_mask=attacked_mask,
            target_unit="timestep_node",
        )
        attacked_target_outcomes.append(target_outcome)

        slice_mask = torch.ones(labels.numel(), dtype=torch.bool, device=device)
        roc_clean = roc_auc_binary(logits_clean, labels, slice_mask)
        roc_adv = roc_auc_binary(logits_adv, labels, slice_mask)
        clean_m = binary_classification_metrics(labels, pred_clean)
        adv_m = binary_classification_metrics(labels, pred_adv)

        conf_drop, n_used = mean_confidence_drop(
            labels, logits_clean, logits_adv, attacked_mask, only_clean_correct=True
        )

        if len(res.injected_node_ids) > 0:
            if surrogate_is_static:
                perturbation_dim = raw_feature_dim
                clean_inj = invert_static_surrogate_feature_transform(
                    res.x_injected_base[:, :perturbation_dim],
                    static_surrogate_feature_transform,
                )
                adv_source = invert_static_surrogate_feature_transform(
                    res.x_adv,
                    static_surrogate_feature_transform,
                )
            else:
                perturbation_dim = raw_feature_dim
                clean_inj = res.x_injected_base[:, :raw_feature_dim]
                adv_source = x_adv_transfer[:, :raw_feature_dim]
            adv_inj = adv_source[res.injected_node_ids, :perturbation_dim]
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

        per_slice.append({
            "t": t,
            "n_labeled": int(labels.numel()),
            "n_targets_available": int(targets.numel()),
            "n_budgeted_selected_targets": int(targets.numel()),
            "n_attacked_targets": int(target_outcome["n_attacked_targets"]),
            "n_injected_nodes_budgeted": assigned_n_inject,
            "n_injected_nodes": len(res.injected_node_ids),
            "edges_added": len(res.injected_edges),
            "attacked_target_ids": attacked_target_ids,
            "target_outcome": target_outcome,
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

        pooled_y_true.append(labels.detach().cpu())
        pooled_pred_clean.append(pred_clean.detach().cpu())
        pooled_pred_adv.append(pred_adv.detach().cpu())
        pooled_logits_clean.append(logits_clean.detach().cpu())
        pooled_logits_adv.append(logits_adv.detach().cpu())
        pooled_attack_mask.append(attacked_mask.detach().cpu())

        asr_obj = target_outcome["asr"]
        asr_value = asr_obj["value"]
        asr_text = "nan" if asr_value is None else f"{asr_value:.4f}"
        print(
            f"t={t:2d}  n_labeled={labels.numel():5d}  n_targets={targets.numel():4d}  "
            f"ASR={asr_text} ({asr_obj['success']}/{asr_obj['attempted_clean_correct']})  "
            f"F1_pos {clean_m['f1_pos']:.3f}->{adv_m['f1_pos']:.3f}  "
            f"ROC {roc_clean if roc_clean==roc_clean else float('nan'):.3f}->{roc_adv if roc_adv==roc_adv else float('nan'):.3f}"
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
        n_selected_targets_total = int(sum(item.get("n_targets_available", 0) for item in per_slice))
        n_budgeted_selected_targets_total = int(
            sum(item.get("n_budgeted_selected_targets", 0) for item in per_slice)
        )
        n_eligible_timesteps = int(sum(1 for item in per_slice if item.get("n_targets_available", 0) > 0))
        n_budgeted_timesteps = int(sum(1 for item in per_slice if item.get("n_injected_nodes_budgeted", 0) > 0))
        target_outcome_c = aggregate_attacked_target_outcomes(
            attacked_target_outcomes,
            target_unit="timestep_node",
        )
        coverage_c = coverage_metrics(
            n_selected_targets=n_selected_targets_total,
            n_budgeted_selected_targets=n_budgeted_selected_targets_total,
            n_attacked_targets=int(target_outcome_c["n_attacked_targets"]),
            target_unit="timestep_node",
        )
        budget_efficiency_c = budget_efficiency_metrics(
            n_attacked_targets=int(target_outcome_c["n_attacked_targets"]),
            n_success=int(target_outcome_c["asr"]["success"]),
            n_injected_nodes=total_injected_nodes,
            n_logical_injected_edges=total_edges_added,
        )
        concat_attack = {
            "target_outcome": target_outcome_c,
            "coverage": coverage_c,
            "budget_efficiency": budget_efficiency_c,
            "n_eligible_timesteps": n_eligible_timesteps,
            "n_budgeted_timesteps": n_budgeted_timesteps,
            "mean_confidence_drop": {
                "scope": "attacked_targets",
                "value": conf_drop_c,
                "n_clean_correct": conf_drop_n_c,
            },
            "perturbation_l2_on_injected_nodes": pert_l2_c,
            "perturbation_l2_n_injected_nodes": pert_l2_n_c,
            "avg_perturbation": avg_perturbation_c,
            "avg_perturbation_signed": avg_perturbation_signed_c,
        }

    print()
    print(f"TDGIA CoSemiGNN | total injected nodes={total_injected_nodes}, edges={total_edges_added}")
    if concat_classification is not None and concat_attack is not None:
        asr_obj = concat_attack["target_outcome"]["asr"]
        asr_value = asr_obj["value"]
        asr_text = "nan" if asr_value is None else f"{asr_value:.4f}"
        coverage_value = concat_attack["coverage"]["attacked_target_coverage"]
        coverage_text = "nan" if coverage_value is None else f"{coverage_value:.4f}"
        print(f"[concatenated across timesteps, n={concat_classification['n_total']}]")
        print(
            f"  Attacked-target ASR: {asr_text} "
            f"({asr_obj['success']}/{asr_obj['attempted_clean_correct']})"
        )
        print(f"  Attacked-target coverage: {coverage_text}")
        print(f"  F1_pos: {concat_classification['f1_pos_clean']:.4f} -> {concat_classification['f1_pos_adv']:.4f}")
    print(f"Attack time (total over test timesteps): {attack_time_seconds:.4f} s")

    run_dir, ts = make_run_dir(MODEL_NAME, SURROGATE_MODEL_NAME)
    config = {
        "timestamp": ts,
        "attack": "TDGIA-BlackBox",
        "threat_model": {
            "access": "black_box_transfer",
            "attack_stage": "evasion",
            "victim_access_during_attack": "none" if TARGET_SELECTION_MODEL == "surrogate" else "target_selection_only",
            "surrogate": (
                "trained_raw_feature_temporal_gcn"
                if train_temporal_gcn_surrogate_flag
                else "separate_static_checkpoint"
                if surrogate_is_static
                else "separate_cosemignn_checkpoint"
            ),
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
            "raw_feature_dim": raw_feature_dim,
            "feature_dim": feature_in,
            "budget_mode": "global_sequence",
            "persistence": "non_persistent",
            "feature_bounds": "manual_clamp" if CLAMP is not None else "per_feature_data_minmax",
            "static_surrogate_feature_transform": (
                "cosemi_train_slice_raw_feature_standard_scaler"
                if train_temporal_gcn_surrogate_flag
                else "elliptic_standard_scaler_from_surrogate_train_split"
                if surrogate_is_static and static_surrogate_feature_transform is not None
                else "none"
            ),
            "crafting_model": "surrogate_model",
            "evaluation_model": "victim_model",
            "attack_label_source": "true_target_labels",
        },
        "timesteps": {"predict_start": predict_start, "predict_end": predict_end},
        "target_selection": {
            "selection_logits": f"{TARGET_SELECTION_MODEL}_clean_logits",
            "attack_only_illicit": ATTACK_ONLY_ILLICIT,
            "attack_fraction": ATTACK_FRACTION,
            "only_clean_correct": ONLY_CLEAN_CORRECT,
            "seed": SEED,
        },
        "data": data_cfg,
        "model_hparams": model_cfg,
        "surrogate_data": surrogate_ckpt_cfg["data"],
        "surrogate_model_hparams": surrogate_model_cfg,
        "semi_cache_dir": semi_cache_dir,
        "injection_schedule": {
            "strategy": "top_tdgia_defective_score",
            "eligible_timesteps": eligible_timesteps,
            "sampled_timesteps": sampled_injection_timesteps,
            "selected_timesteps": sampled_injection_timesteps,
            "timestep_defective_scores": [
                {
                    "t": int(t),
                    "top_slot_score": float(timestep_slot_scores.get(int(t), [0.0])[0])
                    if timestep_slot_scores.get(int(t), [])
                    else 0.0,
                    "slot_scores": [float(s) for s in timestep_slot_scores.get(int(t), [])],
                }
                for t in sorted(timestep_slot_scores)
            ],
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
            "target_outcome": concat_attack["target_outcome"] if concat_attack else None,
            "coverage": concat_attack["coverage"] if concat_attack else None,
            "budget_efficiency": concat_attack["budget_efficiency"] if concat_attack else None,
            "timesteps": {
                "n_eligible_timesteps": concat_attack["n_eligible_timesteps"] if concat_attack else 0,
                "n_budgeted_timesteps": concat_attack["n_budgeted_timesteps"] if concat_attack else 0,
            },
            "mean_confidence_drop": concat_attack["mean_confidence_drop"] if concat_attack else None,
            "perturbation": {
                "l2_on_injected_nodes": concat_attack["perturbation_l2_on_injected_nodes"] if concat_attack else 0.0,
                "n_injected_nodes": concat_attack["perturbation_l2_n_injected_nodes"] if concat_attack else 0,
                "avg_abs": concat_attack["avg_perturbation"] if concat_attack else 0.0,
                "avg_signed": concat_attack["avg_perturbation_signed"] if concat_attack else 0.0,
            },
        },
        "injection": {
            "total_injected_nodes": total_injected_nodes,
            "total_edges_added": total_edges_added,
            "total_logical_injected_edges": total_edges_added,
            "budget_mode": "global_sequence",
            "schedule_strategy": "top_tdgia_defective_score",
            "n_inject_total_budget": int(N_INJECT),
            "n_selected_targets_total": int(sum(item.get("n_targets_available", 0) for item in per_slice)),
            "n_budgeted_selected_targets_total": int(sum(item.get("n_budgeted_selected_targets", 0) for item in per_slice)),
            "n_attacked_targets_total": (
                int(concat_attack["target_outcome"]["n_attacked_targets"])
                if concat_attack
                else 0
            ),
            "eligible_timesteps": eligible_timesteps,
            "sampled_timesteps": sampled_injection_timesteps,
            "selected_timesteps": sampled_injection_timesteps,
            "timestep_defective_scores": [
                {
                    "t": int(t),
                    "top_slot_score": float(timestep_slot_scores.get(int(t), [0.0])[0])
                    if timestep_slot_scores.get(int(t), [])
                    else 0.0,
                    "slot_scores": [float(s) for s in timestep_slot_scores.get(int(t), [])],
                }
                for t in sorted(timestep_slot_scores)
            ],
            "allocation_by_timestep": [
                {"t": int(t), "n_inject": int(injection_allocation[t])}
                for t in sorted(injection_allocation)
            ],
        },
        "per_timestep": per_slice,
    }
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
