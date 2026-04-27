import os
import json
import torch

from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT
from src.models.chronowave_gnn import ChronoWaveGNN
from src.models.recgnn import RecGNN


def _list_run_dirs(model_dir: str, model_name: str):
    prefix = f"{model_name}_"
    if not os.path.isdir(model_dir):
        return []

    run_dirs = []
    for d in os.listdir(model_dir):
        full = os.path.join(model_dir, d)
        if os.path.isdir(full) and d.startswith(prefix):
            ckpt = os.path.join(full, "model.pt")
            if os.path.exists(ckpt):
                run_dirs.append(full)
    return run_dirs


def _pick_latest_run_dir(run_dirs):
    if not run_dirs:
        return None
    run_dirs = sorted(run_dirs, key=lambda p: os.path.basename(p))
    return run_dirs[-1]


def resolve_checkpoint(model_name: str, model_dir: str = "models", run_id: str | None = None):
    if run_id is not None:
        run_dir = os.path.join(model_dir, f"{model_name}_{run_id}")
        ckpt = os.path.join(run_dir, "model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt, run_dir

    run_dirs = _list_run_dirs(model_dir, model_name)
    run_dir = _pick_latest_run_dir(run_dirs)
    if run_dir is None:
        raise FileNotFoundError(
            f"No checkpoint found for {model_name} in {model_dir}. "
            f"Expected {model_dir}/{model_name}_YYYYMMDD_HHMMSS/model.pt"
        )

    return os.path.join(run_dir, "model.pt"), run_dir


def _load_run_config(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "r") as f:
        return json.load(f)


def _load_yaml_config(config_dir: str, model_name: str):
    try:
        import yaml
    except Exception:
        return None

    path = os.path.join(config_dir, f"{model_name}.yaml")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    if "model" in cfg and isinstance(cfg["model"], dict):
        return cfg
    return {"model": cfg, "training": cfg.get("training", {})}


def build_model(model_name: str, num_features: int, num_classes: int, cfg: dict):
    m = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    if model_name == "gcn":
        return GCN(
            in_dim=num_features,
            hidden_dim=int(m.get("hidden_dim", 64)),
            out_dim=num_classes,
            num_layers=int(m.get("num_layers", 2)),
            dropout=float(m.get("dropout", 0.5)),
            use_norm=bool(m.get("use_norm", True)),
        )

    if model_name == "graphsage":
        return GraphSAGE(
            in_dim=num_features,
            hid_dim=int(m.get("hidden_dim", 64)),
            out_dim=num_classes,
            aggr=m.get("aggr", "mean"),
        )

    if model_name == "gat":
        return GAT(
            in_dim=num_features,
            hidden_dim=int(m.get("hidden_dim", 32)),
            out_dim=num_classes,
            num_layers=int(m.get("num_layers", 2)),
            heads=int(m.get("heads", 4)),
            dropout=float(m.get("dropout", 0.6)),
            use_norm=bool(m.get("use_norm", True)),
        )

    if model_name == "chronowave_gnn":
        return ChronoWaveGNN(
            in_dim=num_features,
            hidden_dim=int(m.get("hidden_dim", 128)),
            out_dim=num_classes,
            time_dim=int(m.get("time_dim", 8)),
            heads=int(m.get("heads", 4)),
            num_layers=int(m.get("num_layers", 3)),
            dropout=float(m.get("dropout", 0.4)),
        )

    if model_name == "recgnn":
        state_rows = m.get("state_rows", None)
        if state_rows is None:
            raise ValueError(
                "RecGNN checkpoints now require model.state_rows in config.json because the paper model "
                "is sequence-based over variable-size timestep graphs."
            )
        return RecGNN(
            in_dim=num_features,
            hidden_dim=int(m.get("hidden_dim", 50)),
            out_dim=num_classes,
            state_rows=int(state_rows),
            dropout=float(m.get("dropout", 0.5)),
        )

    if model_name == "cosemignn":
        from src.models.cosemignn import CoSemiGNN

        return CoSemiGNN(
        feature_in=num_features,
        dim=int(m.get("dim", 128)),
        dim2=int(m.get("dim2", 256)),
        dim3=int(m.get("dim3", 128)),
        num_heads=int(m.get("num_heads", 4)),
    )

    raise ValueError(
        f"Unknown model: {model_name}. Choose from 'gcn', 'graphsage', 'gat', "
        f"'chronowave_gnn', 'recgnn', 'cosemignn'."
    )


def load_model(
    model_name: str,
    num_features: int,
    num_classes: int = 2,
    device: str | torch.device = "cpu",
    model_dir: str = "models",
    config_dir: str = "config/models",
    run_id: str | None = None,
):
    model_name = model_name.lower()
    ckpt_path, run_dir = resolve_checkpoint(model_name, model_dir=model_dir, run_id=run_id)

    cfg = _load_run_config(run_dir)
    if cfg is None:
        ycfg = _load_yaml_config(config_dir, model_name)
        cfg = ycfg if ycfg is not None else {"model": {}, "training": {}}

    model = build_model(model_name, num_features, num_classes, cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"✓ Loaded {model_name.upper()} from {ckpt_path}")
    return model


def load_all_models(
    num_features: int,
    num_classes: int = 2,
    device: str | torch.device = "cpu",
    model_dir: str = "models",
    config_dir: str = "config/models",
    model_names=("gcn", "graphsage", "gat"),
):
    models = {}
    for name in model_names:
        try:
            models[name] = load_model(
                model_name=name,
                num_features=num_features,
                num_classes=num_classes,
                device=device,
                model_dir=model_dir,
                config_dir=config_dir,
                run_id=None,
            )
        except FileNotFoundError as e:
            print(f"⚠ {e}")
    return models
