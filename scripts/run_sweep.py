import os
import sys
import json
import csv
import time
import itertools
import importlib.util
from typing import Any, Dict, List, Tuple

import yaml


# Map (attack, model_family) -> script path. The "static" family covers
# whole-graph models (gcn, gat, graphsage, chronowave_gnn). Temporal models
# (recgnn, evolvegcn_o, cosemignn) have their own per-model drivers.
ATTACK_TO_SCRIPTS: Dict[str, Dict[str, str]] = {
    "fgsm": {
        "static": "scripts/run_fgsm_attack.py",
        "recgnn": "scripts/run_fgsm_attack_recgnn.py",
        "evolvegcn_o": "scripts/run_fgsm_attack_evolvegcn.py",
        "cosemignn": "scripts/run_fgsm_attack_cosemignn.py",
    },
    "pgd": {
        "static": "scripts/run_pgd_attack.py",
        "recgnn": "scripts/run_pgd_attack_recgnn.py",
        "evolvegcn_o": "scripts/run_pgd_attack_evolvegcn.py",
        "cosemignn": "scripts/run_pgd_attack_cosemignn.py",
    },
    "node_injection": {
        "static": "scripts/run_node_injection_attack.py",
        "recgnn": "scripts/run_node_injection_attack_recgnn.py",
        "evolvegcn_o": "scripts/run_node_injection_attack_evolvegcn.py",
        "cosemignn": "scripts/run_node_injection_attack_cosemignn.py",
    },
    "nettack": {
        "static": "scripts/run_nettack_attack.py",
    },
    "monti": {
        "static": "scripts/run_monti_attack.py",
    },
}

STATIC_MODELS = {"gcn", "gat", "graphsage", "chronowave_gnn"}
TEMPORAL_FAMILIES = {"recgnn", "evolvegcn_o", "cosemignn"}

# Map dataset -> model checkpoint directory under repo root.
DATASET_TO_MODEL_DIR = {
    "elliptic": "models/Elliptic",
    "ellipticpp_actors": "models/Elliptic++",
}


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def attacks_root() -> str:
    return os.path.join(repo_root(), "attacks")


def model_family(model_name: str) -> str:
    if model_name in STATIC_MODELS:
        return "static"
    if model_name in TEMPORAL_FAMILIES:
        return model_name
    raise ValueError(
        f"Unknown model '{model_name}'. Static: {sorted(STATIC_MODELS)}, "
        f"temporal: {sorted(TEMPORAL_FAMILIES)}."
    )


def script_for(attack: str, model_name: str) -> str:
    family = model_family(model_name)
    fam_map = ATTACK_TO_SCRIPTS.get(attack)
    if fam_map is None:
        raise ValueError(f"Unknown attack '{attack}'. Choose from: {list(ATTACK_TO_SCRIPTS)}")
    if family not in fam_map:
        raise ValueError(
            f"Attack '{attack}' is not implemented for model '{model_name}' "
            f"(family={family}). Available families: {list(fam_map)}."
        )
    return fam_map[family]


def load_script_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cartesian(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    params: {KEY: [v1,v2], KEY2: [w1,w2], ...}
    returns list of dicts over cartesian product
    """
    if not params:
        return [{}]
    keys = list(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    out = []
    for prod in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, prod)})
    return out


def list_attack_dirs() -> List[str]:
    root = attacks_root()
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def newest_dir(created_after: float, prev_set: set) -> str:
    cand = []
    for d in list_attack_dirs():
        if d in prev_set:
            continue
        try:
            mt = os.path.getmtime(d)
        except Exception:
            continue
        if mt >= created_after - 1e-6:
            cand.append((mt, d))
    if not cand:
        all_dirs = [(os.path.getmtime(d), d) for d in list_attack_dirs()]
        if not all_dirs:
            raise RuntimeError("No attacks/* directory found after run.")
        all_dirs.sort(key=lambda x: x[0])
        return all_dirs[-1][1]
    cand.sort(key=lambda x: x[0])
    return cand[-1][1]


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = obj
    return out


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_attacks_to_csv(out_csv: str):
    rows: List[Dict[str, Any]] = []
    for d in sorted(list_attack_dirs()):
        cfg_path = os.path.join(d, "config.json")
        met_path = os.path.join(d, "metrics.json")
        if not (os.path.exists(cfg_path) and os.path.exists(met_path)):
            continue

        cfg = read_json(cfg_path)
        met = read_json(met_path)

        row = {"run_dir": os.path.relpath(d, repo_root())}
        row.update({f"config.{k}": v for k, v in flatten(cfg).items()})
        row.update({f"metrics.{k}": v for k, v in flatten(met).items()})
        rows.append(row)

    if not rows:
        print("No runs found under attacks/* with config.json + metrics.json.")
        return

    cols = set()
    for r in rows:
        cols |= set(r.keys())
    cols = sorted(cols)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✓ Wrote summary: {os.path.relpath(out_csv, repo_root())} (rows={len(rows)})")


def apply_globals(mod, overrides: Dict[str, Any]):
    for k, v in overrides.items():
        setattr(mod, k, v)


def normalize_pgd_params(grid: Dict[str, Any]) -> Dict[str, Any]:
    if "ALPHA" in grid and isinstance(grid["ALPHA"], str) and grid["ALPHA"].lower() == "auto":
        eps = float(grid["EPS"])
        steps = int(grid["STEPS"])
        grid["ALPHA"] = 2.0 * eps / max(1, steps)
    return grid


def run_one(mod, run_name: str, overrides: Dict[str, Any]) -> str:
    prev = set(list_attack_dirs())
    start = time.time()

    apply_globals(mod, overrides)

    for attempt in range(3):
        try:
            mod.main()
            break
        except FileExistsError:
            time.sleep(0.3)
            if attempt == 2:
                raise
        finally:
            time.sleep(0.2)

    run_dir = newest_dir(start, prev)
    print(f"✓ Completed {run_name} -> {os.path.relpath(run_dir, repo_root())}")
    return run_dir


def main():
    root = repo_root()
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "config", "experiments.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    g = cfg.get("global", {}) or {}
    sweeps = cfg.get("sweeps", []) or []

    models = g.get("models", ["gcn"])
    seeds = g.get("seeds", [0])

    # dataset selection: a list of datasets to sweep (may be a single string).
    datasets_g = g.get("datasets", g.get("dataset", ["elliptic"]))
    if isinstance(datasets_g, str):
        datasets_g = [datasets_g]
    for ds in datasets_g:
        if ds not in DATASET_TO_MODEL_DIR:
            raise ValueError(
                f"Unknown dataset '{ds}'. Supported: {list(DATASET_TO_MODEL_DIR)}."
            )

    # global target selection defaults
    g_split = g.get("split", "test")
    g_only_illicit = bool(g.get("attack_only_illicit", True))
    g_only_clean_correct = bool(g.get("only_clean_correct", True))
    g_attack_fraction = float(g.get("attack_fraction", 0.02))

    # cache loaded modules (path -> module)
    module_cache: Dict[str, Any] = {}

    def get_mod(path: str):
        if path not in module_cache:
            module_cache[path] = load_script_module(
                f"_run_{os.path.splitext(os.path.basename(path))[0]}",
                os.path.join(root, path),
            )
        return module_cache[path]

    # run grid
    for sweep in sweeps:
        attack = str(sweep["attack"]).lower()
        if attack not in ATTACK_TO_SCRIPTS:
            raise ValueError(
                f"Unknown attack '{attack}'. Choose from: {list(ATTACK_TO_SCRIPTS)}"
            )

        param_grid = sweep.get("params", {}) or {}
        # per-sweep override of models / datasets if provided
        sweep_models = sweep.get("models") or models
        sweep_datasets = sweep.get("datasets") or datasets_g
        if isinstance(sweep_datasets, str):
            sweep_datasets = [sweep_datasets]

        combos = cartesian(param_grid)

        for dataset in sweep_datasets:
            model_dir = DATASET_TO_MODEL_DIR[dataset]
            for model_name in sweep_models:
                # Skip combinations with no script registered.
                try:
                    script_path = script_for(attack, model_name)
                except ValueError as e:
                    print(f"skip: {e}")
                    continue

                mod = get_mod(script_path)

                for seed in seeds:
                    for combo in combos:
                        overrides = {
                            "MODEL_NAME": model_name,
                            "MODEL_DIR": model_dir,
                            "DATASET": dataset,
                            "SPLIT": g_split,
                            "SEED": int(seed),
                            "ATTACK_ONLY_ILLICIT": g_only_illicit,
                            "ONLY_CLEAN_CORRECT": g_only_clean_correct,
                            "ATTACK_FRACTION": g_attack_fraction,
                        }
                        overrides.update(combo)

                        if attack == "pgd":
                            overrides = normalize_pgd_params(overrides)

                        # Only set variables that exist in the target script.
                        filtered = {k: v for k, v in overrides.items() if hasattr(mod, k)}

                        run_id = (
                            f"{attack}|model={model_name}|dataset={dataset}|seed={seed}|"
                            + ",".join(
                                f"{k}={filtered[k]}" for k in sorted(filtered)
                                if k not in {"MODEL_NAME", "MODEL_DIR", "DATASET", "SPLIT", "SEED"}
                            )
                        )
                        run_one(mod, run_id, filtered)

    summarize_attacks_to_csv(os.path.join(attacks_root(), "results_summary.csv"))


if __name__ == "__main__":
    main()
