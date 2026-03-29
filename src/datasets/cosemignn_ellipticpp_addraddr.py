from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import yeojohnson
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEMI_MODEL_ORDER = [
    "AdaBoost",
    "DecisionTree",
    "gbBoost",
    "RandomForest",
    "xgboost",
    "SVC",
]

def _map_ellipticpp_labels(series: pd.Series) -> pd.Series:
    """
    Elliptic++ convention wanted by user:
    class-1 (illicit) -> 1
    class-2 (licit)   -> 0
    class-3 (unknown) -> -1
    """
    mapped = pd.to_numeric(series, errors="coerce").map({1: 1, 2: 0, 3: -1})
    return mapped.astype("Int64")

def _pick_column(columns: Iterable[str], preferred: list[str], contains: list[str] | None = None) -> str | None:
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in preferred:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if contains:
        for c in cols:
            cl = c.lower()
            if all(tok in cl for tok in contains):
                return c
    return None


def _detect_time_col(df: pd.DataFrame) -> str | None:
    return (
        _pick_column(df.columns, ["Time step", "time_step", "timeStep", "timestep", "ts"])
        or _pick_column(df.columns, [], contains=["time"])
    )


def _detect_label_col(df: pd.DataFrame) -> str:
    col = (
        _pick_column(df.columns, ["class", "Class", "label", "Label", "labels"])
        or _pick_column(df.columns, [], contains=["class"])
        or _pick_column(df.columns, [], contains=["label"])
    )
    if col is None:
        raise ValueError("Could not detect label/class column.")
    return col


def _detect_id_col(feature_df: pd.DataFrame, class_df: pd.DataFrame, time_col: str | None) -> str:
    common = [c for c in feature_df.columns if c in class_df.columns]
    common = [c for c in common if c != time_col and c.lower() not in {"class", "label", "labels"}]
    if not common:
        raise ValueError("Could not detect shared node-id column between features and classes CSVs.")

    preferred = [
        "address",
        "Address",
        "wallet",
        "walletId",
        "wallet_id",
        "walletAddress",
        "wallet_address",
        "addr",
        "addrId",
        "addr_id",
        "node",
        "node_id",
        "nodeId",
        "txId",
        "id",
    ]
    picked = _pick_column(common, preferred)
    return picked if picked is not None else common[0]


def _detect_edge_cols(edge_df: pd.DataFrame) -> tuple[str, str, str | None]:
    src = (
        _pick_column(edge_df.columns, ["source", "src", "from", "node1", "u", "txId1", "addr1", "address1"])
        or edge_df.columns[0]
    )
    remaining = [c for c in edge_df.columns if c != src]
    dst = (
        _pick_column(remaining, ["target", "dst", "to", "node2", "v", "txId2", "addr2", "address2"])
        or remaining[0]
    )
    time_col = _detect_time_col(edge_df)
    return src, dst, time_col


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(np.float32)

def _aggregate_duplicate_actor_rows(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    label_col: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Elliptic++ actor features can contain multiple temporal-interaction rows
    for the same wallet within the same time step. For CoSemiGNN we need one
    node per (wallet, time), so aggregate duplicates.

    Aggregation rule:
    - numeric feature columns -> mean
    - label -> first non-null value (should be consistent per wallet)
    """
    if df.empty:
        return df

    work = df.copy()

    # Make numeric features numeric before grouping
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_spec = {c: "mean" for c in feature_cols}
    agg_spec[label_col] = "first"

    grouped = (
        work.groupby([id_col, time_col], as_index=False)
        .agg(agg_spec)
        .reset_index(drop=True)
    )

    return grouped

def _auto_minmax_scaler(df: pd.DataFrame, use_percentile: bool = True, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    if use_percentile:
        q_min = df.quantile(lower)
        q_max = df.quantile(upper)
    else:
        q_min = df.min()
        q_max = df.max()
    denom = (q_max - q_min).replace(0, 1e-9)
    clipped = df.clip(lower=q_min, upper=q_max, axis=1)
    scaled = (clipped - q_min) / denom
    return scaled.fillna(0.0).astype(np.float32)


def _yeojohnson_safe(col: np.ndarray) -> np.ndarray:
    if np.allclose(col, col[0]):
        return col
    try:
        return yeojohnson(col)[0]
    except Exception:
        return col


def _cluster_preprocess(features: pd.DataFrame) -> np.ndarray:
    x = _safe_numeric_frame(features).to_numpy()
    if x.shape[0] == 0:
        return x
    x = StandardScaler().fit_transform(x)
    x = normalize(x, norm="l2")
    x = np.apply_along_axis(_yeojohnson_safe, 0, x)
    return x.astype(np.float32)


def _masked_labels_for_self_training(y01: np.ndarray, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y_masked = y01.copy()
    for cls in [0, 1]:
        idx = np.where(y01 == cls)[0]
        if len(idx) == 0:
            continue
        n_mask = int(len(idx) * 0.7)
        if n_mask <= 0:
            continue
        masked = rng.choice(idx, size=n_mask, replace=False)
        y_masked[masked] = -1
    return y_masked


def _constant_or_self_train(base_clf, x: np.ndarray, y01: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    if len(np.unique(y01)) < 2:
        return np.full(len(y01), fill_value=int(y01[0]), dtype=np.int64)

    y_masked = _masked_labels_for_self_training(y01, random_state=0)
    known = y_masked != -1
    if known.sum() < 2 or len(np.unique(y_masked[known])) < 2:
        majority = int(pd.Series(y01).mode().iloc[0])
        return np.full(len(y01), fill_value=majority, dtype=np.int64)

    st = SelfTrainingClassifier(
        base_clf,
        threshold=threshold,
        criterion="threshold",
        k_best=50,
        max_iter=1000,
    ).fit(x, y_masked)

    y_pred = np.asarray(st.transduction_, dtype=np.int64)
    y_pred[y_pred == -1] = 0
    return y_pred


def _fit_xgboost_or_fallback(x: np.ndarray, y01: np.ndarray) -> np.ndarray:
    if len(np.unique(y01)) < 2:
        return np.full(len(y01), fill_value=int(y01[0]), dtype=np.int64)

    y_masked = _masked_labels_for_self_training(y01, random_state=0)
    known = y_masked != -1
    x_train = x[known]
    y_train = y_masked[known]
    if len(np.unique(y_train)) < 2:
        majority = int(pd.Series(y01).mode().iloc[0])
        return np.full(len(y01), fill_value=majority, dtype=np.int64)

    try:
        import xgboost as xgb

        clf = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.3,
            n_estimators=300,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            n_jobs=1,
        )
        clf.fit(x_train, y_train)
        pred = clf.predict_proba(x)[:, 1]
        return (pred > 0.5).astype(np.int64)
    except Exception:
        # fallback: keep the pipeline running even if xgboost is unavailable
        fallback = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
        fallback.fit(x_train, y_train)
        pred = fallback.predict_proba(x)[:, 1]
        return (pred > 0.5).astype(np.int64)


def _build_or_load_semi_predictions(
    slice_df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
    label_col: str,
    time_value: int,
    semi_cache_dir: str,
    rebuild: bool,
) -> pd.DataFrame:
    cache_root = Path(semi_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    cached_frames: dict[str, pd.DataFrame] = {}
    all_exist = True
    for model_name in SEMI_MODEL_ORDER:
        path = cache_root / model_name / f"{model_name}{time_value}.csv"
        if path.exists() and not rebuild:
            cached_frames[model_name] = pd.read_csv(path)
        else:
            all_exist = False

    if all_exist:
        out = pd.DataFrame({id_col: slice_df[id_col].astype(str).values})
        for model_name in SEMI_MODEL_ORDER:
            df = cached_frames[model_name].copy()
            first_col = df.columns[0]
            pred_col = _pick_column(df.columns, ["predict", "prediction", "pred"]) or df.columns[-1]
            df[first_col] = df[first_col].astype(str)
            mapper = dict(zip(df[first_col], df[pred_col]))
            out[model_name] = out[id_col].map(mapper).fillna(1).astype(np.int64)
        return out

    x = _cluster_preprocess(slice_df[feature_cols])
    y01 = slice_df[label_col].astype(int).to_numpy()  # now: illicit=1, licit=0, unknown=-1

    pred_map: dict[str, np.ndarray] = {}
    pred_map["AdaBoost"] = _constant_or_self_train(AdaBoostClassifier(n_estimators=50, random_state=42), x, y01)
    pred_map["DecisionTree"] = _constant_or_self_train(
        DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_split=2, random_state=42), x, y01
    )
    pred_map["gbBoost"] = _constant_or_self_train(
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), x, y01
    )
    pred_map["RandomForest"] = _constant_or_self_train(RandomForestClassifier(n_estimators=100, random_state=42), x, y01)
    pred_map["xgboost"] = _fit_xgboost_or_fallback(x, y01)
    pred_map["SVC"] = _constant_or_self_train(
        SVC(kernel="rbf", gamma=5, probability=True, class_weight={0: 0.8, 1: 0.2}, tol=1e-5), x, y01
    )

    out = pd.DataFrame({id_col: slice_df[id_col].astype(str).values})
    for model_name in SEMI_MODEL_ORDER:
        pred = pred_map[model_name].astype(np.int64)
        out[model_name] = pred
        out_dir = cache_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({id_col: out[id_col].values, "predict": pred}).to_csv(
            out_dir / f"{model_name}{time_value}.csv", index=False
        )

    return out


def _build_edge_index(node_ids: list[str], edge_df: pd.DataFrame, src_col: str, dst_col: str) -> tuple[torch.Tensor, dict[str, int]]:
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    rows: list[int] = []
    cols: list[int] = []

    for u, v in edge_df[[src_col, dst_col]].itertuples(index=False):
        su, sv = str(u), str(v)
        if su in node_to_idx and sv in node_to_idx:
            ui = node_to_idx[su]
            vi = node_to_idx[sv]
            if ui == vi:
                continue
            rows.extend([ui, vi])
            cols.extend([vi, ui])

    if not rows:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index, node_to_idx


def _compute_ca_feature(ca_matrix: np.ndarray, scaled_feature_df: pd.DataFrame, semi_preds: np.ndarray) -> np.ndarray:
    if len(scaled_feature_df) == 0:
        return np.zeros((0, 1 + semi_preds.shape[1]), dtype=np.float32)

    ca_agg = ca_matrix @ scaled_feature_df.to_numpy(dtype=np.float32)
    ca_agg_df = pd.DataFrame(ca_agg, columns=scaled_feature_df.columns)
    ca_scaled = _auto_minmax_scaler(ca_agg_df, use_percentile=True)
    combined = scaled_feature_df.to_numpy(dtype=np.float32) + (1.0 + 1e-8 - ca_scaled.to_numpy(dtype=np.float32)) * scaled_feature_df.to_numpy(dtype=np.float32)

    if combined.shape[0] < 2 or np.allclose(combined, combined[0]):
        pca_col = np.zeros((combined.shape[0], 1), dtype=np.float32)
    else:
        pca_col = PCA(n_components=1).fit_transform(combined).astype(np.float32)

    return np.concatenate([pca_col, semi_preds.astype(np.float32) * 0.9], axis=1)


def load_cosemignn_ellipticpp_addraddr(
    feature_path: str,
    class_path: str,
    edge_path: str,
    semi_cache_dir: str,
    device: str | torch.device = "cpu",
    rebuild_semi: bool = False,
):
    feature_df = pd.read_csv(feature_path)
    class_df = pd.read_csv(class_path)
    edge_df = pd.read_csv(edge_path)

    time_col_feature = _detect_time_col(feature_df)
    time_col_class = _detect_time_col(class_df)
    time_col = time_col_feature or time_col_class
    label_col = _detect_label_col(class_df)
    id_col = _detect_id_col(feature_df, class_df, time_col)
    src_col, dst_col, edge_time_col = _detect_edge_cols(edge_df)

    feature_df = feature_df.copy()
    class_df = class_df.copy()
    edge_df = edge_df.copy()

    if time_col_feature is not None and time_col_class is not None and time_col_feature != time_col_class:
        class_df = class_df.rename(columns={time_col_class: time_col_feature})
        time_col_class = time_col_feature
        time_col = time_col_feature

    feature_df[id_col] = feature_df[id_col].astype(str)
    class_df[id_col] = class_df[id_col].astype(str)
    edge_df[src_col] = edge_df[src_col].astype(str)
    edge_df[dst_col] = edge_df[dst_col].astype(str)

    merge_keys = [id_col]
    if time_col_feature is not None and time_col_class is not None and time_col_feature == time_col_class:
        merge_keys.append(time_col_feature)
    merged = pd.merge(feature_df, class_df, on=merge_keys, how="inner")

    label_col = _detect_label_col(merged)
    if time_col is None:
        raise ValueError("Could not detect time-step column in features/classes CSVs.")

    merged[time_col] = pd.to_numeric(merged[time_col], errors="coerce").astype("Int64")
    if edge_time_col is not None:
        edge_df[edge_time_col] = pd.to_numeric(edge_df[edge_time_col], errors="coerce").astype("Int64")
    merged[label_col] = pd.to_numeric(merged[label_col], errors="coerce").astype("Int64")
    merged = merged[merged[label_col].isin([1, 2])].copy()  # drop unknown class-3
    merged[label_col] = merged[label_col].map({1: 1, 2: 0}).astype("Int64")

    non_feature = set(merge_keys + [label_col])
    if time_col in merged.columns:
        non_feature.add(time_col)
    feature_cols = [c for c in merged.columns if c not in non_feature]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(pd.to_numeric(merged[c], errors="coerce"))]

    merged = _aggregate_duplicate_actor_rows(
    df=merged,
    id_col=id_col,
    time_col=time_col,
    label_col=label_col,
    feature_cols=feature_cols,
)

    time_values = sorted(int(t) for t in merged[time_col].dropna().unique())

    feature_list = [None]
    adj_list = [None]
    label_list = [None]
    ca_matrix_list = [None]
    ca_weights_list = [None]
    semi_result_list = [None]
    fake_label_list = [None]
    ca_feature_list = [None]

    for t in time_values:
        slice_df = merged[merged[time_col] == t].copy().reset_index(drop=True)
        if slice_df.empty:
            feature_list.append(None)
            adj_list.append(None)
            label_list.append(None)
            ca_matrix_list.append(None)
            ca_weights_list.append(None)
            semi_result_list.append(None)
            fake_label_list.append(None)
            ca_feature_list.append(None)
            continue

        # After global aggregation, each time slice should now be unique by wallet id.
        if slice_df[id_col].duplicated().any():
            dup_count = int(slice_df[id_col].duplicated().sum())
            slice_df = (
                _aggregate_duplicate_actor_rows(
                    df=slice_df,
                    id_col=id_col,
                    time_col=time_col,
                    label_col=label_col,
                    feature_cols=feature_cols,
                )
                .reset_index(drop=True)
            )
            print(f"[warn] Aggregated {dup_count} duplicate actor rows at time step {t}.")

        semi_df = _build_or_load_semi_predictions(
            slice_df=slice_df,
            id_col=id_col,
            feature_cols=feature_cols,
            label_col=label_col,
            time_value=t,
            semi_cache_dir=semi_cache_dir,
            rebuild=rebuild_semi,
        )
        semi_np = semi_df[SEMI_MODEL_ORDER].to_numpy(dtype=np.float32)

        n = len(slice_df)
        ca_matrix = np.zeros((n, n), dtype=np.float32)
        for col in SEMI_MODEL_ORDER:
            pred = semi_df[col].to_numpy()
            ca_matrix += (pred[:, None] == pred[None, :]).astype(np.float32)
        ca_matrix /= 2.0  # preserve the original CoSemiGNN loader behavior

        feature_df_slice = _safe_numeric_frame(slice_df[feature_cols])
        scaled_feature_df = _auto_minmax_scaler(feature_df_slice, use_percentile=True)
        feature_np = np.concatenate([scaled_feature_df.to_numpy(dtype=np.float32), semi_np * 0.9], axis=1)
        ca_feature_np = _compute_ca_feature(ca_matrix, scaled_feature_df, semi_np)

        edge_slice = edge_df
        if edge_time_col is not None:
            edge_slice = edge_df[edge_df[edge_time_col] == t]
        edge_index, node_to_idx = _build_edge_index(slice_df[id_col].astype(str).tolist(), edge_slice, src_col, dst_col)

        if edge_index.numel() == 0:
            ca_weights = torch.empty((0,), dtype=torch.float32, device=device)
        else:
            weights = [ca_matrix[int(u), int(v)] for u, v in edge_index.t().tolist()]
            ca_weights = torch.tensor(weights, dtype=torch.float32, device=device)

        labels = torch.tensor(slice_df[label_col].astype(int).to_numpy(), dtype=torch.long, device=device)
        feature_tensor = torch.tensor(feature_np, dtype=torch.float32, device=device)
        ca_feature_tensor = torch.tensor(ca_feature_np, dtype=torch.float32, device=device)
        semi_tensor = torch.tensor(semi_np, dtype=torch.float32, device=device)
        fake_label = torch.tensor(semi_np.sum(axis=1) / max(n, 1), dtype=torch.float32, device=device)

        feature_list.append(feature_tensor)
        adj_list.append(edge_index.to(device))
        label_list.append(labels)
        ca_matrix_list.append(ca_matrix)
        ca_weights_list.append(ca_weights)
        semi_result_list.append(semi_tensor)
        fake_label_list.append(fake_label)
        ca_feature_list.append(ca_feature_tensor)

    return (
        feature_list,
        adj_list,
        label_list,
        ca_matrix_list,
        ca_weights_list,
        semi_result_list,
        fake_label_list,
        ca_feature_list,
    )
