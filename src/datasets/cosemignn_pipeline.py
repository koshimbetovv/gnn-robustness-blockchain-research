from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn
from scipy.stats import yeojohnson
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
except Exception:
    xgb = None


@dataclass
class TransactionSliceConfig:
    txid_col: str
    time_col: str
    label_col: str
    feature_cols: list[str]
    edge_src_col: str
    edge_dst_col: str
    label_values: tuple[int, int] = (0, 1)
    semi_cache_dir: str | None = None
    build_cache: bool = True


def auto_minmax_scaler(
    df: pd.DataFrame,
    use_percentile: bool = True,
    lower: float = 0.01,
    upper: float = 0.99,
    feature_range=(0, 1),
) -> pd.DataFrame:
    df = df.copy()
    a, b = feature_range
    if use_percentile:
        q_min = df.quantile(lower)
        q_max = df.quantile(upper)
    else:
        q_min = df.min()
        q_max = df.max()

    denominator = (q_max - q_min).replace(0, 1e-9)
    df_clipped = df.clip(lower=q_min, upper=q_max, axis=1)
    scaled_df = (df_clipped - q_min) / denominator * (b - a) + a
    return scaled_df


def create_slice_edge_index(
    txids_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    src_col: str,
    dst_col: str,
):
    slice_nodes = txids_df.iloc[:, 0].tolist()
    edge_df = edge_df[
        edge_df[src_col].isin(slice_nodes) &
        edge_df[dst_col].isin(slice_nodes)
    ]

    nodes = sorted(slice_nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    row_indices = []
    col_indices = []

    for u, v in edge_df[[src_col, dst_col]].itertuples(index=False):
        row_indices.append(node_index[u])
        col_indices.append(node_index[v])
        row_indices.append(node_index[v])
        col_indices.append(node_index[u])

    indices = torch.LongTensor([row_indices, col_indices])
    values = torch.FloatTensor(np.ones(len(row_indices), dtype=np.float32))
    shape = torch.Size([len(nodes), len(nodes)])
    sparse_adj = torch.sparse_coo_tensor(indices, values, shape)
    return indices, sparse_adj


def co_association(matrix: np.ndarray, cluster_result: np.ndarray) -> np.ndarray:
    cluster_matrix = cluster_result[:, None] == cluster_result[None, :]
    matrix += cluster_matrix.astype(np.float32)
    return matrix


def _cluster_preprocess(x_df: pd.DataFrame) -> np.ndarray:
    x = StandardScaler().fit_transform(x_df)
    x = sklearn.preprocessing.normalize(x, norm="l2")
    x = np.apply_along_axis(lambda arr: yeojohnson(arr)[0], 0, x)
    return x


def _make_partial_labels(df: pd.DataFrame, label_col: str, random_state: int = 0) -> pd.DataFrame:
    data = df.copy().reset_index(drop=True)
    y = data[label_col].astype(int).values.copy()

    rng = np.random.default_rng(random_state)
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        n_mask = int(len(idx) * 0.7)
        mask_idx = rng.choice(idx, size=n_mask, replace=False)
        y[mask_idx] = -1

    data[label_col] = y
    return data


def _self_training_predictions(base_classifier, data_df: pd.DataFrame, txid_col: str, label_col: str) -> pd.DataFrame:
    data_masked = _make_partial_labels(data_df, label_col)
    x = _cluster_preprocess(data_masked.drop(columns=[txid_col, label_col]))
    y = data_masked[label_col].values

    st = SelfTrainingClassifier(
        base_classifier,
        threshold=0.85,
        criterion="threshold",
        k_best=50,
        max_iter=1000,
    ).fit(x, y)

    y_pred = st.transduction_.copy()
    y_pred[y_pred == -1] = 1

    return pd.DataFrame({
        txid_col: data_df[txid_col].values,
        "predict": y_pred.astype(int),
    })


def _xgboost_predictions(data_df: pd.DataFrame, txid_col: str, label_col: str) -> pd.DataFrame:
    if xgb is None:
        raise ImportError("xgboost is required for the exact six semi-supervised teacher pipeline.")

    data = data_df.copy().reset_index(drop=True)
    y = data[label_col].astype(int).values

    known_mask = np.ones(len(data), dtype=bool)
    rng = np.random.default_rng(0)

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        n_drop = int(len(idx) * 0.7)
        drop_idx = rng.choice(idx, size=n_drop, replace=False)
        known_mask[drop_idx] = False

    X_train = data.loc[known_mask].drop(columns=[txid_col, label_col])
    y_train = data.loc[known_mask, label_col]
    X_total = data.drop(columns=[txid_col, label_col])
    y_total = data[label_col]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtotal = xgb.DMatrix(X_total, label=y_total)

    params = {
        "max_depth": 8,
        "eta": 0.3,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    bst = xgb.train(params, dtrain, num_boost_round=300)
    pred = (bst.predict(dtotal) > 0.5).astype(int)

    return pd.DataFrame({
        txid_col: data[txid_col].values,
        "predict": pred.astype(int),
    })


def train_six_semi_supervised_predictions(
    slice_df: pd.DataFrame,
    txid_col: str,
    label_col: str,
    cache_root: str | None = None,
    cache_tag: str | None = None,
    rebuild: bool = True,
) -> pd.DataFrame:
    """
    Exact teacher family from the official repo, but trained on the fly
    instead of loaded from semi_supervised_results/.
    """
    base_clfs = {
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "DecisionTree": DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_split=2),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gbBoost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "SVC": SVC(kernel="rbf", gamma=5, probability=True, class_weight={0: 0.8, 1: 0.2}, tol=1e-5),
    }

    cache_dir = Path(cache_root) if cache_root is not None else None
    outputs = {txid_col: slice_df[txid_col].values}

    if cache_dir is not None and cache_tag is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for name, clf in base_clfs.items():
        out_path = cache_dir / f"{name}_{cache_tag}.csv" if cache_dir is not None and cache_tag is not None else None

        if out_path is not None and out_path.exists() and not rebuild:
            pred_df = pd.read_csv(out_path)
        else:
            pred_df = _self_training_predictions(
                clone(clf),
                slice_df[[txid_col] + [c for c in slice_df.columns if c != txid_col]],
                txid_col,
                label_col,
            )
            if out_path is not None:
                pred_df.to_csv(out_path, index=False)

        outputs[name] = pred_df["predict"].values

    name = "xgboost"
    out_path = cache_dir / f"{name}_{cache_tag}.csv" if cache_dir is not None and cache_tag is not None else None

    if out_path is not None and out_path.exists() and not rebuild:
        pred_df = pd.read_csv(out_path)
    else:
        pred_df = _xgboost_predictions(slice_df, txid_col, label_col)
        if out_path is not None:
            pred_df.to_csv(out_path, index=False)

    outputs[name] = pred_df["predict"].values

    return pd.DataFrame(outputs)


def _compute_ca_feature(
    ca_matrix: np.ndarray,
    scaled_features_df: pd.DataFrame,
    semi_values: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    ca_feature = ca_matrix @ scaled_features_df.values
    ca_feature = (1 + 1e-8 - auto_minmax_scaler(pd.DataFrame(ca_feature), use_percentile=True)) * scaled_features_df.values
    ca_feature = scaled_features_df.values + ca_feature.values

    pca = PCA(n_components=1)
    ca_feature = pca.fit_transform(ca_feature)
    ca_feature = np.concatenate((ca_feature, semi_values * 0.9), axis=1)
    return torch.tensor(ca_feature, dtype=torch.float32, device=device)


def build_cosemignn_slice_artifacts(
    slice_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    spec: TransactionSliceConfig,
    semi_df: pd.DataFrame,
    device: torch.device,
):
    data_num = len(semi_df[spec.txid_col])
    ca_matrix = np.zeros((data_num, data_num), dtype=np.float32)

    for col in ["AdaBoost", "DecisionTree", "gbBoost", "xgboost", "RandomForest", "SVC"]:
        ca_matrix = co_association(ca_matrix, semi_df[col].to_numpy())

    ca_matrix /= 2.0

    fake_label = semi_df.iloc[:, 1:].values.sum(axis=1) / len(ca_matrix[0])
    semi_result = torch.tensor(semi_df.iloc[:, 1:].values, dtype=torch.float32, device=device)

    txids_df = pd.DataFrame(semi_df[spec.txid_col])
    edge_index, _ = create_slice_edge_index(txids_df, edge_df, spec.edge_src_col, spec.edge_dst_col)
    edge_index = edge_index.to(device)

    ca_weights = []
    for edge_i in range(edge_index.size(1)):
        node1 = edge_index[0, edge_i].item()
        node2 = edge_index[1, edge_i].item()
        ca_weights.append(ca_matrix[node1][node2])
    ca_weights = torch.tensor(ca_weights, dtype=torch.float32, device=device)

    feature_df = slice_df[spec.feature_cols].copy()
    feature_scaled = auto_minmax_scaler(feature_df, use_percentile=True)

    feature = np.concatenate((feature_scaled.values, semi_df.iloc[:, 1:].values * 0.9), axis=1)
    feature_tensor = torch.tensor(feature, dtype=torch.float32, device=device)

    ca_feature_tensor = _compute_ca_feature(
        ca_matrix=ca_matrix,
        scaled_features_df=feature_scaled,
        semi_values=semi_df.iloc[:, 1:].values,
        device=device,
    )

    labels = torch.tensor(slice_df[spec.label_col].astype(int).values, dtype=torch.long, device=device)
    fake_label_tensor = torch.tensor(fake_label, dtype=torch.float32, device=device)

    return {
        "feature": feature_tensor,
        "edge_index": edge_index,
        "labels": labels,
        "ca_matrix": ca_matrix,
        "ca_weights": ca_weights,
        "semi_result": semi_result,
        "fake_label": fake_label_tensor,
        "ca_feature": ca_feature_tensor,
    }