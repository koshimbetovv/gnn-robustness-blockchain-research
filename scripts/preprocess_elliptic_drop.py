import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def main():
    features = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
    edges = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")

    features = features.sort_values(by=0).reset_index(drop=True)

    # Robust label mapping: illicit(class-1)=1, licit(class-2)=0, unknown -> filtered out
    classes["class"] = classes["class"].astype(str).str.strip()
    y_map = {"1": 1, "2": 0}
    classes["y"] = classes["class"].map(y_map)

    # Filter unknown labels BEFORE graph construction to match the paper
    known_classes = classes.dropna(subset=["y"]).copy()
    known_classes["y"] = known_classes["y"].astype(int)
    known_txids = set(known_classes["txId"].values)

    features = features[features.iloc[:, 0].isin(known_txids)].sort_values(by=0).reset_index(drop=True)

    tx_ids = features.iloc[:, 0].values
    t = features.iloc[:, 1].astype(int).values
    x = features.iloc[:, 2:].values.astype("float32")

    # Two-way repo-style chronological split
    k = 34
    train_mask_np = (t >= 1) & (t <= k)
    test_mask_np = (t >= (k + 1)) & (t <= 49)
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    # Fit scaler on TRAIN ONLY, then transform all retained labeled nodes
    scaler = StandardScaler()
    scaler.fit(x[train_mask_np])
    x = scaler.transform(x)

    y_dict = dict(zip(known_classes["txId"].values, known_classes["y"].values))
    y = [y_dict[tx] for tx in tx_ids]

    # Map txId -> node index after filtering known-label nodes
    id2idx = {tx: i for i, tx in enumerate(tx_ids)}

    edge_pairs = []
    for u, v in edges[["txId1", "txId2"]].itertuples(index=False):
        if u in id2idx and v in id2idx:
            edge_pairs.append((id2idx[u], id2idx[v]))

    if not edge_pairs:
        raise ValueError("No edges remain after filtering unknown-label nodes.")

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    torch.save(
        {
            "x": torch.tensor(x, dtype=torch.float),
            "y": torch.tensor(y, dtype=torch.long),
            "edge_index": edge_index,
            "time_step": torch.tensor(t, dtype=torch.long),
            "train_mask": train_mask,
            "test_mask": test_mask,
        },
        "data/processed/elliptic/data.pt",
    )


if __name__ == "__main__":
    main()
