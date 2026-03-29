import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    features = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None)
    classes = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
    edges = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")

    features = features.sort_values(by=0)
    
    # timestep is 2nd column (index 1) in elliptic_txs_features.csv
    t = features.iloc[:, 1].astype(int).values

    # Two-way split:
    # train: 1-k, test: (k+1)-49
    k = 34
    train_mask = torch.tensor((t >= 1) & (t <= k), dtype=torch.bool)
    test_mask  = torch.tensor((t >= (k + 1)) & (t <= 49), dtype=torch.bool)

    x = features.iloc[:, 2:].values.astype("float32")

    # scaler must fit on TRAIN ONLY
    train_mask_np = (t >= 1) & (t <= k)

    scaler = StandardScaler()
    scaler.fit(x[train_mask_np])   # fit ONLY on train

    x = scaler.transform(x)        # transform ALL (train/val/test) using train stats


    # --- robust label mapping ---
    classes["class"] = classes["class"].astype(str).str.strip()
    y_map = {"1": 1, "2": 0}
    classes["y"] = classes["class"].map(y_map).fillna(-1).astype(int)

    # Align labels to features order via txId
    features = features.sort_values(by=0)
    tx_ids = features.iloc[:, 0].values

    y_dict = dict(zip(classes["txId"].values, classes["y"].values))
    y = [y_dict.get(t, -1) for t in tx_ids]


    # Map txId -> node index
    id2idx = {tx: i for i, tx in enumerate(tx_ids)}

    edge_pairs = edges[["txId1", "txId2"]].values
    edge_pairs = [(id2idx[u], id2idx[v]) for u, v in edge_pairs if u in id2idx and v in id2idx]
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
    "data/processed/elliptic/data.pt")


if __name__ == "__main__":
    main()
