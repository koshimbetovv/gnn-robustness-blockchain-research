import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_paper_utils import get_device, train_standard_model
from src.datasets.elliptic import EllipticConfig, EllipticDataset
from src.models.graphsage import GraphSAGE
from src.utils.seed import seed_from_config


CONFIG = {
    "seed": 42,
    "data": {
        "feature_path": "data/raw/elliptic/elliptic_txs_features.csv",
        "class_path": "data/raw/elliptic/elliptic_txs_classes.csv",
        "edge_path": "data/raw/elliptic/elliptic_txs_edgelist.csv",
        "train_start": 1,
        "train_end": 34,
        "test_start": 35,
        "test_end": 49,
        "filter_unknown": True,
    },
    "model": {
        "hidden_dim": 128,
        "aggr": "mean",
        "out_dim": 2,
    },
    "training": {
        "epochs": 500,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,
        "adam_eps": 1e-5,
        "log_every": 50,
        "use_class_weights": True,
    },
    "save": {
        "save_dir": "models",
        "save_run": True,
        "prefix": "graphsage",
        "filename": "graphsage.pt",
    },
}


def main():
    seed_from_config(CONFIG)
    device = get_device()

    data = EllipticDataset(EllipticConfig(**CONFIG["data"])).get_data().to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    model = GraphSAGE(
        in_dim=data.num_features,
        hid_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        aggr=CONFIG["model"]["aggr"],
    ).to(device)

    train_standard_model(
        model=model,
        data=data,
        config=CONFIG,
        model_name="graphsage",
        title="GraphSAGE",
    )


if __name__ == "__main__":
    main()
