import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.training.train_paper_utils import get_device, train_standard_model
from src.datasets.elliptic import EllipticConfig, EllipticDataset
from src.models.gcn import GCN
from src.utils.seed import seed_from_config


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "models", "train_gcn.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def main():
    seed_from_config(CONFIG)
    device = get_device()

    data = EllipticDataset(EllipticConfig(**CONFIG["data"])).get_data().to(device)
    data.x = data.x.float()
    data.edge_index = data.edge_index.long()

    model = GCN(
        in_dim=data.num_features,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        num_layers=CONFIG["model"]["num_layers"],
        dropout=CONFIG["model"]["dropout"],
        use_norm=CONFIG["model"]["use_norm"],
    ).to(device)

    train_standard_model(
        model=model,
        data=data,
        config=CONFIG,
        model_name="gcn",
        title="GCN",
    )


if __name__ == "__main__":
    main()
