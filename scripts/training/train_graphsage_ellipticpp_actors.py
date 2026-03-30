import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.training.train_paper_utils import get_device, train_standard_model
from src.datasets.ellipticpp_actors import EllipticPPActorsConfig, EllipticPPActorsDataset
from src.utils.seed import seed_from_config
from src.models.graphsage import GraphSAGE


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "models", "train_graphsage_ellipticpp_actors.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def main():
    seed_from_config(CONFIG)
    device = get_device()

    dataset = EllipticPPActorsDataset(EllipticPPActorsConfig(**CONFIG["data"]))
    data = dataset.get_data().to(device)
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
        model_name="graphsage_ellipticpp_actors",
        title="GraphSAGE (Elliptic++ Actors)",
    )


if __name__ == "__main__":
    main()
