import os
import sys

from torch.optim import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_paper_utils import get_device
from scripts.train_recgnn_utils import evaluate_recgnn_sequence, train_recgnn_epoch
from src.datasets.recgnn_elliptic import RecGNNEllipticConfig, RecGNNEllipticDataset
from src.models.recgnn import RecGNN
from src.utils.logging import print_epoch_metrics, print_final_metrics, save_run
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
        "filter_unknown": False,
    },
    "model": {
        "hidden_dim": 50,
        "dropout": 0.5,
        "out_dim": 2,
    },
    "training": {
        "epochs": 50,
        "lr": 1.5e-3,
        "weight_decay": 0.0,
        "log_every": 10,
        "use_class_weights": False,
    },
    "save": {
        "save_dir": "models",
        "save_run": True,
        "prefix": "recgnn",
    },
}


def main():
    seed_from_config(CONFIG)
    device = get_device()

    dataset = RecGNNEllipticDataset(RecGNNEllipticConfig(**CONFIG["data"]))
    sequence = dataset.get_sequence()
    CONFIG["model"]["state_rows"] = int(sequence.max_nodes)

    model = RecGNN(
        in_dim=sequence.num_features,
        hidden_dim=CONFIG["model"]["hidden_dim"],
        out_dim=CONFIG["model"]["out_dim"],
        state_rows=CONFIG["model"]["state_rows"],
        dropout=CONFIG["model"]["dropout"],
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"].get("weight_decay", 0.0),
    )

    print("Class weights: None")
    print(
        f"\n=== Training RecGNN (paper-faithful sequence mode) ===\n"
        f"Train graphs: {len(sequence.train_graphs)} | Test graphs: {len(sequence.test_graphs)} | "
        f"state_rows={sequence.max_nodes} | input_dim={sequence.num_features}"
    )

    log_every = CONFIG["training"].get("log_every", 10)
    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        avg_loss = train_recgnn_epoch(model, sequence.train_graphs, optimizer, device)

        if epoch == 1 or epoch % log_every == 0 or epoch == CONFIG["training"]["epochs"]:
            train_metrics = evaluate_recgnn_sequence(model, sequence.train_graphs, device)
            test_metrics = evaluate_recgnn_sequence(model, sequence.test_graphs, device, prime_graphs=sequence.train_graphs)
            print_epoch_metrics(epoch, avg_loss, train_metrics, test_metrics, loss_label="AvgBatchLoss")

    test_metrics = evaluate_recgnn_sequence(model, sequence.test_graphs, device, prime_graphs=sequence.train_graphs)
    print_final_metrics("RecGNN", test_metrics)
    save_run(model, test_metrics, CONFIG, "recgnn")


if __name__ == "__main__":
    main()
