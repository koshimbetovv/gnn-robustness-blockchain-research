import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_evolvegcn_utils import (
    DualAdam,
    WeightedCrossEntropy,
    build_evolvegcn_model,
    evaluate_evolvegcn,
    move_sample,
    resolve_class_weights,
)
from scripts.train_paper_utils import get_device
from src.datasets.evolvegcn_elliptic import EvolveGCNEllipticConfig, EvolveGCNEllipticDataset
from src.utils.logging import print_eval_epoch_metrics, print_final_metrics, save_run
from src.utils.seed import seed_from_config


BASE_CONFIGS = {
    "h": {
        "seed": 42,
        "data": {
            "feature_path": "data/raw/elliptic/elliptic_txs_features.csv",
            "class_path": "data/raw/elliptic/elliptic_txs_classes.csv",
            "edge_path": "data/raw/elliptic/elliptic_txs_edgelist.csv",
            "num_hist_steps": 5,
            "adj_mat_time_window": 1,
            "train_start": 1,
            "train_end": 34,
            "test_start": 35,
            "test_end": 49,
            "filter_unknown": False,
        },
        "model": {
            "variant": "h",
            "layer_1_feats": 64,
            "layer_2_feats": 32,
            "cls_feats": 64,
            "skipfeats": False,
        },
        "training": {
            "epochs": 100,
            "lr": 1e-3,
            "class_weights": "auto",
            "log_every": 20,
            "steps_accum_gradients": 1,
        },
        "save": {
            "save_dir": "models",
            "save_run": True,
            "prefix": "evolvegcn_h",
        },
    },
    "o": {
        "seed": 42,
        "data": {
            "feature_path": "data/raw/elliptic/elliptic_txs_features.csv",
            "class_path": "data/raw/elliptic/elliptic_txs_classes.csv",
            "edge_path": "data/raw/elliptic/elliptic_txs_edgelist.csv",
            "num_hist_steps": 5,
            "adj_mat_time_window": 1,
            "train_start": 1,
            "train_end": 34,
            "test_start": 35,
            "test_end": 49,
        },
        "model": {
            "variant": "o",
            "layer_1_feats": 64,
            "layer_2_feats": 32,
            "cls_feats": 64,
            "skipfeats": False,
        },
        "training": {
            "epochs": 100,
            "lr": 1e-3,
            "class_weights": "auto",
            "log_every": 20,
            "steps_accum_gradients": 1,
        },
        "save": {
            "save_dir": "models",
            "save_run": True,
            "prefix": "evolvegcn_o",
        },
    },
}

# Change this to BASE_CONFIGS["o"] to train the exact IBM-repo EvolveGCN-O variant.
CONFIG = copy.deepcopy(BASE_CONFIGS["h"])


def main():
    seed_from_config(CONFIG)
    device = get_device(allow_mps=False)

    dataset = EvolveGCNEllipticDataset(EvolveGCNEllipticConfig(**CONFIG["data"]))
    sequence = dataset.get_sequence()

    model = build_evolvegcn_model(sequence.num_features, CONFIG).to(device)
    class_weights = resolve_class_weights(sequence.train_samples, CONFIG)
    loss_fn = WeightedCrossEntropy(class_weights, device=device)
    optim = DualAdam(model, lr=CONFIG["training"]["lr"])

    print(f"Class weights: {class_weights}")
    print(
        f"\n=== Training exact EvolveGCN-{CONFIG['model']['variant'].upper()} ===\n"
        f"Train windows: {len(sequence.train_samples)} | Test windows: {len(sequence.test_samples)} | "
        f"num_nodes={sequence.num_nodes} | input_dim={sequence.num_features}"
    )

    log_every = CONFIG["training"].get("log_every", 20)
    grad_acc = max(int(CONFIG["training"].get("steps_accum_gradients", 1)), 1)

    for epoch in range(1, CONFIG["training"]["epochs"] + 1):
        model.train()
        optim.zero_grad(set_to_none=True)

        for step, sample in enumerate(sequence.train_samples, start=1):
            hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx, label_vals = move_sample(sample, device)
            logits = model(hist_adj_list, hist_ndFeats_list, node_mask_list, label_idx)
            loss = loss_fn(logits, label_vals) / grad_acc
            loss.backward()

            if step % grad_acc == 0 or step == len(sequence.train_samples):
                optim.step()
                optim.zero_grad(set_to_none=True)

        if epoch == 1 or epoch % log_every == 0 or epoch == CONFIG["training"]["epochs"]:
            train_metrics = evaluate_evolvegcn(model, sequence.train_samples, loss_fn, device)
            test_metrics = evaluate_evolvegcn(model, sequence.test_samples, loss_fn, device)
            print_eval_epoch_metrics(epoch, train_metrics, test_metrics)

    test_metrics = evaluate_evolvegcn(model, sequence.test_samples, loss_fn, device)
    print_final_metrics(f"EvolveGCN-{CONFIG['model']['variant'].upper()}", test_metrics)
    save_run(model, test_metrics, CONFIG, f"evolvegcn_{CONFIG['model']['variant'].lower()}")


if __name__ == "__main__":
    main()
