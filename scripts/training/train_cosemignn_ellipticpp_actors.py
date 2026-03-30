import os
import sys
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from sklearn import metrics
import numpy as np
from statistics import mean, pstdev

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.cosemignn import CoSemiGNN
from src.models.cmos import create_cmos_model
from src.utils.seed import seed_from_config
from src.datasets.cosemignn_ellipticpp_addraddr import load_cosemignn_ellipticpp_addraddr


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "models", "train_cosemignn_ellipticpp_actors.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def eva(y_true, y_pred, prefix=""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = metrics.accuracy_score(y_true, y_pred)
    precision_per_class = metrics.precision_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    recall_per_class = metrics.recall_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    f1_negative = metrics.f1_score(y_true, y_pred, average="binary", pos_label=0, zero_division=0)
    f1_positive = metrics.f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)

    print(prefix)
    print(f"Total transactions: {len(y_true)}")
    print(f"Accuracy: {acc:.4f}")
    print(
        "Precision: True negative/Predicted negative: {:.4f} True positive/Predicted positive: {:.4f}".format(
            precision_per_class[0], precision_per_class[1]
        )
    )
    print(
        f"Recall negative: {recall_per_class[0]:.4f} | Recall positive: {recall_per_class[1]:.4f} | "
        f"Micro F1: {f1_micro:.4f} | Macro F1: {f1_macro:.4f} | "
        f"Negative F1: {f1_negative:.4f} | Positive F1: {f1_positive:.4f}"
    )
    print("---------------------------------------------")

    return {
        "n_samples": int(len(y_true)),
        "acc": float(acc),
        "f1_negative": float(f1_negative),
        "f1_positive": float(f1_positive),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "precision_negative": float(precision_per_class[0]),
        "precision_positive": float(precision_per_class[1]),
        "recall_negative": float(recall_per_class[0]),
        "recall_positive": float(recall_per_class[1]),
    }


def train_cmos_exact(model, data, train_time_list, learning_rate=1e-4, epochs=200):
    adj_list = data[1]
    label_list = data[2]
    ca_weights_list = data[4]
    ca_feature_list = data[7]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        loss_total = 0.0

        for time in train_time_list:
            optimizer.zero_grad()

            outputs = model(ca_feature_list[time], adj_list[time], ca_weights_list[time])
            labels = label_list[time].float()

            pos_counts = labels.sum(dim=0)
            neg_counts = (1 - labels).sum(dim=0)
            epsilon = 1e-6
            pos_weight = (neg_counts / (pos_counts + epsilon)).to(outputs.device) * 10

            loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            loss_total += float(loss.item())

        print(f"CMOS Epoch [{epoch + 1}/{epochs}] Loss: {loss_total:.8f}")

    return model


def train_cosemignn_exact(model, cmos_model, data, time_list, learning_rate=1e-4, epochs=500, alpha=0.6):
    feature_list = data[0]
    adj_list = data[1]
    label_list = data[2]
    ca_weights_list = data[4]
    ca_feature_list = data[7]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    cmos_model.train()
    #cmos_model.eval()
    #for p in cmos_model.parameters():
    #    p.requires_grad = False

    total_loss_list = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for time in time_list:
            out, _ = model(feature_list[time], adj_list[time], ca_weights_list[time])
            #with torch.no_grad():
            #    cmos_out = cmos_model(ca_feature_list[time], adj_list[time], ca_weights_list[time])
            cmos_out = cmos_model(ca_feature_list[time], adj_list[time], ca_weights_list[time])

            inp = torch.sigmoid(out)
            tgt = torch.sigmoid(cmos_out)

            input_probs = torch.stack([torch.log(1 - inp + 1e-12), torch.log(inp + 1e-12)], dim=1)
            target_probs = torch.stack([1 - tgt, tgt], dim=1)

            kl_loss = F.kl_div(input=input_probs, target=target_probs, reduction="batchmean")

            labels = label_list[time].float()
            pos_counts = labels.sum(dim=0)
            neg_counts = (1 - labels).sum(dim=0)
            epsilon = 1e-6
            pos_weight = (neg_counts / (pos_counts + epsilon)).to(out.device) / 2

            ce_loss = F.binary_cross_entropy_with_logits(out, labels, pos_weight=pos_weight)

            total_loss = ce_loss + (1 - alpha) * kl_loss

            epoch_loss += float(total_loss.item())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        total_loss_list.append(epoch_loss)
        print(f"CoSemiGNN Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}")

    return model, total_loss_list


@torch.no_grad()
def eval_cosemignn_exact(model, data, time_list):
    feature_list = data[0]
    adj_list = data[1]
    label_list = data[2]
    ca_weights_list = data[4]

    model.eval()
    metrics_list = []
    y_true_all = []
    y_pred_all = []

    for time in time_list:
        print(f"time: {time}:")
        out, _ = model(feature_list[time], adj_list[time], ca_weights_list[time])
        predicted = (torch.sigmoid(out) > 0.5).long()

        y_true = label_list[time].detach().cpu().numpy()
        y_pred = predicted.detach().cpu().numpy()

        m = eva(y_true, y_pred, prefix=f"time{time}")
        m["time"] = int(time)
        metrics_list.append(m)

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true_concat = np.concatenate(y_true_all, axis=0)
    y_pred_concat = np.concatenate(y_pred_all, axis=0)

    return metrics_list, y_true_concat, y_pred_concat


def summarize_mean_over_time(metrics_list):
    metric_keys = [
        "acc",
        "f1_negative",
        "f1_positive",
        "f1_macro",
        "f1_micro",
        "precision_negative",
        "precision_positive",
        "recall_negative",
        "recall_positive",
    ]

    summary = {
        "aggregation": "mean_over_test_time_steps",
        "n_time_steps": int(len(metrics_list)),
        "time_steps": [int(m["time"]) for m in metrics_list],
    }

    for key in metric_keys:
        values = [float(m[key]) for m in metrics_list]
        summary[key] = float(mean(values))
        summary[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0

    return summary


def summarize_concat_all_predictions(y_true_concat, y_pred_concat):
    summary = eva(y_true_concat, y_pred_concat, prefix="all_test_times_concatenated")
    summary["aggregation"] = "concatenate_all_test_time_steps"
    return summary


def main():
    seed_from_config(CONFIG)
    device = get_device()

    k = CONFIG["time"]["train_end"]
    train_time_list = [i for i in range(1, k)]
    predict_time_list = [i for i in range(CONFIG["time"]["predict_start"], CONFIG["time"]["predict_end"])]

    data = load_cosemignn_ellipticpp_addraddr(
        feature_path=CONFIG["data"]["feature_path"],
        class_path=CONFIG["data"]["class_path"],
        edge_path=CONFIG["data"]["edge_path"],
        semi_cache_dir=CONFIG["data"]["semi_cache_dir"],
        device=device,
        rebuild_semi=CONFIG["data"]["rebuild_semi"],
    )

    max_slice_nodes = max(x.size(0) for x in data[7][1:] if x is not None and x.numel() > 0)

    cmos_model = create_cmos_model(
        input_size=data[7][1].size(1),
        state_rows=max_slice_nodes,
        device=device,
    )

    if CONFIG["cmos"]["load_pretrain"] and os.path.exists(CONFIG["cmos"]["pretrained_path"]):
        cmos_model.load_state_dict(torch.load(CONFIG["cmos"]["pretrained_path"], map_location=device))
    else:
        cmos_model = train_cmos_exact(
            model=cmos_model,
            data=data,
            train_time_list=train_time_list,
            learning_rate=CONFIG["cmos"]["learning_rate"],
            epochs=CONFIG["cmos"]["epochs"],
        )

    model = CoSemiGNN(
        feature_in=data[0][1].size(1),
        dim=CONFIG["cosemignn"]["dim"],
        dim2=CONFIG["cosemignn"]["dim2"],
        dim3=CONFIG["cosemignn"]["dim3"],
        num_heads=CONFIG["cosemignn"]["num_heads"],
    ).to(device)

    if CONFIG["cosemignn"]["load_pretrain"] and os.path.exists(CONFIG["cosemignn"]["pretrained_path"]):
        model.load_state_dict(torch.load(CONFIG["cosemignn"]["pretrained_path"], map_location=device))
    else:
        model, _ = train_cosemignn_exact(
            model=model,
            cmos_model=cmos_model,
            data=data,
            time_list=train_time_list,
            learning_rate=CONFIG["cosemignn"]["learning_rate"],
            epochs=CONFIG["cosemignn"]["epochs"],
            alpha=CONFIG["cosemignn"]["alpha"],
        )

    metrics_list, y_true_concat, y_pred_concat = eval_cosemignn_exact(
    model=model,
    data=data,
    time_list=predict_time_list,
    )

    mean_over_time_metrics = summarize_mean_over_time(metrics_list)
    concat_metrics = summarize_concat_all_predictions(y_true_concat, y_pred_concat)


    if CONFIG["save"]["save_run"]:
        os.makedirs(CONFIG["save"]["save_dir"], exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(CONFIG["save"]["save_dir"], f"cosemignn_exact_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        torch.save(cmos_model.state_dict(), os.path.join(out_dir, "cmos_model.pt"))
        torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

        metrics_payload = {
            "per_time": metrics_list,
            "summary_mean_over_time": mean_over_time_metrics,
            "summary_concat_all_test_predictions": concat_metrics,
        }

        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics_payload, f, indent=2)

        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)

        print(f"✓ Saved run to {out_dir}")


if __name__ == "__main__":
    main()