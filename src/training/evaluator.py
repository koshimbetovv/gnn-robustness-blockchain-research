import torch
from sklearn.metrics import accuracy_score

def evaluate(model, data, split="test"):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

        split_mask = getattr(data, f"{split}_mask", (data.y != -1))
        mask = split_mask & (data.y != -1)

        pred = logits[mask].argmax(dim=1)
        return accuracy_score(data.y[mask].cpu(), pred.cpu())
