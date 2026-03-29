import torch
from torch_geometric.data import Data

class EllipticDataset:
    def __init__(self, root="data/processed/elliptic"):
        obj = torch.load(f"{root}/data.pt")

        self.data = Data(
            x=obj["x"],
            edge_index=obj["edge_index"],
            y=obj["y"],
        )

        # attach masks if present
        for k in ["train_mask", "test_mask", "time_step"]:
            if k in obj:
                setattr(self.data, k, obj[k])

    def get_data(self):
        return self.data
