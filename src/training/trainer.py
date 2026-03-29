import torch
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, data, device, use_class_weights=True):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.use_class_weights = use_class_weights
        
        # Compute class weights if needed
        if use_class_weights:
            self.class_weights = self._compute_class_weights()
        else:
            self.class_weights = None
    
    def _compute_class_weights(self):
        """Compute class weights to handle imbalanced dataset."""
        # Filter out unlabeled data (label = -1)
        train_mask = getattr(self.data, "train_mask", (self.data.y != -1))
        y_train = self.data.y[train_mask & (self.data.y != -1)]

        if y_train.numel() == 0:
            raise ValueError("train_mask has 0 labeled nodes (y!=-1).")

        num_classes = 2  # 0 (legitimate) and 1 (fraudulent)
        
        # Count samples per class
        class_counts = torch.bincount(y_train, minlength=num_classes).float()
        
        # Guard against missing class in TRAIN split
        if (class_counts == 0).any():
            print(f"⚠ train split missing a class: counts={class_counts.tolist()}, disabling class weights.")
            return None

        # Compute weights: weight_c = total_samples / (num_classes * count_c)
        num_samples = float(y_train.numel())
        class_weights = num_samples / (num_classes * class_counts)
        
        print(f"Train class distribution - Legit: {int(class_counts[0])}, Fraud: {int(class_counts[1])}")
        print(f"Train class weights - Legit: {class_weights[0]:.4f}, Fraud: {class_weights[1]:.4f}")
        
        return class_weights.to(self.device)

    def train(self, epochs=200, lr=0.01, weight_decay=0.0):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in tqdm(range(epochs), desc="Training:"):
            self.model.train()
            opt.zero_grad()

            train_mask = getattr(self.data, "train_mask", (self.data.y != -1))
            mask = train_mask & (self.data.y != -1)

            if mask.sum().item() == 0:
                raise ValueError("No labeled nodes found (all y == -1). Check dataset loading / label mapping.")

            out = self.model(self.data.x, self.data.edge_index)
            loss = F.cross_entropy(out[mask], self.data.y[mask], weight=self.class_weights)


            loss.backward()
            
            # Gradient clipping to prevent numerical instability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            opt.step()
            
            # Debug: print loss every 50 epochs
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    pred = out[self.data.y != -1].argmax(dim=1)
                    accuracy = (pred == self.data.y[self.data.y != -1]).float().mean()
                    fraud_pred = (pred == 1).sum().item()
                tqdm.write(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={accuracy:.4f}, Fraud Predictions={fraud_pred}")
