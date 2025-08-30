import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from src.models.pragmatic_encoder import PragmaticFeatureExtractor
from src.models.strategic_encoder import StrategicFeatureExtractor
from src.models.social_encoder import SocialFeatureExtractor
from src.models.fusionmodel import FusionDeceptionModel
import os
class DiplomacyDataset(Dataset):
    def __init__(self, json_data):
        # Normalize input
        if isinstance(json_data, dict) and "message" in json_data:
            # Single sample dict
            self.samples = [json_data]
        elif isinstance(json_data, dict):
            # Dict keyed by id -> flatten values
            self.samples = [v for v in json_data.values() if isinstance(v, dict) and "message" in v]
        elif isinstance(json_data, list):
            # Already a list of samples
            self.samples = [s for s in json_data if isinstance(s, dict) and "message" in s]
        else:
            raise ValueError("json_data must be a list[dict] or dict[str, dict] with 'message'")
        
        self.pragmatic_encoder = PragmaticFeatureExtractor()
        self.strategic_encoder = StrategicFeatureExtractor()
        self.social_encoder = SocialFeatureExtractor()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]   # now always safe

        pragmatic_vec = self.pragmatic_encoder.extract_features(sample["message"])
        strategic_vec = self.strategic_encoder.extract_features(
            score=sample["score"], 
            score_delta=sample["score_delta"], 
            speaker=sample["speaker"], 
            receiver=sample["receiver"],
            season=sample['season'],
            year=sample['year']
            )
        social_vec = self.social_encoder.extract_features(
            speaker=sample['speaker'],
            receiver=sample['receiver']
        )

        features = torch.cat([
            torch.tensor(pragmatic_vec, dtype=torch.float),
            torch.tensor(strategic_vec, dtype=torch.float),
            torch.tensor(social_vec, dtype=torch.float)
        ])

        label = torch.tensor(0 if sample["sender_label"] else 1, dtype=torch.long)

        return features, label


def train_pipeline(json_data, epochs=5, batch_size=8, test_size=0.2, lr=1e-3):
    dataset = DiplomacyDataset(json_data)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size)

    input_dim = len(dataset[0][0])
    model = FusionDeceptionModel(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss {sum(train_losses)/len(train_losses):.4f} "
              f"| Val Acc {acc:.3f} | Val Prec {prec:.3f} | Val Rec {rec:.3f} | Val F1 {f1:.3f}")
    import os

    os.makedirs("trained_models", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim
    }, os.path.join("trained_models", "model.pt"))

    joblib.dump(dataset.pragmatic_encoder, "pragmatic_encoder.pkl")
    joblib.dump(dataset.strategic_encoder, "strategic_encoder.pkl")
    joblib.dump(dataset.social_encoder, "social_encoder.pkl")

    return model
