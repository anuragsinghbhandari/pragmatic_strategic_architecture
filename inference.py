# pipeline/inference.py
import torch
import joblib
from src.models.fusionmodel import FusionDeceptionModel
from src.models.pragmatic_encoder import PragmaticFeatureExtractor
from src.models.strategic_encoder import StrategicFeatureExtractor
from src.models.social_encoder import SocialFeatureExtractor


class DeceptionPredictor:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        model_path : directory where model + encoders are saved
        """
        self.device = device

        # Load trained model checkpoint
        checkpoint = torch.load(f"{model_path}/model.pt", map_location=device)
        input_dim = checkpoint["input_dim"]

        # Rebuild model architecture
        self.model = FusionDeceptionModel(input_dim=input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        # Load encoders / preprocessors if any
        self.pragmatic_encoder = joblib.load(f"{model_path}/pragmatic_encoder.pkl")
        self.strategic_encoder = joblib.load(f"{model_path}/strategic_encoder.pkl")
        self.social_encoder = joblib.load(f"{model_path}/social_encoder.pkl")

    def predict(self, message_text: str, game_state: dict, sender: str, receiver: str):
        # 1. Extract features
        pragmatic_vec = torch.tensor(
            self.pragmatic_encoder.extract_features(message_text), dtype=torch.float
        ).unsqueeze(0)

        strategic_vec = torch.tensor(
            self.strategic_encoder.extract_features(
                season=game_state["season"],
                year=game_state["year"],
                score=game_state["score"],
                score_delta=game_state["score_delta"],
                speaker=sender,
                receiver=receiver
            ),
            dtype=torch.float
        ).unsqueeze(0)

        social_vec = torch.tensor(
            self.social_encoder.extract_features(sender, receiver), dtype=torch.float
        ).unsqueeze(0)

        # 2. Run model
        with torch.no_grad():
    # Concatenate feature vectors
            x = torch.cat([pragmatic_vec, strategic_vec, social_vec], dim=-1)

            # Forward through model
            logits = self.model(x)
            prob = torch.softmax(logits, dim=-1)[:, 1].item()  # probability of "deceptive"


        return {
            "deception_probability": prob,
            "label": int(prob > 0.5)   # 1=deceptive, 0=truthful
        }

