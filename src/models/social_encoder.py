# encoders/social_encoder.py
import numpy as np
from typing import Dict, List, Tuple

class SocialFeatureExtractor:
    """
    Encodes social/interaction graph features between speaker and receiver.
    """

    def __init__(self):
        # Dictionary of (speaker, receiver) → stats
        self.interactions = {}

    def update(self, speaker: str, receiver: str, 
               sender_label: bool, receiver_label: bool):
        """
        Update interaction graph after each message.
        """
        key = (speaker, receiver)
        if key not in self.interactions:
            self.interactions[key] = {
                "messages": 0,
                "agreements": 0,
                "disagreements": 0,
            }
        self.interactions[key]["messages"] += 1
        if receiver_label is True:
            self.interactions[key]["agreements"] += 1
        elif receiver_label is False:
            self.interactions[key]["disagreements"] += 1

    def extract_features(self, speaker: str, receiver: str) -> np.ndarray:
        """
        Extract features for a given speaker-receiver pair.
        """
        key = (speaker, receiver)
        stats = self.interactions.get(key, {"messages": 0, "agreements": 0, "disagreements": 0})

        messages = stats["messages"]
        agreements = stats["agreements"]
        disagreements = stats["disagreements"]

        agreement_rate = agreements / messages if messages > 0 else 0.0
        disagreement_rate = disagreements / messages if messages > 0 else 0.0

        features = [
            messages,
            agreements,
            disagreements,
            agreement_rate,
            disagreement_rate,
        ]
        return np.array(features, dtype=np.float32)

    def feature_dim(self) -> int:
        return 5
