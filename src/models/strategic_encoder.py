# encoders/strategic_encoder.py
import numpy as np
from typing import Dict

class StrategicFeatureExtractor:
    """
    Extracts strategic game-state features from metadata like scores, deltas, season, year.
    """

    def __init__(self):
        # Map categorical season to fixed index
        self.season_map = {"Spring": 0, "Fall": 1, "Winter": 2}

    def extract_features(self, season: str, year: int, 
                         score: Dict[str, int], score_delta: Dict[str, int],
                         speaker: str, receiver: str) -> np.ndarray:
        """
        Extract numeric vector representing strategic state.
        Focus only on speaker & receiver relative stats for now.
        """
        season_idx = self.season_map.get(season, -1)

        speaker_score = score.get(speaker, 0)
        receiver_score = score.get(receiver, 0)
        speaker_delta = score_delta.get(speaker, 0)
        receiver_delta = score_delta.get(receiver, 0)

        # Features: season, year, speaker_score, receiver_score, difference, deltas
        features = [
            season_idx,
            year,
            speaker_score,
            receiver_score,
            speaker_score - receiver_score,
            speaker_delta,
            receiver_delta,
        ]
        return np.array(features, dtype=np.float32)

    def feature_dim(self) -> int:
        return 7
