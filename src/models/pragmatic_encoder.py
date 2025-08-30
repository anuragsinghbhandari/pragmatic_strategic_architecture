# encoders/pragmatic_encoder.py
import re
import numpy as np
from typing import List

class PragmaticFeatureExtractor:
    """
    Lightweight pragmatic encoder using predefined lexical cues.
    Produces a fixed-length numeric vector for each message.
    """

    def __init__(self):
        # Define pragmatic cue lexicons (expand later if needed)
        self.hedges = {
    "maybe", "perhaps", "possibly", "probably", "i think", "i guess",
    "i suppose", "kind of", "sort of", "around", "roughly",
    "it seems", "it looks like", "could be", "might be", "likely"
}
        self.promises = {
    "i will", "i promise", "trust me", "you can count on",
    "i swear", "i assure you", "rest assured", "i guarantee",
    "believe me", "cross my heart", "mark my words", "i give you my word"
}
        self.negations = {
    "not", "never", "no", "nothing", "n't",
    "cannot", "can't", "won't", "doesn't", "don't",
    "nowhere", "nobody", "none", "without"
}
        self.politeness = {
    "please", "thanks", "thank you", "appreciate", "grateful",
    "sorry", "excuse me", "pardon", "would you mind",
    "much obliged", "kindly", "cheers", "with respect"
}
        self.suspicion = {
    "lie", "lying", "deceive", "betray", "fake", "cheat",
    "fraud", "dishonest", "trick", "scam", "con", "hoax",
    "backstab", "pretend", "untrustworthy", "two-faced", "snake"
}

        # The order matters for feature vector output
        self.feature_names = [
            "hedge_count",
            "promise_count",
            "negation_count",
            "politeness_count",
            "suspicion_count",
            "message_length",
        ]

    def extract_features(self, message: str) -> np.ndarray:
        """
        Extract pragmatic features for a single message.
        """
        msg = message.lower()

        features = []
        features.append(sum(1 for h in self.hedges if h in msg))
        features.append(sum(1 for p in self.promises if p in msg))
        features.append(sum(1 for n in self.negations if n in msg))
        features.append(sum(1 for pol in self.politeness if pol in msg))
        features.append(sum(1 for s in self.suspicion if s in msg))
        features.append(len(msg.split()))  # message length (proxy for verbosity)

        return np.array(features, dtype=np.float32)

    def batch_extract(self, messages: List[str]) -> np.ndarray:
        """
        Process a list of messages into a feature matrix.
        """
        return np.stack([self.extract_features(m) for m in messages])

    def feature_dim(self) -> int:
        """
        Return dimensionality of feature vector.
        """
        return len(self.feature_names)

if __name__=="__main__":
    encoder = PragmaticFeatureExtractor()
    msg = "I think maybe we should ally. Trust me, I promise it will work!"
    print(encoder.extract_features(msg))
    # Output (example): [2 hedges, 2 promise, 0 negations, 0 politeness, 0 suspicion, 12 length]
