from src.utils.pipeline import train_pipeline
import json

def load_jsonl(path):
    """
    Load newline-delimited JSON file into a list of dicts.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

if __name__ == "__main__":
    json_data = load_jsonl("preprocess_data/processed_train.jsonl")   # your file path here
    model = train_pipeline(json_data, epochs=5, batch_size=8)
