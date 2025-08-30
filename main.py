from inference import DeceptionPredictor
import json

# Load predictor from stored model directory
predictor = DeceptionPredictor(model_path="stored_models")

# Load test.jsonl
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

test_data = load_jsonl("preprocess_data/processed_test.jsonl")   # adjust path if needed

# Run inference on all samples
results = []
for sample in test_data:
    msg = sample["message"]
    sender = sample["speaker"]
    receiver = sample["receiver"]
    game_state = {
        "score": sample["score"],
        "score_delta": sample["score_delta"],
        "season": sample["season"],
        "year": sample["year"]
    }
    
    pred = predictor.predict(msg, game_state, sender, receiver)
    results.append({**sample, **pred})

# Save predictions to file
with open("predictions.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("✅ Inference complete. Predictions written to predictions.jsonl")
