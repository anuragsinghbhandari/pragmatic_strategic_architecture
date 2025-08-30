import json

def migrate_dataset(input_path, output_path):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            game = json.loads(line.strip())
            
            # For each message index, make a new row
            for i, msg in enumerate(game["messages"]):
                row = {
                    "game_id": game["game_id"],
                    "season": game["seasons"][i],
                    "year": game["years"][i],
                    "abs_msg_index": game["absolute_message_index"][i],
                    "rel_msg_index": game["relative_message_index"][i],
                    "speaker": game["speakers"][i],
                    "receiver": game["receivers"][i],
                    "message": msg,
                    
                    # Labels
                    "receiver_label": game["receiver_labels"][i],
                    "sender_label": game["sender_labels"][i],   # <-- main target
                    
                    # Scores
                    "score": {
                        game["speakers"][i]: int(game["game_score"][i]),
                        game["receivers"][i]: int(game["game_score"][i])
                    },
                    "score_delta": {
                        game["speakers"][i]: int(game["game_score_delta"][i]),
                        game["receivers"][i]: int(game["game_score_delta"][i])
                    },
                    
                    # Meta
                    "players": sorted(game["players"])
                }
                
                fout.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    migrate_dataset("given_data/validation.jsonl", "preprocess_data/processed_val.jsonl")
