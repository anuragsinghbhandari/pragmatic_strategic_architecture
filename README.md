# Diplomacy Deception Detection

This repository contains code and models for detecting deception in the game Diplomacy using deep learning and feature engineering.

## Features

- Custom PyTorch models for pragmatic, strategic, and social feature extraction
- Fusion model for deception classification
- Data pipeline for preprocessing Diplomacy game logs
- Training and evaluation scripts with metrics and visualizations
- Jupyter notebook for interactive analysis and plotting

## Project Structure

```
gps_dd/
    train.py           # Main training script
    gps_dd.py          # Model definitions
    src/
        models/        # Encoders and fusion model
        utils/         # Dataset and pipeline utilities
    data/              # Game data in JSONL format
    moves/             # Game moves per phase
prepare/
    ...                # Data preparation scripts
```

## Getting Started

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare data**
   - Place your game logs in `data/` and moves in `moves/`.

3. **Train the model**
   ```
   python gps_dd/train.py
   ```

4. **Evaluate and visualize**
   - Use the provided Jupyter notebook (`diplomacy_evaluation.ipynb`) for analysis and plotting.

## Usage

- Modify `train.py` to adjust hyperparameters, batch size, or data paths.
- Use `src/utils/pipeline.py` for feature engineering and advanced training workflows.

## Results

- Training and validation metrics are printed during training.

## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature requests.

## License

See `LICENSE` for details.
