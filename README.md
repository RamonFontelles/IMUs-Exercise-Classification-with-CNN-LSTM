# IMU Exercise Classification (CNN‑LSTM)

End‑to‑end pipeline to classify gym exercises using wrist IMU data with a CNN+LSTM in PyTorch.  
Refactored from a Kaggle notebook into a clean, reproducible repo with a CLI and YAML config.

## Features
- Reusable `IMUDataset` with windowing & preprocessing (smoothing + downsampling)
- Train/test session‑level split with stratification and no leakage
- Standardization fitted **only on train**
- Class weighting for imbalance
- Simple CNN‑LSTM model
- CLI: `python scripts/train.py --config configs/default.yaml`
- Saves metrics, confusion matrix, and best checkpoint to `outputs/`

## Kaggle notebook

This repository is a clean refactor (package + CLI) of the original Kaggle notebook:

- **IMU Exercise Classification with CNN-LSTM** — by Ramón Fontelles  
  https://www.kaggle.com/code/ramonfontelles/imu-exercise-classification-with-cnn-lstm

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data (Kaggle)
By default we download **Gym Workout IMU Dataset** using `kagglehub`:
- Dataset: `shakthisairam123/gym-workout-imu-dataset`

> If your environment requires explicit Kaggle credentials, ensure you are logged in for KaggleHub or switch to the Kaggle CLI and place CSVs under `data/raw/`.

## Quickstart
```bash
# Train with defaults
python scripts/train.py --config configs/default.yaml

# Override on the fly
python scripts/train.py --config configs/default.yaml training.epochs=10 training.batch_size=64
```

## Project layout
```
src/imu_exercise_classifier/
    __init__.py
    datasets.py        # IMUDataset + preprocessing
    model.py           # CNN_LSTM
    split.py           # stratified session split (no leakage)
    pipeline.py        # end-to-end helpers to load, scale, build loaders
    train_loop.py      # training & evaluation utilities
scripts/
    train.py           # CLI entry point
configs/
    default.yaml
notebooks/
    exploration.ipynb  # (optional) EDA notebook
tests/
    test_smoke.py
outputs/               # metrics, checkpoints, plots
```

## Reproducibility
We set Python/NumPy/PyTorch seeds, log shapes, and save the scaler + label mapping to `outputs/` for inference.

## License
MIT
