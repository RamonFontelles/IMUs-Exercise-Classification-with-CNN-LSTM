from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMU_FEATURES = [
    "wristMotion_rotationRateX", "wristMotion_rotationRateY", "wristMotion_rotationRateZ",
    "wristMotion_gravityX", "wristMotion_gravityY", "wristMotion_gravityZ",
    "wristMotion_accelerationX", "wristMotion_accelerationY", "wristMotion_accelerationZ",
    "wristMotion_quaternionW", "wristMotion_quaternionX", "wristMotion_quaternionY", "wristMotion_quaternionZ",
]

def filter_data(df: pd.DataFrame, trim_sec: float = 1.5) -> pd.DataFrame:
    min_time = df["secondsElapsed"].min() + trim_sec
    max_time = df["secondsElapsed"].max() - trim_sec
    return df[(df["secondsElapsed"] >= min_time) & (df["secondsElapsed"] <= max_time)].reset_index(drop=True)

def smooth_columns(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window is None or window <= 1: 
        return df
    df_smoothed = df.copy()
    cols_to_smooth = [c for c in df.columns if c in IMU_FEATURES]
    for col in cols_to_smooth:
        df_smoothed[col] = df[col].rolling(window=window, center=True).mean()
    return df_smoothed

def preprocess_sample(window: np.ndarray, y: int, smooth_kernel=5, downsample_factor=2, downsample_mode="avg"):
    X = window.astype(np.float32)
    if smooth_kernel and smooth_kernel > 1:
        k = int(smooth_kernel)
        if k % 2 == 0: k += 1
        kernel = np.ones(k, dtype=np.float32) / k
        X = np.vstack([np.convolve(X[:, f], kernel, mode="same") for f in range(X.shape[1])]).T.astype(np.float32)
    if downsample_factor and downsample_factor > 1:
        d = int(downsample_factor)
        if downsample_mode == "avg":
            T, F = X.shape
            pad_needed = (-T) % d
            if pad_needed:
                X = np.concatenate([X, np.repeat(X[-1:, :], pad_needed, axis=0)], axis=0)
            X = X.reshape(-1, d, F).mean(axis=1)
        else:
            X = X[::d]
    return X, y

class IMUDataset(Dataset):
    def __init__(self, dataframes, features=IMU_FEATURES, window_size=300, step_size=150,
                 preprocess_fn=preprocess_sample, preprocess_kwargs=None):
        self.samples = []
        self.preprocess_fn = preprocess_fn
        self.preprocess_kwargs = preprocess_kwargs or {}
        for df in dataframes:
            X = df[features].values
            y = int(df["activityEncoded"].iloc[0])
            for start in range(0, len(X) - window_size + 1, step_size):
                end = start + window_size
                window = X[start:end]
                if self.preprocess_fn is not None:
                    window, y_out = self.preprocess_fn(window, y, **self.preprocess_kwargs)
                    y_use = y_out
                else:
                    y_use = y
                self.samples.append((window, y_use))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
