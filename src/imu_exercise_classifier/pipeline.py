from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .datasets import IMU_FEATURES, filter_data, smooth_columns
from .split import stratified_session_split

try:
    import kagglehub
except Exception:
    kagglehub = None

def set_seed(seed: int = 42):
    import torch, random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_dataset_with_kagglehub(dataset_slug: str) -> str:
    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed. Install it or provide local CSVs under data/raw/.")
    path = kagglehub.dataset_download(dataset_slug)
    return path

def load_all_csvs(base_path: str) -> list[pd.DataFrame]:
    data = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".csv"):
                df = pd.read_csv(os.path.join(root, f))
                df = filter_data(df, 1.5)
                data.append(df)
    if not data:
        raise FileNotFoundError(f"No CSVs found under: {base_path}")
    return data

def smooth_and_clean(dataframes: list[pd.DataFrame], smooth_kernel: int = 5) -> list[pd.DataFrame]:
    cleaned = []
    for df in dataframes:
        df = smooth_columns(df, smooth_kernel)
        if df[IMU_FEATURES].isnull().values.any():
            df[IMU_FEATURES] = df[IMU_FEATURES].fillna(0)
        df[IMU_FEATURES] = df[IMU_FEATURES].replace([np.inf, -np.inf], 0)
        cleaned.append(df)
    return cleaned

def label_encode(dataframes: list[pd.DataFrame]):
    all_activities = pd.concat([df["activity"] for df in dataframes], ignore_index=True)
    categories = all_activities.astype("category").cat.categories
    activity_to_id = {cat: idx for idx, cat in enumerate(categories)}
    for i, df in enumerate(dataframes):
        df["activityEncoded"] = df["activity"].map(activity_to_id)
        dataframes[i] = df
    id_to_activity = {v: k for k, v in activity_to_id.items()}
    return dataframes, activity_to_id, id_to_activity

def fit_scaler(train_dfs: list[pd.DataFrame]):
    scaler = StandardScaler()
    stack = pd.concat([df.loc[:, IMU_FEATURES] for df in train_dfs], ignore_index=True)
    scaler.fit(stack)
    return scaler

def apply_scaler(dfs: list[pd.DataFrame], scaler):
    for df in dfs:
        df.loc[:, IMU_FEATURES] = scaler.transform(df.loc[:, IMU_FEATURES])

def compute_class_weights(train_dataset, id_to_activity, device):
    import torch, numpy as np, pandas as pd
    y_train = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    classes = np.array(sorted(id_to_activity.keys()))
    class_weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_counts = np.bincount(y_train, minlength=classes.max() + 1)
    table = pd.DataFrame({
        "Class ID": classes,
        "Activity": [id_to_activity[i] for i in classes],
        "Count": class_counts[classes],
        "Weight": class_weights_np
    }).sort_values("Weight", ascending=False, ignore_index=True)
    return torch.tensor(class_weights_np, dtype=torch.float32, device=device), table
