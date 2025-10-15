from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def resolve_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def train_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 1.0):
    model.train()
    correct = 0
    total = 0
    running = 0.0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = running / max(1, len(loader))
    acc = 100 * correct / max(1, total)
    return {"loss": avg_loss, "acc": acc}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        out = model(X)
        pred = out.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return y_true, y_pred

def save_confusion_matrix(y_true, y_pred, labels, out_png: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def save_report(y_true, y_pred, target_names, out_txt: Path):
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    out_txt.write_text(report, encoding="utf-8")

def save_metrics(metrics: Dict[str, Any], out_json: Path):
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
