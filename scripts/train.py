import argparse, yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from imu_exercise_classifier.datasets import IMU_FEATURES, IMUDataset, preprocess_sample
from imu_exercise_classifier.model import CNN_LSTM
from imu_exercise_classifier.pipeline import (
    set_seed, load_dataset_with_kagglehub, load_all_csvs, smooth_and_clean,
    label_encode, fit_scaler, apply_scaler, compute_class_weights
)
from imu_exercise_classifier.split import stratified_session_split
from imu_exercise_classifier.train_loop import (
    resolve_device, train_epoch, evaluate, save_confusion_matrix, save_report, save_metrics
)

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("overrides", nargs="*", help="Override as key=value, e.g., training.epochs=10")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # simple dot-notation overrides
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        ks = k.split(".")
        d = cfg
        for key in ks[:-1]:
            d = d[key]
        # try convert
        if v.isdigit():
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        d[ks[-1]] = v

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = resolve_device(cfg["training"]["device"])

    # ---------- Load data ----------
    if cfg["data"]["use_kagglehub"]:
        base = load_dataset_with_kagglehub(cfg["data"]["kaggle_dataset"])
    else:
        base = "data/raw"

    data = load_all_csvs(base)
    data = smooth_and_clean(data, cfg["data"]["smooth_kernel"])
    data, activity_to_id, id_to_activity = label_encode(data)

    # ---------- Split at session level ----------
    session_labels = [int(df["activityEncoded"].iloc[0]) for df in data]
    train_idx, test_idx = stratified_session_split(data, session_labels, train_ratio=cfg["data"]["train_split"], seed=42)
    train_dfs = [data[i] for i in train_idx]
    test_dfs  = [data[i] for i in test_idx]

    # ---------- Scale ----------
    scaler = fit_scaler(train_dfs)
    apply_scaler(train_dfs, scaler)
    apply_scaler(test_dfs, scaler)

    # ---------- Datasets & Loaders ----------
    train_ds = IMUDataset(
        dataframes=train_dfs,
        features=IMU_FEATURES,
        window_size=cfg["data"]["window_size"],
        step_size=cfg["data"]["step_size"],
        preprocess_fn=preprocess_sample,
        preprocess_kwargs=dict(
            smooth_kernel=cfg["data"]["smooth_kernel"],
            downsample_factor=cfg["data"]["downsample_factor"],
            downsample_mode=cfg["data"]["downsample_mode"]
        )
    )
    test_ds = IMUDataset(
        dataframes=test_dfs,
        features=IMU_FEATURES,
        window_size=cfg["data"]["window_size"],
        step_size=cfg["data"]["step_size"],
        preprocess_fn=preprocess_sample,
        preprocess_kwargs=dict(
            smooth_kernel=cfg["data"]["smooth_kernel"],
            downsample_factor=cfg["data"]["downsample_factor"],
            downsample_mode=cfg["data"]["downsample_mode"]
        )
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=cfg["training"]["num_workers"])

    # ---------- Model ----------
    num_features = len(IMU_FEATURES)
    num_classes = len(activity_to_id)
    model = CNN_LSTM(num_features=num_features, num_classes=num_classes,
                     hidden_dim=cfg["model"]["hidden_dim"], lstm_layers=cfg["model"]["lstm_layers"]).to(device)

    # ---------- Loss & Optim ----------
    class_weights, df_weights = compute_class_weights(train_ds, id_to_activity, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # ---------- Train loop ----------
    best_acc = -1.0
    history = []
    for epoch in range(cfg["training"]["epochs"]):
        train_metrics = train_epoch(model, train_loader, optim, criterion, device, grad_clip=cfg["training"]["grad_clip"])
        y_true, y_pred = evaluate(model, test_loader, device)

        labels_present = sorted(set(y_true) | set(y_pred))
        target_names = [id_to_activity[i] for i in labels_present]

        # Save epoch artifacts
        save_confusion_matrix(y_true, y_pred, labels_present, out_dir / f"cm_epoch{epoch}.png")
        save_report(y_true, y_pred, target_names, out_dir / f"report_epoch{epoch}.txt")

        test_acc = 100.0 * sum(np.array(y_true) == np.array(y_pred)) / max(1, len(y_true))
        epoch_rec = dict(epoch=epoch, train_loss=train_metrics["loss"], train_acc=train_metrics["acc"], test_acc=test_acc)
        history.append(epoch_rec)
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.2f}% test_acc={test_acc:.2f}%")

        # checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model": model.state_dict(), "config": cfg, "activity_to_id": activity_to_id}, out_dir / "checkpoint_best.pt")

    # Save final metadata
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    df_weights.to_csv(out_dir / "class_weights.csv", index=False)

    print("Done. Artifacts in:", out_dir)

if __name__ == "__main__":
    main()
