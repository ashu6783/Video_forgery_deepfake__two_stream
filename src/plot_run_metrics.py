import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepfakeTwoStreamDataset
from model import TwoStreamDeepfakeNet


def get_latest_run_dir(runs_root: Path) -> Path:
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "history.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No run directories with history.json found in {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_history(history_path: Path):
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_epoch_curves(history, out_dir: Path):
    train = history["train"]
    val = history["val"]
    epochs = np.arange(1, min(len(train), len(val)) + 1)
    keys = ["loss", "accuracy", "precision", "recall", "f1", "auc"]

    for key in keys:
        plt.figure(figsize=(8, 5))
        train_vals = [m[key] for m in train[: len(epochs)]]
        val_vals = [m[key] for m in val[: len(epochs)]]
        plt.plot(epochs, train_vals, label="train", linewidth=2)
        plt.plot(epochs, val_vals, label="val", linewidth=2)
        plt.title(f"{key.upper()} vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(key.upper())
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"metrics_over_epochs_{key}.png", dpi=140)
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [m["f1"] for m in train[: len(epochs)]], label="Train F1", linewidth=2)
    plt.plot(epochs, [m["f1"] for m in val[: len(epochs)]], label="Val F1", linewidth=2)
    plt.plot(epochs, [m["auc"] for m in train[: len(epochs)]], label="Train AUC", linewidth=2)
    plt.plot(epochs, [m["auc"] for m in val[: len(epochs)]], label="Val AUC", linewidth=2)
    plt.title("F1 and AUC Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_auc_curves.png", dpi=140)
    plt.close()


def infer_split(model, loader, device="cpu"):
    y_true, y_prob = [], []
    with torch.no_grad():
        for rgb, flow, labels in tqdm(loader, desc="Infer", leave=False, disable=True):
            rgb, flow = rgb.to(device), flow.to(device)
            logits = model(rgb, flow)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred


def plot_confusion(cm: np.ndarray, title: str, output_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default="outputs/runs")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run dir; defaults to most recent")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = Path(args.run_dir) if args.run_dir else get_latest_run_dir(runs_root)
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"
    model_path = run_dir / "best_model.pt"

    if not history_path.exists():
        raise FileNotFoundError(f"Missing history file: {history_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    history = load_history(history_path)
    plot_epoch_curves(history, run_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    processed_csv = cfg.get("processed_csv", "data/splits/all_splits_processed.csv")
    val_ds = DeepfakeTwoStreamDataset(processed_csv, split="val")
    train_ds = DeepfakeTwoStreamDataset(processed_csv, split="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu"
    model = TwoStreamDeepfakeNet(
        pretrained=cfg.get("pretrained", False),
        dropout=cfg.get("dropout", 0.3),
        temporal_hidden_dim=cfg.get("temporal_hidden_dim", 256),
        fusion_dim=cfg.get("fusion_dim", 512),
        max_frames=cfg.get("max_frames", 32),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    split_outputs = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        y_true, y_prob, y_pred = infer_split(model, loader, device=device)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion(cm, f"{split_name.upper()} Confusion Matrix", run_dir / f"confusion_matrix_{split_name}.png")

        if len(np.unique(y_true)) < 2:
            auc = 0.0
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
        else:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)

        split_outputs[split_name] = {
            "auc": auc,
            "confusion_matrix": cm.tolist(),
            "support": int(len(y_true)),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    plt.figure(figsize=(7, 6))
    for split_name in ["train", "val"]:
        plt.plot(
            split_outputs[split_name]["fpr"],
            split_outputs[split_name]["tpr"],
            label=f"{split_name.upper()} AUC={split_outputs[split_name]['auc']:.4f}",
            linewidth=2,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Train vs Val ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "roc_curve_train_val.png", dpi=140)
    plt.close()

    with open(run_dir / "split_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "train": {
                    "auc": split_outputs["train"]["auc"],
                    "confusion_matrix": split_outputs["train"]["confusion_matrix"],
                    "support": split_outputs["train"]["support"],
                },
                "val": {
                    "auc": split_outputs["val"]["auc"],
                    "confusion_matrix": split_outputs["val"]["confusion_matrix"],
                    "support": split_outputs["val"]["support"],
                },
            },
            f,
            indent=2,
        )

    print(f"Saved plots and metrics to: {run_dir}")


if __name__ == "__main__":
    main()
