import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DeepfakeTwoStreamDataset
from model import TwoStreamDeepfakeNet


def infer_split(model, loader, device="cpu"):
    y_true, y_prob = [], []
    with torch.no_grad():
        for rgb, flow, labels in tqdm(loader, desc="Evaluating", leave=False):
            rgb, flow = rgb.to(device), flow.to(device)
            logits = model(rgb, flow)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_prob)


def tune_threshold(y_true, y_prob):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (y_prob >= thr).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-csv", type=str, default="data/splits/all_splits_processed.csv")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true")
    args = parser.parse_args()

    out_dir = Path("outputs/runs") / datetime.now().strftime("%Y%m%d_%H%M%S_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_ds = DeepfakeTwoStreamDataset(args.processed_csv, split="val")
    test_ds = DeepfakeTwoStreamDataset(args.processed_csv, split="test")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = "cpu"
    model = TwoStreamDeepfakeNet(pretrained=args.pretrained).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    threshold = args.threshold
    threshold_source = "fixed"
    val_tuned_f1 = None
    if args.tune_threshold:
        y_val_true, y_val_prob = infer_split(model, val_loader, device=device)
        threshold, val_tuned_f1 = tune_threshold(y_val_true, y_val_prob)
        threshold_source = "val_f1_tuned"

    y_true, y_prob = infer_split(model, test_loader, device=device)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)
    if len(set(y_true)) < 2:
        auc = 0.0
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    else:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "auc": auc,
                "accuracy": accuracy_score(y_true, y_pred),
                "threshold": threshold,
                "threshold_source": threshold_source,
                "val_tuned_f1": val_tuned_f1,
                "classification_report": report,
            },
            f,
            indent=2,
        )

    print(
        f"Evaluation complete. AUC={auc:.4f}, threshold={threshold:.3f}, "
        f"accuracy={accuracy_score(y_true, y_pred):.4f}. Artifacts: {out_dir}"
    )


if __name__ == "__main__":
    main()