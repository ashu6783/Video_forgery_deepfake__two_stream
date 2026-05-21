import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import DeepfakeTwoStreamDataset
from model import TwoStreamDeepfakeNet


# ================= LOSS =================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()


# ================= METRICS =================

def compute_metrics(labels, probs, preds):
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    if len(set(labels)) < 2:
        metrics["auc"] = 0.0
    else:
        metrics["auc"] = roc_auc_score(labels, probs)
    return metrics


# ================= TRAIN / VAL =================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cpu",
    grad_clip_norm=1.0,
    accum_steps=1,
    scheduler_per_step=None,
):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    if train:
        optimizer.zero_grad()

    for step, (rgb, flow, labels) in enumerate(tqdm(loader, leave=False)):
        rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)

        if not train:
            with torch.no_grad():
                logits = model(rgb, flow)
                loss = criterion(logits, labels)
        else:
            logits = model(rgb, flow)
            loss = criterion(logits, labels)
            (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler_per_step is not None:
                    scheduler_per_step.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs, all_preds)
    metrics["loss"] = avg_loss

    return metrics


def make_balanced_sampler(labels):
    labels = np.asarray(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_warmup_cosine(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


# ================= MAIN =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-csv", type=str, default="data/splits/all_splits_processed.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--monitor-metric", type=str, default="auc", choices=["accuracy", "auc", "f1"])
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--temporal-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--full-finetune", action="store_true",
                        help="If set, all parameters are trainable from the start (default).")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Freeze backbones for first N epochs. 0 = full fine-tune from start.")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Use WeightedRandomSampler to balance classes per batch.")

    args = parser.parse_args()

    run_dir = Path("outputs/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = DeepfakeTwoStreamDataset(args.processed_csv, "train", augment=True)
    val_ds = DeepfakeTwoStreamDataset(args.processed_csv, "val")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty dataset split detected.")

    if train_ds.df["label_id"].nunique() < 2 or val_ds.df["label_id"].nunique() < 2:
        raise ValueError("Each split must contain both classes.")

    if args.balanced_sampler:
        sampler = make_balanced_sampler(train_ds.df["label_id"].values)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = "cpu"

    model = TwoStreamDeepfakeNet(
        pretrained=args.pretrained,
        dropout=args.dropout,
        temporal_hidden_dim=args.temporal_hidden_dim,
        fusion_dim=args.fusion_dim,
        max_frames=args.max_frames,
    ).to(device)

    if args.freeze_epochs > 0:
        model.freeze_backbones()

    counts = train_ds.df["label_id"].value_counts().sort_index()
    total = float(counts.sum())
    weights = torch.tensor(
        [total / (2 * max(counts.get(i, 1), 1)) for i in [0, 1]],
        dtype=torch.float32,
    ).to(device)

    if args.loss_type == "focal":
        criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, args.accum_steps))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    if args.scheduler == "cosine":
        per_step_scheduler = build_warmup_cosine(optimizer, warmup_steps, total_steps)
        plateau_scheduler = None
    else:
        per_step_scheduler = None
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train": [], "val": []}
    best_metric = -np.inf
    wait = 0

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.5,
                weight_decay=args.weight_decay,
            )
            if args.scheduler == "cosine":
                remaining_steps = steps_per_epoch * (args.epochs - epoch + 1)
                per_step_scheduler = build_warmup_cosine(optimizer, max(1, int(remaining_steps * 0.05)), remaining_steps)
            else:
                plateau_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip_norm,
            accum_steps=args.accum_steps,
            scheduler_per_step=per_step_scheduler,
        )
        val_metrics = run_epoch(model, val_loader, criterion, None, device)

        if plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics[args.monitor_metric])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("Train:", train_metrics)
        print("Val:", val_metrics)

        if val_metrics[args.monitor_metric] > best_metric:
            best_metric = val_metrics[args.monitor_metric]
            wait = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                break

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
