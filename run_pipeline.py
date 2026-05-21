import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("Running:", cmd)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--seed", type=int,
     default=42)
    parser.add_argument("--fake-per-group", type=int, default=400)
    parser.add_argument("--target-real", type=int, default=1600)

    # Preprocess
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)

    # Training
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)

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
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--balanced-sampler", action="store_true")

    # Control
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--processed-csv", type=str, default=None)

    args = parser.parse_args()

    python = sys.executable
    root = Path(__file__).parent

    # ================= PREPARE =================
    if not args.skip_prepare:
        run([
            python,
            str(root / "src" / "prepare_data.py"),
            "--out-dir", str(root / "data"),
            "--seed", str(args.seed),
            "--fake-per-group", str(args.fake_per_group),
            "--target-real", str(args.target_real),
        ])

    # ================= PREPROCESS =================
    if not args.skip_preprocess:
        run([
            python,
            str(root / "src" / "preprocess.py"),
            "--split-csv", str(root / "data" / "splits" / "all_splits.csv"),
            "--out-dir", str(root / "data" / "processed"),
            "--num-frames", str(args.num_frames),
            "--image-size", str(args.image_size),
        ])

    processed_csv = args.processed_csv or str(root / "data" / "splits" / "all_splits_processed.csv")
    print("Using processed CSV:", processed_csv)

    # ================= TRAIN =================
    train_cmd = [
        python,
        str(root / "src" / "train.py"),
        "--processed-csv",
        processed_csv,
        "--batch-size",
        str(args.batch_size),
        "--accum-steps",
        str(args.accum_steps),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--monitor-metric",
        args.monitor_metric,
        "--num-workers",
        str(args.num_workers),
        "--loss-type",
        args.loss_type,
        "--focal-gamma",
        str(args.focal_gamma),
        "--label-smoothing",
        str(args.label_smoothing),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--dropout",
        str(args.dropout),
        "--temporal-hidden-dim",
        str(args.temporal_hidden_dim),
        "--fusion-dim",
        str(args.fusion_dim),
        "--max-frames",
        str(args.max_frames),
        "--scheduler",
        args.scheduler,
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--freeze-epochs",
        str(args.freeze_epochs),
    ]
    if args.pretrained:
        train_cmd.append("--pretrained")
    if args.balanced_sampler:
        train_cmd.append("--balanced-sampler")

    run(train_cmd)

    print("\n✅ Pipeline completed successfully!")
    print("👉 Now run evaluation using:")
    print("python src/evaluate.py --model-path outputs/runs/.../best_model.pt --tune-threshold")


if __name__ == "__main__":
    main()
