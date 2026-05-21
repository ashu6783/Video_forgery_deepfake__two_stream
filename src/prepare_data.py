import argparse
import random
from collections import defaultdict
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def download_dataset() -> Path:
    path = kagglehub.dataset_download("xdxd003/ff-c23")
    return Path(path)


def list_videos(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]


def infer_label_and_group(video_path: Path):
    parts = [p.lower() for p in video_path.parts]
    # FaceForensics++ real videos are often stored under "original" paths.
    if any(("real" in p) or ("original" in p) for p in parts):
        return "real", "real"
    # Pick nearest parent folder for fake type signal.
    fake_group = video_path.parent.name
    # If folder name is generic, try one level above.
    if fake_group.lower() in {"videos", "video", "c23", "ff-c23"} and video_path.parent.parent != video_path.parent:
        fake_group = video_path.parent.parent.name
    return "fake", fake_group


def audit_dataset(root: Path):
    videos = list_videos(root)
    rows = []
    for vp in videos:
        label, fake_group = infer_label_and_group(vp)
        rows.append(
            {
                "video_path": str(vp),
                "label": label,
                "fake_group": fake_group,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No videos found under: {root}")
    return df


def sample_balanced(df: pd.DataFrame, fake_per_group: int, target_real: int, seed: int):
    rng = random.Random(seed)

    fake_df = df[df["label"] == "fake"].copy()
    real_df = df[df["label"] == "real"].copy()

    fake_by_group = defaultdict(list)
    for idx, row in fake_df.iterrows():
        fake_by_group[row["fake_group"]].append(idx)

    # Prioritize top 5 fake groups by available size.
    group_sizes = sorted(fake_by_group.items(), key=lambda x: len(x[1]), reverse=True)
    selected_groups = [g for g, _ in group_sizes[:5]]

    selected_fake_idx = []
    for group in selected_groups:
        candidates = fake_by_group[group]
        rng.shuffle(candidates)
        # Safe sampling in case a group has fewer videos than requested.
        take = min(fake_per_group, len(candidates))
        selected_fake_idx.extend(candidates[:take])

    sampled_fake = fake_df.loc[selected_fake_idx].reset_index(drop=True)
    fake_count = len(sampled_fake)

    real_indices = list(real_df.index)
    rng.shuffle(real_indices)
    # Do not force real count to match fake count.
    real_take = min(target_real, len(real_indices))
    sampled_real = real_df.loc[real_indices[:real_take]].reset_index(drop=True)

    sampled = pd.concat([sampled_fake, sampled_real], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled["label_id"] = (sampled["label"] == "fake").astype(int)
    print("Final dataset size:", len(sampled))
    print("Fake samples:", len(sampled_fake))
    print("Real samples:", len(sampled_real))
    return sampled, selected_groups


def stratified_split(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label_id"],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fake-per-group", type=int, default=400)
    parser.add_argument("--target-real", type=int, default=1600)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    split_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = download_dataset()
    print(f"Dataset root: {dataset_root}")

    audit_df = audit_dataset(dataset_root)
    audit_df.to_csv(split_dir / "audit_all_videos.csv", index=False)

    sampled_df, fake_groups = sample_balanced(
        audit_df,
        fake_per_group=args.fake_per_group,
        target_real=args.target_real,
        seed=args.seed,
    )

    train_df, val_df, test_df = stratified_split(sampled_df, seed=args.seed)
    full_split = pd.concat([train_df, val_df, test_df], ignore_index=True)

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    full_split.to_csv(split_dir / "all_splits.csv", index=False)

    print("Selected fake groups:", fake_groups)
    print("Sampled counts:")
    print(sampled_df["label"].value_counts())
    print("Split counts:")
    print(full_split.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()
