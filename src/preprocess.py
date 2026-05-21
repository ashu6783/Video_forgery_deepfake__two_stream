import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def sample_frame_indices(total_frames: int, num_samples: int):
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()


def load_selected_frames(video_path: str, num_frames: int):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = set(sample_frame_indices(total_frames, num_frames))
    frames = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in indices:
            frames.append(frame)
        frame_id += 1

    cap.release()
    return frames


def get_face_detector():
    try:
        from retinaface import RetinaFace
    except ImportError as exc:
        raise ImportError(
            "RetinaFace is required. Install it with `pip install retina-face`."
        ) from exc
    return RetinaFace


def compute_flow(prev_bgr, curr_bgr):
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow.astype(np.float32)


def normalize_flow_2ch(flow: np.ndarray, clip_value: float = 20.0):
    # Keep raw 2D optical flow (dx, dy), clipped and scaled to [-1, 1].
    flow = np.clip(flow, -clip_value, clip_value) / clip_value
    return flow.astype(np.float16)


def process_one_video(video_path: str, num_frames: int, image_size: int, face_detector):
    frames = load_selected_frames(video_path, num_frames=num_frames)

    if len(frames) < 2:
        return None

    first_frame = frames[0]

    # 🔥 Faster detection (resize first)
    small = cv2.resize(first_frame, (320, 180))
    faces = face_detector.detect_faces(small)

    if isinstance(faces, dict) and len(faces) > 0:
        face_items = sorted(faces.values(), key=lambda f: f.get("score", 0.0), reverse=True)
        x1, y1, x2, y2 = face_items[0]["facial_area"]

        # scale back
        scale_x = first_frame.shape[1] / 320
        scale_y = first_frame.shape[0] / 180

        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        # ✅ padding + clipping (FIX)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        pad = int(0.3 * max(w, h))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(first_frame.shape[1], x2 + pad)
        y2 = min(first_frame.shape[0], y2 + pad)

    else:
        x1, y1, x2, y2 = 0, 0, first_frame.shape[1], first_frame.shape[0]

    aligned = []
    for f in frames:
        crop = f[y1:y2, x1:x2]
        if crop.size == 0:
            crop = f
        crop = cv2.resize(crop, (image_size, image_size))
        aligned.append(crop)

    rgb = np.stack([cv2.cvtColor(a, cv2.COLOR_BGR2RGB) for a in aligned]).astype(np.uint8)

    flow_list = [np.zeros((image_size, image_size, 2), dtype=np.float16)]
    for i in range(1, len(aligned)):
        prev = aligned[i - 1]
        curr = aligned[i]
        flow = compute_flow(prev, curr)
        flow_list.append(normalize_flow_2ch(flow))

    flow = np.stack(flow_list).astype(np.float16)

    return rgb, flow


def stratified_resplit(df: pd.DataFrame, seed: int = 42):
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
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def load_hash_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r") as f:
        return json.load(f)


def save_hash_manifest(manifest_path: Path, manifest: dict):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _manifest_owner(entry):
    if isinstance(entry, str):
        # Backward compatibility with old manifest format.
        return entry
    if isinstance(entry, dict):
        return entry.get("video_path")
    return None


def _build_hashes(video_path: str, preprocess_sig: str):
    # New hash is config-aware. Legacy hash is path-only.
    config_hash = hashlib.md5(f"{video_path}|{preprocess_sig}".encode()).hexdigest()
    legacy_hash = hashlib.md5(video_path.encode()).hexdigest()
    return config_hash, legacy_hash


def resolve_hashed_npz_path(video_path: str, out_dir: Path, hash_manifest: dict, preprocess_sig: str):
    config_hash, legacy_hash = _build_hashes(video_path, preprocess_sig)

    # 1) Reuse any existing cache from either new or legacy naming.
    for full_hash in (config_hash, legacy_hash):
        for hash_len in (12, 16, 20, 24, 32):
            key = full_hash[:hash_len]
            owner = _manifest_owner(hash_manifest.get(key))
            npz_path = out_dir / f"{key}.npz"
            if npz_path.exists() and (owner is None or owner == video_path):
                hash_manifest[key] = {"video_path": video_path, "preprocess_sig": preprocess_sig}
                return npz_path

    # 2) Allocate a fresh key from config-aware hash only.
    for hash_len in (12, 16, 20, 24, 32):
        key = config_hash[:hash_len]
        owner = _manifest_owner(hash_manifest.get(key))
        npz_path = out_dir / f"{key}.npz"

        if owner is None:
            hash_manifest[key] = {"video_path": video_path, "preprocess_sig": preprocess_sig}
            return npz_path

        if owner == video_path:
            hash_manifest[key] = {"video_path": video_path, "preprocess_sig": preprocess_sig}
            return npz_path

    raise RuntimeError("Hash collision overflow")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "hash_manifest.json"
    manifest = load_hash_manifest(manifest_path)
    preprocess_sig = f"nf{args.num_frames}_is{args.image_size}_flow2ch_v1"

    df = pd.read_csv(args.split_csv)
    face_detector = get_face_detector()

    npz_paths = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["video_path"]

        if not Path(video_path).exists():
            npz_paths.append("")
            failed += 1
            continue

        npz_path = resolve_hashed_npz_path(video_path, out_dir, manifest, preprocess_sig)

        if npz_path.exists():
            npz_paths.append(str(npz_path))
            continue

        result = process_one_video(video_path, args.num_frames, args.image_size, face_detector)

        if result is None:
            npz_paths.append("")
            failed += 1
            continue

        rgb, flow = result

        np.savez_compressed(npz_path, rgb=rgb, flow=flow, label=int(row["label_id"]))
        npz_paths.append(str(npz_path))

    save_hash_manifest(manifest_path, manifest)

    df["npz_path"] = npz_paths
    clean_df = df[df["npz_path"] != ""].copy()

    clean_df = stratified_resplit(clean_df)
    clean_df.to_csv(Path(args.split_csv).with_name("all_splits_processed.csv"), index=False)

    print(f"Done. Kept {len(clean_df)}/{len(df)} samples. Failed: {failed}")
    print(clean_df.groupby(["split", "label_id"]).size())


if __name__ == "__main__":
    main()