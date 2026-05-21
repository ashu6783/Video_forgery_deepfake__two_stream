from datetime import datetime
from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch

from model import TwoStreamDeepfakeNet
from preprocess import compute_flow, get_face_detector, load_selected_frames, process_one_video


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@st.cache_resource(show_spinner=False)
def load_face_detector():
    return get_face_detector()


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, pretrained: bool, max_frames: int):
    model = TwoStreamDeepfakeNet(pretrained=pretrained, max_frames=max_frames).to("cpu")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_tensors(video_path: str, num_frames: int, image_size: int, detector):
    result = process_one_video(video_path, num_frames=num_frames, image_size=image_size, face_detector=detector)
    if result is None:
        return None

    rgb_np, flow_np = result
    rgb = torch.from_numpy(rgb_np).float() / 255.0
    flow = torch.from_numpy(flow_np).float()

    rgb = rgb.permute(0, 3, 1, 2).unsqueeze(0)
    flow = flow.permute(0, 3, 1, 2).unsqueeze(0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb, flow


def run_inference(model, rgb, flow, threshold: float):
    with torch.no_grad():
        logits = model(rgb, flow)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        fake_prob = float(probs[1].item())
        pred = 1 if fake_prob >= threshold else 0
    return pred, fake_prob


def _flow_to_rgb(flow_xy: np.ndarray):
    fx = flow_xy[..., 0]
    fy = flow_xy[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((flow_xy.shape[0], flow_xy.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle / 2) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_intermediate_artifacts(video_path: str, num_frames: int, image_size: int, detector, output_root: str):
    out_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S_ui_infer")
    sampled_dir = out_dir / "sampled_frames"
    crops_dir = out_dir / "face_crops"
    flow_dir = out_dir / "optical_flow_preview"
    sampled_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)

    frames = load_selected_frames(video_path, num_frames=num_frames)
    if len(frames) < 2:
        return None

    first_frame = frames[0]
    small = cv2.resize(first_frame, (320, 180))
    faces = detector.detect_faces(small)
    if isinstance(faces, dict) and len(faces) > 0:
        face_items = sorted(faces.values(), key=lambda f: f.get("score", 0.0), reverse=True)
        x1, y1, x2, y2 = face_items[0]["facial_area"]
        scale_x = first_frame.shape[1] / 320
        scale_y = first_frame.shape[0] / 180
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
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
    for i, frame in enumerate(frames):
        boxed = frame.copy()
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(sampled_dir / f"frame_{i:03d}.jpg"), boxed)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        crop_resized = cv2.resize(crop, (image_size, image_size))
        aligned.append(crop_resized)
        cv2.imwrite(str(crops_dir / f"crop_{i:03d}.jpg"), crop_resized)

    zero_flow = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    cv2.imwrite(str(flow_dir / "flow_000.jpg"), zero_flow)
    for i in range(1, len(aligned)):
        flow = compute_flow(aligned[i - 1], aligned[i])
        flow_vis = _flow_to_rgb(flow)
        cv2.imwrite(str(flow_dir / f"flow_{i:03d}.jpg"), flow_vis)

    return out_dir


def main():
    st.set_page_config(page_title="Deepfake Detector", page_icon="🎭", layout="centered")
    st.title("🎭 Deepfake Video Detector")
    st.caption("Upload a video to classify it as real or fake using your trained two-stream model.")

    default_model_path = "outputs/runs/20260428_171917/best_model.pt"

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path (.pt)", value=default_model_path)
        threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        num_frames = st.number_input("Frames sampled per video", min_value=2, max_value=64, value=8, step=1)
        image_size = st.number_input("Face crop size", min_value=96, max_value=512, value=224, step=16)
        max_frames = st.number_input("Model max_frames", min_value=2, max_value=128, value=32, step=1)
        pretrained = st.checkbox("Model uses pretrained EfficientNet weights", value=False)
        save_intermediates = st.checkbox("Save intermediate images", value=True)
        artifacts_root = st.text_input("Intermediate output root", value="outputs/ui_artifacts")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
    if uploaded is not None:
        st.video(uploaded)

    if st.button("Predict", type="primary", use_container_width=True):
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            st.error(f"Model not found: {model_path_obj}")
            return
        if uploaded is None:
            st.warning("Please upload a video first.")
            return

        with st.spinner("Loading detector and model..."):
            try:
                detector = load_face_detector()
            except ImportError as exc:
                st.error(str(exc))
                st.info("Install dependency: pip install retina-face")
                return

            try:
                model = load_model(str(model_path_obj), pretrained=pretrained, max_frames=int(max_frames))
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                return

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temp_video_path = tmp.name

        with st.spinner("Processing video and running inference..."):
            try:
                tensors = prepare_tensors(
                    temp_video_path,
                    num_frames=int(num_frames),
                    image_size=int(image_size),
                    detector=detector,
                )
            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")
                return

            if tensors is None:
                st.error("Could not extract enough frames from the video.")
                return

            rgb, flow = tensors
            try:
                pred, fake_prob = run_inference(model, rgb, flow, threshold=threshold)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

            artifacts_dir = None
            if save_intermediates:
                try:
                    artifacts_dir = save_intermediate_artifacts(
                        temp_video_path,
                        num_frames=int(num_frames),
                        image_size=int(image_size),
                        detector=detector,
                        output_root=artifacts_root,
                    )
                except Exception as exc:
                    st.warning(f"Could not save intermediate images: {exc}")

        label = "FAKE" if pred == 1 else "REAL"
        confidence = fake_prob if pred == 1 else (1.0 - fake_prob)

        if pred == 1:
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(fake_prob, text=f"Fake probability: {fake_prob:.4f}")
        st.caption(f"Threshold used: {threshold:.2f}")
        if save_intermediates and artifacts_dir is not None:
            st.success(f"Intermediate images saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
