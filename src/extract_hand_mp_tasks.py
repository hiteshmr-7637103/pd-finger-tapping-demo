"""
Hand landmark extraction using the MediaPipe Tasks API (HandLandmarker).

The original pipeline (src/extract_pose.py) was built against the legacy
`mediapipe.solutions.hands` API. That legacy API has been removed from the
mediapipe wheels available for this machine's Python (3.13) -- only the
Tasks API ships now. This module re-implements just the hand-landmark part
with the same output schema as src/extract_pose.py's
`extract_hand_landmarks`, so the rest of the existing pipeline
(calculate_kinematics.calc_tapping_features, the trained model) can be
reused unmodified.

Model file: mp_models/hand_landmarker.task (downloaded from Google's
MediaPipe model zoo, float16 variant, ~7.8 MB).
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MID_MCP = 9

_MODEL_PATH = Path(__file__).parent.parent / "mp_models" / "hand_landmarker.task"


def _hand_csv_header():
    cols = ["frame"]
    for i in range(21):
        cols += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
    cols += ["thumb_index_dist_px", "hand_size_px", "norm_dist"]
    return cols


def euclidean(a, b):
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def extract_hand_landmarks(video_path: Path, out_path: Path | None = None, max_num_hands: int = 1) -> dict:
    """
    Process a finger-tapping video with MediaPipe Tasks HandLandmarker.
    Returns (meta, rows) where rows is a list matching _hand_csv_header(),
    and optionally also writes rows to out_path as CSV if given.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    detected = 0
    rows = []
    frame_idx = 0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            row = [frame_idx]

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                detected += 1

                coords = [(lm.x * w, lm.y * h, lm.z) for lm in lms]
                for c in coords:
                    row += list(c)

                thumb_px = coords[THUMB_TIP][:2]
                index_px = coords[INDEX_TIP][:2]
                wrist_px = coords[WRIST][:2]
                mid_mcp = coords[MID_MCP][:2]

                dist = euclidean(thumb_px, index_px)
                hand_size = euclidean(wrist_px, mid_mcp)
                norm_dist = dist / hand_size if hand_size > 1e-6 else 0.0

                row += [dist, hand_size, norm_dist]
            else:
                row += [float("nan")] * (21 * 3 + 3)

            rows.append(row)
            frame_idx += 1

    cap.release()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_hand_csv_header())
            writer.writerows(rows)

    meta = {
        "video": str(video_path),
        "n_frames": n_frames,
        "fps": fps,
        "detection_rate": detected / max(frame_idx, 1),
    }
    return meta, rows
