"""
Finger Tapping PD Risk Score Dashboard -- demo build.

Upload a finger-tapping video -> extracts the thumb-index distance wave,
computes the same kinematic features used to train the model, and shows
a predicted MDS-UPDRS 3.4 severity score with a risk gauge.

Research prototype / demo only -- not a validated clinical diagnostic tool.
"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

DEMO_DIR = Path(__file__).parent
MODELS_DIR = DEMO_DIR / "models"
SAMPLE_DIR = DEMO_DIR / "samples"
SRC_DIR = DEMO_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from calculate_kinematics import calc_tapping_features, _lowpass, _remove_nan_gaps  # noqa: E402
from train_model import TAPPING_FEATURES  # noqa: E402
from extract_hand_mp_tasks import extract_hand_landmarks  # noqa: E402

import pickle
from tensorflow import keras

UPDRS_DESCRIPTIONS = {
    0: "Normal: no impairment.",
    1: "Slight: 1-2 interruptions, slight slowing, or amplitude decrement near the end.",
    2: "Mild: 3-5 interruptions, mild slowing, or amplitude decrement from midway.",
    3: "Moderate: more than 5 interruptions / a longer freeze, or decrement starting after the 1st tap.",
    4: "Severe: cannot or can only barely perform the task.",
}

RISK_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c", 4: "#8e0000"}


@st.cache_resource
def load_dl_model():
    model = keras.models.load_model(MODELS_DIR / "tapping_dl_model.keras")
    with open(MODELS_DIR / "tapping_dl_scaler.pkl", "rb") as f:
        bundle = pickle.load(f)
    return model, bundle["scaler"], bundle["features"]


def list_sample_videos():
    samples = []
    for hand in ("left", "right"):
        label_file = SAMPLE_DIR / f"{hand}hand_val.txt"
        video_dir = SAMPLE_DIR / f"{hand}hand_videos"
        if not label_file.exists():
            continue
        for line in label_file.read_text().strip().splitlines()[:8]:
            parts = line.split()
            if len(parts) != 2:
                continue
            vid_id, label = parts
            video_path = video_dir / f"{vid_id}.avi"
            if video_path.exists():
                samples.append((f"{hand} hand - {vid_id} (true label {label})", video_path))
    return samples


def run_pipeline(video_path: Path):
    """Extract landmarks, compute features, run the model. Returns (feats, pred, proba, classes, wave)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lm_csv = Path(tmpdir) / "landmarks.csv"
        meta, _ = extract_hand_landmarks(video_path, lm_csv)
        fps = float(meta.get("fps", 25.0)) or 25.0
        feats = calc_tapping_features(lm_csv, fps=fps, label=-1)

        df = pd.read_csv(lm_csv)
        raw = _remove_nan_gaps(df["norm_dist"].values.astype(float))
        wave = _lowpass(raw, fps, cutoff=10.0) if len(raw) >= 10 else raw
        time_axis = np.arange(len(wave)) / fps

    model, scaler, dl_features = load_dl_model()
    X_raw = np.array([[feats.get(f, 0.0) for f in dl_features]], dtype=np.float32)
    X = scaler.transform(X_raw)[..., np.newaxis]

    proba_arr = model.predict(X, verbose=0)[0]
    pred = int(np.argmax(proba_arr))
    classes = list(range(len(proba_arr)))

    return feats, pred, dict(zip(classes, proba_arr.tolist())), time_axis, wave, meta


# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PD Finger Tapping Risk Score", page_icon="🖐️", layout="centered")

st.title("🖐️ Finger Tapping PD Motor Risk Score")
st.warning(
    "⚠️ Research prototype / demo only. Not a validated clinical diagnostic tool. "
    "Trained on PDMotorDB (academic-use dataset); not tested on arbitrary real-world videos.",
    icon="⚠️",
)

st.markdown(
    "Upload a finger-tapping video (hand cropped, similar to PDMotorDB clips) or pick a "
    "sample below. The app tracks the thumb-index distance, computes kinematic features, "
    "and predicts an MDS-UPDRS 3.4 severity score (0-3 seen in training data)."
)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader("Upload a video", type=["avi", "mp4", "mov"])
with col2:
    samples = list_sample_videos()
    sample_labels = ["-- none --"] + [s[0] for s in samples]
    sample_choice = st.selectbox("...or pick a sample clip", sample_labels)

video_path = None
if uploaded is not None:
    tmp_path = Path(tempfile.mkdtemp()) / uploaded.name
    tmp_path.write_bytes(uploaded.read())
    video_path = tmp_path
elif sample_choice != "-- none --":
    video_path = dict(samples)[sample_choice]

if video_path is not None:
    st.video(str(video_path))

    with st.spinner("Tracking hand landmarks and scoring..."):
        try:
            feats, pred, proba, time_axis, wave, meta = run_pipeline(video_path)
        except Exception as e:
            st.error(f"Could not process this video: {e}")
            st.stop()

    if meta["detection_rate"] < 0.5:
        st.warning(
            f"Hand detected in only {meta['detection_rate']:.0%} of frames -- "
            "result may be unreliable for this video.",
            icon="⚠️",
        )

    # Risk score gauge
    score_col, graph_col = st.columns([1, 2])
    with score_col:
        color = RISK_COLORS.get(pred, "#888")
        st.markdown(
            f"<div style='text-align:center;padding:20px;border-radius:12px;"
            f"background:{color}22;border:2px solid {color}'>"
            f"<div style='font-size:14px;color:#888'>Predicted MDS-UPDRS 3.4 score</div>"
            f"<div style='font-size:64px;font-weight:bold;color:{color}'>{pred}</div>"
            f"<div style='font-size:14px'>confidence {proba[pred]:.0%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(UPDRS_DESCRIPTIONS.get(pred, ""))

        st.markdown("**Class probabilities**")
        for c in sorted(proba):
            st.progress(float(proba[c]), text=f"Score {c}: {proba[c]:.1%}")

    with graph_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_axis, y=wave, mode="lines", name="thumb-index distance"))
        fig.update_layout(
            title="Thumb-Index Distance Wave",
            xaxis_title="Time (s)",
            yaxis_title="Normalized distance",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Extracted feature values"):
        feat_table = pd.DataFrame(
            [{"feature": k, "value": round(v, 4)} for k, v in feats.items() if k in TAPPING_FEATURES]
        )
        st.dataframe(feat_table, use_container_width=True, hide_index=True)
else:
    st.info("Upload a video or pick a sample clip to run the demo.")
