"""
Finger Tapping PD Risk Score Dashboard -- demo build.

Upload a finger-tapping video -> extracts the thumb-index distance wave,
computes the same kinematic features used to train the model, and shows
a predicted MDS-UPDRS 3.4 severity score with a risk gauge.
"""
import subprocess
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

UPDRS_LABELS = {0: "Normal", 1: "Slight", 2: "Mild", 3: "Moderate", 4: "Severe"}

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


@st.cache_data(show_spinner=False)
def to_browser_mp4(video_path_str: str) -> str:
    """
    Transcode to a browser-playable format for the <video> preview player only.
    AVI (or other non-mp4/webm) containers aren't reliably playable in
    browsers' native HTML5 video element, even though OpenCV reads them
    fine for the actual scoring pipeline -- this is display-only.
    Tries H.264/mp4 first, falls back to VP8/webm if libx264 isn't
    available in this ffmpeg build.
    """
    video_path = Path(video_path_str)
    if video_path.suffix.lower() in (".mp4", ".webm"):
        return str(video_path)

    tmpdir = Path(tempfile.mkdtemp())

    mp4_path = tmpdir / "preview.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(mp4_path),
            ],
            check=True, capture_output=True, timeout=30,
        )
        return str(mp4_path)
    except Exception:
        pass

    webm_path = tmpdir / "preview.webm"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "libvpx", "-b:v", "1M", str(webm_path)],
            check=True, capture_output=True, timeout=30,
        )
        return str(webm_path)
    except Exception:
        return str(video_path)  # last resort: original (likely won't play, but scoring still works)


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
st.set_page_config(page_title="PD Finger Tapping Risk Score", page_icon="🖐️", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1400px; }
    h1 { font-size: 3rem !important; margin-bottom: 0.3rem !important; }
    .subtitle { color: #b0b6bd; font-size: 1.25rem; margin-bottom: 2rem; line-height: 1.5; }
    div[data-testid="stExpander"] { border-radius: 12px; font-size: 1.05rem; }
    div[data-testid="stFileUploader"] label, div[data-testid="stSelectbox"] label {
        font-size: 1.1rem !important; font-weight: 600;
    }
    .prob-row { display:flex; align-items:center; gap:14px; margin-bottom:10px; }
    .prob-label { width: 230px; font-size: 1.0rem; font-weight: 600; flex-shrink:0; }
    .prob-pct { width: 64px; font-size: 1.05rem; color:#b0b6bd; text-align:right; flex-shrink:0; }
    .prob-track { flex:1; height: 20px; border-radius: 10px; background: rgba(255,255,255,0.08); overflow:hidden; }
    .prob-fill { height: 100%; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🖐️ Finger Tapping PD Motor Risk Score")
st.markdown(
    "<div class='subtitle'>Upload a finger-tapping video or pick a sample clip. "
    "The app tracks the thumb-index distance, computes kinematic features, and predicts "
    "an MDS-UPDRS 3.4 severity score using a CNN-BiLSTM-Attention model.</div>",
    unsafe_allow_html=True,
)

with st.container(border=True):
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
    with st.spinner("Tracking hand landmarks and scoring..."):
        try:
            feats, pred, proba, time_axis, wave, meta = run_pipeline(video_path)
        except Exception as e:
            st.error(f"Could not process this video: {e}")
            st.stop()

    st.divider()

    if meta["detection_rate"] < 0.5:
        st.warning(
            f"Hand detected in only {meta['detection_rate']:.0%} of frames -- "
            "result may be unreliable for this video.",
            icon="⚠️",
        )

    top_col1, top_col2 = st.columns([1, 1.4], gap="large")
    with top_col1:
        st.video(to_browser_mp4(str(video_path)))
    with top_col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis, y=wave, mode="lines", name="thumb-index distance",
            line=dict(color="#4f8cff", width=3),
        ))
        fig.update_layout(
            title=dict(text="Thumb-Index Distance Wave", font=dict(size=22)),
            xaxis_title="Time (s)",
            yaxis_title="Normalized distance",
            height=440,
            margin=dict(l=30, r=30, t=60, b=40),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=15),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    score_col, prob_col = st.columns([1, 1.4], gap="large")

    with score_col:
        color = RISK_COLORS.get(pred, "#888")
        label = UPDRS_LABELS.get(pred, "")
        st.markdown(
            f"<div style='text-align:center;padding:40px 24px;border-radius:18px;"
            f"background:{color}1a;border:2px solid {color}80'>"
            f"<div style='font-size:16px;letter-spacing:.05em;text-transform:uppercase;color:#b0b6bd'>"
            f"Predicted MDS-UPDRS 3.4 score</div>"
            f"<div style='font-size:120px;font-weight:800;color:{color};line-height:1.05;margin:10px 0'>{pred}</div>"
            f"<div style='font-size:28px;font-weight:700;color:{color};margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:20px;color:#b0b6bd;margin-bottom:10px'>confidence {proba[pred]:.0%}</div>"
            f"<div style='font-size:16px;color:#d8dbe0;max-width:380px;margin:0 auto'>{UPDRS_DESCRIPTIONS.get(pred, '')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with prob_col:
        st.markdown("<div style='font-size:1.2rem;font-weight:700;margin-bottom:14px'>Class probabilities</div>", unsafe_allow_html=True)
        # Show all 5 clinical classes (0-4), even though the model only has
        # 4 output units (no training patient scored a 4) -- class 4 is
        # pinned at 0% rather than omitted, so the full scale stays visible.
        full_proba = {c: proba.get(c, 0.0) for c in range(5)}
        rows_html = ""
        for c in sorted(full_proba):
            pct = full_proba[c]
            bar_color = RISK_COLORS.get(c, "#888")
            label = UPDRS_LABELS.get(c, "")
            rows_html += (
                f"<div class='prob-row'>"
                f"<div class='prob-label'>{c} &middot; {label}</div>"
                f"<div class='prob-track'><div class='prob-fill' style='width:{pct*100:.1f}%;background:{bar_color}'></div></div>"
                f"<div class='prob-pct'>{pct:.1%}</div>"
                f"</div>"
            )
        st.markdown(rows_html, unsafe_allow_html=True)

    st.write("")
    with st.expander("Extracted feature values"):
        feat_table = pd.DataFrame(
            [{"feature": k, "value": round(v, 4)} for k, v in feats.items() if k in TAPPING_FEATURES]
        )
        st.dataframe(feat_table, use_container_width=True, hide_index=True)
else:
    st.info("Upload a video or pick a sample clip to run the demo.")
