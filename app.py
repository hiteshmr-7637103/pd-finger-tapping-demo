"""
Finger Tapping PD Risk Score Dashboard -- demo build (Gradio).

Upload a finger-tapping video -> extracts the thumb-index distance wave,
computes the same kinematic features used to train the model, and shows
a predicted MDS-UPDRS 3.4 severity score with a risk gauge.

Research prototype / demo only -- not a validated clinical diagnostic tool.

Gradio is used instead of Streamlit because it's the SDK Hugging Face
Spaces auto-detects and hosts for free; the Docker SDK (which Streamlit
would otherwise need) is gated behind a paid plan on this account.
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

warnings.filterwarnings("ignore")
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

_model_cache = {}


def load_dl_model():
    if "bundle" not in _model_cache:
        model = keras.models.load_model(MODELS_DIR / "tapping_dl_model.keras")
        with open(MODELS_DIR / "tapping_dl_scaler.pkl", "rb") as f:
            bundle = pickle.load(f)
        _model_cache["bundle"] = (model, bundle["scaler"], bundle["features"])
    return _model_cache["bundle"]


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
                samples.append((f"{hand} hand - {vid_id} (true label {label})", str(video_path)))
    return samples


def run_pipeline(video_path):
    """Extract landmarks, compute features, run the model. Returns (feats, pred, proba, time_axis, wave, meta)."""
    video_path = Path(video_path)
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


def score_gauge_html(pred: int, confidence: float) -> str:
    color = RISK_COLORS.get(pred, "#888")
    return (
        f"<div style='text-align:center;padding:20px;border-radius:12px;"
        f"background:{color}22;border:2px solid {color}'>"
        f"<div style='font-size:14px;color:#888'>Predicted MDS-UPDRS 3.4 score</div>"
        f"<div style='font-size:64px;font-weight:bold;color:{color}'>{pred}</div>"
        f"<div style='font-size:14px'>confidence {confidence:.0%}</div>"
        f"<div style='font-size:13px;margin-top:8px;color:#aaa'>{UPDRS_DESCRIPTIONS.get(pred, '')}</div>"
        f"</div>"
    )


def wave_plot(time_axis, wave):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=wave, mode="lines", name="thumb-index distance"))
    fig.update_layout(
        title="Thumb-Index Distance Wave",
        xaxis_title="Time (s)",
        yaxis_title="Normalized distance",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def predict(uploaded_video, sample_choice):
    video_path = uploaded_video or (dict(list_sample_videos()).get(sample_choice) if sample_choice else None)
    if video_path is None:
        return (
            None, gr.update(value="Upload a video or pick a sample clip to run the demo."),
            None, None, None,
        )

    try:
        feats, pred, proba, time_axis, wave, meta = run_pipeline(video_path)
    except Exception as e:
        return None, gr.update(value=f"Could not process this video: {e}"), None, None, None

    warn_html = ""
    if meta["detection_rate"] < 0.5:
        warn_html = (
            f"<div style='color:#e67e22'>⚠️ Hand detected in only "
            f"{meta['detection_rate']:.0%} of frames -- result may be unreliable.</div>"
        )

    gauge_html = warn_html + score_gauge_html(pred, proba[pred])
    label_dict = {f"Score {c}": float(p) for c, p in proba.items()}
    fig = wave_plot(time_axis, wave)
    feat_df = pd.DataFrame(
        [{"feature": k, "value": round(v, 4)} for k, v in feats.items() if k in TAPPING_FEATURES]
    )

    return video_path, gauge_html, label_dict, fig, feat_df


with gr.Blocks(title="PD Finger Tapping Risk Score") as demo:
    gr.Markdown("# 🖐️ Finger Tapping PD Motor Risk Score")
    gr.Markdown(
        "⚠️ **Research prototype / demo only. Not a validated clinical diagnostic tool.** "
        "Trained on PDMotorDB (academic-use dataset); not tested on arbitrary real-world videos."
    )
    gr.Markdown(
        "Upload a finger-tapping video (hand cropped, similar to PDMotorDB clips) or pick a "
        "sample below. The app tracks the thumb-index distance, computes kinematic features, "
        "and predicts an MDS-UPDRS 3.4 severity score (0-3 seen in training data) using a "
        "CNN-BiLSTM-Attention model."
    )

    with gr.Row():
        video_in = gr.Video(label="Upload a video", sources=["upload"])
        sample_dropdown = gr.Dropdown(
            choices=[s[0] for s in list_sample_videos()],
            label="...or pick a sample clip",
        )

    run_btn = gr.Button("Run", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            video_preview = gr.Video(label="Input video", interactive=False)
            gauge = gr.HTML()
            probs = gr.Label(label="Class probabilities")
        with gr.Column(scale=2):
            plot = gr.Plot(label="Thumb-Index Distance Wave")

    with gr.Accordion("Extracted feature values", open=False):
        feat_table = gr.Dataframe(headers=["feature", "value"])

    run_btn.click(
        fn=predict,
        inputs=[video_in, sample_dropdown],
        outputs=[video_preview, gauge, probs, plot, feat_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
