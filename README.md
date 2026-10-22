# Finger Tapping PD Risk Score — Demo

Implements `docs/finger_tapping_demo_plan.md`.

This folder **is the git repo that gets hosted** — the project root above it
is never tracked by git. Everything the app needs lives inside here with no
references outside this folder (no `../src`, `../models`, `../PDMotorDB(1)`).

## Layout
```
demo/
├── app.py                  # entry point (Gradio Blocks app; HF Spaces auto-detects this)
├── requirements.txt        # pinned deps, also at root (required by HF Spaces)
├── src/
│   ├── extract_hand_mp_tasks.py   # hand-landmark extraction (see deviation note below)
│   ├── calculate_kinematics.py    # kinematic feature engineering (copy of project src/)
│   ├── train_model.py             # TAPPING_FEATURES list (copy; LightGBM baseline no longer used by app.py)
│   └── train_dl_model.py          # trains the CNN-BiLSTM-Attention model used by the app
├── models/
│   ├── tapping_dl_model.keras    # trained CNN-BiLSTM-Attention model (DL, used by app.py)
│   ├── tapping_dl_scaler.pkl     # StandardScaler fit on train split
│   └── tapping_dl_report.json    # training/val metrics
├── mp_models/
│   └── hand_landmarker.task  # MediaPipe Tasks API model (downloaded from Google)
├── samples/
│   ├── lefthand_val.txt / righthand_val.txt   # trimmed label files
│   └── lefthand_videos/ / righthand_videos/   # 16 sample clips total
├── docs/
│   └── finger_tapping_demo_plan.md
└── .gitignore               # excludes .venv/, __pycache__/
```

## What's done (Steps 1-3 of the plan)
- **Feature extraction / inference**: `src/calculate_kinematics.py` provides
  the 18 handcrafted kinematic features (amplitude, cycle speed, sequence
  effect, hesitation/CV metrics, tapping frequency, jitter) described in the
  plan.
- **Deviation from the plan (hand tracking)**: the project's original
  hand-tracking code uses the legacy `mediapipe.solutions.hands` API, which
  is not available in the mediapipe build for this machine's Python (3.13) —
  only the newer Tasks API ships. `src/extract_hand_mp_tasks.py`
  re-implements just the hand-landmark extraction using `HandLandmarker`
  (Tasks API) with the exact same CSV schema as the original, so the rest of
  the pipeline (feature calc + model) did not need to change.
- **Model: CNN-BiLSTM-Attention (the DL architecture the plan specifies),
  not the LightGBM fallback.** `src/train_dl_model.py` trains
  `Conv1D(64,k=3) -> MaxPool1D -> BiLSTM(64) -> Attention -> BiLSTM ->
  concat(CNN+BiLSTM) -> Dense(250) -> Softmax` (342K params, ~1.3 MB) on the
  18-feature vectors from `data/features/tapping_features.csv`
  (left+right hand combined, 611 clips).
  - **Found and fixed a real bug on the way**: that precomputed feature
    CSV's `split` column was not patient-grouped — 81/368 patients had their
    left-hand clip in train and right-hand clip in val (or vice versa),
    exactly the leakage risk the plan's Step 2 calls out. `train_dl_model.py`
    ignores that column and re-splits with `GroupShuffleSplit` grouped by
    patient id (80/20), so no patient's clips cross the split.
  - **Result (on the corrected, leak-free split): val balanced accuracy
    0.857, val macro-F1 0.860** (`models/tapping_dl_report.json`). This is a
    fair, non-inflated number — the earlier LightGBM model's 0.886 was
    measured on the leaky split and is not directly comparable.
  - Verified end-to-end on real sample clips through `app.py`'s own
    pipeline (5/6 correct on a spot-check).
- **Sample clips**: `samples/` holds 16 clips (8 left-hand, 8 right-hand)
  from PDMotorDB's val split — enough for the in-app dropdown without
  redistributing the full academic-use dataset.
- **Gradio app**: `app.py` — upload a video or pick a sample clip, shows
  the thumb-index distance wave (Plotly), a colored risk gauge for the
  predicted MDS-UPDRS 3.4 score (0-3), per-class probabilities, and the
  extracted feature table. Disclaimer banner at top.
  - **Deviation from the plan (hosting SDK)**: the plan assumed Streamlit +
    Hugging Face Spaces. On this HF account, the Docker SDK (which a
    Streamlit Space would need, since there's no native Streamlit card in
    the New Space flow here) is gated behind a paid plan. Gradio is the free,
    natively auto-detected SDK, so the UI was rewritten in Gradio `Blocks` —
    same backend (`run_pipeline`), same features, no Dockerfile needed.

## Retraining the DL model
```
cd demo
.venv/bin/python3 src/train_dl_model.py
```
Reads `../data/features/tapping_features.csv` from the project root (not
copied into `demo/` since it's an intermediate artifact, not needed at
inference time) and overwrites `models/tapping_dl_model.keras` /
`tapping_dl_scaler.pkl` / `tapping_dl_report.json`.

## Running locally
```
cd demo
.venv/bin/python3 app.py
```
(`.venv` here is a fresh Python 3.13 venv with gradio/mediapipe/opencv/
tensorflow-cpu/plotly installed. It's gitignored — on a fresh clone, recreate
it with `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.)

## Hosting on Hugging Face Spaces
1. Create a new Space: SDK **Gradio**, hardware **CPU Basic (free)**.
2. From inside `demo/` (already a git repo, `main` branch, initial commit
   done): `git remote add origin <your HF Space git URL>`.
3. `git push -u origin main` (use an HF **write** access token as the
   password when prompted).
4. Watch the Space's build logs; a few minutes for TensorFlow/mediapipe to
   install.
5. Test one upload (or the sample-clip dropdown) on the hosted Space before
   the panel sees it — library versions can differ from this local `.venv`.

## Known limits (explicitly out of scope per the plan)
- No live webcam capture.
- No clinical validation — labeled "research prototype" in the UI.
- Only tested on PDMotorDB-style cropped hand clips, not arbitrary phone
  videos.
- Postural stability is untouched (label anomaly from the earlier audit is
  unresolved).
