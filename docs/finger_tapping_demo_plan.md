| **Name: Hitesh M R** | **Reg No.: 3122245002028** |
|---|---|
| **Date: 22.10.2026** | **Topic: Finger Tapping PD Risk Score Dashboard — Demo Build Plan** |

# Finger Tapping PD Risk Score Dashboard — Demo Build Plan

## Context
Tomorrow's panel review needs a working, hosted demo. Scope is deliberately narrow:

- Train a small classifier on PDMotorDB finger-tapping videos (labels confirmed real/usable).
- Build a Streamlit dashboard: **upload a test video** (not live webcam) -> shows the
  thumb-index distance graph + a risk score.
- Host it (Hugging Face Spaces, free CPU tier) so the panel can open a link.
- **Not in scope for tomorrow:** live webcam capture, clinical validation, dataset-size
  concerns, generalization to real patient videos. This is a working research-prototype demo.
- Dataset root: `<FILL IN — e.g. /mnt/d_drive/bradykinesia_model>`
- Prior audit already confirmed: 368 finger-tapping patients (lefthand_videos: 368 files,
  righthand_videos: 243 files), 100% label coverage, no corrupt/duplicate files, fixed
  5.0s clips at 25fps. See `audit_pdmotordb.py` / `audit_report.md` / `audit_files.csv`
  already produced in that folder.

## Time budget (~2 hours total)
1. Feature extraction: ~10-15 min (CPU, parallelizable across videos)
2. Model training: <1 min (tiny model, 368 samples)
3. Streamlit app build: 30-45 min
4. Hosting/deploy: 10-15 min
5. Buffer for debugging

## Reference: closest published approach
Yang et al. (IEEE Access 2022, the PDMotorDB paper itself) pipeline:
hand pose estimation (21 keypoints) -> neurologist-defined features -> classifier.
Reported micro-F1: 88% left hand, 84% right hand (exact split protocol unknown — treat
as a rough target, not a guarantee).

A separate paper (CNN-BiLSTM with attention, trained on a *different* dataset — not
PDMotorDB) used this architecture on handcrafted features:
Conv1D(64, k=3) -> MaxPool1D -> BiLSTM(64) -> Attention -> BiLSTM -> concat(CNN+BiLSTM)
-> Dense(250) -> Softmax. This is small (well under 1M params) and trains in ~30s on
CPU. Good candidate architecture for tomorrow given the time constraint — simple, fast,
proven on this kind of feature vector.

## Step 1: Feature extraction script (`extract_features.py`)
For every video in `lefthand_videos/` and `righthand_videos/`:

1. Run MediaPipe Hands (or existing hand-tracking already used in the audit, if any) to
   get landmark positions per frame for thumb tip and index fingertip.
2. Compute the thumb-index Euclidean distance per frame -> this is "the wave."
3. Detect taps (local minima/maxima of the distance signal) and derive handcrafted
   features per clip, e.g.:
   - tap count / tap rate
   - mean amplitude, max amplitude
   - amplitude decrement (early taps vs late taps in the 5s clip)
   - mean/variance of inter-tap interval (rhythm/hesitation proxy)
   - mean tap speed (velocity between min/max)
   - number of freezes/hesitations (gaps > some threshold between taps)
4. Handle videos where hand detection fails or confidence is low — log them, don't
   silently drop (print a warning, keep a `detection_failed` flag column).
5. Output: one row per video to `features.csv` with columns:
   `patient_id, hand (left/right), label, n_frames_detected, <feature columns>`
6. Pull `label` from whatever label file/format the earlier audit already confirmed
   (train/val label lines, 100% coverage per the audit report — reuse that parsing logic).
7. Print a summary: total videos processed, how many failed detection, feature value
   ranges (sanity check for NaNs/outliers).

## Step 2: Train script (`train_model.py`)
1. Load `features.csv`.
2. **Patient-level split**: group by `patient_id` before splitting train/val (80/20 or
   similar) — a patient's left and right hand clips must stay in the same split. Do NOT
   split by video row directly (this was flagged as a risk in the audit follow-up).
3. Normalize/scale features (StandardScaler, fit on train only).
4. Model: small CNN-BiLSTM-attention classifier as described above (Keras/TensorFlow or
   PyTorch, whichever is faster to get working in the time available — prefer whichever
   the environment already has installed to avoid install delays).
   - If time is tight, a simpler fallback is fine: LightGBM/RandomForest on the same
     feature vector trains in seconds and is easier to debug/host. Use this as a
     safety-net fallback if the CNN-BiLSTM has issues — don't burn the demo on a
     training bug.
5. Train, evaluate on validation split: accuracy, macro F1, per-class recall, and
   accuracy-within-1-class (useful since severity is ordinal).
6. Save the trained model to disk (`model.pkl` / `model.pt` / `model.h5`) plus the
   fitted scaler, so the Streamlit app can load both without retraining.
7. Print final metrics clearly so they can be quoted in tomorrow's demo/slide.

## Step 3: Streamlit app (`app.py`)
1. Simple single-page layout:
   - File uploader for a video (`st.file_uploader`, accept mp4/avi/mov).
   - On upload: run the SAME feature-extraction logic from Step 1 on the uploaded video
     (import/reuse the function, don't duplicate code).
   - Plot the thumb-index distance wave over time (`st.line_chart` or matplotlib/plotly)
     with detected taps marked.
   - Load the trained model + scaler, predict the severity class (0-4) and show it as
     a clear risk score / gauge (e.g. `st.metric` or a simple colored gauge — green/
     yellow/orange/red bands across 0-4).
   - Show the extracted feature values in a small table underneath, for transparency.
2. Add a visible one-line disclaimer at the top: this is a research prototype/demo, not
   a validated clinical diagnostic tool.
3. Optional (if time permits): let the user pick a few sample videos from PDMotorDB's
   test split directly in the app (dropdown) so the panel can click through examples
   live without needing their own video file.
4. Keep the UI minimal — title, uploader, graph, score, feature table, disclaimer.
   Don't over-engineer styling given the time budget.

## Step 4: Hosting (Hugging Face Spaces, free CPU tier)
1. Create `requirements.txt` with exact packages used (streamlit, mediapipe, opencv-
   python-headless, numpy, pandas, scikit-learn, and tensorflow/torch + lightgbm if used).
2. Create a new Space (Streamlit SDK) on huggingface.co, push `app.py`, model files,
   scaler, and any helper modules.
3. **Do not upload the full PDMotorDB dataset to the Space** — only the trained model/
   scaler and (if doing the optional sample-picker) a handful of representative demo
   clips, keeping total upload small and respecting the dataset's academic-use license.
4. Verify the deployed Space builds and loads correctly, then test end-to-end with one
   uploaded video to confirm the full pipeline works on the hosted instance (not just
   locally — library versions can differ).
5. Note the Space's public URL for tomorrow's demo link.

## Explicitly out of scope for tomorrow
- Live webcam / real-time capture.
- Clinical validation, IRB/ethics sign-off, any claim this is diagnostic.
- Handling arbitrary real-world patient videos (different camera/lighting/angle) robustly
  — the demo works on PDMotorDB-style test videos.
- Addressing the postural-stability label anomaly from the earlier audit — out of scope,
  finger tapping only.

## After tomorrow (not now)
- Confirm finger-tapping label distribution/quality more rigorously, check for label
  noise.
- Add input normalization/cropping so arbitrary uploaded videos (not just dataset clips)
  score reliably.
- Consider ordinal loss/head (e.g. CORAL) instead of plain softmax, since severity is
  ordinal.
- Real-time webcam capture if the panel wants that as a next iteration.
