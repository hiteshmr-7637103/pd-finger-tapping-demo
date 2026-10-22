"""
CNN-BiLSTM-Attention classifier for finger-tapping MDS-UPDRS severity.

Architecture per docs/finger_tapping_demo_plan.md:
  Conv1D(64, k=3) -> MaxPool1D -> BiLSTM(64) -> Attention -> BiLSTM
  -> concat(CNN branch + BiLSTM branch) -> Dense(250) -> Softmax

Input: the 18 handcrafted kinematic features per video (same feature set
used by the LightGBM baseline), reshaped to a (18, 1) pseudo-sequence.

Data: data/features/tapping_features.csv (pre-computed from both hands,
patient-level train/val split already baked into the 'split' column).

Outputs:
  models/tapping_dl_model.keras
  models/tapping_dl_scaler.pkl
  models/tapping_dl_report.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURES = [
    "average_amplitude",
    "avg_cycle_duration", "avg_cycle_max_speed", "avg_cycle_avg_speed",
    "amplitude_slope", "speed_slope", "cycle_duration_slope",
    "amplitude_decrement", "speed_decrement",
    "cv_amplitude", "cv_cycle_duration", "cv_cms", "cv_cas", "n_interruptions",
    "tapping_frequency", "rhythm_power",
    "jitter_ratio", "peak_to_peak_var",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "tapping_features.csv"
MODELS_DIR = DEMO_DIR / "models"


def build_model(n_features: int, n_classes: int) -> keras.Model:
    inp = keras.Input(shape=(n_features, 1), name="features")

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    cnn_branch = layers.Flatten()(x)

    lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    attn = layers.Attention()([lstm1, lstm1])
    lstm2 = layers.Bidirectional(layers.LSTM(64))(attn)

    merged = layers.Concatenate()([cnn_branch, lstm2])
    dense = layers.Dense(250, activation="relu")(merged)
    dense = layers.Dropout(0.3)(dense)
    out = layers.Dense(n_classes, activation="softmax")(dense)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    df = pd.read_csv(FEATURES_CSV)
    # video_id is the patient id; it's shared between a patient's left- and
    # right-hand rows, and pandas may have inferred it as int (dropping
    # zero-padding) -- normalize to string so grouping works correctly.
    df["video_id"] = df["video_id"].astype(str).str.zfill(5)

    # The precomputed 'split' column was NOT patient-grouped: 81/368 patients
    # have their left-hand row in train and right-hand row in val (or vice
    # versa), which is exactly the leakage risk the demo plan calls out.
    # Re-split here at the patient level instead of trusting that column.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["video_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df.video_id) & set(val_df.video_id)
    if overlap:
        raise RuntimeError(f"Train/val leakage detected for video_ids: {overlap}")

    X_train_raw = train_df[FEATURES].values.astype(np.float32)
    X_val_raw = val_df[FEATURES].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)[..., np.newaxis]
    X_val = scaler.transform(X_val_raw)[..., np.newaxis]

    n_classes = int(max(y_train.max(), y_val.max())) + 1
    class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    model = build_model(len(FEATURES), n_classes)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    f1_mac = f1_score(y_val, y_pred, average="macro", zero_division=0)
    cr = classification_report(y_val, y_pred, zero_division=0)

    print(f"\nVal balanced accuracy: {bal_acc:.4f}")
    print(f"Val macro F1: {f1_mac:.4f}")
    print(cr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "tapping_dl_model.keras")
    with open(MODELS_DIR / "tapping_dl_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "features": FEATURES}, f)

    report = {
        "task": "tapping",
        "architecture": "CNN-BiLSTM-Attention",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "val_bal_acc": float(bal_acc),
        "val_f1_macro": float(f1_mac),
        "classification_report": cr,
        "epochs_run": len(history.history["loss"]),
    }
    with open(MODELS_DIR / "tapping_dl_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'tapping_dl_model.keras'}")
    print(f"Saved scaler to {MODELS_DIR / 'tapping_dl_scaler.pkl'}")
    print(f"Saved report to {MODELS_DIR / 'tapping_dl_report.json'}")


if __name__ == "__main__":
    main()
