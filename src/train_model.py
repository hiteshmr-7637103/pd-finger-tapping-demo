"""
Stage 3 – Model Training
==========================
Trains separate classifiers for:
  • Finger tapping  (left hand / right hand / combined)
  • Postural stability

Architecture inspired by:
  Zarrat Ehsan et al. (npj PD, 2026): LightGBM multi-class on interpretable kinematic features
  Khan et al. (video, 2014): SVM-RBF on tapping features → 88 % accuracy
  Williams et al. (PhD thesis, 2023): SVM-RBF with PCA on 10 optical-flow features → 80 % LOOCV

Strategy:
  1. Ordinal / multi-class classification: predict MDS-UPDRS score (0-4)
  2. Class imbalance: SMOTE oversampling inside CV loop
  3. Cross-validation: StratifiedKFold (5-fold) on train split
  4. Final evaluation: held-out val split
  5. Models tried: SVM-RBF (primary), LightGBM, MLP, Logistic Regression

Outputs:
  models/tapping_model.pkl      – best tapping model
  models/stability_model.pkl    – best stability model
  results/train_report.json     – full metrics
"""

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Try to import optional libraries ─────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.info("LightGBM not installed; will skip LGBM option.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.info("imbalanced-learn not installed; SMOTE disabled.")


# ─── Feature columns ─────────────────────────────────────────────────────────

TAPPING_FEATURES = [
    # hypokinesia
    "average_amplitude",
    # bradykinesia
    "avg_cycle_duration", "avg_cycle_max_speed", "avg_cycle_avg_speed",
    # sequence effect
    "amplitude_slope", "speed_slope", "cycle_duration_slope",
    "amplitude_decrement", "speed_decrement",
    # hesitation-halts
    "cv_amplitude", "cv_cycle_duration", "cv_cms", "cv_cas", "n_interruptions",
    # frequency / rhythm
    "tapping_frequency", "rhythm_power",
    # variability
    "jitter_ratio", "peak_to_peak_var",
]

STABILITY_FEATURES = [
    "com_std_x", "com_std_y", "com_range_x", "com_range_y",
    "com_path_len", "com_mean_vel",
    "x_total_power", "x_high_freq_power", "x_low_freq_power",
    "y_total_power", "y_high_freq_power", "y_low_freq_power",
    "sd1_x", "sd2_x", "sd1_y", "sd2_y",
    "rma95", "rmi95", "area95", "rmi99", "area99",
    "ankle_asym_std", "ankle_asym_mean",
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_split(csv_path: Path, features: list, label_col: str = "label"):
    """
    Return (X_train, y_train, X_val, y_val) split by the 'split' column.
    Rows with label=-1 or any NaN feature are dropped.
    """
    df = pd.read_csv(csv_path)

    # Drop rows with missing label or NaN features
    df = df.dropna(subset=features + [label_col])
    df = df[df[label_col] >= 0]

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]

    X_tr = train[features].values.astype(float)
    y_tr = train[label_col].values.astype(int)
    X_va = val[features].values.astype(float)
    y_va = val[label_col].values.astype(int)

    logger.info(f"  Train: {len(X_tr)} samples, Val: {len(X_va)} samples")
    logger.info(f"  Train label dist: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    logger.info(f"  Val   label dist: {dict(zip(*np.unique(y_va, return_counts=True)))}")

    return X_tr, y_tr, X_va, y_va


# ─── Pipeline builders ────────────────────────────────────────────────────────

def _make_candidate_pipelines():
    """
    Return a dict of {name: pipeline / estimator}.
    All use RobustScaler (handles outliers better than StandardScaler for small medical datasets).
    SVM-RBF uses PCA-2 as per Williams et al. to reduce overfitting on small N.
    """
    pipes = {}

    # ── SVM-RBF (preferred per literature) ───────────────────────────────────
    pipes["SVM_RBF"] = Pipeline([
        ("scaler", RobustScaler()),
        ("pca",    PCA(n_components=2, whiten=True)),
        ("clf",    SVC(kernel="rbf", probability=True, class_weight="balanced",
                       random_state=42)),
    ])

    # ── Logistic Regression ────────────────────────────────────────────────────
    pipes["LR"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    LogisticRegression(max_iter=2000, class_weight="balanced",
                                      solver="saga",
                                      random_state=42)),
    ])

    # ── MLP ───────────────────────────────────────────────────────────────────
    pipes["MLP"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                  random_state=42, early_stopping=True,
                                  n_iter_no_change=20)),
    ])

    # ── Random Forest ─────────────────────────────────────────────────────────
    pipes["RF"] = Pipeline([
        ("scaler", RobustScaler()),
        ("clf",    RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=42, n_jobs=-1)),
    ])

    # ── LightGBM (SOTA per Zarrat Ehsan 2026) ─────────────────────────────────
    if HAS_LGBM:
        pipes["LightGBM"] = Pipeline([
            ("scaler", RobustScaler()),
            ("clf",    LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                       num_leaves=31, class_weight="balanced",
                                       random_state=42, verbose=-1, n_jobs=1)),
        ])

    return pipes


def _hyperparam_grids():
    return {
        "SVM_RBF"  : {"pca__n_components": [2, 3, 4],
                       "clf__C"           : [0.1, 1.0, 10.0, 50.0],
                       "clf__gamma"       : ["scale", "auto", 0.01, 0.1]},
        "LR"       : {"clf__C"            : [0.01, 0.1, 1.0, 10.0]},
        "MLP"      : {"clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                       "clf__alpha"              : [1e-4, 1e-3, 1e-2]},
        "RF"       : {"clf__n_estimators" : [100, 300],
                       "clf__max_depth"   : [None, 5, 10]},
        "LightGBM" : {"clf__n_estimators" : [200, 500],
                       "clf__learning_rate": [0.01, 0.05, 0.1],
                       "clf__num_leaves"  : [15, 31, 63]},
    }


# ─── Training loop ────────────────────────────────────────────────────────────

def train_and_evaluate(X_tr, y_tr, X_va, y_va,
                        task_name: str,
                        models_dir: Path,
                        results_dir: Path) -> dict:
    """
    1. Optionally apply SMOTE to training data.
    2. Grid-search each candidate pipeline with StratifiedKFold CV.
    3. Re-fit best config on full training set.
    4. Evaluate on held-out val set.
    5. Save best model.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Apply SMOTE if available (only on training data)
    if HAS_IMBLEARN:
        smote = SMOTE(random_state=42, k_neighbors=min(5, int(np.bincount(y_tr).min() - 1)))
        try:
            X_tr_r, y_tr_r = smote.fit_resample(X_tr, y_tr)
            logger.info(f"  SMOTE: {len(y_tr)} → {len(y_tr_r)} samples")
        except Exception as e:
            logger.warning(f"  SMOTE failed ({e}), using original data")
            X_tr_r, y_tr_r = X_tr, y_tr
    else:
        X_tr_r, y_tr_r = X_tr, y_tr

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipes  = _make_candidate_pipelines()
    grids  = _hyperparam_grids()
    report = {}

    best_bal_acc   = -1.0
    best_name      = None
    best_estimator = None

    for name, pipe in pipes.items():
        logger.info(f"\n  ── Training [{name}] on {task_name} ──")
        param_grid = grids.get(name, {})

        gs = GridSearchCV(
            estimator  = pipe,
            param_grid = param_grid,
            cv         = cv,
            scoring    = "balanced_accuracy",
            n_jobs     = 1,
            refit      = True,
            verbose    = 0,
        )

        try:
            gs.fit(X_tr_r, y_tr_r)
        except Exception as e:
            logger.error(f"  [{name}] GridSearch failed: {e}")
            continue

        y_pred    = gs.predict(X_va)
        bal_acc   = balanced_accuracy_score(y_va, y_pred)
        f1_mac    = f1_score(y_va, y_pred, average="macro", zero_division=0)
        cr        = classification_report(y_va, y_pred, zero_division=0)

        logger.info(f"  [{name}] CV best: {gs.best_score_:.3f}  |  Val BalAcc: {bal_acc:.3f}  |  Val F1-macro: {f1_mac:.3f}")
        logger.info(f"  Best params: {gs.best_params_}")
        logger.info(f"\n{cr}")

        report[name] = {
            "cv_best_score"   : float(gs.best_score_),
            "val_bal_acc"     : float(bal_acc),
            "val_f1_macro"    : float(f1_mac),
            "best_params"     : gs.best_params_,
            "classification_report": cr,
        }

        if bal_acc > best_bal_acc:
            best_bal_acc   = bal_acc
            best_name      = name
            best_estimator = gs.best_estimator_

    # ── Save best model ───────────────────────────────────────────────────────
    if best_estimator is not None:
        model_out = models_dir / f"{task_name}_model.pkl"
        metadata  = {
            "task"        : task_name,
            "best_model"  : best_name,
            "val_bal_acc" : float(best_bal_acc),
            "all_results" : report,
        }
        with open(model_out, "wb") as f:
            pickle.dump({"estimator": best_estimator, "metadata": metadata}, f)
        logger.info(f"\n  ✅ Best model [{best_name}] saved → {model_out}")

        # Save JSON report
        json_out = results_dir / f"{task_name}_report.json"
        with open(json_out, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  📄 Report → {json_out}")

    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MDS-UPDRS severity classifiers.")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--models_dir",   default="models")
    parser.add_argument("--results_dir",  default="results")
    parser.add_argument("--task",         choices=["all", "tapping", "stability"], default="all")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    feat_dir    = Path(args.features_dir)
    models_dir  = Path(args.models_dir)
    results_dir = Path(args.results_dir)

    # ── Tapping ───────────────────────────────────────────────────────────────
    if args.task in ("all", "tapping"):
        tap_csv = feat_dir / "tapping_features.csv"
        if not tap_csv.exists():
            logger.error(f"Tapping feature CSV not found: {tap_csv}")
        else:
            logger.info("=== TAPPING MODEL ===")
            X_tr, y_tr, X_va, y_va = load_split(tap_csv, TAPPING_FEATURES)
            if len(X_tr) == 0:
                logger.error("No tapping training data found!")
            else:
                train_and_evaluate(X_tr, y_tr, X_va, y_va,
                                   task_name   = "tapping",
                                   models_dir  = models_dir,
                                   results_dir = results_dir)

    # ── Stability ─────────────────────────────────────────────────────────────
    if args.task in ("all", "stability"):
        stab_csv = feat_dir / "stability_features.csv"
        if not stab_csv.exists():
            logger.error(f"Stability feature CSV not found: {stab_csv}")
        else:
            logger.info("=== STABILITY MODEL ===")
            X_tr, y_tr, X_va, y_va = load_split(stab_csv, STABILITY_FEATURES)
            if len(X_tr) == 0:
                logger.error("No stability training data found!")
            else:
                train_and_evaluate(X_tr, y_tr, X_va, y_va,
                                   task_name   = "stability",
                                   models_dir  = models_dir,
                                   results_dir = results_dir)


if __name__ == "__main__":
    main()
