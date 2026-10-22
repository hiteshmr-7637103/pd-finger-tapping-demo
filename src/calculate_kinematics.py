"""
Stage 2 – Kinematic Feature Engineering
=========================================
Converts per-frame MediaPipe landmark CSVs into one tabular feature row
per video, implementing the clinically interpretable feature set from:

  Zarrat Ehsan et al. (npj Parkinson's Disease, 2026) [SOTA]:
    • Hypokinesia   : average_amplitude
    • Bradykinesia  : avg_cycle_duration, avg_cycle_max_speed (CMS), avg_cycle_avg_speed (CAS)
    • Sequence effect: amplitude_slope, speed_slope, cycle_duration_slope
    • Hesitation-halts: cv_amplitude, cv_cycle_duration, cv_cms, cv_cas, n_interruptions

  Guarín et al. (npj Parkinson's Disease, 2025):
    • amplitude_decrement_ratio (last_seg_amp / first_seg_amp)
    • speed_decrement_ratio     (last_seg_speed / first_seg_speed)
    • tapping_frequency         (dominant FFT peak)

  Williams et al. (PhD thesis, 2023, Chapter 4):
    • peak_to_peak_variability  (std of inter-tap intervals)
    • jitter_ratio              (#maxima / #minima)
    • rhythm_power_score        (power of dominant + ±0.2Hz neighbours)

For stability videos the CoM trajectory is used (analogous to CoP in Jung et al. 2024):
    • com_sway_range_x/y, com_path_length, com_mean_velocity
    • com_std_x/y (sway amplitude proxies)
    • step_asymmetry           (|left_ankle_x - right_ankle_x| variance)

Output:  data/features/tapping_features.csv
         data/features/stability_features.csv
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Signal utilities ─────────────────────────────────────────────────────────

def _lowpass(signal: np.ndarray, fps: float, cutoff: float = 10.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (same cutoff as Guarín et al.)."""
    nyq = fps / 2.0
    cutoff = min(cutoff, nyq * 0.95)
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def _remove_nan_gaps(signal: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN sections (hand not detected)."""
    nans = np.isnan(signal)
    if nans.any():
        idx   = np.arange(len(signal))
        valid = ~nans
        if valid.sum() < 4:
            return np.zeros(len(signal))
        signal = np.interp(idx, idx[valid], signal[valid])
    return signal


def _detect_peaks_minima(signal: np.ndarray):
    """Return indices of maxima and minima using scipy find_peaks."""
    peaks, _  = find_peaks(signal,  prominence=0.05)
    troughs,_ = find_peaks(-signal, prominence=0.05)
    return peaks, troughs


def _compute_cycle_stats(signal: np.ndarray, fps: float):
    """
    Compute per-cycle statistics from the normalised distance signal.

    Returns dict with lists of per-cycle:
      amplitudes, durations (s), max_speeds, avg_speeds
    Plus peaks/troughs indices.
    """
    peaks, troughs = _detect_peaks_minima(signal)

    amplitudes, durations, max_speeds, avg_speeds = [], [], [], []

    # Pair consecutive peak–trough–peak = one full cycle
    all_events = sorted([(i, "p") for i in peaks] + [(i, "t") for i in troughs])
    prev_peak  = None

    for idx, etype in all_events:
        if etype == "p":
            if prev_peak is not None:
                start, end = prev_peak, idx
                dur  = (end - start) / fps
                if dur < 0.1 or dur > 3.0:  # physiologically impossible
                    prev_peak = idx
                    continue
                seg  = signal[start:end+1]
                amp  = float(np.max(seg) - np.min(seg))
                # Cycle Average Speed (CAS): amplitude / cycle_duration
                spd  = amp / dur
                # Cycle Max Speed (CMS): 95th percentile of frame-wise speeds
                # Zarrat Ehsan et al. 2026, eq. 11 — robust to keypoint jitter
                frame_speeds = np.abs(np.diff(seg)) * fps
                peak_speed   = float(np.percentile(frame_speeds, 95)) if len(frame_speeds) > 0 else 0.0
                amplitudes.append(amp)
                durations.append(dur)
                max_speeds.append(peak_speed)
                avg_speeds.append(spd)
            prev_peak = idx

    return {
        "amplitudes"  : amplitudes,
        "durations"   : durations,
        "max_speeds"  : max_speeds,
        "avg_speeds"  : avg_speeds,
        "peaks"       : peaks,
        "troughs"     : troughs,
    }


def _slope(values: list) -> float:
    """Linear regression slope (normalised index → value). Returns 0 if < 3 points."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float) / len(values)
    return float(np.polyfit(x, values, 1)[0])


def _cv(values: list) -> float:
    """Coefficient of variation (std / mean). 0 if mean ≈ 0."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    m = np.mean(arr)
    return float(np.std(arr) / m) if m > 1e-9 else 0.0


def _tapping_freq_fft(signal: np.ndarray, fps: float) -> float:
    """Dominant tapping frequency (Hz) via FFT."""
    N    = len(signal)
    fft  = np.abs(np.fft.rfft(signal - signal.mean()))
    freq = np.fft.rfftfreq(N, d=1.0/fps)
    band = (freq >= 0.5) & (freq <= 6.0)
    if not band.any():
        return 0.0
    return float(freq[band][np.argmax(fft[band])])


def _rhythm_power(signal: np.ndarray, fps: float, bandwidth: float = 0.2) -> float:
    """
    Williams thesis (Study 2): power of the dominant frequency peak ± bandwidth.
    A more regular tapping concentrates power in a narrow band.
    """
    N    = len(signal)
    fft  = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    freq = np.fft.rfftfreq(N, d=1.0/fps)
    f0   = _tapping_freq_fft(signal, fps)
    if f0 == 0:
        return 0.0
    band = (freq >= f0 - bandwidth) & (freq <= f0 + bandwidth)
    return float(fft[band].sum() / (fft.sum() + 1e-12))


def _n_interruptions(cycle_durations: list) -> int:
    """
    Count 'interruption' cycles following Zarrat Ehsan et al. 2026 (§Methods):
      A cycle is an interruption if its duration > 2 × median(all cycle durations).
    This is more robust than a fixed-time threshold.
    """
    if len(cycle_durations) < 2:
        return 0
    arr       = np.array(cycle_durations, dtype=float)
    threshold = 2.0 * float(np.median(arr))
    return int(np.sum(arr > threshold))


def _decrement_ratio(values: list, n_segments: int = 3) -> float:
    """
    Guarín et al. 2025: ratio of last-segment mean to first-segment mean.
    > 1 indicates decrement (sequence effect).
    """
    if len(values) < n_segments:
        return 1.0
    arr  = np.array(values, dtype=float)
    sz   = len(arr) // n_segments
    if sz == 0:
        return 1.0
    first = arr[:sz].mean()
    last  = arr[-sz:].mean()
    return float(last / first) if first > 1e-9 else 1.0


# ─── Tapping feature extractor ────────────────────────────────────────────────

def calc_tapping_features(csv_path: Path, fps: float = 30.0, label: int = -1) -> dict:
    """
    Compute the full 15-feature vector for one finger-tapping video CSV.
    Returns a flat dict of floats + metadata.
    """
    df = pd.read_csv(csv_path)
    if "norm_dist" not in df.columns:
        raise ValueError(f"Missing 'norm_dist' column in {csv_path}")

    raw = df["norm_dist"].values.astype(float)
    raw = _remove_nan_gaps(raw)
    if len(raw) < 10:
        raise ValueError(f"Too few valid frames in {csv_path}")

    # Low-pass filter (10 Hz, per Guarín et al.)
    sig = _lowpass(raw, fps, cutoff=10.0)

    # ── Cycle-level stats ─────────────────────────────────────────────────────
    cycles = _compute_cycle_stats(sig, fps)
    amps   = cycles["amplitudes"]
    durs   = cycles["durations"]
    cms    = cycles["max_speeds"]
    cas    = cycles["avg_speeds"]
    peaks  = cycles["peaks"]
    troughs= cycles["troughs"]

    def safe_mean(lst):  return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):   return float(np.std(lst))  if lst else 0.0

    # ── Hypokinesia ───────────────────────────────────────────────────────────
    average_amplitude   = safe_mean(amps)

    # ── Bradykinesia ──────────────────────────────────────────────────────────
    avg_cycle_duration  = safe_mean(durs)
    avg_cycle_max_speed = safe_mean(cms)
    avg_cycle_avg_speed = safe_mean(cas)

    # ── Sequence effect ───────────────────────────────────────────────────────
    amplitude_slope      = _slope(amps)
    speed_slope          = _slope(cas)
    cycle_duration_slope = _slope(durs)
    amplitude_decrement  = _decrement_ratio(amps)
    speed_decrement      = _decrement_ratio(cas)

    # ── Hesitation-halts ──────────────────────────────────────────────────────
    cv_amplitude         = _cv(amps)
    cv_cycle_duration    = _cv(durs)
    cv_cms               = _cv(cms)
    cv_cas               = _cv(cas)
    n_interruptions      = _n_interruptions(durs)

    # ── Frequency / rhythm (Williams thesis) ──────────────────────────────────
    tapping_frequency    = _tapping_freq_fft(sig, fps)
    rhythm_power         = _rhythm_power(sig, fps)

    # ── Ratio maxima:minima jitter (Williams) ─────────────────────────────────
    jitter_ratio = (len(peaks) / float(len(troughs))) if len(troughs) > 0 else 1.0

    # ── Inter-tap interval std (peak-to-peak variability) ────────────────────
    if len(peaks) >= 2:
        ipi = np.diff(peaks) / fps   # inter-peak intervals in seconds
        peak_to_peak_var = float(np.std(ipi))
    else:
        peak_to_peak_var = 0.0

    return {
        # meta
        "video_id"            : csv_path.stem,
        "label"               : label,
        "n_cycles"            : len(amps),
        # hypokinesia
        "average_amplitude"   : average_amplitude,
        # bradykinesia
        "avg_cycle_duration"  : avg_cycle_duration,
        "avg_cycle_max_speed" : avg_cycle_max_speed,
        "avg_cycle_avg_speed" : avg_cycle_avg_speed,
        # sequence effect
        "amplitude_slope"     : amplitude_slope,
        "speed_slope"         : speed_slope,
        "cycle_duration_slope": cycle_duration_slope,
        "amplitude_decrement" : amplitude_decrement,
        "speed_decrement"     : speed_decrement,
        # hesitation-halts
        "cv_amplitude"        : cv_amplitude,
        "cv_cycle_duration"   : cv_cycle_duration,
        "cv_cms"              : cv_cms,
        "cv_cas"              : cv_cas,
        "n_interruptions"     : float(n_interruptions),
        # frequency / rhythm
        "tapping_frequency"   : tapping_frequency,
        "rhythm_power"        : rhythm_power,
        # variability
        "jitter_ratio"        : jitter_ratio,
        "peak_to_peak_var"    : peak_to_peak_var,
    }


# ─── Stability feature extractor ──────────────────────────────────────────────

def calc_stability_features(csv_path: Path, fps: float = 30.0, label: int = -1) -> dict:
    """
    Compute postural-stability features from a pose landmark CSV.
    Uses CoM trajectory as a surrogate for center-of-pressure (Jung et al. 2024).
    """
    df = pd.read_csv(csv_path)

    if "com_x" not in df.columns:
        raise ValueError(f"Missing 'com_x' column in {csv_path}")

    com_x = _remove_nan_gaps(df["com_x"].values.astype(float))
    com_y = _remove_nan_gaps(df["com_y"].values.astype(float))

    # Low-pass filter at 10 Hz
    com_x_f = _lowpass(com_x, fps, cutoff=10.0)
    com_y_f = _lowpass(com_y, fps, cutoff=10.0)

    # ── Time-domain ──────────────────────────────────────────────────────────
    com_std_x      = float(np.std(com_x_f))
    com_std_y      = float(np.std(com_y_f))
    com_range_x    = float(np.ptp(com_x_f))   # peak-to-peak (sway range)
    com_range_y    = float(np.ptp(com_y_f))

    # ── Path length & velocity ────────────────────────────────────────────────
    dx = np.diff(com_x_f)
    dy = np.diff(com_y_f)
    step_lengths   = np.sqrt(dx**2 + dy**2)
    com_path_len   = float(step_lengths.sum())
    com_mean_vel   = float(step_lengths.mean()) * fps if len(step_lengths) else 0.0

    # ── Frequency-domain (Jung et al. 2024 – high-freq power ≥3.5 Hz) ────────
    def _freq_power(sig, fps_):
        N    = len(sig)
        fft  = np.abs(np.fft.rfft(sig - sig.mean())) ** 2
        freq = np.fft.rfftfreq(N, d=1.0/fps_)
        total = fft.sum() + 1e-12
        return {
            "total_power"     : float(fft.sum()),
            "high_freq_power" : float(fft[freq >= 3.5].sum() / total),
            "low_freq_power"  : float(fft[freq < 1.0].sum() / total),
        }

    fp_x = _freq_power(com_x_f, fps)
    fp_y = _freq_power(com_y_f, fps)

    # ── Nonlinear – using Poincaré SD concepts ────────────────────────────────
    def _poincare_sd(sig):
        if len(sig) < 2:
            return 0.0, 0.0
        rr1 = sig[:-1]
        rr2 = sig[1:]
        diff_ = rr2 - rr1
        summ_ = rr2 + rr1
        sd1 = float(np.std(diff_) / np.sqrt(2))
        sd2 = float(np.std(summ_) / np.sqrt(2))
        return sd1, sd2

    sd1_x, sd2_x = _poincare_sd(com_x_f)
    sd1_y, sd2_y = _poincare_sd(com_y_f)

    # ── 95% confidence ellipse area (Jung et al. 2024) ────────────────────────
    cov  = np.cov(com_x_f, com_y_f)
    try:
        eigenvals = np.linalg.eigvals(cov)
        eigenvals = np.abs(eigenvals)
        chi2_95   = 5.991   # chi² with 2 DOF at 95%
        chi2_99   = 9.210
        rma95 = float(np.sqrt(chi2_95 * eigenvals.max()))
        rmi95 = float(np.sqrt(chi2_95 * eigenvals.min()))
        rma99 = float(np.sqrt(chi2_99 * eigenvals.max()))
        rmi99 = float(np.sqrt(chi2_99 * eigenvals.min()))
        area95 = float(np.pi * rma95 * rmi95)
        area99 = float(np.pi * rma99 * rmi99)
    except Exception:
        rma95 = rmi95 = rma99 = rmi99 = area95 = area99 = 0.0

    # ── Ankle asymmetry ───────────────────────────────────────────────────────
    for col in ["left_ankle_x", "right_ankle_x"]:
        if col not in df.columns:
            df[col] = 0.0
    la_x = _remove_nan_gaps(df["left_ankle_x"].values.astype(float))
    ra_x = _remove_nan_gaps(df["right_ankle_x"].values.astype(float))
    ankle_asym_std  = float(np.std(la_x - ra_x))
    ankle_asym_mean = float(np.mean(np.abs(la_x - ra_x)))

    return {
        "video_id"           : csv_path.stem,
        "label"              : label,
        # time-domain
        "com_std_x"          : com_std_x,
        "com_std_y"          : com_std_y,
        "com_range_x"        : com_range_x,
        "com_range_y"        : com_range_y,
        "com_path_len"       : com_path_len,
        "com_mean_vel"       : com_mean_vel,
        # frequency-domain
        "x_total_power"      : fp_x["total_power"],
        "x_high_freq_power"  : fp_x["high_freq_power"],
        "x_low_freq_power"   : fp_x["low_freq_power"],
        "y_total_power"      : fp_y["total_power"],
        "y_high_freq_power"  : fp_y["high_freq_power"],
        "y_low_freq_power"   : fp_y["low_freq_power"],
        # nonlinear Poincaré
        "sd1_x"              : sd1_x,
        "sd2_x"              : sd2_x,
        "sd1_y"              : sd1_y,
        "sd2_y"              : sd2_y,
        # ellipse
        "rma95"              : rma95,
        "rmi95"              : rmi95,
        "area95"             : area95,
        "rmi99"              : rmi99,
        "area99"             : area99,
        # ankle
        "ankle_asym_std"     : ankle_asym_std,
        "ankle_asym_mean"    : ankle_asym_mean,
    }


# ─── Batch processors ─────────────────────────────────────────────────────────

def _read_manifest(landmark_dir: Path):
    """Load manifest.json → {vid_id: {label, fps, ...}}"""
    mpath = landmark_dir / "manifest.json"
    if not mpath.exists():
        return {}
    return json.loads(mpath.read_text())


def build_tapping_dataset(landmark_base: Path, out_csv: Path, fps_default: float = 30.0):
    """
    Collect feature rows from left_train / left_val / right_train / right_val,
    tagging each row with hand + split.
    """
    rows = []
    for hand in ("left", "right"):
        for split in ("train", "val"):
            lm_dir   = landmark_base / "hand" / f"{hand}_{split}"
            manifest = _read_manifest(lm_dir)

            for vid_id, meta in manifest.items():
                label    = int(meta.get("label", -1))
                fps      = float(meta.get("fps", fps_default)) or fps_default
                csv_path = lm_dir / f"{vid_id}.csv"
                if not csv_path.exists():
                    continue
                try:
                    feats = calc_tapping_features(csv_path, fps=fps, label=label)
                    feats["hand"]  = hand
                    feats["split"] = split
                    rows.append(feats)
                    logger.info(f"  [tap] {hand}/{split}/{vid_id}  label={label}")
                except Exception as e:
                    logger.warning(f"  [tap SKIP] {vid_id}: {e}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Tapping features saved → {out_csv}  ({len(df)} rows)")
    return df


def build_stability_dataset(landmark_base: Path, out_csv: Path, fps_default: float = 30.0):
    """Collect stability feature rows from train + val."""
    rows = []
    for split in ("train", "val"):
        lm_dir   = landmark_base / "pose" / split
        manifest = _read_manifest(lm_dir)

        for vid_id, meta in manifest.items():
            label    = int(meta.get("label", -1))
            fps      = float(meta.get("fps", fps_default)) or fps_default
            csv_path = lm_dir / f"{vid_id}.csv"
            if not csv_path.exists():
                continue
            try:
                feats = calc_stability_features(csv_path, fps=fps, label=label)
                feats["split"] = split
                rows.append(feats)
                logger.info(f"  [stab] {split}/{vid_id}  label={label}")
            except Exception as e:
                logger.warning(f"  [stab SKIP] {vid_id}: {e}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Stability features saved → {out_csv}  ({len(df)} rows)")
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute kinematic features from landmark CSVs.")
    parser.add_argument("--landmark_dir", default="data/landmarks")
    parser.add_argument("--out_dir",      default="data/features")
    parser.add_argument("--task",         choices=["all","tapping","stability"], default="all")
    parser.add_argument("--fps",          type=float, default=30.0)
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    lm_base = Path(args.landmark_dir)
    out_dir = Path(args.out_dir)

    if args.task in ("all", "tapping"):
        build_tapping_dataset(lm_base, out_dir / "tapping_features.csv", fps_default=args.fps)

    if args.task in ("all", "stability"):
        build_stability_dataset(lm_base, out_dir / "stability_features.csv", fps_default=args.fps)


if __name__ == "__main__":
    main()
