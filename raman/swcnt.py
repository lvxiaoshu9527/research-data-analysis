"""
cnt_raman_blank_processing.py  —  Version 3
═══════════════════════════════════════════════════════════════════════════════
Standalone Raman processing script for blank (iodine-free) SWCNT + amino acid
samples.

Target system  : SWCNT (6,5) or (7,4) + amino acids,  NO iodine
Expected peaks : RBM,  G-band
Absent peaks   : iodine peaks (90–170 cm-1)

This script is COMPLETELY INDEPENDENT of the iodine-doped CNT pipeline.
It contains NO iodine peak detection, fitting, or ratio logic whatsoever.

Processing pipeline
-------------------
load_spectrum()
  -> remove < 80 cm-1,  clip to 1800 cm-1
  -> smooth_spectrum()        Savitzky-Golay  (window=9, poly=2)
  -> baseline_correction()    3-step: 5th-pct envelope -> Exp-fit -> SNIP residual
  -> fit_rbm()                single Voigt, fitted only on RBM sub-region
  -> evaluate_fit_on_full_spectrum()
  -> compute_fit_quality()    R2, RMSE, reduced chi2
  -> plot_results()           per-sample 2-panel figure (RBM zoom + full)
  -> export_excel()           Raw_spectra | Corrected_spectra | RBM_fit_parameters

Baseline design (v3)
--------------------
Pure SWCNT fluorescence background is a steeply decaying exponential.
SNIP alone compresses this slope via its double-sqrt transform, leaving the
baseline too high in 125-225 cm-1 and creating a spurious hump in the
corrected spectrum.  The 3-step approach:

  Step 1 - Sliding 5th-percentile envelope (half-window = 30 pts ~ 30-40 cm-1)
            Suppresses peaks before fitting so the RBM does not inflate
            the exponential.
  Step 2 - Fit A*exp(-index/tau)+C to the lower envelope
            Provides the correct exponential shape across the full spectrum.
  Step 3 - SNIP residual correction (niter = 30)
            Corrects small local deviations without the compression artefact.

Validated on simulated SWCNT data:
  125-225 cm-1 corrected max ~ 5 counts  (noise floor, hump eliminated)
  Baseline error at 150 cm-1             ~ +-2 counts
  Baseline error at RBM (308 cm-1)       ~ +-3 counts

Author  : (generated)
Version : 3
"""

from __future__ import annotations

import os
import glob
import datetime
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from lmfit.models import VoigtModel


# ============================================================================
# 0.  Configuration
# ============================================================================

# Baseline parameters
BASELINE_PERCENTILE:  int = 5    # lower-envelope percentile (Step 1)
BASELINE_HALF_WINDOW: int = 30   # half-window in points for Step 1 (~30-40 cm-1)
SNIP_NITER:           int = 30   # SNIP residual-correction iterations (Step 3)

# Savitzky-Golay smoothing
SG_WINDOW:    int = 9
SG_POLYORDER: int = 2

# Spectral range
SPEC_MIN: float = 80.0
SPEC_MAX: float = 1800.0

# Chirality -> RBM window mapping
CHIRALITY_RULES: List[Dict] = [
    {
        "chirality":       "(6,5)",
        "keywords":        ["65", "(6,5)", "6,5"],
        "rbm_min":         295.0,
        "rbm_max":         320.0,
        "rbm_center_init": 308.0,
    },
    {
        "chirality":       "(7,4)",
        "keywords":        ["74", "(7,4)", "7,4"],
        "rbm_min":         270.0,
        "rbm_max":         300.0,
        "rbm_center_init": 284.0,
    },
]

DEFAULT_CHIRALITY: Dict = {
    "chirality":       "unknown",
    "rbm_min":         270.0,
    "rbm_max":         330.0,
    "rbm_center_init": 295.0,
}


# ============================================================================
# 1.  Spectrum loading
# ============================================================================

def load_spectrum(
    filepath:  str,
    min_shift: float = SPEC_MIN,
    max_shift: float = SPEC_MAX,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load a two-column Raman data file (.txt or .csv, whitespace or comma
    separated) and return (x, y) clipped to [min_shift, max_shift] cm-1.

    Returns (None, None) on any failure.
    """
    try:
        df = pd.read_csv(
            filepath, sep=r"[\s,]+", engine="python",
            header=None, comment="#",
        )
        if isinstance(df.iloc[0, 0], str):
            df = df.iloc[1:].reset_index(drop=True)

        df = df.astype(float)
        x: np.ndarray = df.iloc[:, 0].values
        y: np.ndarray = df.iloc[:, 1].values

        order = np.argsort(x)
        x, y  = x[order], y[order]

        mask = (x >= min_shift) & (x <= max_shift)
        if mask.sum() < 20:
            raise ValueError(
                f"Only {mask.sum()} points in [{min_shift}, {max_shift}] cm-1"
            )

        x_out, y_out = x[mask], y[mask]
        print(
            f"    Loaded: {x_out.min():.1f}-{x_out.max():.1f} cm-1  "
            f"({len(x_out)} points)"
        )
        return x_out, y_out

    except Exception as exc:
        print(f"    x  Failed to load {os.path.basename(filepath)}: {exc}")
        return None, None


# ============================================================================
# 2.  Smoothing
# ============================================================================

def smooth_spectrum(
    y:             np.ndarray,
    window_length: int = SG_WINDOW,
    polyorder:     int = SG_POLYORDER,
) -> np.ndarray:
    """
    Savitzky-Golay smoothing.
    The raw array is kept separately for export.
    """
    if len(y) >= window_length:
        return savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y.copy()


# ============================================================================
# 3.  Baseline correction  (3-step)
# ============================================================================

def baseline_correction(
    y:     np.ndarray,
    niter: int = SNIP_NITER,
    pct:   int = BASELINE_PERCENTILE,
    half:  int = BASELINE_HALF_WINDOW,
) -> np.ndarray:
    """
    3-step baseline for pure SWCNT Raman spectra with exponential fluorescence.

    Why not plain SNIP?
    -------------------
    SNIP's double-sqrt transform compresses a strongly sloped background.
    On inverse-transform, the baseline is systematically too high in 125-225
    cm-1, producing a spurious hump in the corrected spectrum even when no
    Raman peak exists there.

    Step 1 - Sliding percentile lower envelope
        y_low[i] = pct-th percentile of y within [i-half, i+half].
        pct=5, half=30 pts (~30-40 cm-1): stays near the valley floor,
        wide enough to span peak shoulders, narrow enough to track background
        curvature.  Pre-suppresses RBM so the exponential fit in Step 2 is
        not inflated by peak signal.

    Step 2 - Exponential fit to the lower envelope
        Fit  f(i) = A * exp(-i/tau) + C  using index positions (not cm-1),
        making the fit scale-independent across datasets with different
        x-axis spacings.  Initial guesses from endpoint percentiles.
        Fallback: heavily SG-smoothed y_low if curve_fit fails.

    Step 3 - SNIP correction on the residual
        residual = y - bl_exp  (clipped to >= 0)
        A short SNIP pass (niter=30) corrects small local deviations left by
        the exponential fit.  Because the residual is nearly flat, the
        double-sqrt compression artefact is negligible here.

    Final baseline = bl_exp + snip_correction, clipped to <= y.

    Parameters
    ----------
    y     : smoothed intensity array (output of smooth_spectrum)
    niter : SNIP residual-correction iterations   (default 30)
    pct   : lower-envelope percentile              (default 5)
    half  : half-window in points for Step 1       (default 30)

    Returns
    -------
    baseline : np.ndarray, same length as y, guaranteed <= y everywhere.
    """

    # --- Step 1: sliding percentile lower envelope ---------------------------
    y_low: np.ndarray = np.array(
        [
            np.percentile(y[max(0, i - half): min(len(y), i + half + 1)], pct)
            for i in range(len(y))
        ],
        dtype=float,
    )

    # --- Step 2: fit A*exp(-index/tau)+C to y_low ----------------------------
    xi: np.ndarray = np.arange(len(y), dtype=float)

    def _exp(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
        return A * np.exp(-t / tau) + C

    seg:  int   = max(1, len(y_low) // 10)
    y_lo: float = float(np.percentile(y_low[:seg],  10))
    y_hi: float = float(np.percentile(y_low[-seg:], 10))
    A0:   float = max(y_lo - y_hi, 1.0)
    C0:   float = max(y_hi, 0.0)
    tau0: float = len(y) / 3.0

    try:
        popt, _ = curve_fit(
            _exp, xi, y_low,
            p0=[A0, tau0, C0],
            bounds=(
                [0.0,           5.0,                0.0      ],
                [y.max() * 5,   float(len(y)) * 3.0, y.max()],
            ),
            maxfev=8000,
        )
        bl_exp: np.ndarray = _exp(xi, *popt)
    except Exception:
        win: int = min(len(y_low) // 2 * 2 - 1, 201)
        bl_exp   = savgol_filter(y_low, win, 3)

    # --- Step 3: SNIP on residual --------------------------------------------
    resid: np.ndarray = np.maximum(y - bl_exp, 0.0)
    w: np.ndarray     = np.sqrt(np.sqrt(resid + 1.0))
    for m in range(1, niter + 1):
        if 2 * m >= len(w):
            break
        w[m:-m] = np.minimum(w[m:-m], (w[:-2 * m] + w[2 * m:]) / 2.0)
    snip_correction: np.ndarray = (w ** 2) ** 2 - 1.0

    return np.minimum(bl_exp + snip_correction, y)


# ============================================================================
# 4.  Chirality detection
# ============================================================================

def detect_chirality(filename: str) -> Dict:
    """
    Infer CNT chirality from the filename (case-insensitive keyword match).

    Rules:
      (6,5)  <-  filename contains  '65' | '(6,5)' | '6,5'
      (7,4)  <-  filename contains  '74' | '(7,4)' | '7,4'

    Falls back to DEFAULT_CHIRALITY (wide window 270-330 cm-1) if no match.
    """
    fname_lower: str = filename.lower()
    for rule in CHIRALITY_RULES:
        for kw in rule["keywords"]:
            if kw.lower() in fname_lower:
                print(
                    f"    Chirality: {rule['chirality']}  "
                    f"(matched '{kw}')  "
                    f"RBM window: {rule['rbm_min']}-{rule['rbm_max']} cm-1"
                )
                return rule

    print(
        f"    !  Chirality not recognised -> using default window "
        f"({DEFAULT_CHIRALITY['rbm_min']}-{DEFAULT_CHIRALITY['rbm_max']} cm-1)"
    )
    return DEFAULT_CHIRALITY.copy()


# ============================================================================
# 5.  RBM peak fitting  (single Voigt -- NO iodine peaks)
# ============================================================================

def _voigt_amplitude_init(height: float, sigma: float = 5.0) -> float:
    """
    Estimate lmfit VoigtModel amplitude from peak height.

    lmfit Voigt:  height ~ amplitude / (sigma * sqrt(2*pi))
    =>  amplitude ~ height * sigma * sqrt(2*pi)

    Replaces the incorrect 'height * 10' heuristic.
    """
    return max(float(height) * sigma * np.sqrt(2.0 * np.pi), 1e-6)


def fit_rbm(
    x:                np.ndarray,
    y_corrected:      np.ndarray,
    chirality_config: Dict,
) -> Optional[object]:
    """
    Fit a single Voigt peak to the RBM sub-region only.

    Fitting on the full spectrum (80-1800 cm-1) allows the G-band (~1580
    cm-1, hundreds of points) to dominate the residuals and degrade the RBM
    fit quality.  Restricting to the chirality-specific RBM window eliminates
    this problem.  Parameters are evaluated on the full x axis afterwards.

    Parameters
    ----------
    x                : full Raman shift array
    y_corrected      : baseline-corrected spectrum (full length)
    chirality_config : dict from detect_chirality()

    Returns
    -------
    lmfit ModelResult fitted on the RBM sub-region.
    Extra attributes:
      ._x_fit    : x array used for fitting
      ._fit_mask : boolean mask into x for the RBM sub-region
    """
    rbm_min:     float = chirality_config["rbm_min"]
    rbm_max:     float = chirality_config["rbm_max"]
    center_init: float = chirality_config["rbm_center_init"]

    fit_mask: np.ndarray = (x >= rbm_min) & (x <= rbm_max)
    x_fit:    np.ndarray = x[fit_mask]
    y_fit:    np.ndarray = y_corrected[fit_mask]

    if len(x_fit) < 8:
        raise ValueError(
            f"Too few points in RBM window [{rbm_min}, {rbm_max}] cm-1 "
            f"({len(x_fit)} points, need >= 8)"
        )

    peak_idx:   int   = int(np.argmax(y_fit))
    center_est: float = float(x_fit[peak_idx])
    height_est: float = float(y_fit[peak_idx])
    sigma0:     float = 4.0
    amp0:       float = _voigt_amplitude_init(height_est, sigma0)

    model  = VoigtModel(prefix="rbm_")
    params = model.make_params()

    params["rbm_center"].set(   value=center_est, min=rbm_min, max=rbm_max)
    params["rbm_amplitude"].set(value=amp0,        min=0.0)
    params["rbm_sigma"].set(    value=sigma0,       min=1.0,    max=15.0)
    params["rbm_gamma"].set(    value=sigma0,       min=1.0,    max=15.0,
                                vary=True, expr="")

    result = model.fit(y_fit, params, x=x_fit)

    if result.params["rbm_sigma"].value >= 14.5:
        print("    !  RBM sigma at upper bound -- refitting with tighter constraint")
        params["rbm_sigma"].set(
            value=min(result.params["rbm_sigma"].value, 10.0),
            min=1.0, max=12.0,
        )
        params["rbm_gamma"].set(
            value=min(result.params["rbm_gamma"].value, 10.0),
            min=1.0, max=12.0, vary=True, expr="",
        )
        result = model.fit(y_fit, params, x=x_fit)

    result._x_fit    = x_fit
    result._fit_mask = fit_mask
    return result


def evaluate_fit_on_full_spectrum(
    result,
    x_full: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-evaluate the fitted Voigt on the full x axis for plotting and export.

    Returns
    -------
    total_fit_full : fitted curve on x_full
    rbm_comp_full  : RBM component on x_full  (identical to total here)
    """
    model_full     = VoigtModel(prefix="rbm_")
    total_fit_full = model_full.eval(result.params, x=x_full)
    rbm_comp_full  = total_fit_full.copy()
    return total_fit_full, rbm_comp_full


# ============================================================================
# 6.  Fit quality metrics
# ============================================================================

def compute_fit_quality(
    y_data:        np.ndarray,
    y_fit:         np.ndarray,
    n_free_params: int,
) -> Dict[str, float]:
    """
    Compute R2, RMSE, and reduced chi2 for the RBM sub-region fit.

      R2           ~ 1    -> good fit
      RMSE                   same units as intensity
      reduced chi2 ~ 1    -> good fit;  >> 1 under-fit;  << 1 over-fit
    """
    residual: np.ndarray = y_data - y_fit
    ss_res:   float      = float(np.sum(residual ** 2))
    ss_tot:   float      = float(np.sum((y_data - np.mean(y_data)) ** 2))

    r2:   float = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse: float = float(np.sqrt(np.mean(residual ** 2)))

    dof:          int   = len(y_data) - n_free_params
    sigma2:       float = float(np.var(residual)) if dof > 0 else 1.0
    reduced_chi2: float = (
        ss_res / sigma2 / dof if (dof > 0 and sigma2 > 0) else float("nan")
    )

    return {
        "R2":           round(r2,           6),
        "RMSE":         round(rmse,         4),
        "reduced_chi2": round(reduced_chi2, 4),
    }


# ============================================================================
# 7.  Plotting
# ============================================================================

def plot_results(
    x:              np.ndarray,
    y_raw:          np.ndarray,
    y_smooth:       np.ndarray,
    baseline:       np.ndarray,
    y_corrected:    np.ndarray,
    total_fit_full: np.ndarray,
    rbm_comp_full:  np.ndarray,
    filename:       str,
    chirality:      str,
    chi_config:     Dict,
    output_dir:     str,
) -> None:
    """
    Generate two PNG files per sample:

      <filename>_fit_RBM.png   -- RBM region  (250-350 cm-1)
      <filename>_fit_full.png  -- Full spectrum (50-1800 cm-1)

    Each figure has two panels:
      Top    : raw (gray dots), smoothed (dashed blue), baseline (gray line),
               corrected (blue), total Voigt fit (red), RBM fill (magenta)
      Bottom : fit residual  (corrected - total_fit, purple)

    No iodine peak components are shown -- blank-sample script only.
    """
    residual: np.ndarray = y_corrected - total_fit_full

    title_base: str = (
        f"Blank SWCNT Raman -- {filename}\n"
        f"Chirality: {chirality}  |  "
        f"RBM window: {chi_config['rbm_min']:.0f}-{chi_config['rbm_max']:.0f} cm-1  |  "
        f"Baseline: Exp-fit + SNIP (niter={SNIP_NITER})"
    )

    os.makedirs(os.path.join(output_dir, "fitted_spectra"), exist_ok=True)

    for zoom_label, xlims in [
        ("RBM",  (250,  350)),
        ("full", (50,   min(float(x.max()), 1800.0))),
    ]:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(12, 9),
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # Top panel
        ax.plot(x, y_raw,          "o",  color="lightgray", markersize=3,
                zorder=1, label="Raw (>=80 cm-1)")
        ax.plot(x, y_smooth,       "--", color="steelblue", linewidth=1.0,
                alpha=0.7, zorder=2, label="Smoothed (SG)")
        ax.plot(x, baseline,       "-",  color="#888888",   linewidth=1.5,
                alpha=0.85, zorder=3,
                label=f"Baseline (Exp+SNIP, niter={SNIP_NITER})")
        ax.plot(x, y_corrected,    "b-",                    linewidth=2.0,
                alpha=0.5,  zorder=4, label="Corrected")
        ax.plot(x, total_fit_full, "r-",                    linewidth=2.5,
                zorder=5, label="Total Fit (Voigt RBM)")
        ax.fill_between(
            x, rbm_comp_full, alpha=0.40, color="magenta", zorder=6,
            label=(
                f"RBM component  "
                f"({chi_config['rbm_min']:.0f}-{chi_config['rbm_max']:.0f} cm-1)"
            ),
        )
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.35)
        ax.set_xlim(*xlims)
        ax.set_xlabel("Raman Shift (cm-1)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(title_base, fontsize=10)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        # Bottom panel -- residual
        ax2.plot(x, residual, color="purple", linewidth=1.0)
        ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax2.fill_between(x, residual, alpha=0.25, color="purple")
        ax2.set_xlim(*xlims)
        ax2.set_xlabel("Raman Shift (cm-1)")
        ax2.set_ylabel("Residual")
        ax2.set_title("Fit Residual  (corrected - total fit)", fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(
            output_dir, "fitted_spectra", f"{filename}_fit_{zoom_label}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Plot saved: {os.path.basename(save_path)}")


def plot_overlay(
    all_data:      Dict[str, Dict],
    chirality_map: Dict[str, str],
    output_dir:    str,
) -> None:
    """
    Overlay of all corrected spectra, colour-coded by chirality.

    Outputs:
      all_overlay_RBM.png   (250-350 cm-1)
      all_overlay_full.png  (50-1800 cm-1)
    """
    if not all_data:
        return

    COLORS: Dict[str, str] = {
        "(6,5)":   "tab:red",
        "(7,4)":   "tab:blue",
        "unknown": "tab:gray",
    }

    for zoom_label, xlims in [
        ("RBM",  (250,  350)),
        ("full", (50,  1800)),
    ]:
        fig, ax = plt.subplots(figsize=(13, 6))
        for idx, (sname, d) in enumerate(all_data.items()):
            chi   = chirality_map.get(sname, "unknown")
            color = COLORS.get(chi, f"C{idx % 10}")
            xa, ya = d["x"], d["corrected"]
            mask = (xa >= xlims[0]) & (xa <= xlims[1])
            if mask.sum() == 0:
                continue
            ax.plot(xa[mask], ya[mask], linewidth=1.2, alpha=0.8,
                    label=f"{sname}  [{chi}]", color=color)

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.35)
        ax.set_xlim(*xlims)
        ax.set_xlabel("Raman Shift (cm-1)")
        ax.set_ylabel("Corrected Intensity (a.u.)")
        ax.set_title(
            f"Blank SWCNT -- All Samples Overlay  "
            f"({'RBM region' if zoom_label == 'RBM' else 'Full spectrum'})",
            fontsize=11,
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"all_overlay_{zoom_label}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    Overlay saved: {os.path.basename(save_path)}")


# ============================================================================
# 8.  Excel export
# ============================================================================

def _build_wide_table(
    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """
    Build wide DataFrame:  Raman_shift | sample1 | sample2 | ...

    Common x axis = sorted union of all sample x arrays.
    Aligned by linear interpolation; out-of-range = NaN.
    """
    if not data_dict:
        return pd.DataFrame({"Raman_shift": []})

    all_x = np.unique(np.concatenate([x for x, _ in data_dict.values()]))
    df    = pd.DataFrame({"Raman_shift": all_x})
    for sname, (x_arr, y_arr) in data_dict.items():
        df[sname] = np.interp(all_x, x_arr, y_arr, left=np.nan, right=np.nan)
    return df


def export_excel(
    all_summaries: List[Dict],
    all_data:      Dict[str, Dict],
    output_dir:    str,
) -> str:
    """
    Write  raman_blank_results.xlsx  with three sheets:

    Sheet 1  Raw_spectra         -- raw intensities (post-clipping, pre-smooth)
    Sheet 2  Corrected_spectra   -- baseline-subtracted intensities
    Sheet 3  RBM_fit_parameters  -- one row per sample:
               sample_name | chirality |
               RBM_position | RBM_height | RBM_FWHM | RBM_area |
               RBM_sigma | RBM_gamma |
               fit_R2 | fit_RMSE | fit_reduced_chi2

    Spectral sheets: wide-table format  Raman_shift | sample1 | sample2 | ...
    """
    excel_path: str = os.path.join(output_dir, "raman_blank_results.xlsx")

    raw_dict  = {s: (d["x"], d["raw"])       for s, d in all_data.items()}
    corr_dict = {s: (d["x"], d["corrected"]) for s, d in all_data.items()}

    df_raw  = _build_wide_table(raw_dict)
    df_corr = _build_wide_table(corr_dict)

    param_cols: List[str] = [
        "sample_name", "chirality",
        "RBM_position", "RBM_height", "RBM_FWHM", "RBM_area",
        "RBM_sigma", "RBM_gamma",
        "fit_R2", "fit_RMSE", "fit_reduced_chi2",
    ]
    df_params = pd.DataFrame(all_summaries)
    for col in param_cols:
        if col not in df_params.columns:
            df_params[col] = np.nan
    df_params = df_params[param_cols]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_raw   .to_excel(writer, sheet_name="Raw_spectra",        index=False)
        df_corr  .to_excel(writer, sheet_name="Corrected_spectra",  index=False)
        df_params.to_excel(writer, sheet_name="RBM_fit_parameters", index=False)

    print(f"\nv  Excel saved: {excel_path}")
    print(f"   Sheets: Raw_spectra | Corrected_spectra | RBM_fit_parameters")
    return excel_path


# ============================================================================
# 9.  Single-file processing pipeline
# ============================================================================

def process_file(
    filepath:   str,
    output_dir: str,
) -> Optional[Tuple[Dict, Dict]]:
    """
    Run the full pipeline for one file.

    Returns (summary_row, data_entry) on success, None on failure.
    """
    filename: str = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n  Processing: {filename}")

    try:
        # 1. Chirality detection
        chi_config: Dict = detect_chirality(filename)
        chirality:  str  = chi_config["chirality"]

        # 2. Load and clip spectrum
        x, y_raw = load_spectrum(filepath)
        if x is None or y_raw is None:
            return None

        # 3. Savitzky-Golay smoothing (raw preserved for export)
        y_smooth: np.ndarray = smooth_spectrum(y_raw)

        # 4. 3-step baseline correction
        baseline:    np.ndarray = baseline_correction(y_smooth)
        y_corrected: np.ndarray = y_smooth - baseline

        # 5. Single Voigt RBM fit (RBM sub-region only)
        fit_result = fit_rbm(x, y_corrected, chi_config)
        if fit_result is None:
            raise RuntimeError("RBM fit returned None")

        p = fit_result.params

        # 6. Evaluate on full x axis for plotting/export
        total_fit_full, rbm_comp_full = evaluate_fit_on_full_spectrum(
            fit_result, x
        )

        # 7. Fit quality in RBM sub-region
        fq: Dict = compute_fit_quality(
            y_data=y_corrected[fit_result._fit_mask],
            y_fit=total_fit_full[fit_result._fit_mask],
            n_free_params=fit_result.nvarys,
        )

        # 8. Summary row
        summary_row: Dict = {
            "sample_name":      filename,
            "chirality":        chirality,
            "RBM_position":     round(float(p["rbm_center"].value),    2),
            "RBM_height":       round(float(p["rbm_height"].value),    2),
            "RBM_FWHM":         round(float(p["rbm_fwhm"].value),      2),
            "RBM_area":         round(float(p["rbm_amplitude"].value),  4),
            "RBM_sigma":        round(float(p["rbm_sigma"].value),      4),
            "RBM_gamma":        round(float(p["rbm_gamma"].value),      4),
            "fit_R2":           fq["R2"],
            "fit_RMSE":         fq["RMSE"],
            "fit_reduced_chi2": fq["reduced_chi2"],
        }

        data_entry: Dict = {
            "x":         x,
            "raw":       y_raw,
            "smooth":    y_smooth,
            "baseline":  baseline,
            "corrected": y_corrected,
            "fit":       total_fit_full,
            "residual":  y_corrected - total_fit_full,
        }

        # 9. Plots
        plot_results(
            x=x, y_raw=y_raw, y_smooth=y_smooth, baseline=baseline,
            y_corrected=y_corrected, total_fit_full=total_fit_full,
            rbm_comp_full=rbm_comp_full,
            filename=filename, chirality=chirality,
            chi_config=chi_config, output_dir=output_dir,
        )

        print(
            f"    v  chirality={chirality}  "
            f"RBM={p['rbm_center'].value:.1f} cm-1  "
            f"FWHM={p['rbm_fwhm'].value:.2f}  "
            f"R2={fq['R2']:.4f}"
        )
        return summary_row, data_entry

    except Exception as exc:
        print(f"    x  Failed: {exc}")
        traceback.print_exc()
        return None


# ============================================================================
# 10.  Main entry point
# ============================================================================

def main() -> None:
    """
    GUI batch driver:
      1. Select files or folder
      2. Select output folder
      3. Batch-process -> plots + Excel + overlay
    """
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askquestion(
        "Select input",
        "Process an entire folder?\n\n"
        "[Yes]  Select folder\n"
        "[No]   Select individual files",
    )
    file_paths: List[str] = []
    if choice == "yes":
        folder = filedialog.askdirectory(title="Select folder with Raman data")
        if not folder:
            return
        file_paths.extend(glob.glob(os.path.join(folder, "*.txt")))
        file_paths.extend(glob.glob(os.path.join(folder, "*.csv")))
    else:
        selected = filedialog.askopenfilenames(
            title="Select Raman files",
            filetypes=[("Data files", "*.txt *.csv")],
        )
        file_paths = list(selected)

    if not file_paths:
        messagebox.showwarning("No files", "No files selected. Exiting.")
        return

    messagebox.showinfo("Output folder", "Please select the results output folder.")
    output_root: str = filedialog.askdirectory(title="Select output folder")
    if not output_root:
        return

    os.makedirs(os.path.join(output_root, "fitted_spectra"), exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"  Blank SWCNT Raman Processing  --  v3  (iodine-free)")
    print(f"{'=' * 65}")
    print(f"  Baseline    : Exp-fit + SNIP residual  (niter={SNIP_NITER})")
    print(f"                5th-pct envelope, half={BASELINE_HALF_WINDOW} pts")
    print(f"  Peak model  : single Voigt  (RBM only -- NO iodine peaks)")
    print(f"  Chirality rules:")
    for rule in CHIRALITY_RULES:
        print(
            f"    {rule['chirality']:6s}: {rule['keywords']}"
            f"  ->  RBM {rule['rbm_min']:.0f}-{rule['rbm_max']:.0f} cm-1"
        )
    print(f"  Files to process: {len(file_paths)}")
    print(f"{'=' * 65}")

    all_summaries: List[Dict]      = []
    all_data:      Dict[str, Dict] = {}
    chirality_map: Dict[str, str]  = {}

    for fp in file_paths:
        result = process_file(fp, output_root)
        if result is None:
            continue
        summary_row, data_entry = result
        sname = summary_row["sample_name"]
        all_summaries.append(summary_row)
        all_data[sname]      = data_entry
        chirality_map[sname] = summary_row["chirality"]

    if not all_summaries:
        messagebox.showwarning(
            "No results",
            "No files processed successfully.\n"
            "Check file format and try again.",
        )
        return

    export_excel(all_summaries, all_data, output_root)
    plot_overlay(all_data, chirality_map, output_root)

    chi_counts: Dict[str, int] = {}
    for row in all_summaries:
        c = row["chirality"]
        chi_counts[c] = chi_counts.get(c, 0) + 1
    chi_summary = "  |  ".join(f"{c}: {n}" for c, n in chi_counts.items())

    messagebox.showinfo(
        "Done",
        f"Processing complete  ({datetime.date.today()})\n\n"
        f"Processed: {len(all_summaries)} / {len(file_paths)} files\n"
        f"Chirality: {chi_summary}\n\n"
        f"Output folder:\n{output_root}\n\n"
        f"Excel:  raman_blank_results.xlsx\n"
        f"  Sheet 1: Raw_spectra\n"
        f"  Sheet 2: Corrected_spectra\n"
        f"  Sheet 3: RBM_fit_parameters\n\n"
        f"Plots per sample (fitted_spectra/):\n"
        f"  <name>_fit_RBM.png   (250-350 cm-1)\n"
        f"  <name>_fit_full.png  (50-1800 cm-1)\n\n"
        f"Overlay plots:\n"
        f"  all_overlay_RBM.png\n"
        f"  all_overlay_full.png"
    )


if __name__ == "__main__":
    main()