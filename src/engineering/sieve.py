"""
sieve.py — Sieve Analysis Processing Pipeline
SNI 03-1968-1990 / ASTM C136 / ASTM D422

Provides:
  - process_sieve_data(): full pipeline from raw data to classified result
  - interpolate_diameter(): log-linear D-value interpolation
  - calculate_gradation_coefficients(): Cu and Cc
  - classify_from_gradation(): USCS soil group from gradation + Atterberg
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .soil_classification import (
    classify_soil,
    extract_gradation_params,
    ClassificationResult,
    GradationParams,
)

# ---------------------------------------------------------------------------
# Standard sieve series (ASTM E11 / SNI 03-1968-1990)
# Listed in mm, largest to smallest
# ---------------------------------------------------------------------------
STANDARD_SIEVES_MM: list[float] = [
    75.0,    # 3 inch
    50.0,    # 2 inch
    37.5,    # 1.5 inch
    25.0,    # 1 inch
    19.0,    # 3/4 inch
    12.5,    # 1/2 inch
    9.5,     # 3/8 inch
    4.75,    # No. 4
    2.36,    # No. 8
    2.00,    # No. 10
    1.18,    # No. 16
    0.600,   # No. 30
    0.425,   # No. 40
    0.300,   # No. 50
    0.150,   # No. 100
    0.075,   # No. 200
]

# Map sieve sizes to common labels
SIEVE_LABELS: dict[float, str] = {
    75.0:  "75 mm (3\")",
    50.0:  "50 mm (2\")",
    37.5:  "37.5 mm (1½\")",
    25.0:  "25 mm (1\")",
    19.0:  "19 mm (¾\")",
    12.5:  "12.5 mm (½\")",
    9.5:   "9.5 mm (⅜\")",
    4.75:  "4.75 mm (No.4)",
    2.36:  "2.36 mm (No.8)",
    2.00:  "2.00 mm (No.10)",
    1.18:  "1.18 mm (No.16)",
    0.600: "0.600 mm (No.30)",
    0.425: "0.425 mm (No.40)",
    0.300: "0.300 mm (No.50)",
    0.150: "0.150 mm (No.100)",
    0.075: "0.075 mm (No.200)",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SieveProcessedData:
    """Fully processed sieve analysis for one sample."""

    sample_id: str
    sieve_sizes_mm: list[float]
    mass_retained_g: list[float]
    percent_retained: list[float]
    cumulative_retained: list[float]
    percent_passing: list[float]
    total_mass_g: float
    gradation: GradationParams
    classification: ClassificationResult
    soil_description: str


# ---------------------------------------------------------------------------
# Core interpolation
# ---------------------------------------------------------------------------

def interpolate_diameter(
    percent_passing_target: float,
    sieve_sizes_mm: list[float],
    percent_passing: list[float],
) -> Optional[float]:
    """
    Find the sieve diameter (mm) corresponding to a target percent-passing
    value using **log-linear interpolation**.

    Interpolation is performed in log10(sieve_size) space, which reflects the
    geometric (logarithmic) spacing of standard sieve series.

    Parameters
    ----------
    percent_passing_target : float
        The percent-passing percentage to find the diameter for (e.g. 10, 30, 60).
    sieve_sizes_mm : list[float]
        Sieve opening sizes in mm, in ANY order (will be sorted ascending).
    percent_passing : list[float]
        Corresponding percent-passing values, same order as sieve_sizes_mm.

    Returns
    -------
    float or None
        Interpolated sieve size in mm, or None if target is outside data range.

    Notes
    -----
    Both lists must have the same length and at least 2 entries.
    Values that exactly match a table entry are returned without interpolation.
    """
    if len(sieve_sizes_mm) != len(percent_passing):
        raise ValueError("sieve_sizes_mm and percent_passing must be the same length.")
    if len(sieve_sizes_mm) < 2:
        raise ValueError("At least 2 data points are required for interpolation.")

    # Sort by ascending sieve size
    pairs = sorted(zip(sieve_sizes_mm, percent_passing), key=lambda x: x[0])
    sizes = [p[0] for p in pairs]
    passings = [p[1] for p in pairs]

    target = float(percent_passing_target)

    # Check bounds
    if target < min(passings) or target > max(passings):
        return None

    # Exact match
    for s, p in zip(sizes, passings):
        if abs(p - target) < 1e-9:
            return float(s)

    # Find bracketing interval
    for i in range(len(passings) - 1):
        p_lo = passings[i]
        p_hi = passings[i + 1]
        s_lo = sizes[i]
        s_hi = sizes[i + 1]

        if p_lo <= target <= p_hi or p_hi <= target <= p_lo:
            if abs(p_hi - p_lo) < 1e-12:
                return float(s_lo)

            frac = (target - p_lo) / (p_hi - p_lo)

            # Log-linear in sieve size space
            if s_lo > 0 and s_hi > 0:
                log_s = math.log10(s_lo) + frac * (math.log10(s_hi) - math.log10(s_lo))
                return 10.0 ** log_s
            else:
                return s_lo + frac * (s_hi - s_lo)

    return None


# ---------------------------------------------------------------------------
# Gradation coefficients
# ---------------------------------------------------------------------------

def calculate_gradation_coefficients(
    D10: Optional[float],
    D30: Optional[float],
    D60: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute Cu (Coefficient of Uniformity) and Cc (Coefficient of Curvature).

    Cu = D60 / D10
    Cc = D30² / (D10 × D60)

    Parameters
    ----------
    D10 : float or None
        Diameter at 10% passing (mm).
    D30 : float or None
        Diameter at 30% passing (mm).
    D60 : float or None
        Diameter at 60% passing (mm).

    Returns
    -------
    tuple[float | None, float | None]
        (Cu, Cc) — either may be None if required D-values are unavailable.
    """
    Cu: Optional[float] = None
    Cc: Optional[float] = None

    if D10 is not None and D60 is not None and D10 > 0.0:
        Cu = D60 / D10

    if (
        D10 is not None
        and D30 is not None
        and D60 is not None
        and D10 > 0.0
        and D60 > 0.0
    ):
        Cc = (D30 ** 2) / (D10 * D60)

    return (
        round(Cu, 3) if Cu is not None else None,
        round(Cc, 3) if Cc is not None else None,
    )


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_sieve_data(
    df: pd.DataFrame,
    col_sieve_size: str,
    col_mass_retained: str,
    sample_id: str = "S-1",
    total_mass_g: Optional[float] = None,
    liquid_limit: Optional[float] = None,
    plastic_limit: Optional[float] = None,
    col_sieve_label: Optional[str] = None,
) -> SieveProcessedData:
    """
    Full sieve analysis pipeline: raw retained-mass data → classified result.

    Pipeline Steps
    --------------
    1. Validate and clean input DataFrame.
    2. Sort sieves from largest to smallest (descending).
    3. Compute percent retained per sieve.
    4. Compute cumulative percent retained.
    5. Compute percent passing = 100 − cumulative retained.
    6. Interpolate D10, D30, D60 using log-linear method.
    7. Compute Cu and Cc.
    8. Classify soil (USCS + AASHTO) using soil_classification module.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sieve data.
    col_sieve_size : str
        Column name containing sieve opening sizes in mm.
    col_mass_retained : str
        Column name containing mass retained on each sieve (grams).
    sample_id : str, optional
        Label for this sample.
    total_mass_g : float, optional
        Known total dry mass.  If None, it is computed as sum of retained + pan.
    liquid_limit : float, optional
        For Atterberg-based USCS classification of fines.
    plastic_limit : float, optional
        For Atterberg-based USCS classification of fines.
    col_sieve_label : str, optional
        Column with sieve label strings (e.g. "No. 200"); used for display only.

    Returns
    -------
    SieveProcessedData
        Fully processed and classified sieve analysis result.

    Raises
    ------
    ValueError
        If required columns are missing or no valid data rows exist.
    """
    # -----------------------------------------------------------------------
    # 1. Validate inputs
    # -----------------------------------------------------------------------
    if col_sieve_size not in df.columns:
        raise ValueError(f"Column '{col_sieve_size}' not found in DataFrame.")
    if col_mass_retained not in df.columns:
        raise ValueError(f"Column '{col_mass_retained}' not found in DataFrame.")

    work = df[[col_sieve_size, col_mass_retained]].copy()
    work.columns = ["size_mm", "mass_g"]

    # Coerce to numeric, drop invalid
    work["size_mm"] = pd.to_numeric(work["size_mm"], errors="coerce")
    work["mass_g"] = pd.to_numeric(work["mass_g"], errors="coerce")
    work = work.dropna(subset=["size_mm", "mass_g"])
    work = work[work["size_mm"] > 0.0]
    work = work[work["mass_g"] >= 0.0]

    if work.empty:
        raise ValueError("No valid sieve data rows found after cleaning.")

    # -----------------------------------------------------------------------
    # 2. Sort descending (largest sieve first)
    # -----------------------------------------------------------------------
    work = work.sort_values("size_mm", ascending=False).reset_index(drop=True)

    sizes_mm: list[float] = work["size_mm"].tolist()
    masses_g: list[float] = work["mass_g"].tolist()

    # -----------------------------------------------------------------------
    # 3. Total mass
    # -----------------------------------------------------------------------
    sum_retained = sum(masses_g)
    if total_mass_g is None or total_mass_g <= 0.0:
        total_mass_g = sum_retained
        if total_mass_g <= 0.0:
            raise ValueError("Total mass is zero — check mass_retained column.")
    else:
        # Validate: warn if supplied total differs from sum by > 2%
        diff_pct = abs(total_mass_g - sum_retained) / total_mass_g * 100.0
        if diff_pct > 2.0:
            warnings.warn(
                f"Supplied total_mass_g ({total_mass_g:.1f} g) differs from "
                f"sum of retained masses ({sum_retained:.1f} g) by {diff_pct:.1f}%.",
                stacklevel=2,
            )

    # -----------------------------------------------------------------------
    # 4–6. Compute retained %, cumulative retained %, percent passing
    # -----------------------------------------------------------------------
    pct_retained: list[float] = []
    cum_retained: list[float] = []
    pct_passing: list[float] = []

    cumulative = 0.0
    for mass in masses_g:
        pct_ret = (mass / total_mass_g) * 100.0
        pct_retained.append(round(pct_ret, 2))
        cumulative += pct_ret
        cum_retained.append(round(cumulative, 2))
        pct_pass = max(0.0, 100.0 - cumulative)
        pct_passing.append(round(pct_pass, 2))

    # -----------------------------------------------------------------------
    # 7. D-values via log-linear interpolation
    # -----------------------------------------------------------------------
    # For interpolation, sort ascending
    sizes_asc = list(reversed(sizes_mm))
    pass_asc = list(reversed(pct_passing))

    D10 = interpolate_diameter(10.0, sizes_asc, pass_asc)
    D30 = interpolate_diameter(30.0, sizes_asc, pass_asc)
    D60 = interpolate_diameter(60.0, sizes_asc, pass_asc)

    Cu, Cc = calculate_gradation_coefficients(D10, D30, D60)

    # Fraction percentages
    fines_pct = pct_passing[-1]   # % passing smallest sieve (0.075 mm if present)
    # Use proper 4.75 mm interpolation
    pct_at_4_75 = interpolate_diameter(50.0, sizes_asc, pass_asc)  # size where 50% passes
    # Better: interpolate passing at 4.75 mm
    def _pass_at(target_size: float) -> float:
        if target_size <= sizes_asc[0]:
            return float(pass_asc[0])
        if target_size >= sizes_asc[-1]:
            return float(pass_asc[-1])
        for i in range(len(sizes_asc) - 1):
            s_lo, s_hi = sizes_asc[i], sizes_asc[i + 1]
            p_lo, p_hi = pass_asc[i], pass_asc[i + 1]
            if s_lo <= target_size <= s_hi:
                if abs(s_hi - s_lo) < 1e-12:
                    return float(p_lo)
                if s_lo > 0 and s_hi > 0:
                    frac = (math.log10(target_size) - math.log10(s_lo)) / (
                        math.log10(s_hi) - math.log10(s_lo)
                    )
                else:
                    frac = (target_size - s_lo) / (s_hi - s_lo)
                return float(p_lo + frac * (p_hi - p_lo))
        return float(pass_asc[-1])

    passing_4_75 = _pass_at(4.75)
    passing_0_075 = _pass_at(0.075)

    gravel_pct = max(0.0, 100.0 - passing_4_75)
    fines_pct_std = max(0.0, passing_0_075)
    sand_pct = max(0.0, passing_4_75 - fines_pct_std)

    gradation = GradationParams(
        D10_mm=round(D10, 4) if D10 is not None else None,
        D30_mm=round(D30, 4) if D30 is not None else None,
        D60_mm=round(D60, 4) if D60 is not None else None,
        Cu=Cu,
        Cc=Cc,
        fines_percent=round(fines_pct_std, 1),
        gravel_percent=round(gravel_pct, 1),
        sand_percent=round(sand_pct, 1),
    )

    # -----------------------------------------------------------------------
    # 8. Classification
    # -----------------------------------------------------------------------
    classification = classify_from_gradation(
        gradation=gradation,
        sieve_sizes_mm=sizes_asc,
        percent_passing=pass_asc,
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
    )

    # Build soil description
    parts = []
    if gravel_pct > 50.0:
        parts.append("Gravel-dominant")
    elif sand_pct > 50.0:
        parts.append("Sand-dominant")
    else:
        parts.append("Silt/Clay-dominant")

    if fines_pct_std > 12.0:
        parts.append("high fines content")
    elif fines_pct_std > 5.0:
        parts.append("some fines content")
    else:
        parts.append("bersih")

    soil_description = f"{classification.uscs_symbol} — " + ", ".join(parts)

    return SieveProcessedData(
        sample_id=sample_id,
        sieve_sizes_mm=sizes_mm,
        mass_retained_g=masses_g,
        percent_retained=pct_retained,
        cumulative_retained=cum_retained,
        percent_passing=pct_passing,
        total_mass_g=total_mass_g,
        gradation=gradation,
        classification=classification,
        soil_description=soil_description,
    )


# ---------------------------------------------------------------------------
# Convenience: classify from pre-computed gradation
# ---------------------------------------------------------------------------

def classify_from_gradation(
    gradation: GradationParams,
    sieve_sizes_mm: list[float],
    percent_passing: list[float],
    liquid_limit: Optional[float] = None,
    plastic_limit: Optional[float] = None,
) -> ClassificationResult:
    """
    Classify a soil sample from GradationParams and optional Atterberg limits.

    Delegates to ``classify_soil()`` in soil_classification module.

    Parameters
    ----------
    gradation : GradationParams
        Pre-computed gradation parameters (D10/D30/D60, Cu, Cc, fractions).
    sieve_sizes_mm : list[float]
        Sieve sizes (ascending) for passing to geolysis.
    percent_passing : list[float]
        Percent-passing values (same order).
    liquid_limit : float, optional
        Atterberg LL.
    plastic_limit : float, optional
        Atterberg PL.

    Returns
    -------
    ClassificationResult
        USCS + AASHTO classification.
    """
    return classify_soil(
        sieve_sizes_mm=sieve_sizes_mm,
        percent_passing=percent_passing,
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
    )


# ---------------------------------------------------------------------------
# Helper: build a clean DataFrame for display
# ---------------------------------------------------------------------------

def to_display_dataframe(processed: SieveProcessedData) -> pd.DataFrame:
    """
    Convert SieveProcessedData to a tidy DataFrame for Streamlit display.

    Returns columns: Sieve Size (mm), Label, Mass Retained (g),
    % Retained, % Cumulative Retained, % Passing.
    """
    labels = [SIEVE_LABELS.get(s, f"{s:.3f} mm") for s in processed.sieve_sizes_mm]

    df = pd.DataFrame({
        "Sieve Size (mm)": processed.sieve_sizes_mm,
        "Sieve Label": labels,
        "Mass Retained (g)": [round(m, 2) for m in processed.mass_retained_g],
        "% Retained": processed.percent_retained,
        "% Cumulative Retained": processed.cumulative_retained,
        "% Passing": processed.percent_passing,
    })
    return df