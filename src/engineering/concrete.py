"""
concrete.py — Concrete Compressive Strength Calculator
SNI 1974:2011 / ASTM C39 / ACI 318

Provides:
  - CORRECTION_FACTORS: h/d ratio → correction factor (SNI 1974:2011 Table 1)
  - QUALITY_GRADES: Indonesian concrete quality grades (K-xxx)
  - STANDARD_FC_TARGETS: standard f'c targets K-175 to K-500
  - calculate_compressive_strength(): single specimen fc (MPa)
  - analyze_concrete_batch(): batch statistics from DataFrame
  - strength_age_regression(): logarithmic age-strength model
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SNI 1974:2011 — Correction factors for h/d ratio
# h/d ratio → correction factor
# Interpolated at 0.05 increments from 1.00 to 2.00 (21 entries)
# Source: SNI 1974:2011 Table 1 with linear interpolation
# ---------------------------------------------------------------------------
CORRECTION_FACTORS: dict[float, float] = {
    1.00: 0.870,
    1.05: 0.873,
    1.10: 0.876,
    1.15: 0.879,
    1.20: 0.882,
    1.25: 0.930,
    1.30: 0.934,
    1.35: 0.938,
    1.40: 0.942,
    1.45: 0.946,
    1.50: 0.960,
    1.55: 0.964,
    1.60: 0.968,
    1.65: 0.972,
    1.70: 0.976,
    1.75: 0.980,
    1.80: 0.984,
    1.85: 0.988,
    1.90: 0.992,
    1.95: 0.996,
    2.00: 1.000,
}

# ---------------------------------------------------------------------------
# Indonesian concrete quality grades (K-xxx → f'c MPa)
# K-value is characteristic cube strength in kgf/cm²
# f'c (cylinder) ≈ K × 0.83 / 10.197 (approx MPa conversion)
# Standard conversion: f'c = K × 0.0814 (simplified)
# ---------------------------------------------------------------------------
QUALITY_GRADES: dict[str, float] = {
    "K-175": 14.5,
    "K-200": 16.6,
    "K-225": 18.7,
    "K-250": 20.8,
    "K-275": 22.7,
    "K-300": 24.9,
    "K-325": 27.0,
    "K-350": 29.1,
    "K-375": 31.2,
    "K-400": 33.2,
    "K-425": 35.3,
    "K-450": 37.4,
    "K-500": 41.5,
}

# ---------------------------------------------------------------------------
# Standard f'c target values (MPa) — common design strengths
# ---------------------------------------------------------------------------
STANDARD_FC_TARGETS: dict[str, float] = {
    "K-175": 14.5,
    "K-200": 16.6,
    "K-225": 18.7,
    "K-250": 20.8,
    "K-275": 22.7,
    "K-300": 24.9,
    "K-350": 29.1,
    "K-400": 33.2,
    "K-450": 37.4,
    "K-500": 41.5,
    "C20": 20.0,
    "C25": 25.0,
    "C30": 30.0,
    "C35": 35.0,
    "C40": 40.0,
    "C45": 45.0,
    "C50": 50.0,
}

# ---------------------------------------------------------------------------
# Age correction factors for concrete strength (fraction of 28-day strength)
# Source: ACI 209R-92
# ---------------------------------------------------------------------------
AGE_FACTORS: dict[int, float] = {
    1:  0.16,
    3:  0.40,
    7:  0.65,
    14: 0.88,
    21: 0.95,
    28: 1.00,
    56: 1.10,
    90: 1.17,
    180: 1.23,
    365: 1.27,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConcreteSpecimenResult:
    """Single concrete specimen test result."""

    diameter_mm: float
    height_mm: float
    hd_ratio: float
    max_load_kn: float
    area_mm2: float
    fc_raw_mpa: float           # Before correction
    correction_factor: float
    fc_corrected_mpa: float     # After h/d correction


@dataclass
class ConcreteStrengthResult:
    """Complete concrete batch analysis result."""

    fc_values: list[float]          # Individual corrected strengths (MPa)
    fc_mean: float                  # Mean compressive strength (MPa)
    fc_std: float                   # Standard deviation (MPa)
    fc_char: float                  # Characteristic strength = mean - 1.645×std
    fc_min: float                   # Minimum fc in batch
    fc_max: float                   # Maximum fc in batch
    sample_count: int               # Total number of specimens
    pass_count: int                 # Count above target fc
    fail_count: int                 # Count below target fc
    cv: float                       # Coefficient of variation (%)
    target_fc: float                # Target compressive strength (MPa)
    grade_assessment: str           # "PASS" or "FAIL" per SNI criteria
    specimens: list[ConcreteSpecimenResult]
    regression_params: Optional[tuple] = None  # (a, b, r_squared) if age data


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def _get_correction_factor(hd_ratio: float) -> float:
    """
    Get h/d correction factor for a given height/diameter ratio.

    Uses linear interpolation between the closest table entries
    in CORRECTION_FACTORS (SNI 1974:2011).

    Parameters
    ----------
    hd_ratio : float
        Height-to-diameter ratio of the specimen.

    Returns
    -------
    float
        Correction factor (0.87 – 1.00).

    Raises
    ------
    ValueError
        If hd_ratio is outside the range [1.00, 2.00].
    """
    if hd_ratio < 1.00 or hd_ratio > 2.00:
        raise ValueError(
            f"h/d ratio {hd_ratio:.2f} outside valid range [1.00, 2.00]. "
            "Standard specimens have h/d = 2.0; capped specimens must have h/d ≥ 1.0."
        )

    # Exact match
    rounded = round(hd_ratio, 2)
    if rounded in CORRECTION_FACTORS:
        return CORRECTION_FACTORS[rounded]

    # Linear interpolation between nearest entries
    sorted_ratios = sorted(CORRECTION_FACTORS.keys())
    for i in range(len(sorted_ratios) - 1):
        r_low = sorted_ratios[i]
        r_high = sorted_ratios[i + 1]
        if r_low <= hd_ratio <= r_high:
            cf_low = CORRECTION_FACTORS[r_low]
            cf_high = CORRECTION_FACTORS[r_high]
            fraction = (hd_ratio - r_low) / (r_high - r_low)
            return cf_low + fraction * (cf_high - cf_low)

    # Fallback (should not reach here)
    return 1.00


def calculate_compressive_strength(
    diameter_mm: float,
    max_load_kn: float,
    height_mm: Optional[float] = None,
    correction: bool = True,
) -> ConcreteSpecimenResult:
    """
    Calculate concrete compressive strength for a single specimen.

    f'c = (P / A) × CF

    where:
        P = max_load_kn × 1000 (convert to N)
        A = π × (diameter_mm/2)² (mm²)
        CF = correction factor from h/d ratio (SNI 1974:2011)

    Parameters
    ----------
    diameter_mm : float
        Diameter of the cylindrical specimen in mm.
    max_load_kn : float
        Maximum compressive load in kN.
    height_mm : float, optional
        Height of the specimen in mm. If None, assumes h/d = 2.0.
    correction : bool
        Whether to apply h/d correction factor. Default True.

    Returns
    -------
    ConcreteSpecimenResult
        Dataclass with raw and corrected fc values.

    Raises
    ------
    ValueError
        If diameter_mm or max_load_kn are not positive.
    """
    if diameter_mm <= 0:
        raise ValueError(f"diameter_mm must be positive, got {diameter_mm}")
    if max_load_kn <= 0:
        raise ValueError(f"max_load_kn must be positive, got {max_load_kn}")

    # Default height: h/d = 2.0
    if height_mm is None:
        height_mm = diameter_mm * 2.0

    if height_mm <= 0:
        raise ValueError(f"height_mm must be positive, got {height_mm}")

    hd_ratio = height_mm / diameter_mm

    # Area in mm²
    area_mm2 = math.pi * (diameter_mm / 2.0) ** 2

    # Load in N
    load_n = max_load_kn * 1000.0

    # Raw compressive strength (MPa = N/mm²)
    fc_raw = load_n / area_mm2

    # Apply correction factor
    if correction and hd_ratio < 2.00:
        cf = _get_correction_factor(hd_ratio)
    else:
        cf = 1.00

    fc_corrected = fc_raw * cf

    return ConcreteSpecimenResult(
        diameter_mm=round(diameter_mm, 1),
        height_mm=round(height_mm, 1),
        hd_ratio=round(hd_ratio, 3),
        max_load_kn=round(max_load_kn, 2),
        area_mm2=round(area_mm2, 2),
        fc_raw_mpa=round(fc_raw, 3),
        correction_factor=round(cf, 4),
        fc_corrected_mpa=round(fc_corrected, 3),
    )


def analyze_concrete_batch(
    df: pd.DataFrame,
    diameter_col: str,
    load_col: str,
    height_col: Optional[str] = None,
    target_fc: float = 25.0,
    age_col: Optional[str] = None,
    strength_col: Optional[str] = None,
) -> ConcreteStrengthResult:
    """
    Analyze a batch of concrete specimens from a DataFrame.

    Runs calculate_compressive_strength for each row and computes
    batch statistics per SNI 1974:2011.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with specimen data.
    diameter_col : str
        Column name for specimen diameter (mm).
    load_col : str
        Column name for maximum compressive load (kN).
    height_col : str, optional
        Column name for specimen height (mm). If None, h/d = 2.0.
    target_fc : float
        Target compressive strength in MPa. Default 25.0.
    age_col : str, optional
        Column for specimen age in days (for regression).
    strength_col : str, optional
        Column for pre-computed strength (for regression only).

    Returns
    -------
    ConcreteStrengthResult
        Complete batch analysis result.

    Raises
    ------
    ValueError
        If DataFrame is empty or required columns missing.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if diameter_col not in df.columns:
        raise ValueError(f"Column '{diameter_col}' not found in DataFrame.")
    if load_col not in df.columns:
        raise ValueError(f"Column '{load_col}' not found in DataFrame.")
    if height_col is not None and height_col not in df.columns:
        raise ValueError(f"Column '{height_col}' not found in DataFrame.")

    specimens: list[ConcreteSpecimenResult] = []
    fc_values: list[float] = []

    for _, row in df.iterrows():
        diameter = float(row[diameter_col])
        load = float(row[load_col])
        height = float(row[height_col]) if height_col else None

        try:
            result = calculate_compressive_strength(
                diameter_mm=diameter,
                max_load_kn=load,
                height_mm=height,
            )
            specimens.append(result)
            fc_values.append(result.fc_corrected_mpa)
        except ValueError as e:
            warnings.warn(f"Skipping row: {e}", stacklevel=2)

    if not fc_values:
        raise ValueError("No valid specimens processed.")

    fc_arr = np.array(fc_values)
    fc_mean = float(np.mean(fc_arr))
    fc_std = float(np.std(fc_arr, ddof=1)) if len(fc_arr) > 1 else 0.0
    fc_char = fc_mean - 1.645 * fc_std  # SNI characteristic strength
    fc_min = float(np.min(fc_arr))
    fc_max = float(np.max(fc_arr))
    cv = (fc_std / fc_mean * 100.0) if fc_mean > 0 else 0.0

    pass_count = int(np.sum(fc_arr >= target_fc))
    fail_count = len(fc_values) - pass_count

    # Grade assessment per SNI 2847:2019 criteria:
    # 1) fc_mean >= fc' + 1.34×s  AND  2) every individual fc >= fc' - 3.5 MPa
    crit_1 = fc_mean >= target_fc + 1.34 * fc_std
    crit_2 = fc_min >= target_fc - 3.5
    grade_assessment = "PASS" if (crit_1 and crit_2) else "FAIL"

    # Optional: age-strength regression
    regression_params = None
    if age_col and strength_col:
        if age_col in df.columns and strength_col in df.columns:
            ages = df[age_col].dropna().values.astype(float)
            strengths = df[strength_col].dropna().values.astype(float)
            if len(ages) >= 3 and len(ages) == len(strengths):
                regression_params = strength_age_regression(ages, strengths)

    return ConcreteStrengthResult(
        fc_values=[round(v, 3) for v in fc_values],
        fc_mean=round(fc_mean, 3),
        fc_std=round(fc_std, 3),
        fc_char=round(fc_char, 3),
        fc_min=round(fc_min, 3),
        fc_max=round(fc_max, 3),
        sample_count=len(fc_values),
        pass_count=pass_count,
        fail_count=fail_count,
        cv=round(cv, 2),
        target_fc=target_fc,
        grade_assessment=grade_assessment,
        specimens=specimens,
        regression_params=regression_params,
    )


def strength_age_regression(
    ages: np.ndarray,
    strengths: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit a logarithmic age-strength model to concrete test data.

    Model: f'c(t) = a × ln(t) + b

    Parameters
    ----------
    ages : np.ndarray
        Specimen ages in days (must be > 0).
    strengths : np.ndarray
        Compressive strength values in MPa.

    Returns
    -------
    tuple[float, float, float]
        (a, b, r_squared) where a is slope, b is intercept,
        r_squared is coefficient of determination.

    Raises
    ------
    ValueError
        If ages contain zero or negative values, or arrays have
        different lengths, or fewer than 2 data points.
    """
    ages = np.asarray(ages, dtype=float)
    strengths = np.asarray(strengths, dtype=float)

    if len(ages) != len(strengths):
        raise ValueError(
            f"ages ({len(ages)}) and strengths ({len(strengths)}) must have equal length."
        )
    if len(ages) < 2:
        raise ValueError("At least 2 data points required for regression.")
    if np.any(ages <= 0):
        raise ValueError("All age values must be positive (> 0 days).")

    # Logarithmic transform
    ln_ages = np.log(ages)

    # Linear regression on ln(t) vs f'c
    n = len(ages)
    sum_x = np.sum(ln_ages)
    sum_y = np.sum(strengths)
    sum_xy = np.sum(ln_ages * strengths)
    sum_x2 = np.sum(ln_ages ** 2)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-15:
        warnings.warn("Degenerate regression (all ages identical).", stacklevel=2)
        return (0.0, float(np.mean(strengths)), 0.0)

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    # R-squared
    y_pred = a * ln_ages + b
    ss_res = np.sum((strengths - y_pred) ** 2)
    ss_tot = np.sum((strengths - np.mean(strengths)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return (round(a, 4), round(b, 4), round(r_squared, 6))


def predict_strength_at_age(
    regression_params: tuple[float, float, float],
    target_age: int = 28,
) -> float:
    """
    Predict concrete strength at a given age using regression parameters.

    Parameters
    ----------
    regression_params : tuple
        (a, b, r_squared) from strength_age_regression.
    target_age : int
        Target age in days. Default 28.

    Returns
    -------
    float
        Predicted f'c in MPa at the target age.
    """
    if target_age <= 0:
        raise ValueError(f"target_age must be positive, got {target_age}")

    a, b, _ = regression_params
    return round(a * math.log(target_age) + b, 3)
