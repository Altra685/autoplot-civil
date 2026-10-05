"""
compaction.py — Proctor Compaction Test Analysis
SNI 1742:2008 / ASTM D698 (Standard) / ASTM D1557 (Modified)

Provides:
  - calculate_dry_density(): from wet density and water content
  - fit_proctor_curve(): polynomial curve fitting with scipy
  - zero_air_voids_line(): ZAV curve for Gs = 2.65 (default)
  - find_mdd_omc(): MDD and OMC from polynomial peak
  - CompactionResult: dataclass with all fields
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompactionPoint:
    """Single compaction test data point."""

    water_content_pct: float
    wet_density_gcm3: float
    dry_density_gcm3: float


@dataclass
class ProctorCurveResult:
    """Fitted Proctor curve coefficients and key results."""

    mdd_gcm3: float            # Maximum Dry Density (g/cm³)
    omc_pct: float             # Optimum Moisture Content (%)
    degree_of_compaction: Optional[float]  # If field density provided
    poly_coefficients: list[float]         # Degree-3 polynomial coefficients
    r_squared: float
    water_contents_fit: list[float]         # Dense x-grid for plotting
    dry_densities_fit: list[float]          # Fitted y-values for plotting
    zav_water_contents: list[float]         # ZAV x-values
    zav_dry_densities: list[float]          # ZAV y-values
    Gs: float                              # Specific gravity used for ZAV
    points: list[CompactionPoint]


@dataclass
class CompactionResult:
    """Complete compaction analysis result."""

    sample_id: str
    n_points: int
    mdd_gcm3: float
    omc_pct: float
    proctor_curve: ProctorCurveResult
    field_density_gcm3: Optional[float]
    field_water_content_pct: Optional[float]
    field_dry_density_gcm3: Optional[float]
    degree_of_compaction_pct: Optional[float]
    compaction_standard: str   # "Standard" | "Modified"
    notes: str


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def calculate_dry_density(
    wet_density_gcm3: float,
    water_content_pct: float,
) -> float:
    """
    Compute dry density from wet density and water content.

    Formula
    -------
    γ_d = γ_wet / (1 + w/100)

    where:
      γ_d     = dry density (g/cm³)
      γ_wet   = wet (bulk) density (g/cm³)
      w       = water content (%)

    Parameters
    ----------
    wet_density_gcm3 : float
        Wet (bulk) density in g/cm³.
    water_content_pct : float
        Gravimetric water content in percent (e.g. 12.5 for 12.5%).

    Returns
    -------
    float
        Dry density in g/cm³.

    Raises
    ------
    ValueError
        If inputs are physically invalid.
    """
    if wet_density_gcm3 <= 0.0:
        raise ValueError(f"wet_density_gcm3 must be positive, got {wet_density_gcm3}.")
    if water_content_pct < 0.0:
        raise ValueError(f"water_content_pct cannot be negative, got {water_content_pct}.")
    if water_content_pct > 100.0:
        warnings.warn(
            f"water_content_pct = {water_content_pct:.1f}% is unusually high.",
            stacklevel=2,
        )

    dry_density = wet_density_gcm3 / (1.0 + water_content_pct / 100.0)
    return round(dry_density, 4)


def zero_air_voids_line(
    water_content_range: tuple[float, float] = (5.0, 35.0),
    n_points: int = 200,
    Gs: float = 2.65,
    rho_w: float = 1.0,
) -> tuple[list[float], list[float]]:
    """
    Compute Zero Air Voids (ZAV) curve for a given specific gravity.

    The ZAV line represents the theoretical maximum dry density at each water
    content assuming no air voids (S = 100%):

    γ_d_ZAV = (Gs · ρ_w) / (1 + Gs · w/100)

    Parameters
    ----------
    water_content_range : tuple[float, float], optional
        (min_w, max_w) in percent, default (5.0, 35.0).
    n_points : int, optional
        Number of points for the ZAV curve, default 200.
    Gs : float, optional
        Specific gravity of soil solids, default 2.65.
    rho_w : float, optional
        Density of water (g/cm³), default 1.0.

    Returns
    -------
    tuple[list[float], list[float]]
        (water_contents_pct, zav_dry_densities_gcm3)
    """
    w_min, w_max = water_content_range
    if w_min >= w_max:
        raise ValueError("water_content_range: min must be less than max.")
    if Gs <= 0.0:
        raise ValueError(f"Gs must be positive, got {Gs}.")

    water_contents = np.linspace(w_min, w_max, n_points)
    zav_densities = (Gs * rho_w) / (1.0 + Gs * (water_contents / 100.0))

    return (
        [round(float(w), 3) for w in water_contents],
        [round(float(d), 4) for d in zav_densities],
    )


def fit_proctor_curve(
    water_contents_pct: list[float],
    dry_densities_gcm3: list[float],
    degree: int = 3,
    Gs: float = 2.65,
) -> ProctorCurveResult:
    """
    Fit a polynomial to Proctor compaction data and find MDD/OMC.

    A degree-3 polynomial is the standard choice for Proctor curves.
    The peak is found by analytically solving the derivative of the fitted
    polynomial.

    Parameters
    ----------
    water_contents_pct : list[float]
        Water content values (%) — at least 4 data points recommended.
    dry_densities_gcm3 : list[float]
        Corresponding dry density values (g/cm³).
    degree : int, optional
        Polynomial degree, default 3.  Increase to 4 for unusual curves.
    Gs : float, optional
        Specific gravity for ZAV overlay, default 2.65.

    Returns
    -------
    ProctorCurveResult
        Fitted curve, MDD, OMC, ZAV line, and input points.

    Raises
    ------
    ValueError
        If fewer than (degree + 1) data points are provided.
    RuntimeError
        If no real peak is found in the valid domain.
    """
    n = len(water_contents_pct)
    if n != len(dry_densities_gcm3):
        raise ValueError("water_contents_pct and dry_densities_gcm3 must be same length.")
    if n < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} data points for degree-{degree} polynomial, got {n}."
        )

    wc = np.array(water_contents_pct, dtype=float)
    dd = np.array(dry_densities_gcm3, dtype=float)

    # Fit polynomial (highest degree first)
    coeffs = np.polyfit(wc, dd, degree)

    # R² calculation
    dd_pred = np.polyval(coeffs, wc)
    ss_res = float(np.sum((dd - dd_pred) ** 2))
    ss_tot = float(np.sum((dd - np.mean(dd)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Find peak: solve derivative = 0
    mdd, omc = find_mdd_omc(coeffs, wc_min=float(wc.min()), wc_max=float(wc.max()))

    # Dense grid for plotting
    wc_fit = np.linspace(float(wc.min()) - 1.0, float(wc.max()) + 1.0, 300)
    dd_fit = np.polyval(coeffs, wc_fit)

    # ZAV line
    zav_wc_range = (max(1.0, float(wc.min()) - 2.0), float(wc.max()) + 5.0)
    zav_wc, zav_dd = zero_air_voids_line(water_content_range=zav_wc_range, Gs=Gs)

    # Build CompactionPoint list
    points = [
        CompactionPoint(
            water_content_pct=float(w),
            wet_density_gcm3=round(float(d) * (1.0 + w / 100.0), 4),
            dry_density_gcm3=float(d),
        )
        for w, d in zip(water_contents_pct, dry_densities_gcm3)
    ]

    return ProctorCurveResult(
        mdd_gcm3=round(mdd, 4),
        omc_pct=round(omc, 2),
        degree_of_compaction=None,
        poly_coefficients=[round(float(c), 6) for c in coeffs],
        r_squared=round(r_squared, 4),
        water_contents_fit=[round(float(w), 3) for w in wc_fit],
        dry_densities_fit=[round(float(d), 4) for d in dd_fit],
        zav_water_contents=zav_wc,
        zav_dry_densities=zav_dd,
        Gs=Gs,
        points=points,
    )


def find_mdd_omc(
    poly_coefficients: list[float],
    wc_min: float,
    wc_max: float,
) -> tuple[float, float]:
    """
    Find Maximum Dry Density (MDD) and Optimum Moisture Content (OMC) by
    analytically solving the derivative of the polynomial.

    For a degree-3 polynomial p(x) = ax³ + bx² + cx + d:
    p'(x) = 3ax² + 2bx + c = 0  → solve quadratic.

    For general degree, numpy.polyder is used.

    Parameters
    ----------
    poly_coefficients : list[float]
        Polynomial coefficients from numpy.polyfit (highest degree first).
    wc_min : float
        Minimum water content in the data range (%).
    wc_max : float
        Maximum water content in the data range (%).

    Returns
    -------
    tuple[float, float]
        (mdd_gcm3, omc_pct)

    Raises
    ------
    RuntimeError
        If no valid peak is found within [wc_min, wc_max].
    """
    coeffs_arr = np.array(poly_coefficients, dtype=float)
    deriv_coeffs = np.polyder(coeffs_arr)

    # Find roots of the derivative
    roots = np.roots(deriv_coeffs)

    # Keep only real roots within [wc_min, wc_max] with some tolerance
    tol = (wc_max - wc_min) * 0.05  # 5% tolerance
    valid_roots = [
        float(r.real)
        for r in roots
        if abs(r.imag) < 1e-6 and (wc_min - tol) <= r.real <= (wc_max + tol)
    ]

    if not valid_roots:
        # Fall back to numerical peak search on a dense grid
        wc_grid = np.linspace(wc_min, wc_max, 1000)
        dd_grid = np.polyval(coeffs_arr, wc_grid)
        peak_idx = int(np.argmax(dd_grid))
        omc = float(wc_grid[peak_idx])
        mdd = float(dd_grid[peak_idx])
        warnings.warn(
            "No analytical peak found in valid domain — using numerical peak.", stacklevel=2
        )
        return mdd, omc

    # Evaluate polynomial at each valid root and pick the maximum
    best_mdd = -np.inf
    best_omc = valid_roots[0]

    for root in valid_roots:
        dd_at_root = float(np.polyval(coeffs_arr, root))
        if dd_at_root > best_mdd:
            best_mdd = dd_at_root
            best_omc = root

    if best_mdd <= 0:
        raise RuntimeError(
            f"Found peak dry density {best_mdd:.4f} g/cm³ is non-positive — "
            "check your input data."
        )

    return best_mdd, best_omc


def analyze_compaction(
    df: pd.DataFrame,
    col_water_content: str,
    col_wet_density: str,
    sample_id: str = "PC-1",
    Gs: float = 2.65,
    field_density_gcm3: Optional[float] = None,
    field_water_content_pct: Optional[float] = None,
    compaction_standard: str = "Standard",
    degree: int = 3,
) -> CompactionResult:
    """
    Full compaction analysis pipeline from raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with at least water content and wet density columns.
    col_water_content : str
        Column name for gravimetric water content (%).
    col_wet_density : str
        Column name for wet (bulk) density (g/cm³).
    sample_id : str
        Sample label.
    Gs : float
        Specific gravity for ZAV line.
    field_density_gcm3 : float, optional
        Measured in-situ density for degree of compaction calculation.
    field_water_content_pct : float, optional
        In-situ water content for field dry density calculation.
    compaction_standard : str
        "Standard" (ASTM D698) or "Modified" (ASTM D1557).
    degree : int
        Polynomial degree for curve fitting.

    Returns
    -------
    CompactionResult
    """
    if col_water_content not in df.columns:
        raise ValueError(f"Column '{col_water_content}' not found in DataFrame.")
    if col_wet_density not in df.columns:
        raise ValueError(f"Column '{col_wet_density}' not found in DataFrame.")

    work = df[[col_water_content, col_wet_density]].copy()
    work.columns = ["wc", "wd"]
    work["wc"] = pd.to_numeric(work["wc"], errors="coerce")
    work["wd"] = pd.to_numeric(work["wd"], errors="coerce")
    work = work.dropna()
    work = work[(work["wc"] > 0.0) & (work["wd"] > 0.0)]

    if work.empty:
        raise ValueError("No valid compaction data rows after cleaning.")

    # Compute dry densities
    work["dd"] = work.apply(
        lambda row: calculate_dry_density(row["wd"], row["wc"]), axis=1
    )

    wc_list = work["wc"].tolist()
    dd_list = work["dd"].tolist()

    # Fit Proctor curve
    curve = fit_proctor_curve(
        water_contents_pct=wc_list,
        dry_densities_gcm3=dd_list,
        degree=degree,
        Gs=Gs,
    )

    # Field dry density and degree of compaction
    field_dd: Optional[float] = None
    doc: Optional[float] = None

    if field_density_gcm3 is not None and field_water_content_pct is not None:
        field_dd = calculate_dry_density(field_density_gcm3, field_water_content_pct)
        doc = (field_dd / curve.mdd_gcm3) * 100.0
        curve.degree_of_compaction = round(doc, 1)

    notes_parts = [
        f"Standar: {compaction_standard} Proctor.",
        f"MDD = {curve.mdd_gcm3:.3f} g/cm³, OMC = {curve.omc_pct:.1f}%.",
        f"R² fit polinomial = {curve.r_squared:.4f}.",
    ]
    if doc is not None:
        status = "LULUS ✓" if doc >= 95.0 else "GAGAL ✗"
        notes_parts.append(
            f"Derajat Kepadatan = {doc:.1f}% — {status} (syarat ≥ 95%)."
        )

    return CompactionResult(
        sample_id=sample_id,
        n_points=len(wc_list),
        mdd_gcm3=curve.mdd_gcm3,
        omc_pct=curve.omc_pct,
        proctor_curve=curve,
        field_density_gcm3=field_density_gcm3,
        field_water_content_pct=field_water_content_pct,
        field_dry_density_gcm3=round(field_dd, 4) if field_dd else None,
        degree_of_compaction_pct=round(doc, 1) if doc else None,
        compaction_standard=compaction_standard,
        notes=" ".join(notes_parts),
    )