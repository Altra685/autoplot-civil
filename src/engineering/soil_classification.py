"""
soil_classification.py — USCS & AASHTO Soil Classification
SNI 03-3637-1994 / ASTM D2487 / AASHTO M 145

Provides:
  - USCS_NAMES: USCS soil type names (all 15 types)
  - classify_soil(): wrapper around geolysis + fallback manual classification
  - extract_gradation_params(): D10, D30, D60 via log-linear interpolation
  - _manual_uscs_classify(): standalone fallback (no external libs required)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# USCS classification names (ASTM D2487 terminology)
# All 15 USCS soil types
# ---------------------------------------------------------------------------
USCS_NAMES: dict[str, str] = {
    # Coarse-grained soils — Gravels
    "GW": "Well-graded Gravel",
    "GP": "Poorly-graded Gravel",
    "GM": "Silty Gravel",
    "GC": "Clayey Gravel",
    # Coarse-grained soils — Sands
    "SW": "Well-graded Sand",
    "SP": "Poorly-graded Sand",
    "SM": "Silty Sand",
    "SC": "Clayey Sand",
    # Fine-grained soils — Silts and Clays (LL < 50%) — Low plasticity
    "ML": "Inorganic Silt, Low Plasticity",
    "CL": "Inorganic Clay, Low–Medium Plasticity",
    "OL": "Organic Silt/Clay, Low Plasticity",
    # Fine-grained soils — Silts and Clays (LL ≥ 50%) — High plasticity
    "MH": "Inorganic Silt, High Plasticity / Elastic Silt",
    "CH": "Inorganic Clay, High Plasticity / Fat Clay",
    "OH": "Organic Clay, Medium–High Plasticity",
    # Highly organic soils
    "Pt": "Peat / Highly Organic Soil",
}

# ---------------------------------------------------------------------------
# AASHTO classification descriptions
# ---------------------------------------------------------------------------
AASHTO_NAMES: dict[str, str] = {
    "A-1-a": "Selected Granular Material (Crushed Stone / Coarse Gravel) — A-1-a",
    "A-1-b": "Selected Granular Material (Coarse Sand) — A-1-b",
    "A-2-4": "Silty/Clayey Gravel & Sand (LL≤40, PI≤10) — A-2-4",
    "A-2-5": "Silty/Clayey Gravel & Sand (LL>40, PI≤10) — A-2-5",
    "A-2-6": "Silty/Clayey Gravel & Sand (LL≤40, PI>10) — A-2-6",
    "A-2-7": "Silty/Clayey Gravel & Sand (LL>40, PI>10) — A-2-7",
    "A-3":   "Fine Sand — A-3",
    "A-4":   "Silty Soil — A-4",
    "A-5":   "Silty Soil, Highly Plastic — A-5",
    "A-6":   "Clayey Soil — A-6",
    "A-7-5": "Plastic Clayey Soil — A-7-5",
    "A-7-6": "Highly Plastic Clayey Soil — A-7-6",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GradationParams:
    """Particle size characteristic diameters and gradation coefficients."""

    D10_mm: Optional[float]   # Effective size
    D30_mm: Optional[float]   # Diameter at 30% passing
    D60_mm: Optional[float]   # Controlling size
    Cu: Optional[float]       # Coefficient of Uniformity = D60/D10
    Cc: Optional[float]       # Coefficient of Curvature = D30²/(D10·D60)
    fines_percent: float      # % passing 0.075 mm (No. 200 sieve)
    gravel_percent: float     # % retained on 4.75 mm (No. 4 sieve)
    sand_percent: float       # % between 4.75 and 0.075 mm


@dataclass
class ClassificationResult:
    """Complete soil classification result."""

    uscs_symbol: str
    uscs_name: str
    aashto_symbol: str
    aashto_name: str
    liquid_limit: Optional[float]
    plastic_limit: Optional[float]
    plasticity_index: Optional[float]
    gradation: Optional[GradationParams]
    classification_method: str   # "geolysis" | "manual"
    notes: str


# ---------------------------------------------------------------------------
# Gradation parameter extraction
# ---------------------------------------------------------------------------

def extract_gradation_params(
    sieve_sizes_mm: list[float],
    percent_passing: list[float],
) -> GradationParams:
    """
    Extract D10, D30, D60 and gradation coefficients from sieve analysis data.

    Uses **log-linear interpolation** (interpolation in log-sieve-size space,
    linear in percent-passing space) — standard practice per ASTM D2487.

    Parameters
    ----------
    sieve_sizes_mm : list[float]
        Sieve opening sizes in millimetres, in ANY order (will be sorted).
    percent_passing : list[float]
        Corresponding percent passing values [0–100], same order as sieve_sizes.

    Returns
    -------
    GradationParams
        D10, D30, D60, Cu, Cc, and fraction percentages.

    Notes
    -----
    - D-values outside the tested range are returned as None.
    - Fines (< 0.075 mm), sand (0.075–4.75 mm), gravel (> 4.75 mm) percentages
      are estimated by interpolating at the standard boundaries.
    """
    if len(sieve_sizes_mm) != len(percent_passing):
        raise ValueError("sieve_sizes_mm and percent_passing must have the same length.")
    if len(sieve_sizes_mm) < 2:
        raise ValueError("At least 2 sieve data points are required.")

    # Sort by sieve size ascending
    pairs = sorted(zip(sieve_sizes_mm, percent_passing), key=lambda x: x[0])
    sizes = np.array([p[0] for p in pairs], dtype=float)
    passings = np.array([p[1] for p in pairs], dtype=float)

    # Clamp passing to [0, 100]
    passings = np.clip(passings, 0.0, 100.0)

    def interpolate_at_passing(target_percent: float) -> Optional[float]:
        """
        Find the sieve size (mm) corresponding to a given percent-passing target.
        Log-linear interpolation in log(size) space.
        """
        if target_percent < passings[0] or target_percent > passings[-1]:
            return None

        # Find bracketing indices
        for i in range(len(passings) - 1):
            p_lo, p_hi = passings[i], passings[i + 1]
            s_lo, s_hi = sizes[i], sizes[i + 1]

            if p_lo <= target_percent <= p_hi:
                if p_hi == p_lo:
                    return float(s_lo)
                # Linear interpolation fraction in percent-passing space
                frac = (target_percent - p_lo) / (p_hi - p_lo)
                # Log-linear in size space
                log_s = math.log(s_lo) + frac * (math.log(s_hi) - math.log(s_lo))
                return math.exp(log_s)

        return None

    def interpolate_passing_at_size(target_size_mm: float) -> float:
        """Return percent passing at a given sieve size (log-linear)."""
        if target_size_mm <= sizes[0]:
            return float(passings[0])
        if target_size_mm >= sizes[-1]:
            return float(passings[-1])

        for i in range(len(sizes) - 1):
            s_lo, s_hi = sizes[i], sizes[i + 1]
            p_lo, p_hi = passings[i], passings[i + 1]
            if s_lo <= target_size_mm <= s_hi:
                if s_hi == s_lo:
                    return float(p_lo)
                # Log-linear
                frac = (math.log(target_size_mm) - math.log(s_lo)) / (
                    math.log(s_hi) - math.log(s_lo)
                )
                return float(p_lo + frac * (p_hi - p_lo))

        return float(passings[-1])

    D10 = interpolate_at_passing(10.0)
    D30 = interpolate_at_passing(30.0)
    D60 = interpolate_at_passing(60.0)

    # Gradation coefficients
    Cu: Optional[float] = None
    Cc: Optional[float] = None
    if D10 is not None and D60 is not None and D10 > 0:
        Cu = D60 / D10
    if D10 is not None and D30 is not None and D60 is not None and D10 > 0 and D60 > 0:
        Cc = (D30 ** 2) / (D10 * D60)

    # Fraction percentages
    fines_pct = interpolate_passing_at_size(0.075)      # < 0.075 mm
    no4_pct = interpolate_passing_at_size(4.75)          # % passing No. 4
    gravel_pct = 100.0 - no4_pct                        # retained on No. 4
    sand_pct = no4_pct - fines_pct                      # between 0.075 & 4.75

    return GradationParams(
        D10_mm=round(D10, 4) if D10 is not None else None,
        D30_mm=round(D30, 4) if D30 is not None else None,
        D60_mm=round(D60, 4) if D60 is not None else None,
        Cu=round(Cu, 2) if Cu is not None else None,
        Cc=round(Cc, 2) if Cc is not None else None,
        fines_percent=round(fines_pct, 1),
        gravel_percent=round(gravel_pct, 1),
        sand_percent=round(sand_pct, 1),
    )


# ---------------------------------------------------------------------------
# Manual USCS classification (no external libraries)
# ---------------------------------------------------------------------------

def _manual_uscs_classify(
    fines_pct: float,
    gravel_pct: float,
    Cu: Optional[float],
    Cc: Optional[float],
    liquid_limit: Optional[float],
    plasticity_index: Optional[float],
) -> str:
    """
    Classify soil using USCS (ASTM D2487) rules without geolysis.

    Parameters
    ----------
    fines_pct : float
        Percent passing 0.075 mm sieve.
    gravel_pct : float
        Percent retained on 4.75 mm sieve.
    Cu : float or None
        Coefficient of uniformity (D60/D10).
    Cc : float or None
        Coefficient of curvature.
    liquid_limit : float or None
        Atterberg LL in percent.
    plasticity_index : float or None
        Atterberg PI in percent.

    Returns
    -------
    str
        USCS symbol (e.g. "GW", "CL", "Pt").
    """
    # Highly organic / peat — check before everything else
    # Field judgement: if LL is very high and organic, use "Pt";
    # we rely on caller to flag peat separately.

    # -----------------------------------------------------------------------
    # Coarse-grained: < 50% fines
    # -----------------------------------------------------------------------
    if fines_pct < 50.0:
        # Gravel dominant: gravel fraction > 50% of coarse fraction
        coarse_pct = 100.0 - fines_pct
        gravel_fraction = gravel_pct  # retained on 4.75 mm

        if coarse_pct > 0 and (gravel_fraction / coarse_pct) >= 0.5:
            # GRAVEL
            if fines_pct < 5.0:
                # Clean gravel — check Cu, Cc
                if Cu is not None and Cc is not None and Cu >= 4.0 and 1.0 <= Cc <= 3.0:
                    return "GW"
                else:
                    return "GP"
            elif fines_pct <= 12.0:
                # Borderline — dual symbol; use single based on Atterberg
                if liquid_limit is not None and plasticity_index is not None:
                    # A-line: PI = 0.73·(LL - 20)
                    a_line_pi = 0.73 * (liquid_limit - 20.0)
                    if plasticity_index < 4.0 or plasticity_index < a_line_pi:
                        return "GM"
                    else:
                        return "GC"
                else:
                    # Default to GM when no Atterberg
                    return "GM"
            else:
                # > 12% fines
                if liquid_limit is not None and plasticity_index is not None:
                    a_line_pi = 0.73 * (liquid_limit - 20.0)
                    if plasticity_index < 4.0 or plasticity_index < a_line_pi:
                        return "GM"
                    else:
                        return "GC"
                else:
                    return "GM"
        else:
            # SAND
            if fines_pct < 5.0:
                if Cu is not None and Cc is not None and Cu >= 6.0 and 1.0 <= Cc <= 3.0:
                    return "SW"
                else:
                    return "SP"
            elif fines_pct <= 12.0:
                if liquid_limit is not None and plasticity_index is not None:
                    a_line_pi = 0.73 * (liquid_limit - 20.0)
                    if plasticity_index < 4.0 or plasticity_index < a_line_pi:
                        return "SM"
                    else:
                        return "SC"
                else:
                    return "SM"
            else:
                if liquid_limit is not None and plasticity_index is not None:
                    a_line_pi = 0.73 * (liquid_limit - 20.0)
                    if plasticity_index < 4.0 or plasticity_index < a_line_pi:
                        return "SM"
                    else:
                        return "SC"
                else:
                    return "SM"

    # -----------------------------------------------------------------------
    # Fine-grained: ≥ 50% fines
    # -----------------------------------------------------------------------
    else:
        if liquid_limit is None or plasticity_index is None:
            return "ML"  # Default if no Atterberg data

        a_line_pi = 0.73 * (liquid_limit - 20.0)

        if liquid_limit < 50.0:
            # Low plasticity region (L)
            if plasticity_index > 7.0 and plasticity_index >= a_line_pi:
                return "CL"
            elif plasticity_index < 4.0 or plasticity_index < a_line_pi:
                return "ML"
            else:
                # Hatched zone (4 ≤ PI ≤ 7)
                return "CL" if plasticity_index >= a_line_pi else "ML"
        else:
            # High plasticity region (H)
            if plasticity_index >= a_line_pi:
                return "CH"
            else:
                return "MH"


def _manual_aashto_classify(
    fines_pct: float,
    gravel_pct: float,
    liquid_limit: Optional[float],
    plasticity_index: Optional[float],
    D60: Optional[float],
) -> str:
    """
    Classify soil per AASHTO M 145 (manual implementation).

    Returns
    -------
    str
        AASHTO group symbol (e.g. "A-1-a", "A-6").
    """
    ll = liquid_limit if liquid_limit is not None else 0.0
    pi = plasticity_index if plasticity_index is not None else 0.0

    # A-1-a: ≤ 10% passing #200, ≤ 30% #40, ≤ 15% #10, PI ≤ 6
    # Simplified per M-145 Table 1
    passing_40 = 100.0 - gravel_pct  # rough estimate
    passing_10 = max(0.0, passing_40 - 10.0)

    if fines_pct <= 35.0:
        # Granular materials (A-1, A-2, A-3)
        if fines_pct <= 10.0:
            if pi <= 6.0:
                return "A-1-a"
            else:
                return "A-1-b"
        elif fines_pct <= 25.0:
            if pi <= 6.0:
                return "A-1-b"
            else:
                return "A-3"
        elif fines_pct <= 35.0:
            if ll <= 40.0:
                if pi <= 10.0:
                    return "A-2-4"
                else:
                    return "A-2-6"
            else:
                if pi <= 10.0:
                    return "A-2-5"
                else:
                    return "A-2-7"
    else:
        # Silt-clay materials (A-4 through A-7)
        if ll <= 40.0:
            if pi <= 10.0:
                return "A-4"
            else:
                return "A-6"
        else:
            if pi <= 10.0:
                return "A-5"
            else:
                # A-7-5: PI ≤ LL - 30; A-7-6: PI > LL - 30
                if pi <= ll - 30.0:
                    return "A-7-5"
                else:
                    return "A-7-6"

    return "A-3"  # Default fallback


# ---------------------------------------------------------------------------
# Primary classification entry point
# ---------------------------------------------------------------------------

def classify_soil(
    sieve_sizes_mm: Optional[list[float]] = None,
    percent_passing: Optional[list[float]] = None,
    liquid_limit: Optional[float] = None,
    plastic_limit: Optional[float] = None,
    is_organic: bool = False,
) -> ClassificationResult:
    """
    Classify soil by USCS and AASHTO using geolysis if available, else manual.

    Parameters
    ----------
    sieve_sizes_mm : list[float], optional
        Sieve opening sizes in mm.
    percent_passing : list[float], optional
        Corresponding percent-passing values.
    liquid_limit : float, optional
        Liquid limit (Atterberg), percent.
    plastic_limit : float, optional
        Plastic limit (Atterberg), percent.
    is_organic : bool, optional
        Flag peat / highly organic soil.

    Returns
    -------
    ClassificationResult
        Full classification with USCS + AASHTO symbols and descriptions.
    """
    # Compute PI
    plasticity_index: Optional[float] = None
    if liquid_limit is not None and plastic_limit is not None:
        plasticity_index = liquid_limit - plastic_limit

    # Extract gradation params if sieve data provided
    gradation: Optional[GradationParams] = None
    fines_pct = 0.0
    gravel_pct = 0.0
    Cu: Optional[float] = None
    Cc: Optional[float] = None
    D60: Optional[float] = None

    if sieve_sizes_mm is not None and percent_passing is not None and len(sieve_sizes_mm) >= 2:
        gradation = extract_gradation_params(sieve_sizes_mm, percent_passing)
        fines_pct = gradation.fines_percent
        gravel_pct = gradation.gravel_percent
        Cu = gradation.Cu
        Cc = gradation.Cc
        D60 = gradation.D60_mm

    # Peat override
    if is_organic and (liquid_limit is None or liquid_limit > 500):
        return ClassificationResult(
            uscs_symbol="Pt",
            uscs_name=USCS_NAMES["Pt"],
            aashto_symbol="A-8",
            aashto_name="Peat / Highly Organic Soil — A-8",
            liquid_limit=liquid_limit,
            plastic_limit=plastic_limit,
            plasticity_index=plasticity_index,
            gradation=gradation,
            classification_method="manual",
            notes="Peat / highly organic soil — identified from field description.",
        )

    # Attempt geolysis classification
    method = "manual"
    uscs_symbol = ""
    aashto_symbol = ""
    notes = ""

    try:
        import geolysis  # type: ignore

        # geolysis API: uscs_classification and aashto_classification
        from geolysis.soil_classifier import (  # type: ignore
            USCSClassifier,
            AASHTOClassifier,
        )

        uscs_obj = USCSClassifier(
            liquid_limit=liquid_limit or 0.0,
            plasticity_index=plasticity_index or 0.0,
            fines=fines_pct,
            sand=100.0 - fines_pct - gravel_pct,
            d10=gradation.D10_mm if gradation and gradation.D10_mm else 0.0,
            d30=gradation.D30_mm if gradation and gradation.D30_mm else 0.0,
            d60=gradation.D60_mm if gradation and gradation.D60_mm else 0.0,
        )
        uscs_symbol = str(uscs_obj.soil_class)

        aashto_obj = AASHTOClassifier(
            liquid_limit=liquid_limit or 0.0,
            plasticity_index=plasticity_index or 0.0,
            fines=fines_pct,
        )
        aashto_symbol = str(aashto_obj.soil_class)
        method = "geolysis"

    except Exception:
        # Fallback to manual classification
        warnings.warn(
            "geolysis not available or raised exception — using manual USCS/AASHTO classification.",
            stacklevel=2,
        )
        uscs_symbol = _manual_uscs_classify(
            fines_pct=fines_pct,
            gravel_pct=gravel_pct,
            Cu=Cu,
            Cc=Cc,
            liquid_limit=liquid_limit,
            plasticity_index=plasticity_index,
        )
        aashto_symbol = _manual_aashto_classify(
            fines_pct=fines_pct,
            gravel_pct=gravel_pct,
            liquid_limit=liquid_limit,
            plasticity_index=plasticity_index,
            D60=D60,
        )
        notes = "Manual classification (geolysis not available)."

    uscs_name = USCS_NAMES.get(uscs_symbol, f"Unknown symbol: {uscs_symbol}")
    aashto_name = AASHTO_NAMES.get(
        aashto_symbol, f"Unknown symbol: {aashto_symbol}"
    )

    return ClassificationResult(
        uscs_symbol=uscs_symbol,
        uscs_name=uscs_name,
        aashto_symbol=aashto_symbol,
        aashto_name=aashto_name,
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
        plasticity_index=plasticity_index,
        gradation=gradation,
        classification_method=method,
        notes=notes,
    )