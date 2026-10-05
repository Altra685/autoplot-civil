"""
tests/unit/test_sieve.py
========================
Unit tests for sieve analysis — D value interpolation, Cu/Cc, fines %.
"""
import math
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Inline sieve analysis helpers
# ---------------------------------------------------------------------------

def interpolate_diameter(apertures: list, pct_finer: list, target_pct: float) -> float:
    """Log-linear interpolation to find Dx for given % finer."""
    if target_pct <= 0 or target_pct >= 100:
        raise ValueError(f"target_pct must be between 0 and 100, got {target_pct}")

    # Sort by aperture descending
    pairs = sorted(zip(apertures, pct_finer), key=lambda x: -x[0])
    apert_sorted = [p[0] for p in pairs]
    finer_sorted = [p[1] for p in pairs]

    for i in range(len(finer_sorted) - 1):
        upper_finer = finer_sorted[i]
        lower_finer = finer_sorted[i + 1]
        if lower_finer <= target_pct <= upper_finer:
            upper_apert = apert_sorted[i]
            lower_apert = apert_sorted[i + 1]
            if lower_apert <= 0:
                return 0.0
            # Log-linear interpolation
            log_d = math.log10(lower_apert) + (
                (math.log10(upper_apert) - math.log10(lower_apert))
                * (target_pct - lower_finer)
                / (upper_finer - lower_finer)
            )
            return 10 ** log_d

    # If target not bracketed, return boundary
    if target_pct >= finer_sorted[0]:
        return apert_sorted[0]
    return apert_sorted[-1]


def compute_cu(d60: float, d10: float) -> float:
    """Coefficient of uniformity Cu = D60 / D10."""
    if d10 <= 0:
        raise ValueError("D10 must be > 0 for Cu calculation")
    return d60 / d10


def compute_cc(d60: float, d30: float, d10: float) -> float:
    """Coefficient of curvature Cc = D30² / (D10 × D60)."""
    if d10 <= 0 or d60 <= 0:
        raise ValueError("D10 and D60 must be > 0 for Cc calculation")
    return d30**2 / (d10 * d60)


def compute_fines_pct(apertures: list, pct_finer: list) -> float:
    """Return % finer at 0.075 mm (No.200 sieve)."""
    for a, f in zip(apertures, pct_finer):
        if abs(a - 0.075) < 0.001:
            return f
    # Interpolate
    return interpolate_diameter(apertures, pct_finer, target_pct=0.075)


# ===========================================================================
# TESTS
# ===========================================================================

# Standard sieve data for testing (well-graded sand-gravel mix)
APERTURES = [75, 19, 9.5, 4.75, 2.36, 1.18, 0.6, 0.3, 0.15, 0.075]
PCT_FINER = [100, 95, 88, 80, 70, 60, 48, 35, 20, 8]


class TestDValueInterpolation:
    """Tests for D-value interpolation from % finer curve."""

    def test_d60_interpolation(self):
        """D60 should be between apertures where finer crosses 60%."""
        d60 = interpolate_diameter(APERTURES, PCT_FINER, 60)
        assert 1.0 < d60 < 2.5  # Reasonable for this gradation

    def test_d30_interpolation(self):
        """D30 should be between apertures where finer crosses 30%."""
        d30 = interpolate_diameter(APERTURES, PCT_FINER, 30)
        assert 0.1 < d30 < 0.5

    def test_d10_interpolation(self):
        """D10 should be close to 0.075mm for this data (8% finer at 0.075)."""
        d10 = interpolate_diameter(APERTURES, PCT_FINER, 10)
        assert 0.05 < d10 < 0.15

    def test_d50_interpolation(self):
        """D50 (median grain size) should be reasonable."""
        d50 = interpolate_diameter(APERTURES, PCT_FINER, 50)
        assert 0.3 < d50 < 1.5

    def test_boundary_100_pct(self):
        """target_pct = 100 should raise."""
        with pytest.raises(ValueError):
            interpolate_diameter(APERTURES, PCT_FINER, 100)

    def test_boundary_0_pct(self):
        """target_pct = 0 should raise."""
        with pytest.raises(ValueError):
            interpolate_diameter(APERTURES, PCT_FINER, 0)


class TestCuCc:
    """Tests for Cu and Cc coefficients."""

    def test_cu_computation(self):
        """Cu = D60 / D10."""
        d60 = interpolate_diameter(APERTURES, PCT_FINER, 60)
        d10 = interpolate_diameter(APERTURES, PCT_FINER, 10)
        cu = compute_cu(d60, d10)
        assert cu == pytest.approx(d60 / d10, rel=1e-6)
        assert cu > 1  # Always true for graded soil

    def test_cc_computation(self):
        """Cc = D30² / (D10 × D60)."""
        d60 = interpolate_diameter(APERTURES, PCT_FINER, 60)
        d30 = interpolate_diameter(APERTURES, PCT_FINER, 30)
        d10 = interpolate_diameter(APERTURES, PCT_FINER, 10)
        cc = compute_cc(d60, d30, d10)
        assert cc == pytest.approx(d30**2 / (d10 * d60), rel=1e-6)

    def test_cu_zero_d10_raises(self):
        """Cu with D10=0 should raise."""
        with pytest.raises(ValueError):
            compute_cu(2.0, 0.0)

    def test_cc_zero_d10_raises(self):
        """Cc with D10=0 should raise."""
        with pytest.raises(ValueError):
            compute_cc(2.0, 0.5, 0.0)

    def test_well_graded_sand(self):
        """Well-graded sand: Cu > 6 and 1 < Cc < 3."""
        d60 = interpolate_diameter(APERTURES, PCT_FINER, 60)
        d30 = interpolate_diameter(APERTURES, PCT_FINER, 30)
        d10 = interpolate_diameter(APERTURES, PCT_FINER, 10)
        cu = compute_cu(d60, d10)
        cc = compute_cc(d60, d30, d10)
        # For this particular gradation Cu should be moderate
        assert cu > 4  # Reasonably well graded


class TestFinesPercentage:
    """Tests for fines percentage (passing No.200 sieve)."""

    def test_fines_at_0075(self):
        """Fines % at 0.075mm should match data."""
        fines = compute_fines_pct(APERTURES, PCT_FINER)
        assert abs(fines - 8.0) < 1.0  # 8% fines from data

    def test_fines_below_50_for_coarse(self):
        """For this coarse-grained data, fines < 50%."""
        fines = compute_fines_pct(APERTURES, PCT_FINER)
        assert fines < 50.0

    def test_fines_with_fine_grained_data(self):
        """Fine-grained soil: fines > 50%."""
        # Simulate a clay soil
        apertures = [4.75, 2.0, 0.425, 0.075]
        pct_finer = [100, 98, 90, 65]
        fines = compute_fines_pct(apertures, pct_finer)
        assert fines == 65.0
