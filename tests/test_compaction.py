"""
tests/unit/test_compaction.py
=============================
Unit tests for compaction/Proctor calculations — dry density, MDD/OMC, ZAV line.
"""
import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Inline compaction helpers
# ---------------------------------------------------------------------------

def calculate_dry_density(wet_density: float, water_content_pct: float) -> float:
    """γd = γwet / (1 + w/100)."""
    if wet_density <= 0:
        raise ValueError("wet_density must be > 0")
    if water_content_pct < 0:
        raise ValueError("water_content_pct must be >= 0")
    return wet_density / (1 + water_content_pct / 100)


def find_mdd_omc(water_contents: list, dry_densities: list) -> tuple:
    """Find MDD and OMC from Proctor test data using polynomial fit."""
    if len(water_contents) < 3:
        raise ValueError("Need at least 3 data points")
    wc = np.array(water_contents, dtype=float)
    gd = np.array(dry_densities, dtype=float)
    # Quadratic polynomial fit
    coeffs = np.polyfit(wc, gd, 2)
    # Peak at -b / 2a
    a, b, c = coeffs
    if a >= 0:
        raise ValueError("Curve does not have a maximum (not concave down)")
    omc = -b / (2 * a)
    mdd = np.polyval(coeffs, omc)
    return float(mdd), float(omc)


def zero_air_voids_line(
    water_contents: list,
    Gs: float = 2.65,
    gamma_w: float = 1.0,
) -> list:
    """ZAV line: γd_sat = Gs × γw / (1 + Gs × w/100)."""
    results = []
    for w in water_contents:
        gd_sat = (Gs * gamma_w) / (1 + Gs * w / 100)
        results.append(gd_sat)
    return results


# ===========================================================================
# TESTS
# ===========================================================================

class TestDryDensity:
    """Tests for dry density formula."""

    def test_basic_calculation(self):
        """γwet=1.9, w=15% → γd ≈ 1.652 g/cm³."""
        gd = calculate_dry_density(1.9, 15.0)
        expected = 1.9 / (1 + 15.0 / 100)
        assert abs(gd - expected) < 0.001
        assert abs(gd - 1.6522) < 0.01

    def test_zero_water_content(self):
        """w=0% → γd = γwet."""
        gd = calculate_dry_density(1.8, 0.0)
        assert abs(gd - 1.8) < 0.001

    def test_high_water_content(self):
        """w=30% → γd is noticeably lower than γwet."""
        gd = calculate_dry_density(2.0, 30.0)
        assert gd < 2.0
        assert abs(gd - 2.0 / 1.3) < 0.001

    def test_negative_wet_density_raises(self):
        """Negative wet density should raise."""
        with pytest.raises(ValueError):
            calculate_dry_density(-1.5, 10.0)

    def test_negative_water_content_raises(self):
        """Negative water content should raise."""
        with pytest.raises(ValueError):
            calculate_dry_density(1.8, -5.0)


class TestMDDOMC:
    """Tests for MDD and OMC from polynomial fit."""

    def test_known_parabola(self):
        """Known parabola data → MDD/OMC accurate."""
        # Create data that follows γd = -0.01*(w-14)^2 + 1.82
        wc = [8, 10, 12, 14, 16, 18, 20]
        gd = [-0.01 * (w - 14)**2 + 1.82 for w in wc]
        mdd, omc = find_mdd_omc(wc, gd)
        assert abs(omc - 14.0) < 0.5
        assert abs(mdd - 1.82) < 0.02

    def test_realistic_proctor_data(self):
        """Realistic Proctor test data."""
        wc = [10, 12, 14, 16, 18]
        gd = [1.65, 1.74, 1.82, 1.78, 1.70]
        mdd, omc = find_mdd_omc(wc, gd)
        assert 1.70 < mdd < 1.90  # MDD in realistic range
        assert 12 < omc < 16      # OMC in realistic range

    def test_insufficient_points_raises(self):
        """Less than 3 points should raise."""
        with pytest.raises(ValueError):
            find_mdd_omc([10, 12], [1.65, 1.74])

    def test_mdd_is_peak(self):
        """MDD should be the maximum dry density."""
        wc = [8, 10, 12, 14, 16, 18, 20]
        gd = [1.55, 1.65, 1.74, 1.82, 1.78, 1.70, 1.60]
        mdd, omc = find_mdd_omc(wc, gd)
        assert mdd >= max(gd) - 0.05  # Fitted peak close to data max


class TestZAVLine:
    """Tests for Zero Air Voids line."""

    def test_zav_decreasing(self):
        """ZAV density should decrease with increasing water content."""
        wc = [8, 10, 12, 14, 16, 18, 20]
        zav = zero_air_voids_line(wc, Gs=2.65)
        for i in range(len(zav) - 1):
            assert zav[i] > zav[i + 1]

    def test_zav_formula(self):
        """Verify ZAV formula: γd_sat = Gs×γw / (1 + Gs×w/100)."""
        Gs = 2.65
        w = 14.0
        expected = Gs * 1.0 / (1 + Gs * w / 100)
        result = zero_air_voids_line([w], Gs=Gs)
        assert abs(result[0] - expected) < 0.001

    def test_zav_at_zero_water(self):
        """At w=0%, ZAV = Gs×γw = 2.65 g/cm³."""
        zav = zero_air_voids_line([0.0], Gs=2.65)
        assert abs(zav[0] - 2.65) < 0.001

    def test_zav_above_proctor(self):
        """ZAV line should always be above actual dry densities."""
        wc = [10, 12, 14, 16, 18]
        gd = [1.65, 1.74, 1.82, 1.78, 1.70]
        zav = zero_air_voids_line(wc, Gs=2.65)
        for z, g in zip(zav, gd):
            assert z > g, "ZAV must be above measured dry density"
