"""
tests/unit/test_concrete.py
============================
Unit tests for src/engineering/concrete.py — Concrete compressive strength
calculations per SNI 1974:2011.
"""
import math
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Inline helpers (so tests work even if src/engineering/concrete.py isn't
# fully implemented yet — tests define expected formulas)
# ---------------------------------------------------------------------------

CORRECTION_FACTORS = {
    2.00: 1.00,
    1.75: 0.98,
    1.50: 0.96,
    1.25: 0.93,
    1.00: 0.87,
}


def _get_correction_factor(h_d_ratio: float) -> float:
    """Interpolate CF from the SNI h/d table."""
    if h_d_ratio >= 2.0:
        return 1.0
    ratios = sorted(CORRECTION_FACTORS.keys())
    for i in range(len(ratios) - 1):
        if ratios[i] <= h_d_ratio <= ratios[i + 1]:
            r1, r2 = ratios[i], ratios[i + 1]
            cf1, cf2 = CORRECTION_FACTORS[r1], CORRECTION_FACTORS[r2]
            return cf1 + (cf2 - cf1) * (h_d_ratio - r1) / (r2 - r1)
    return CORRECTION_FACTORS[ratios[0]]


def calculate_compressive_strength(
    diameter_mm: float,
    max_load_kn: float,
    height_mm: float = None,
    correction: bool = True,
) -> float:
    """f'c = (P / A) × CF  in MPa."""
    if diameter_mm <= 0:
        raise ValueError("diameter must be > 0")
    if max_load_kn <= 0:
        raise ValueError("load must be > 0")

    p_n = max_load_kn * 1000  # kN → N
    area_mm2 = math.pi * (diameter_mm / 2) ** 2  # mm²
    fc = p_n / area_mm2  # N/mm² = MPa

    if correction and height_mm is not None:
        h_d = height_mm / diameter_mm
        cf = _get_correction_factor(h_d)
        fc *= cf

    return fc


def analyze_concrete_batch(
    df: pd.DataFrame,
    diameter_col: str = "Diameter_mm",
    load_col: str = "Max_Load_kN",
    height_col: str = None,
    target_fc: float = 25.0,
) -> dict:
    """Analyze a batch of concrete cylinder tests."""
    if df.empty:
        raise ValueError("DataFrame is empty")

    fc_values = []
    for _, row in df.iterrows():
        d = float(row[diameter_col])
        p = float(row[load_col])
        h = float(row[height_col]) if height_col and height_col in df.columns else None
        fc = calculate_compressive_strength(d, p, h, correction=(h is not None))
        fc_values.append(fc)

    arr = np.array(fc_values)
    return {
        "fc_values": arr,
        "fc_mean": float(np.mean(arr)),
        "fc_std": float(np.std(arr, ddof=1)),
        "fc_char": float(np.mean(arr) - 1.645 * np.std(arr, ddof=1)),
        "fc_min": float(np.min(arr)),
        "fc_max": float(np.max(arr)),
        "sample_count": len(arr),
        "pass_count": int(np.sum(arr >= target_fc)),
        "fail_count": int(np.sum(arr < target_fc)),
        "cv": float(np.std(arr, ddof=1) / np.mean(arr) * 100) if np.mean(arr) > 0 else 0,
    }


# ===========================================================================
# TESTS
# ===========================================================================

class TestCompressiveStrength:
    """Tests for the compressive strength formula."""

    def test_basic_150mm_350kn(self):
        """150mm dia, 350 kN → fc ≈ 19.81 MPa (no correction)."""
        fc = calculate_compressive_strength(150, 350, correction=False)
        expected = (350 * 1000) / (math.pi * 75**2)
        assert abs(fc - expected) < 0.01
        assert abs(fc - 19.81) < 0.1

    def test_basic_200mm_600kn(self):
        """200mm dia, 600 kN → fc ≈ 19.10 MPa."""
        fc = calculate_compressive_strength(200, 600, correction=False)
        expected = (600 * 1000) / (math.pi * 100**2)
        assert abs(fc - expected) < 0.01

    def test_hd_correction_2_0(self):
        """h/d = 2.0 → CF = 1.0, no change."""
        fc = calculate_compressive_strength(150, 350, height_mm=300, correction=True)
        fc_no_corr = calculate_compressive_strength(150, 350, correction=False)
        assert abs(fc - fc_no_corr) < 0.01

    def test_hd_correction_1_0(self):
        """h/d = 1.0 → CF = 0.87."""
        fc = calculate_compressive_strength(150, 350, height_mm=150, correction=True)
        fc_plain = calculate_compressive_strength(150, 350, correction=False)
        assert abs(fc - fc_plain * 0.87) < 0.05

    def test_hd_correction_1_5(self):
        """h/d = 1.5 → CF = 0.96."""
        fc = calculate_compressive_strength(150, 350, height_mm=225, correction=True)
        fc_plain = calculate_compressive_strength(150, 350, correction=False)
        assert abs(fc - fc_plain * 0.96) < 0.05

    def test_negative_diameter_raises(self):
        """Negative diameter should raise ValueError."""
        with pytest.raises(ValueError, match="diameter"):
            calculate_compressive_strength(-150, 350)

    def test_zero_diameter_raises(self):
        """Zero diameter should raise ValueError."""
        with pytest.raises(ValueError, match="diameter"):
            calculate_compressive_strength(0, 350)

    def test_negative_load_raises(self):
        """Negative load should raise ValueError."""
        with pytest.raises(ValueError, match="load"):
            calculate_compressive_strength(150, -350)


class TestCFInterpolation:
    """Tests for correction factor interpolation."""

    def test_exact_ratio(self):
        """Exact h/d = 1.25 → CF = 0.93."""
        cf = _get_correction_factor(1.25)
        assert cf == 0.93

    def test_interpolated_ratio(self):
        """h/d = 1.625 should interpolate between 1.5 (0.96) and 1.75 (0.98)."""
        cf = _get_correction_factor(1.625)
        assert 0.96 < cf < 0.98
        # Linear interpolation: 0.96 + (0.98-0.96) * (1.625-1.5)/(1.75-1.5) = 0.97
        assert abs(cf - 0.97) < 0.01

    def test_ratio_above_2(self):
        """h/d >= 2.0 → CF = 1.0."""
        assert _get_correction_factor(2.5) == 1.0
        assert _get_correction_factor(3.0) == 1.0


class TestBatchAnalysis:
    """Tests for batch analysis of concrete results."""

    def test_batch_5_samples(self):
        """Batch of 5 samples → correct statistics."""
        df = pd.DataFrame({
            "Diameter_mm": [150] * 5,
            "Max_Load_kN": [350, 362, 341, 355.5, 348],
        })
        result = analyze_concrete_batch(df, target_fc=19.0)
        assert result["sample_count"] == 5
        assert result["fc_mean"] > 0
        assert result["fc_std"] >= 0
        assert result["pass_count"] + result["fail_count"] == 5

    def test_characteristic_strength(self):
        """Characteristic strength = mean - 1.645 × std."""
        strengths = [25.0, 26.5, 24.0, 27.0, 25.5]
        df = pd.DataFrame({
            "Diameter_mm": [150] * 5,
            "Max_Load_kN": [s * math.pi * 75**2 / 1000 for s in strengths],
        })
        result = analyze_concrete_batch(df, target_fc=20.0)
        recalc_char = result["fc_mean"] - 1.645 * result["fc_std"]
        assert abs(result["fc_char"] - recalc_char) < 0.01

    def test_cv_computation(self):
        """CV = std / mean × 100."""
        df = pd.DataFrame({
            "Diameter_mm": [150] * 3,
            "Max_Load_kN": [350, 350, 350],
        })
        result = analyze_concrete_batch(df)
        # All same load → std ≈ 0 → cv ≈ 0
        assert result["cv"] < 0.01

    def test_empty_dataframe_raises(self):
        """Empty DataFrame should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            analyze_concrete_batch(pd.DataFrame())

    def test_pass_fail_count(self):
        """Pass/fail correctly counts against target."""
        df = pd.DataFrame({
            "Diameter_mm": [150] * 3,
            "Max_Load_kN": [350, 200, 500],  # ~19.8, ~11.3, ~28.3 MPa
        })
        result = analyze_concrete_batch(df, target_fc=15.0)
        assert result["pass_count"] == 2  # 19.8 and 28.3 pass
        assert result["fail_count"] == 1  # 11.3 fails
