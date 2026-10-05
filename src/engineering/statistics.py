"""
statistics.py — Statistical Analysis for Engineering Data
EN 1990 / ISO 2394 / General engineering statistics

Provides:
  - StatisticalSummary: dataclass with all statistical measures
  - calculate_statistics(): mean, std, CV, skewness, kurtosis, CI
  - test_normality(): Shapiro-Wilk normality test
  - calculate_characteristic_strength(): EN 1990 characteristic value
  - identify_outliers(): IQR + Z-score outlier detection
  - regression_analysis(): linear, logarithmic, power, polynomial
  - RegressionResult: dataclass for fitted models
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from scipy import stats as sp_stats
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Student's t critical values for 95% CI (two-tailed, α=0.05)
# df → t_critical
T_CRITICAL_95: dict[int, float] = {
    1:   12.706,
    2:    4.303,
    3:    3.182,
    4:    2.776,
    5:    2.571,
    6:    2.447,
    7:    2.365,
    8:    2.306,
    9:    2.262,
    10:   2.228,
    11:   2.201,
    12:   2.179,
    13:   2.160,
    14:   2.145,
    15:   2.131,
    16:   2.120,
    17:   2.110,
    18:   2.101,
    19:   2.093,
    20:   2.086,
    25:   2.060,
    30:   2.042,
    40:   2.021,
    50:   2.009,
    60:   2.000,
    80:   1.990,
    100:  1.984,
    120:  1.980,
    200:  1.972,
    500:  1.965,
    1000: 1.962,
}

# EN 1990 kn factors for characteristic values
# n (sample size) → kn (for known Vx | for unknown Vx)
EN1990_KN_FACTORS: dict[int, tuple[float, float]] = {
    # n: (kn_known, kn_unknown)
    1:   (2.31, float("inf")),
    2:   (2.01, float("inf")),
    3:   (1.89, 3.37),
    4:   (1.83, 2.63),
    5:   (1.80, 2.33),
    6:   (1.77, 2.18),
    8:   (1.74, 2.00),
    10:  (1.72, 1.92),
    12:  (1.71, 1.87),
    15:  (1.70, 1.82),
    20:  (1.68, 1.76),
    25:  (1.67, 1.74),
    30:  (1.67, 1.73),
    40:  (1.66, 1.70),
    50:  (1.66, 1.68),
    100: (1.64, 1.66),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StatisticalSummary:
    """Complete statistical summary of a dataset."""

    n: int                              # Sample size
    mean: float                         # Arithmetic mean
    median: float                       # Median
    std: float                          # Standard deviation (sample, ddof=1)
    variance: float                     # Sample variance
    cv: float                           # Coefficient of variation (%)
    skewness: float                     # Sample skewness
    kurtosis: float                     # Excess kurtosis
    minimum: float
    maximum: float
    range: float                        # max - min
    q1: float                           # 25th percentile
    q3: float                           # 75th percentile
    iqr: float                          # Interquartile range
    ci_lower: float                     # 95% CI lower bound
    ci_upper: float                     # 95% CI upper bound
    ci_level: float                     # Confidence level (default 0.95)
    geometric_mean: Optional[float]     # For positive datasets
    harmonic_mean: Optional[float]      # For positive datasets


@dataclass
class NormalityTestResult:
    """Normality test result."""

    test_name: str                      # "Shapiro-Wilk" or "D'Agostino-Pearson"
    statistic: float                    # Test statistic
    p_value: float                      # p-value
    is_normal: bool                     # True if p > alpha
    alpha: float                        # Significance level
    interpretation: str                 # Human-readable result


@dataclass
class OutlierResult:
    """Outlier detection result."""

    n_outliers_iqr: int
    n_outliers_zscore: int
    outlier_indices_iqr: list[int]
    outlier_indices_zscore: list[int]
    outlier_values_iqr: list[float]
    outlier_values_zscore: list[float]
    iqr_lower_fence: float
    iqr_upper_fence: float
    zscore_threshold: float
    method_used: str                    # "IQR", "Z-score", or "both"


@dataclass
class RegressionResult:
    """Regression analysis result."""

    model_type: str                     # "linear", "logarithmic", "power", "polynomial"
    coefficients: list[float]           # Model coefficients
    r_squared: float                    # Coefficient of determination
    adjusted_r_squared: float           # Adjusted R²
    rmse: float                         # Root mean squared error
    equation: str                       # Human-readable equation string
    predictions: list[float]            # Fitted values
    residuals: list[float]              # Residuals (observed - predicted)
    n: int                              # Number of data points
    p_degree: Optional[int]             # Polynomial degree (if applicable)


@dataclass
class CharacteristicStrengthResult:
    """Characteristic strength per EN 1990."""

    characteristic_value: float         # Xk = mean - kn × std
    mean: float
    std: float
    kn_factor: float
    sample_size: int
    method: str                         # "EN 1990 known Vx" or "EN 1990 unknown Vx"
    fractile: str                       # "5%" typically


# ---------------------------------------------------------------------------
# Core Statistical Functions
# ---------------------------------------------------------------------------

def _get_t_critical(df: int, confidence: float = 0.95) -> float:
    """Get Student's t critical value for given df and confidence."""
    if HAS_SCIPY:
        alpha = 1.0 - confidence
        return float(sp_stats.t.ppf(1 - alpha / 2, df))  # type: ignore[possibly-unbound]

    # Fallback: interpolation from table
    sorted_dfs = sorted(T_CRITICAL_95.keys())

    if df <= 0:
        return 12.706

    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]

    # Find surrounding entries
    for i in range(len(sorted_dfs) - 1):
        d_low = sorted_dfs[i]
        d_high = sorted_dfs[i + 1]
        if d_low < df < d_high:
            t_low = T_CRITICAL_95[d_low]
            t_high = T_CRITICAL_95[d_high]
            frac = (df - d_low) / (d_high - d_low)
            return t_low + frac * (t_high - t_low)

    # Beyond table — use z-value
    return 1.960


def _get_en1990_kn(n: int, vx_known: bool = False) -> float:
    """Get EN 1990 kn factor for characteristic value calculation."""
    sorted_ns = sorted(EN1990_KN_FACTORS.keys())
    idx = 0 if vx_known else 1

    if n in EN1990_KN_FACTORS:
        return EN1990_KN_FACTORS[n][idx]

    # Interpolate
    for i in range(len(sorted_ns) - 1):
        n_low = sorted_ns[i]
        n_high = sorted_ns[i + 1]
        if n_low < n < n_high:
            kn_low = EN1990_KN_FACTORS[n_low][idx]
            kn_high = EN1990_KN_FACTORS[n_high][idx]
            frac = (n - n_low) / (n_high - n_low)
            return kn_low + frac * (kn_high - kn_low)

    # Beyond table
    if n > sorted_ns[-1]:
        return EN1990_KN_FACTORS[sorted_ns[-1]][idx]

    return EN1990_KN_FACTORS[sorted_ns[0]][idx]


def calculate_statistics(
    data: list[float] | np.ndarray,
    confidence_level: float = 0.95,
) -> StatisticalSummary:
    """
    Calculate comprehensive descriptive statistics.

    Parameters
    ----------
    data : list or ndarray
        Numerical data array.
    confidence_level : float
        Confidence level for CI calculation. Default 0.95.

    Returns
    -------
    StatisticalSummary

    Raises
    ------
    ValueError
        If data is empty or has fewer than 2 points.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]  # Remove NaN

    n = len(arr)
    if n < 1:
        raise ValueError("Data is empty after removing NaN values.")
    if n < 2:
        warnings.warn("Only 1 data point — std, CI, skewness, kurtosis undefined.", stacklevel=2)

    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    variance = std ** 2
    cv = (std / mean * 100.0) if abs(mean) > 1e-15 else 0.0

    # Skewness (Fisher's definition)
    if n >= 3 and std > 0:
        skew = float(np.sum(((arr - mean) / std) ** 3) * n / ((n - 1) * (n - 2)))
    else:
        skew = 0.0

    # Excess kurtosis (Fisher's definition)
    if n >= 4 and std > 0:
        m4 = float(np.mean((arr - mean) ** 4))
        kurt = m4 / (std ** 4) - 3.0  # ddof=0 based fourth moment ratio
    else:
        kurt = 0.0

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    range_val = max_val - min_val

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    # Confidence interval
    if n > 1:
        df = n - 1
        t_crit = _get_t_critical(df, confidence_level)
        se = std / math.sqrt(n)
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
    else:
        ci_lower = mean
        ci_upper = mean

    # Geometric and harmonic means (only for positive data)
    geo_mean: Optional[float] = None
    harm_mean: Optional[float] = None

    if np.all(arr > 0):
        geo_mean = round(float(np.exp(np.mean(np.log(arr)))), 6)
        harm_mean = round(float(n / np.sum(1.0 / arr)), 6)

    return StatisticalSummary(
        n=n,
        mean=round(mean, 6),
        median=round(median, 6),
        std=round(std, 6),
        variance=round(variance, 6),
        cv=round(cv, 4),
        skewness=round(skew, 6),
        kurtosis=round(kurt, 6),
        minimum=round(min_val, 6),
        maximum=round(max_val, 6),
        range=round(range_val, 6),
        q1=round(q1, 6),
        q3=round(q3, 6),
        iqr=round(iqr, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        ci_level=confidence_level,
        geometric_mean=geo_mean,
        harmonic_mean=harm_mean,
    )


def test_normality(
    data: list[float] | np.ndarray,
    alpha: float = 0.05,
    method: str = "shapiro",
) -> NormalityTestResult:
    """
    Test data for normality.

    Shapiro-Wilk test (recommended for n < 5000).
    D'Agostino-Pearson test (for larger samples).

    Parameters
    ----------
    data : list or ndarray
        Numerical data array.
    alpha : float
        Significance level. Default 0.05.
    method : str
        "shapiro" or "dagostino".

    Returns
    -------
    NormalityTestResult

    Raises
    ------
    ValueError
        If insufficient data points.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) < 3:
        raise ValueError("At least 3 data points required for normality test.")

    if not HAS_SCIPY:
        warnings.warn(
            "scipy not available. Normality test approximated via skewness/kurtosis.",
            stacklevel=2,
        )
        # Fallback: approximate via skewness and kurtosis
        stats = calculate_statistics(arr.tolist())
        # Simple heuristic: |skew| < 2 and |kurt| < 7
        is_normal = abs(stats.skewness) < 2.0 and abs(stats.kurtosis) < 7.0
        return NormalityTestResult(
            test_name="Skewness-Kurtosis (fallback)",
            statistic=stats.skewness,
            p_value=0.05 if is_normal else 0.01,
            is_normal=is_normal,
            alpha=alpha,
            interpretation=(
                "Data appears approximately normal (skew/kurtosis within limits)"
                if is_normal else
                "Data appears non-normal (skew/kurtosis exceed limits)"
            ),
        )

    if method == "shapiro":
        stat, p_value = sp_stats.shapiro(arr)  # type: ignore[possibly-unbound]
        test_name = "Shapiro-Wilk"
    elif method == "dagostino":
        if len(arr) < 8:
            raise ValueError("D'Agostino test requires at least 8 data points.")
        stat, p_value = sp_stats.normaltest(arr)  # type: ignore[possibly-unbound]
        test_name = "D'Agostino-Pearson"
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'shapiro' or 'dagostino'.")

    is_normal = p_value > alpha

    interpretation = (
        f"Data is normally distributed (p={p_value:.4f} > α={alpha})"
        if is_normal else
        f"Data is NOT normally distributed (p={p_value:.4f} ≤ α={alpha})"
    )

    return NormalityTestResult(
        test_name=test_name,
        statistic=round(float(stat), 6),
        p_value=round(float(p_value), 6),
        is_normal=is_normal,
        alpha=alpha,
        interpretation=interpretation,
    )


def calculate_characteristic_strength(
    data: list[float] | np.ndarray,
    vx_known: bool = False,
    fractile: float = 0.05,
) -> CharacteristicStrengthResult:
    """
    Calculate characteristic strength per EN 1990 Annex D.

    Xk = mean - kn × s

    where kn depends on sample size and whether the coefficient
    of variation Vx is known a priori.

    Parameters
    ----------
    data : list or ndarray
        Strength values.
    vx_known : bool
        True if Vx is known from prior information.
    fractile : float
        Target fractile (default 0.05 = 5%).

    Returns
    -------
    CharacteristicStrengthResult

    Raises
    ------
    ValueError
        If fewer than 3 data points.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]

    n = len(arr)
    if n < 3:
        raise ValueError("At least 3 data points required for EN 1990 characteristic value.")

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    kn = _get_en1990_kn(n, vx_known)

    if math.isinf(kn):
        warnings.warn(
            f"kn factor is infinite for n={n} with unknown Vx. "
            "Cannot determine characteristic value reliably.",
            stacklevel=2,
        )
        xk = mean  # Conservative fallback
    else:
        xk = mean - kn * std

    method = "EN 1990 known Vx" if vx_known else "EN 1990 unknown Vx"

    return CharacteristicStrengthResult(
        characteristic_value=round(xk, 4),
        mean=round(mean, 4),
        std=round(std, 4),
        kn_factor=round(kn, 4) if not math.isinf(kn) else float("inf"),
        sample_size=n,
        method=method,
        fractile=f"{int(fractile*100)}%",
    )


def identify_outliers(
    data: list[float] | np.ndarray,
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    method: str = "both",
) -> OutlierResult:
    """
    Identify outliers using IQR and/or Z-score methods.

    IQR method:
        Lower fence = Q1 - k × IQR
        Upper fence = Q3 + k × IQR
        Default k = 1.5

    Z-score method:
        Outlier if |z| > threshold (default 3.0)

    Parameters
    ----------
    data : list or ndarray
        Numerical data array.
    iqr_multiplier : float
        IQR multiplier k. Default 1.5.
    zscore_threshold : float
        Z-score threshold. Default 3.0.
    method : str
        "iqr", "zscore", or "both".

    Returns
    -------
    OutlierResult
    """
    arr = np.asarray(data, dtype=float)
    arr_clean = arr[~np.isnan(arr)]

    if len(arr_clean) < 3:
        return OutlierResult(
            n_outliers_iqr=0, n_outliers_zscore=0,
            outlier_indices_iqr=[], outlier_indices_zscore=[],
            outlier_values_iqr=[], outlier_values_zscore=[],
            iqr_lower_fence=0.0, iqr_upper_fence=0.0,
            zscore_threshold=zscore_threshold, method_used=method,
        )

    q1 = float(np.percentile(arr_clean, 25))
    q3 = float(np.percentile(arr_clean, 75))
    iqr = q3 - q1
    lower_fence = q1 - iqr_multiplier * iqr
    upper_fence = q3 + iqr_multiplier * iqr

    mean = float(np.mean(arr_clean))
    std = float(np.std(arr_clean, ddof=1))

    # IQR outliers
    iqr_indices: list[int] = []
    iqr_values: list[float] = []
    if method in ("iqr", "both"):
        for i, v in enumerate(arr_clean):
            if v < lower_fence or v > upper_fence:
                iqr_indices.append(i)
                iqr_values.append(round(float(v), 6))

    # Z-score outliers
    zscore_indices: list[int] = []
    zscore_values: list[float] = []
    if method in ("zscore", "both") and std > 0:
        for i, v in enumerate(arr_clean):
            z = abs((v - mean) / std)
            if z > zscore_threshold:
                zscore_indices.append(i)
                zscore_values.append(round(float(v), 6))

    return OutlierResult(
        n_outliers_iqr=len(iqr_indices),
        n_outliers_zscore=len(zscore_indices),
        outlier_indices_iqr=iqr_indices,
        outlier_indices_zscore=zscore_indices,
        outlier_values_iqr=iqr_values,
        outlier_values_zscore=zscore_values,
        iqr_lower_fence=round(lower_fence, 6),
        iqr_upper_fence=round(upper_fence, 6),
        zscore_threshold=zscore_threshold,
        method_used=method,
    )


def regression_analysis(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    model: str = "linear",
    degree: int = 2,
) -> RegressionResult:
    """
    Perform regression analysis.

    Models:
    - "linear":      y = a×x + b
    - "logarithmic": y = a×ln(x) + b
    - "power":       y = a×x^b  (log-log linear)
    - "polynomial":  y = a_n×x^n + ... + a_1×x + a_0

    Parameters
    ----------
    x : list or ndarray
        Independent variable.
    y : list or ndarray
        Dependent variable.
    model : str
        Model type. Default "linear".
    degree : int
        Polynomial degree (only for "polynomial"). Default 2.

    Returns
    -------
    RegressionResult

    Raises
    ------
    ValueError
        If x and y have different lengths or insufficient data.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Remove NaN pairs
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    n = len(x_arr)
    if n < 2:
        raise ValueError("At least 2 data points required for regression.")
    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"x ({len(x_arr)}) and y ({len(y_arr)}) must have equal length."
        )

    if model == "linear":
        # y = a×x + b
        coeffs = np.polyfit(x_arr, y_arr, 1)
        a, b = float(coeffs[0]), float(coeffs[1])
        y_pred = a * x_arr + b
        equation = f"y = {a:.4f}×x + {b:.4f}"
        coeff_list = [a, b]
        p_deg = 1

    elif model == "logarithmic":
        # y = a×ln(x) + b
        if np.any(x_arr <= 0):
            raise ValueError("Logarithmic model requires all x > 0.")
        ln_x = np.log(x_arr)
        coeffs = np.polyfit(ln_x, y_arr, 1)
        a, b = float(coeffs[0]), float(coeffs[1])
        y_pred = a * ln_x + b
        equation = f"y = {a:.4f}×ln(x) + {b:.4f}"
        coeff_list = [a, b]
        p_deg = None

    elif model == "power":
        # y = a×x^b → log(y) = log(a) + b×log(x)
        if np.any(x_arr <= 0) or np.any(y_arr <= 0):
            raise ValueError("Power model requires all x > 0 and y > 0.")
        log_x = np.log(x_arr)
        log_y = np.log(y_arr)
        coeffs = np.polyfit(log_x, log_y, 1)
        b_exp = float(coeffs[0])
        a_coeff = math.exp(float(coeffs[1]))
        y_pred = a_coeff * x_arr ** b_exp
        equation = f"y = {a_coeff:.4f}×x^{b_exp:.4f}"
        coeff_list = [a_coeff, b_exp]
        p_deg = None

    elif model == "polynomial":
        if degree < 1:
            raise ValueError(f"Polynomial degree must be ≥ 1, got {degree}")
        if degree >= n:
            raise ValueError(
                f"Polynomial degree ({degree}) must be less than n ({n})."
            )
        coeffs = np.polyfit(x_arr, y_arr, degree)
        y_pred = np.polyval(coeffs, x_arr)
        coeff_list = [float(c) for c in coeffs]

        # Build equation string
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power > 1:
                terms.append(f"{c:.4f}×x^{power}")
            elif power == 1:
                terms.append(f"{c:.4f}×x")
            else:
                terms.append(f"{c:.4f}")
        equation = "y = " + " + ".join(terms)
        p_deg = degree

    else:
        raise ValueError(
            f"Unknown model '{model}'. Use 'linear', 'logarithmic', 'power', or 'polynomial'."
        )

    # R-squared
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Adjusted R-squared
    k = len(coeff_list)  # Number of parameters
    if n - k - 1 > 0:
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1)
    else:
        adj_r_squared = r_squared

    # RMSE
    rmse = math.sqrt(ss_res / n)

    # Residuals
    residuals = (y_arr - y_pred).tolist()

    return RegressionResult(
        model_type=model,
        coefficients=coeff_list,
        r_squared=round(r_squared, 6),
        adjusted_r_squared=round(adj_r_squared, 6),
        rmse=round(rmse, 6),
        equation=equation,
        predictions=[round(float(v), 6) for v in y_pred],
        residuals=[round(float(v), 6) for v in residuals],
        n=n,
        p_degree=p_deg,
    )
