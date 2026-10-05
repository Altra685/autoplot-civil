"""
Civil engineering calculation modules.

Unit conversion, concrete compressive strength, soil classification, sieve
analysis, and Proctor compaction.
"""
from __future__ import annotations

from .units import convert, format_unit_value
from .concrete import (
    CORRECTION_FACTORS, QUALITY_GRADES, STANDARD_FC_TARGETS, AGE_FACTORS,
    calculate_compressive_strength, analyze_concrete_batch,
    strength_age_regression, predict_strength_at_age,
)
from .soil_classification import (
    classify_soil, extract_gradation_params, ClassificationResult, GradationParams,
)
from .sieve import (
    STANDARD_SIEVES_MM, process_sieve_data, calculate_gradation_coefficients,
    classify_from_gradation, to_display_dataframe,
)
from .compaction import (
    calculate_dry_density, zero_air_voids_line, fit_proctor_curve,
    find_mdd_omc, analyze_compaction,
)
from .statistics import (
    calculate_statistics, test_normality, identify_outliers, regression_analysis,
)

__all__ = [
    "convert", "format_unit_value",
    "calculate_compressive_strength", "analyze_concrete_batch",
    "classify_soil", "extract_gradation_params",
    "process_sieve_data", "calculate_gradation_coefficients",
    "calculate_dry_density", "fit_proctor_curve", "find_mdd_omc", "analyze_compaction",
    "calculate_statistics", "test_normality", "identify_outliers", "regression_analysis",
]