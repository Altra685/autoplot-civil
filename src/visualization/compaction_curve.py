"""compaction_curve.py - Proctor compaction curve chart for AutoPlot Civil."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from .chart_builder import CHART_COLORS, CHART_LAYOUT_BASE, _AXIS_STYLE

_GS_DEFAULT = 2.65  # Specific gravity of soil solids
_RHO_W = 1.0  # Density of water, Mg/m3


def fit_compaction_curve(wc: np.ndarray, dry_density: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fit degree-2 polynomial to compaction data. Returns (coeffs, MDD, OMC)."""
    if len(wc) < 3:
        raise ValueError("Need at least 3 data points to fit compaction curve.")
    coeffs = np.polyfit(wc, dry_density, 2)
    if coeffs[0] >= 0:
        raise ValueError("Polynomial fit did not produce a concave-down curve. Check data.")
    omc = -coeffs[1] / (2 * coeffs[0])
    mdd = float(np.polyval(coeffs, omc))
    return coeffs, mdd, float(omc)


def calculate_zav_line(wc_range: np.ndarray, Gs: float = _GS_DEFAULT, rho_w: float = _RHO_W) -> np.ndarray:
    """Zero Air Voids line: rho_d_ZAV = Gs*rho_w / (1 + Gs*w/100) where w is in %."""
    w_frac = wc_range / 100.0
    return (Gs * rho_w) / (1.0 + Gs * w_frac)


def create_compaction_chart(
    df: pd.DataFrame,
    wc_col: str = "water_content",
    dry_density_col: str = "dry_density",
    height: int = 520,
    Gs: float = _GS_DEFAULT,
    show_saturation_lines: bool = True,
    standard_mdd: Optional[float] = None,
) -> go.Figure:
    """Create Proctor compaction curve with MDD/OMC annotation and ZAV line."""
    df = df.copy()
    for col in [wc_col, dry_density_col]:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
    wc_arr = df[wc_col].values.astype(float)
    rd_arr = df[dry_density_col].values.astype(float)

    fig = go.Figure()

    # Data scatter points
    fig.add_trace(go.Scatter(
        x=wc_arr, y=rd_arr, mode="markers", name="Test Data",
        marker=dict(size=9, color=CHART_COLORS["primary"],
                    symbol="circle", line=dict(color="white", width=1)),
        hovertemplate="w: %{x:.1f}%<br>rho_d: %{y:.3f} Mg/m3<extra></extra>",
    ))

    # Polynomial fit
    try:
        coeffs, mdd, omc = fit_compaction_curve(wc_arr, rd_arr)
        wc_smooth = np.linspace(wc_arr.min() - 1, wc_arr.max() + 1, 200)
        rd_smooth = np.polyval(coeffs, wc_smooth)
        fig.add_trace(go.Scatter(
            x=wc_smooth, y=rd_smooth, mode="lines", name="Compaction Curve (Poly Fit)",
            line=dict(color=CHART_COLORS["primary"], width=2.5),
            hovertemplate="w: %{x:.1f}%<br>rho_d: %{y:.3f} Mg/m3<extra></extra>",
        ))
        # MDD and OMC markers
        fig.add_vline(x=omc, line=dict(color=CHART_COLORS["accent"], width=1.5, dash="dash"),
                      annotation_text=f"OMC = {omc:.1f}%",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=CHART_COLORS["accent"]))
        fig.add_hline(y=mdd, line=dict(color="#FF0000", width=1.5, dash="dash"),
                      annotation_text=f"MDD = {mdd:.3f} Mg/m3",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color="#FF0000"))
        # MDD point marker
        fig.add_trace(go.Scatter(
            x=[omc], y=[mdd], mode="markers", name=f"MDD = {mdd:.3f} Mg/m3 @ OMC = {omc:.1f}%",
            marker=dict(size=14, color="#FF0000", symbol="star",
                        line=dict(color="white", width=1)),
        ))
    except ValueError as e:
        mdd, omc = None, None

    # Zero Air Voids line
    wc_range = np.linspace(max(0, wc_arr.min() - 3), wc_arr.max() + 3, 100)
    zav = calculate_zav_line(wc_range, Gs=Gs)
    fig.add_trace(go.Scatter(
        x=wc_range, y=zav, mode="lines", name=f"Zero Air Voids (Gs={Gs})",
        line=dict(color="rgba(0,0,0,0.6)", width=1.5, dash="dashdot"),
        hovertemplate="w: %{x:.1f}%<br>ZAV rho_d: %{y:.3f} Mg/m3<extra></extra>",
    ))

    # Relative compaction lines (80, 90, 95%) - only if MDD available
    if show_saturation_lines and mdd is not None:
        for pct, color in [(0.80, "rgba(255,165,0,0.7)"), (0.90, "rgba(255,100,0,0.7)"), (0.95, "rgba(255,0,0,0.7)")]:
            fig.add_hline(
                y=mdd * pct,
                line=dict(color=color, width=1, dash="dot"),
                annotation_text=f"{int(pct*100)}% RC = {mdd*pct:.3f} Mg/m3",
                annotation_position="right",
                annotation_font=dict(size=8, color=color),
            )

    # Standard MDD line (from lab specification, if provided)
    if standard_mdd is not None:
        fig.add_hline(
            y=standard_mdd,
            line=dict(color="#7030A0", width=2, dash="longdash"),
            annotation_text=f"Standard MDD = {standard_mdd:.3f} Mg/m3",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#7030A0"),
        )

    fig.update_layout(**{
        **CHART_LAYOUT_BASE,
        "title": {**CHART_LAYOUT_BASE["title"], "text": "Proctor Compaction Curve"},
        "height": height,
        "xaxis": {**_AXIS_STYLE, "title_text": "Water Content (%)"},
        "yaxis": {**_AXIS_STYLE, "title_text": "Dry Density (Mg/m³)"},
    })
    return fig
