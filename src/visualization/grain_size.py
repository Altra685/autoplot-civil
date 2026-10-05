"""grain_size.py - Grain size distribution chart for AutoPlot Civil."""
from __future__ import annotations
from typing import Optional
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from .chart_builder import CHART_COLORS, CHART_LAYOUT_BASE, _AXIS_STYLE


def calculate_gradation_params(aperture_mm: pd.Series, pct_finer: pd.Series) -> dict:
    """Interpolate D10, D30, D60 and compute Cu, Cc from grain size curve."""
    arr_mm = np.array(aperture_mm, dtype=float)
    arr_pf = np.array(pct_finer, dtype=float)
    sort_idx = np.argsort(arr_pf)
    arr_mm = arr_mm[sort_idx]
    arr_pf = arr_pf[sort_idx]
    result: dict = {"D10": None, "D30": None, "D60": None, "Cu": None, "Cc": None}
    try:
        d10 = float(np.interp(10.0, arr_pf, arr_mm))
        d30 = float(np.interp(30.0, arr_pf, arr_mm))
        d60 = float(np.interp(60.0, arr_pf, arr_mm))
        result["D10"] = d10
        result["D30"] = d30
        result["D60"] = d60
        if d10 > 0:
            result["Cu"] = round(d60 / d10, 2)
        if d10 > 0 and d60 > 0:
            result["Cc"] = round((d30 ** 2) / (d10 * d60), 2)
    except Exception:
        pass
    return result


def create_grain_size_chart(
    df: pd.DataFrame,
    sieve_col: str = "aperture_mm",
    pct_finer_col: str = "pct_finer",
    sample_id_col: Optional[str] = None,
    height: int = 520,
) -> go.Figure:
    """Create grain size distribution chart with log X-axis and zone bands."""
    df = df.copy()
    required = [sieve_col, pct_finer_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    fig = go.Figure()

    # Zone bands (vertical)
    zones = [
        (4.75, 100.0, "rgba(139, 90, 43, 0.12)", "Gravel"),
        (0.075, 4.75, "rgba(255, 215, 0, 0.12)", "Sand"),
        (0.002, 0.075, "rgba(144, 238, 144, 0.12)", "Silt"),
        (0.0001, 0.002, "rgba(173, 216, 230, 0.15)", "Clay"),
    ]
    for x0, x1, color, label in zones:
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=color, layer="below", line_width=0,
            annotation_text=label, annotation_position="top left",
            annotation_font=dict(size=9, color="#555555"),
        )

    # Zone boundary vertical lines
    for bx, blabel in [(4.75, "4.75mm"), (0.075, "0.075mm"), (0.002, "0.002mm")]:
        fig.add_vline(
            x=bx, line=dict(color="#999999", width=1, dash="dot"),
            annotation_text=blabel, annotation_position="bottom right",
            annotation_font=dict(size=8, color="#999999"),
        )

    # Plot curves
    colors = CHART_COLORS["sequential"]
    if sample_id_col and sample_id_col in df.columns:
        samples = df[sample_id_col].unique()
        for i, sid in enumerate(samples):
            sub = df[df[sample_id_col] == sid].sort_values(sieve_col)
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=sub[sieve_col], y=sub[pct_finer_col],
                mode="lines+markers", name=str(sid),
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                hovertemplate="Size: %{x:.4f} mm<br>% Finer: %{y:.1f}%<extra></extra>",
            ))
            # D values for each sample
            params = calculate_gradation_params(sub[sieve_col], sub[pct_finer_col])
            _add_d_value_lines(fig, params, color, str(sid))
    else:
        df_sorted = df.sort_values(sieve_col)
        fig.add_trace(go.Scatter(
            x=df_sorted[sieve_col], y=df_sorted[pct_finer_col],
            mode="lines+markers", name="Sample",
            line=dict(color=CHART_COLORS["primary"], width=2.5),
            marker=dict(size=7, color=CHART_COLORS["primary"]),
            hovertemplate="Size: %{x:.4f} mm<br>% Finer: %{y:.1f}%<extra></extra>",
        ))
        params = calculate_gradation_params(df_sorted[sieve_col], df_sorted[pct_finer_col])
        _add_d_value_lines(fig, params, CHART_COLORS["primary"], "")

    # Layout
    fig.update_layout(**{
        **CHART_LAYOUT_BASE,
        "title": {**CHART_LAYOUT_BASE["title"], "text": "Grain Size Distribution Curve"},
        "height": height,
        "xaxis": {**_AXIS_STYLE, "type": "log", "title_text": "Particle Size (mm)",
                  "range": [-4, 2], "dtick": 1},
        "yaxis": {**_AXIS_STYLE, "title_text": "Percent Finer (%)", "range": [0, 105]},
    })
    return fig


def _add_d_value_lines(fig: go.Figure, params: dict, color: str, sample_label: str) -> None:
    """Add D10, D30, D60 vertical dashed marker lines to the figure."""
    label_sfx = f" ({sample_label})" if sample_label else ""
    for d_key, pct_val in [("D10", 10), ("D30", 30), ("D60", 60)]:
        val = params.get(d_key)
        if val is None or val <= 0:
            continue
        fig.add_vline(
            x=val,
            line=dict(color=color, width=1, dash="dash"),
            annotation_text=f"{d_key}={val:.3f}mm{label_sfx}",
            annotation_position="bottom left",
            annotation_font=dict(size=8, color=color),
        )
        # Horizontal line at pct_val
        fig.add_hline(
            y=pct_val,
            line=dict(color="rgba(150,150,150,0.4)", width=0.8, dash="dot"),
        )
