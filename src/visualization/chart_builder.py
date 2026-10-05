"""
chart_builder.py
================
Core Plotly chart builder utilities for AutoPlot Civil.
Provides base figures, layout presets, annotations, and export helpers
used by all domain-specific chart modules.
"""

from __future__ import annotations

import io
import base64
from typing import Literal, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import streamlit as st
except ImportError:
    st = None

# ---------------------------------------------------------------------------
# COLOR PALETTE
# ---------------------------------------------------------------------------
CHART_COLORS: dict[str, str | list[str]] = {
    # Primary brand color
    "primary": "#1F4E78",
    # Accent / highlight
    "accent": "#2E75B6",
    # Sequential palette for multi-series
    "sequential": [
        "#1F4E78",
        "#2E75B6",
        "#ED7D31",
        "#A9D18E",
        "#FF0000",
        "#7030A0",
        "#00B0F0",
        "#FFC000",
    ],
    # Status colors
    "pass_green": "#70AD47",
    "fail_red": "#FF0000",
    "warning_orange": "#ED7D31",
    # Soil classification fills
    "gravel": "rgba(139, 90, 43, 0.25)",
    "sand": "rgba(255, 215, 0, 0.25)",
    "silt": "rgba(144, 238, 144, 0.25)",
    "clay": "rgba(173, 216, 230, 0.25)",
    # Safety band
    "safety_band": "rgba(255, 165, 0, 0.15)",
    "danger_band": "rgba(255, 0, 0, 0.10)",
    # Gridline
    "grid": "rgba(200, 200, 200, 0.5)",
    # Background
    "plot_bg": "#FFFFFF",
    "paper_bg": "#FFFFFF",
}

# ---------------------------------------------------------------------------
# AXIS STYLE (shared by all domain chart modules)
# ---------------------------------------------------------------------------
_AXIS_STYLE: dict = {
    "showgrid": True,
    "gridcolor": "rgba(200,200,200,0.5)",
    "gridwidth": 1,
    "showline": True,
    "linecolor": "#AAAAAA",
    "linewidth": 1,
    "mirror": True,
    "ticks": "outside",
    "tickfont": {"size": 11},
    "titlefont": {"size": 12, "color": "#1F4E78"},
    "zeroline": False,
}

# ---------------------------------------------------------------------------
# LAYOUT BASE
# ---------------------------------------------------------------------------
CHART_LAYOUT_BASE: dict = {
    "template": "plotly_white",
    "font": {
        "family": "Arial, Helvetica, sans-serif",
        "size": 12,
        "color": "#1A1A1A",
    },
    "title": {
        "font": {"size": 16, "color": "#1F4E78", "family": "Arial, Helvetica, sans-serif"},
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    "margin": {"l": 70, "r": 40, "t": 70, "b": 70},
    "plot_bgcolor": "#FFFFFF",
    "paper_bgcolor": "#FFFFFF",
    "legend": {
        "bgcolor": "rgba(255,255,255,0.9)",
        "bordercolor": "#CCCCCC",
        "borderwidth": 1,
        "font": {"size": 11},
    },
    "hoverlabel": {
        "bgcolor": "white",
        "bordercolor": "#1F4E78",
        "font": {"size": 12, "color": "#1A1A1A"},
    },
    "xaxis": {**_AXIS_STYLE},
    "yaxis": {**_AXIS_STYLE},
}


def _is_dark_theme() -> bool:
    if st is None:
        return False
    settings = st.session_state.get("apc_settings", {})
    return str(settings.get("theme", "light")).lower() == "dark"


def _get_axis_style() -> dict:
    if _is_dark_theme():
        return {
            **_AXIS_STYLE,
            "gridcolor": "rgba(123, 146, 181, 0.30)",
            "linecolor": "#5B6E8A",
            "tickfont": {"size": 11, "color": "#DCE6F5"},
            "titlefont": {"size": 12, "color": "#CFE0FF"},
        }
    return {**_AXIS_STYLE}


def _get_layout_base() -> dict:
    if not _is_dark_theme():
        base = {**CHART_LAYOUT_BASE}
        base["xaxis"] = _get_axis_style()
        base["yaxis"] = _get_axis_style()
        return base

    return {
        **CHART_LAYOUT_BASE,
        "template": "plotly_dark",
        "font": {
            "family": "Arial, Helvetica, sans-serif",
            "size": 12,
            "color": "#E8EDF6",
        },
        "title": {
            "font": {"size": 16, "color": "#CFE0FF", "family": "Arial, Helvetica, sans-serif"},
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        "plot_bgcolor": "rgba(13, 18, 32, 0.55)",
        "paper_bgcolor": "rgba(13, 18, 32, 0.25)",
        "legend": {
            "bgcolor": "rgba(13, 18, 32, 0.90)",
            "bordercolor": "#465B7A",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#E8EDF6"},
        },
        "hoverlabel": {
            "bgcolor": "rgba(13, 18, 32, 0.95)",
            "bordercolor": "#4FC3F7",
            "font": {"size": 12, "color": "#E8EDF6"},
        },
        "xaxis": _get_axis_style(),
        "yaxis": _get_axis_style(),
    }


def get_theme_layout_base() -> dict:
    return _get_layout_base()


def get_theme_axis_style() -> dict:
    return _get_axis_style()


def get_theme_tokens() -> dict:
    if _is_dark_theme():
        return {
            "annotation_text": "#E8EDF6",
            "muted_text": "#C7D4E8",
            "annotation_bg": "rgba(13, 18, 32, 0.90)",
            "annotation_border": "#4A5A75",
            "line_muted": "#9FB0C8",
            "marker_outline": "#0D1220",
        }
    return {
        "annotation_text": "#333333",
        "muted_text": "#666666",
        "annotation_bg": "rgba(255,255,255,0.85)",
        "annotation_border": "#CCCCCC",
        "line_muted": "#999999",
        "marker_outline": "white",
    }


# ---------------------------------------------------------------------------
# BASE FIGURE FACTORY
# ---------------------------------------------------------------------------

def create_base_figure(
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    height: int = 520,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Create a single-panel Plotly figure with the AutoPlot Civil engineering
    layout preset applied.

    Parameters
    ----------
    title : str
        Chart title text.
    x_title : str
        X-axis label text.
    y_title : str
        Y-axis label text.
    height : int
        Figure height in pixels.
    width : int or None
        Figure width in pixels. None means responsive.

    Returns
    -------
    go.Figure
    """
    base_layout = _get_layout_base()
    layout_kwargs = {**base_layout}
    layout_kwargs["title"] = {**base_layout["title"], "text": title}
    layout_kwargs["height"] = height
    if width is not None:
        layout_kwargs["width"] = width

    fig = go.Figure()
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(title_text=x_title, **_get_axis_style())
    fig.update_yaxes(title_text=y_title, **_get_axis_style())
    return fig


# ---------------------------------------------------------------------------
# THRESHOLD / BAND HELPERS
# ---------------------------------------------------------------------------

def add_threshold_line(
    fig: go.Figure,
    value: float,
    orientation: Literal["h", "v"] = "h",
    label: str = "",
    color: str = "#FF0000",
    dash: str = "dash",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add a horizontal or vertical threshold line with an annotation.

    Parameters
    ----------
    fig : go.Figure
    value : float
        Position of the line on the relevant axis.
    orientation : 'h' or 'v'
        'h' → horizontal line (constant y), 'v' → vertical line (constant x).
    label : str
        Annotation text shown near the line.
    color : str
        Line color hex string.
    dash : str
        Plotly dash style: 'dash', 'dot', 'dashdot', 'solid'.
    row, col : int or None
        Target subplot row/col (1-indexed). None targets the whole figure.
    """
    line_kwargs: dict = {}
    if row is not None:
        line_kwargs["row"] = row
    if col is not None:
        line_kwargs["col"] = col

    if orientation == "h":
        fig.add_hline(
            y=value,
            line=dict(color=color, width=1.5, dash=dash),
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=10, color=color),
            **line_kwargs,
        )
    else:
        fig.add_vline(
            x=value,
            line=dict(color=color, width=1.5, dash=dash),
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=10, color=color),
            **line_kwargs,
        )
    return fig


def add_safety_band(
    fig: go.Figure,
    y_lower: float,
    y_upper: float,
    fill_color: str = "rgba(255, 165, 0, 0.15)",
    label: str = "Acceptable Range",
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Add a horizontal rectangular safety band (hrect) between y_lower and y_upper.

    Parameters
    ----------
    fig : go.Figure
    y_lower : float
        Lower bound of the band.
    y_upper : float
        Upper bound of the band.
    fill_color : str
        RGBA fill color string.
    label : str
        Annotation text displayed inside the band.
    row, col : int or None
        Target subplot row/col (1-indexed). None for whole figure.
    """
    hrect_kwargs: dict = {
        "y0": y_lower,
        "y1": y_upper,
        "fillcolor": fill_color,
        "layer": "below",
        "line_width": 0,
        "annotation_text": label,
        "annotation_position": "top left",
        "annotation_font": dict(size=9, color="#555555"),
    }
    if row is not None:
        hrect_kwargs["row"] = row
    if col is not None:
        hrect_kwargs["col"] = col

    fig.add_hrect(**hrect_kwargs)
    return fig


# ---------------------------------------------------------------------------
# MULTI-AXIS FIGURE FACTORY
# ---------------------------------------------------------------------------

def create_multi_axis_figure(
    title: str = "",
    x_title: str = "",
    y1_title: str = "",
    y2_title: str = "",
    height: int = 520,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Create a dual Y-axis Plotly figure with engineering layout preset.

    The primary Y-axis (left) is used for the main signal.
    The secondary Y-axis (right) is used for the secondary signal (e.g. rainfall).

    Parameters
    ----------
    title : str
    x_title : str
    y1_title : str
        Label for left Y-axis.
    y2_title : str
        Label for right Y-axis.
    height : int
    width : int or None

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    base_layout = _get_layout_base()
    layout_kwargs = {**base_layout}
    layout_kwargs["title"] = {**base_layout["title"], "text": title}
    layout_kwargs["height"] = height
    if width is not None:
        layout_kwargs["width"] = width

    fig.update_layout(**layout_kwargs)
    axis_style = _get_axis_style()
    fig.update_xaxes(title_text=x_title, **axis_style)
    fig.update_yaxes(title_text=y1_title, secondary_y=False, **axis_style)
    fig.update_yaxes(title_text=y2_title, secondary_y=True, **axis_style)
    return fig


# ---------------------------------------------------------------------------
# SUBPLOT FIGURE FACTORY
# ---------------------------------------------------------------------------

def create_subplot_figure(
    rows: int = 1,
    cols: int = 1,
    title: str = "",
    subplot_titles: Optional[list[str]] = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    horizontal_spacing: float = 0.08,
    vertical_spacing: float = 0.10,
    height: int = 520,
    width: Optional[int] = None,
) -> go.Figure:
    """Create a subplot figure with the standard engineering layout."""
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )
    base_layout = _get_layout_base()
    layout_kwargs = {**base_layout}
    layout_kwargs["title"] = {**base_layout["title"], "text": title}
    layout_kwargs["height"] = height
    if width is not None:
        layout_kwargs["width"] = width

    fig.update_layout(**layout_kwargs)
    return fig


# ---------------------------------------------------------------------------
# CHART EXPORT HELPER
# ---------------------------------------------------------------------------

def export_chart_to_bytes(
    fig: go.Figure,
    fmt: str = "png",
    width: int = 1200,
    height: int = 700,
    scale: float = 2.0,
) -> bytes:
    """
    Export a Plotly figure to image bytes.

    Parameters
    ----------
    fig : go.Figure
    fmt : str
        Image format: 'png', 'svg', 'webp', 'jpeg'.
    width : int
    height : int
    scale : float
        Resolution multiplier for raster formats.

    Returns
    -------
    bytes
        Raw image bytes or empty bytes on failure.
    """
    try:
        kwargs = {"format": fmt, "width": width, "height": height, "engine": "kaleido"}
        if fmt not in ("svg",):
            kwargs["scale"] = scale
        return fig.to_image(**kwargs)
    except Exception:
        # Fallback: return the figure as JSON bytes
        try:
            import plotly.io as pio
            return pio.to_json(fig).encode("utf-8")
        except Exception:
            return b""


# ---------------------------------------------------------------------------
# ENGINEERING LAYOUT APPLICATOR
# ---------------------------------------------------------------------------

def apply_engineering_layout(
    fig: go.Figure,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    height: Optional[int] = None,
    width: Optional[int] = None,
    legend_orientation: str = "h",
    legend_position: str = "bottom",
) -> go.Figure:
    """
    Apply or re-apply the standard AutoPlot Civil engineering layout to any figure.

    Useful for normalizing figures created outside of create_base_figure().

    Parameters
    ----------
    fig : go.Figure
    title : str
    x_title : str
    y_title : str
    height : int or None
    width : int or None
    legend_orientation : str
        'h' for horizontal, 'v' for vertical.
    legend_position : str
        'bottom', 'top', 'right'.

    Returns
    -------
    go.Figure
    """
    base_layout = _get_layout_base()
    layout_kwargs = {**base_layout}
    if title:
        layout_kwargs["title"] = {**base_layout["title"], "text": title}
    if height is not None:
        layout_kwargs["height"] = height
    if width is not None:
        layout_kwargs["width"] = width

    # Legend positioning
    legend_configs = {
        "bottom": dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        "top": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        "right": dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    }
    legend_base = {
        **base_layout.get("legend", {}),
        "font": {"size": 11, "color": base_layout.get("font", {}).get("color", "#1A1A1A")},
    }
    legend_pos = legend_configs.get(legend_position, legend_configs["bottom"])
    if legend_orientation == "v":
        legend_pos["orientation"] = "v"
    layout_kwargs["legend"] = {**legend_base, **legend_pos}

    fig.update_layout(**layout_kwargs)
    if x_title:
        fig.update_xaxes(title_text=x_title, **_get_axis_style())
    if y_title:
        fig.update_yaxes(title_text=y_title, **_get_axis_style())
    return fig
