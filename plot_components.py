"""Publication-quality plotting functions for the MDA Web Calculator."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Style constants – designed for colorblind-friendly, journal-ready output
# ---------------------------------------------------------------------------

FONT_FAMILY = "Arial, Helvetica, sans-serif"

# Matplotlib reference mapping:  "D"->diamond, "v"->triangle-down, "s"->square,
# "^"->triangle-up, "P"->cross, "X"->x, "."->circle(small), "p"->pentagon,
# "*"->star, "o"->circle-open.  Colors match the user's publication palette.

METHOD_ORDER = ["YSG", "YSP", "YC1\u03c3", "YC2\u03c3", "Y3Za", "Y3Zo_2\u03c3", "YDZ", "YPP", "\u03c4", "MLA"]

METHOD_STYLE: Dict[str, Dict] = {
    "YSG":          {"symbol": "diamond",       "color": "#EE6677", "filled": True,  "size": 9},
    "YSP":          {"symbol": "triangle-down", "color": "#AA3377", "filled": False, "size": 9},
    "YC1\u03c3":    {"symbol": "square",        "color": "#228833", "filled": True,  "size": 8},
    "YC2\u03c3":    {"symbol": "triangle-up",   "color": "#66CCEE", "filled": False, "size": 9},
    "Y3Za":         {"symbol": "cross",         "color": "#CCBB44", "filled": True,  "size": 10},
    "Y3Zo_2\u03c3": {"symbol": "x",            "color": "#BBBBBB", "filled": True,  "size": 10},
    "YDZ":          {"symbol": "circle",        "color": "#333333", "filled": True,  "size": 6},
    "YPP":          {"symbol": "pentagon",      "color": "#4477AA", "filled": False, "size": 9},
    "\u03c4":       {"symbol": "star",          "color": "#EE7733", "filled": True,  "size": 12},
    "MLA":          {"symbol": "circle-open",   "color": "#009988", "filled": False, "size": 9},
}

BAND_COLORS = [
    "rgba(244,244,244,0.6)",  # light grey (matches reference)
    "rgba(255,255,255,0)",    # transparent
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def method_interval(result: Dict, metric_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract (mid, lo, hi) from a metric result dict."""
    if metric_name == "YDZ":
        mid = result.get("median")
        return mid, None, None  # single point, no error bars
    if metric_name == "YPP":
        mid = result.get("peak_age")
        return mid, None, None

    age = result.get("age")
    age2 = result.get("age_2sigma")
    if age is None:
        return None, None, None

    half = (age2 / 2.0) if age2 is not None else None
    if half is None:
        return age, None, None
    return age, age - half, age + half


# ---------------------------------------------------------------------------
# Publication-quality individual grain plot
# ---------------------------------------------------------------------------

def make_publication_plot(
    ages: np.ndarray,
    sigma2_abs: np.ndarray,
    metric_name: str,
    result: Dict,
    sample_name: str,
) -> go.Figure:
    """Scatter plot of individual grains ranked by age with method overlay."""

    idx = np.arange(1, len(ages) + 1)
    sort_idx = np.argsort(ages)
    ages_sorted = ages[sort_idx]
    err_sorted = sigma2_abs[sort_idx] / 2.0
    idx_sorted = idx[sort_idx]

    style = METHOD_STYLE.get(metric_name, {"symbol": "circle", "color": "#555", "size": 8})

    fig = go.Figure()

    # Individual grains
    fig.add_trace(
        go.Scatter(
            x=ages_sorted,
            y=idx_sorted,
            mode="markers",
            marker=dict(size=5.5, color="rgba(30,41,59,0.72)"),
            error_x=dict(
                type="data",
                array=err_sorted,
                visible=True,
                color="rgba(100,116,139,0.40)",
                thickness=1.2,
            ),
            name="Individual grains (\u00b11\u03c3)",
            hovertemplate="Age: %{x:.2f} Ma<br>Grain rank: %{y}<extra></extra>",
        )
    )

    # Method result overlay
    mid, lo, hi = method_interval(result, metric_name)
    if mid is not None:
        fig.add_vline(
            x=mid,
            line_dash="dash",
            line_width=2.2,
            line_color=style["color"],
        )
        fig.add_annotation(
            x=mid, y=1, yref="paper",
            text=f"<b>{metric_name}</b>: {mid:.2f} Ma",
            showarrow=False,
            yshift=20,
            font=dict(size=12, color=style["color"], family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=3,
        )
    if lo is not None and hi is not None:
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=style["color"].replace(")", ",0.10)").replace("rgb", "rgba")
            if "rgb" in style["color"]
            else _hex_to_rgba(style["color"], 0.10),
            line_width=0,
            annotation_text=f"\u00b1 interval",
            annotation_position="top left",
            annotation_font=dict(size=10, color=style["color"]),
        )

    # Highlight grains inside method cluster
    indices = result.get("indices")
    if indices is not None and len(indices) > 0:
        cluster_ages = ages[indices]
        cluster_sigma = sigma2_abs[np.array(indices)] / 2.0
        cluster_ranks = np.searchsorted(ages_sorted, cluster_ages, side="left") + 1
        fig.add_trace(
            go.Scatter(
                x=cluster_ages,
                y=cluster_ranks,
                mode="markers",
                marker=dict(
                    size=8, color=style["color"],
                    symbol=style["symbol"],
                    line=dict(width=1, color="#1e293b"),
                ),
                error_x=dict(
                    type="data", array=cluster_sigma,
                    visible=True, color=style["color"], thickness=1.5,
                ),
                name=f"{metric_name} cluster (N={len(indices)})",
                hovertemplate="Age: %{x:.2f} Ma<br>Cluster grain<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{sample_name}</b> \u2022 {metric_name}",
            font=dict(size=15, family=FONT_FAMILY),
        ),
        template="plotly_white",
        font=dict(family=FONT_FAMILY),
        height=460,
        margin=dict(l=25, r=25, t=72, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        title=dict(text="Age (Ma)", font=dict(size=13)),
        showgrid=True, gridcolor="rgba(148,163,184,0.22)",
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        title=dict(text="Grain rank (youngest \u2192 oldest)", font=dict(size=13)),
        autorange="reversed", showgrid=False,
        tickfont=dict(size=11),
    )
    return fig


# ---------------------------------------------------------------------------
# Publication-quality forest plot (multi-sample summary)
# ---------------------------------------------------------------------------

def make_forest_plot(
    summary_df: pd.DataFrame,
    ref_age: Optional[float] = None,
    ref_label: Optional[str] = None,
    ref_age_2: Optional[float] = None,
    ref_label_2: Optional[str] = None,
    show_title: bool = False,
) -> go.Figure:
    """
    Forest plot comparing MDA methods across samples.

    Designed to match publication standards: distinct marker symbols per method,
    alternating horizontal bands, optional reference lines, journal-ready
    typography.
    """

    samples = list(summary_df["Sample"].unique())
    n_samples = len(samples)
    sample_to_y = {s: i for i, s in enumerate(samples)}

    # Use canonical order; only include methods actually present in data
    all_methods = summary_df["Method"].unique()
    plotted_methods = [m for m in METHOD_ORDER if m in all_methods]
    # Append any methods not in METHOD_ORDER (future-proofing)
    plotted_methods += [m for m in all_methods if m not in plotted_methods and m is not None]
    n_methods = len(plotted_methods)

    fig = go.Figure()

    # ---- Alternating horizontal bands ----
    x_min = summary_df["Lower (Ma)"].min()
    x_max = summary_df["Upper (Ma)"].max()
    x_pad = (x_max - x_min) * 0.06
    x_lo = x_min - x_pad - 5
    x_hi = x_max + x_pad + 5

    for i, sample in enumerate(samples):
        fig.add_shape(
            type="rect",
            x0=x_lo, x1=x_hi,
            y0=i - 0.5, y1=i + 0.5,
            fillcolor=BAND_COLORS[i % 2],
            line_width=0,
            layer="below",
        )

    # ---- Data traces per method ----
    for mi, method in enumerate(plotted_methods):
        sub = summary_df[
            (summary_df["Method"] == method) & summary_df["Age (Ma)"].notna()
        ].copy()
        if sub.empty:
            continue

        style = METHOD_STYLE.get(method, {"symbol": "circle", "color": "#555", "size": 9, "filled": True})
        filled = style.get("filled", True)

        # Jitter offset to avoid overlapping markers within a sample band
        jitter = (mi - n_methods / 2.0) * 0.042

        y_vals = [sample_to_y[s] + jitter for s in sub["Sample"]]

        err_plus = (sub["Upper (Ma)"] - sub["Age (Ma)"]).fillna(0).clip(lower=0).to_numpy()
        err_minus = (sub["Age (Ma)"] - sub["Lower (Ma)"]).fillna(0).clip(lower=0).to_numpy()

        # Filled vs open markers — NEVER use white (invisible on white bg).
        # Open markers get a light tint of their method color.
        marker_edge = style["color"]
        marker_color = style["color"] if filled else _hex_to_rgba(style["color"], 0.18)

        fig.add_trace(
            go.Scatter(
                x=sub["Age (Ma)"].to_numpy(),
                y=y_vals,
                mode="markers",
                name=method,
                marker=dict(
                    symbol=style["symbol"],
                    size=style["size"],
                    color=marker_color,
                    line=dict(width=1.0, color=marker_edge),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    visible=True,
                    thickness=1.0,
                    color=style["color"],
                ),
                hovertemplate=(
                    "Sample: %{text}<br>"
                    f"Method: {method}<br>"
                    "Age: %{x:.2f} Ma<extra></extra>"
                ),
                text=sub["Sample"].tolist(),
            )
        )

    # ---- Reference line 1 ----
    if ref_age is not None:
        label = ref_label or f"{ref_age} Ma"
        fig.add_vline(
            x=ref_age,
            line_dash="dot",
            line_width=1.6,
            line_color="#1e293b",
        )
        fig.add_annotation(
            x=ref_age, y=1.0, yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False, yshift=16,
            font=dict(size=10.5, color="#1e293b", family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
        )

    # ---- Reference line 2 (e.g. CA-ID-TIMS) ----
    if ref_age_2 is not None:
        label2 = ref_label_2 or f"{ref_age_2} Ma"
        fig.add_annotation(
            x=ref_age_2, y=n_samples - 1, yref="y",
            ax=ref_age_2 - 8, ay=n_samples - 0.3, ayref="y",
            text=f"<b>{label2}</b>",
            showarrow=True,
            arrowhead=2, arrowsize=1.2,
            arrowwidth=1.4, arrowcolor="#1e293b",
            font=dict(size=10, color="#1e293b", family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[ref_age_2],
                y=[n_samples - 1],
                mode="markers",
                marker=dict(symbol="star", size=13, color="#9b59b6", line=dict(width=1, color="#1e293b")),
                showlegend=False,
                hovertemplate=f"{label2}<extra></extra>",
            )
        )

    # ---- Layout ----
    plot_height = max(460, 100 + 60 * n_samples)

    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12),
        height=plot_height,
        title=dict(
            text="<b>MDA method comparison</b>" if show_title else "",
            font=dict(size=15, family=FONT_FAMILY),
        ),
        margin=dict(l=30, r=160, t=65 if show_title else 45, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title=dict(text="<b>Method</b>", font=dict(size=11)),
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            font=dict(size=10.5),
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.9)",
            itemsizing="constant",
        ),
    )

    # Y-axis: numeric with sample-name labels
    fig.update_yaxes(
        tickvals=list(range(n_samples)),
        ticktext=samples,
        range=[n_samples - 0.5, -0.5],  # top-to-bottom ordering
        showgrid=False,
        tickfont=dict(size=12, family=FONT_FAMILY),
        title=dict(text=""),
    )

    # "top" / "base" annotations
    if n_samples > 1:
        fig.add_annotation(
            x=-0.01, y=-0.5, xref="paper", yref="y",
            text="<i>top</i>", showarrow=False,
            font=dict(size=10, color="#64748b", family=FONT_FAMILY),
            xanchor="right",
        )
        fig.add_annotation(
            x=-0.01, y=n_samples - 0.5, xref="paper", yref="y",
            text="<i>base</i>", showarrow=False,
            font=dict(size=10, color="#64748b", family=FONT_FAMILY),
            xanchor="right",
        )

    fig.update_xaxes(
        title=dict(text="<b>Age (Ma)</b>", font=dict(size=13)),
        showgrid=True,
        gridcolor="rgba(148,163,184,0.30)",
        griddash="dot",
        dtick=10,
        zeroline=False,
        tickfont=dict(size=11),
        range=[x_lo, x_hi],
    )

    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_plot_pdf(fig: go.Figure, width: int = 1400, height: int = 700) -> bytes:
    """Export a Plotly figure as PDF bytes (requires kaleido)."""
    return fig.to_image(format="pdf", width=width, height=height, scale=2)


def export_plot_png(fig: go.Figure, width: int = 1400, height: int = 700) -> bytes:
    """Export a Plotly figure as high-res PNG bytes."""
    return fig.to_image(format="png", width=width, height=height, scale=3)


def export_plot_svg(fig: go.Figure, width: int = 1400, height: int = 700) -> bytes:
    """Export a Plotly figure as SVG bytes."""
    return fig.to_image(format="svg", width=width, height=height)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
