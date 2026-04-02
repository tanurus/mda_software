import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mda_methods import compute_all_metrics

st.set_page_config(page_title="Detrital Zircon MDA Calculator", layout="wide")
st.title("Maximum Depositional Age (MDA) Web Calculator")
st.caption(
    "Excel-like paste, flexible column mapping, publication-style plots, and exportable results for scientific workflows."
)

INPUT_MODES = [
    "1σ absolute",
    "2σ absolute",
    "1σ percent",
    "2σ percent",
]

OUTPUT_MODES = INPUT_MODES

DEFAULT_EXAMPLE = """sample1_age,sample1_sigma,sample2_age,sample2_sigma
518.37,11.03,610.1,8.2
525.66,11.10,603.4,7.9
529.80,11.44,599.7,8.1
535.77,11.16,590.1,7.8
"""

if "input_text" not in st.session_state:
    st.session_state.input_text = DEFAULT_EXAMPLE
if "grid_df" not in st.session_state:
    st.session_state.grid_df = None


@st.cache_data
def parse_table(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text.strip()), sep=None, engine="python")


def to_internal_sigma2_abs(ages: np.ndarray, sigma: np.ndarray, mode: str) -> np.ndarray:
    if mode == "1σ absolute":
        return sigma * 2.0
    if mode == "2σ absolute":
        return sigma
    if mode == "1σ percent":
        return (sigma / 100.0) * ages * 2.0
    if mode == "2σ percent":
        return (sigma / 100.0) * ages
    raise ValueError("Unknown mode")


def from_internal_sigma2_abs(ages: np.ndarray, sigma2_abs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "1σ absolute":
        return sigma2_abs / 2.0
    if mode == "2σ absolute":
        return sigma2_abs
    if mode == "1σ percent":
        return (sigma2_abs / 2.0) / ages * 100.0
    if mode == "2σ percent":
        return sigma2_abs / ages * 100.0
    raise ValueError("Unknown mode")


def split_samples(
    df: pd.DataFrame, input_mode: str, first_col_idx: int = 0, pair_stride: int = 2
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out = []
    cols = list(df.columns)
    if len(cols) < first_col_idx + 2:
        return out

    for i in range(first_col_idx, len(cols) - 1, pair_stride):
        age_col = cols[i]
        sig_col = cols[i + 1]
        sub = df[[age_col, sig_col]].dropna()
        if sub.empty:
            continue
        ages = pd.to_numeric(sub.iloc[:, 0], errors="coerce").to_numpy()
        sig = pd.to_numeric(sub.iloc[:, 1], errors="coerce").to_numpy()
        sigma2_abs = to_internal_sigma2_abs(ages, sig, input_mode)
        mask = np.isfinite(ages) & np.isfinite(sigma2_abs) & (ages > 0) & (sigma2_abs > 0)
        ages = ages[mask]
        sigma2_abs = sigma2_abs[mask]
        if len(ages) == 0:
            continue
        out.append((str(age_col), ages, sigma2_abs))
    return out


def summarize_metric(metric_name: str, result: Dict, output_mode: str):
    if metric_name in {"YDZ", "YPP"}:
        if metric_name == "YDZ":
            return {
                "age": result.get("median"),
                "aux": f"p2.5={result.get('p2_5'):.3f}, p97.5={result.get('p97_5'):.3f}",
            }
        return {"age": result.get("peak_age"), "aux": "KDE peak age"}

    age = result.get("age")
    age2 = result.get("age_2sigma")
    if age is None:
        return {"age": None, "uncertainty": None, "aux": result.get("reason", "No valid subset")}

    u = float(from_internal_sigma2_abs(np.array([age]), np.array([age2]), output_mode)[0]) if age2 else None
    return {
        "age": age,
        "uncertainty": u,
        "aux": f"N={result.get('N', '-')}, MSWD={result.get('MSWD', np.nan):.3f}"
        if result.get("MSWD") is not None
        else f"N={result.get('N', '-')}",
    }


def method_interval(result: Dict, metric_name: str) -> Tuple[float | None, float | None, float | None]:
    if metric_name == "YDZ":
        mid = result.get("median")
        lo = result.get("p2_5")
        hi = result.get("p97_5")
        return mid, lo, hi
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


def make_publication_plot(ages: np.ndarray, sigma2_abs: np.ndarray, metric_name: str, result: Dict, sample_name: str):
    idx = np.arange(1, len(ages) + 1)
    sort_idx = np.argsort(ages)
    ages_sorted = ages[sort_idx]
    err_sorted = (sigma2_abs[sort_idx] / 2.0)
    idx_sorted = idx[sort_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ages_sorted,
            y=idx_sorted,
            mode="markers",
            marker=dict(size=6, color="rgba(30,41,59,0.7)"),
            error_x=dict(type="data", array=err_sorted, visible=True, color="rgba(71,85,105,0.45)", thickness=1),
            name="Individual grains (±1σ)",
            hovertemplate="Age: %{x:.2f} Ma<br>Grain rank: %{y}<extra></extra>",
        )
    )

    mid, lo, hi = method_interval(result, metric_name)
    if mid is not None:
        fig.add_vline(x=mid, line_dash="dash", line_width=2, line_color="#ef4444")
        fig.add_annotation(
            x=mid,
            y=1,
            yref="paper",
            text=f"{metric_name}: {mid:.2f} Ma",
            showarrow=False,
            yshift=18,
            font=dict(size=12, color="#991b1b"),
        )
    if lo is not None and hi is not None:
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor="rgba(239,68,68,0.12)",
            line_width=0,
            annotation_text="Method interval",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"{sample_name} • {metric_name}",
        template="plotly_white",
        height=480,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="Age (Ma)", showgrid=True, gridcolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(title="Grain rank (youngest → oldest)", autorange="reversed", showgrid=False)
    return fig


def make_forest_plot(summary_df: pd.DataFrame):
    colors = [
        "#0f766e",
        "#1d4ed8",
        "#7c3aed",
        "#ea580c",
        "#be123c",
        "#4338ca",
        "#0284c7",
        "#15803d",
        "#4f46e5",
        "#b45309",
    ]
    fig = go.Figure()
    plotted_methods = [m for m in summary_df["Method"].unique() if m is not None]

    for i, method in enumerate(plotted_methods):
        sub = summary_df[(summary_df["Method"] == method) & (summary_df["Age (Ma)"].notna())].copy()
        if sub.empty:
            continue
        err_plus = sub["Upper (Ma)"] - sub["Age (Ma)"]
        err_minus = sub["Age (Ma)"] - sub["Lower (Ma)"]
        fig.add_trace(
            go.Scatter(
                x=sub["Age (Ma)"],
                y=sub["Sample"],
                mode="markers",
                name=method,
                marker=dict(symbol="diamond", size=9, color=colors[i % len(colors)], line=dict(width=0.7, color="#111827")),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=np.maximum(err_plus.fillna(0).to_numpy(), 0),
                    arrayminus=np.maximum(err_minus.fillna(0).to_numpy(), 0),
                    visible=True,
                    thickness=1.6,
                ),
                hovertemplate="Sample: %{y}<br>Method: " + method + "<br>Age: %{x:.2f} Ma<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=max(420, 90 + 52 * summary_df["Sample"].nunique()),
        title="MDA method comparison by sample (publication-style forest plot)",
        xaxis_title="Age (Ma)",
        yaxis_title="Sample",
        legend_title="Method",
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.30)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(226,232,240,0.5)", categoryorder="array", categoryarray=list(summary_df["Sample"].unique())[::-1])
    return fig


with st.expander("Input format help", expanded=False):
    st.markdown(
        "- Paste CSV/TSV copied from Excel with alternating columns: `age, sigma, age, sigma, ...`\n"
        "- If your sheet has extra leading columns, use **First age column** to set where age/sigma pairs begin.\n"
        "- Use discordancy-filtered grains before pasting (e.g., 10% filter)."
    )
    st.code(DEFAULT_EXAMPLE)

with st.container(border=True):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2])
    with col1:
        in_mode = st.selectbox("Input uncertainty", INPUT_MODES, index=1)
    with col2:
        out_mode = st.selectbox("Output uncertainty", OUTPUT_MODES, index=1)
    with col3:
        pair_stride = st.selectbox("Column stride", [2, 1], index=0, help="2 = age/sigma pairs; 1 = every adjacent column pair")
    with col4:
        run = st.button("Plot and calculate", type="primary", use_container_width=True)

    tab_paste, tab_grid = st.tabs(["Paste data", "Spreadsheet editor"])

    with tab_paste:
        st.text_area(
            "Paste table (Excel-friendly)",
            height=250,
            key="input_text",
            placeholder="Paste directly from Excel or CSV/TSV.",
        )
        p1, p2, p3 = st.columns([1.2, 1, 2.8])
        with p1:
            if st.button("Build spreadsheet from paste", use_container_width=True):
                try:
                    st.session_state.grid_df = parse_table(st.session_state.input_text)
                    st.success("Pasted table loaded into spreadsheet editor.")
                except Exception as exc:
                    st.error(f"Could not parse pasted text: {exc}")
        with p2:
            if st.button("Load example to grid", use_container_width=True):
                st.session_state.input_text = DEFAULT_EXAMPLE
                st.session_state.grid_df = parse_table(DEFAULT_EXAMPLE)
                st.rerun()

    with tab_grid:
        if st.session_state.grid_df is None:
            try:
                st.session_state.grid_df = parse_table(st.session_state.input_text)
            except Exception:
                st.session_state.grid_df = pd.DataFrame(columns=["age", "sigma"])

        st.caption("Excel-like input table: click any cell and paste from Excel. You can also add/remove rows.")
        st.session_state.grid_df = st.data_editor(
            st.session_state.grid_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="excel_like_grid",
        )

    a, b, c = st.columns([1, 1, 3])
    with a:
        if st.button("Clear", use_container_width=True):
            st.session_state.input_text = ""
            st.session_state.grid_df = pd.DataFrame(columns=["age", "sigma"])
            st.rerun()
    with b:
        if st.button("Load example", use_container_width=True):
            st.session_state.input_text = DEFAULT_EXAMPLE
            st.session_state.grid_df = parse_table(DEFAULT_EXAMPLE)
            st.rerun()


summary_rows = []
forest_fig = None

if run:
    df = st.session_state.grid_df.copy() if st.session_state.grid_df is not None else pd.DataFrame()
    if df.empty:
        st.error("Input table is empty. Paste data or add rows in the spreadsheet editor.")
        st.stop()

    st.subheader("Parsed input preview")
    st.dataframe(df.head(40), use_container_width=True, height=260)

    first_col_idx = st.selectbox(
        "First age column (Excel-like mapping)",
        options=list(range(len(df.columns))),
        index=0,
        format_func=lambda i: f"{i}: {df.columns[i]}",
        help="Choose the first age column. The next column is treated as sigma, then repeats by stride.",
    )

    samples = split_samples(df, in_mode, first_col_idx=first_col_idx, pair_stride=pair_stride)
    if not samples:
        st.warning("No valid sample pairs found. Check the first age column and your age/sigma pattern.")
        st.stop()

    for sample_name, ages, sigma2_abs in samples:
        st.header(f"Sample: {sample_name}")
        results = compute_all_metrics(ages, sigma2_abs)

        rows = []
        for mname, res in results.items():
            s = summarize_metric(mname, res, out_mode)
            mid, lo, hi = method_interval(res, mname)
            if mid is not None:
                summary_rows.append(
                    {
                        "Sample": sample_name,
                        "Method": mname,
                        "Age (Ma)": float(mid),
                        "Lower (Ma)": float(lo) if lo is not None else float(mid),
                        "Upper (Ma)": float(hi) if hi is not None else float(mid),
                    }
                )
            rows.append(
                {
                    "Method": mname,
                    "Age (Ma)": None if s["age"] is None else round(float(s["age"]), 4),
                    f"Uncertainty ({out_mode})": None
                    if s.get("uncertainty") is None
                    else round(float(s["uncertainty"]), 4),
                    "Details": s.get("aux", ""),
                }
            )

        table_df = pd.DataFrame(rows)
        st.dataframe(table_df, use_container_width=True)
        st.download_button(
            label=f"Export {sample_name} results CSV",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{sample_name}_mda_results.csv",
            mime="text/csv",
            use_container_width=False,
        )

        with st.expander(f"Plots for {sample_name}", expanded=True):
            methods = list(results.keys())
            selected = st.multiselect(
                f"Select methods to render ({sample_name})",
                methods,
                default=["YSG", "YC2σ", "YSP", "MLA", "YPP", "τ"],
                key=f"plot_{sample_name}",
            )
            for method in selected:
                fig = make_publication_plot(ages, sigma2_abs, method, results[method], sample_name)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(
                    label=f"Export {sample_name}-{method} plot (HTML)",
                    data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                    file_name=f"{sample_name}_{method}_plot.html",
                    mime="text/html",
                )

    if summary_rows:
        st.subheader("Multi-sample scientific summary plot")
        summary_df = pd.DataFrame(summary_rows)
        forest_fig = make_forest_plot(summary_df)
        st.plotly_chart(forest_fig, use_container_width=True)

        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "Export combined summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="mda_summary_all_samples.csv",
                mime="text/csv",
            )
        with exp2:
            st.download_button(
                "Export summary plot (HTML)",
                data=forest_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                file_name="mda_summary_plot.html",
                mime="text/html",
            )

st.markdown("---")
st.caption("Note: MLA is implemented as a Gaussian-mixture/BIC approximation to IsoplotR youngest-component unmixing.")
