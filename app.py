import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mda_methods import compute_all_metrics

st.set_page_config(page_title="Detrital Zircon MDA Calculator", layout="wide")
st.title("Maximum Depositional Age (MDA) Web Calculator")
st.caption("Paste multi-sample ages/uncertainties, compute 10 MDA methods, and visualize per-method outputs.")

INPUT_MODES = [
    "1σ absolute",
    "2σ absolute",
    "1σ percent",
    "2σ percent",
]

OUTPUT_MODES = INPUT_MODES


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


def split_samples(df: pd.DataFrame, input_mode: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out = []
    cols = list(df.columns)
    if len(cols) < 2:
        return out
    n_pairs = len(cols) // 2
    for i in range(n_pairs):
        age_col = cols[2 * i]
        sig_col = cols[2 * i + 1]
        sub = df[[age_col, sig_col]].dropna()
        if sub.empty:
            continue
        ages = sub.iloc[:, 0].astype(float).to_numpy()
        sig = sub.iloc[:, 1].astype(float).to_numpy()
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
    return {"age": age, "uncertainty": u, "aux": f"N={result.get('N', '-')}, MSWD={result.get('MSWD', np.nan):.3f}" if result.get("MSWD") is not None else f"N={result.get('N', '-')}"}


def make_plot(ages: np.ndarray, sigma2_abs: np.ndarray, metric_name: str, result: Dict):
    fig = go.Figure()
    idx = np.arange(1, len(ages) + 1)
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=ages,
            mode="markers",
            error_y=dict(type="data", array=sigma2_abs / 2.0, visible=True),
            name="Zircon age ±1σ",
        )
    )

    mda_age = None
    if metric_name == "YDZ":
        mda_age = result.get("median")
    elif metric_name == "YPP":
        mda_age = result.get("peak_age")
    else:
        mda_age = result.get("age")

    if mda_age is not None:
        fig.add_hline(y=float(mda_age), line_dash="dash", annotation_text=f"{metric_name}: {mda_age:.2f} Ma")

    fig.update_layout(
        title=f"{metric_name} visualization",
        xaxis_title="Grain index",
        yaxis_title="Age (Ma)",
        height=420,
    )
    return fig


example = """sample1_age,sample1_sigma,sample2_age,sample2_sigma
518.37,11.03,610.1,8.2
525.66,11.10,603.4,7.9
529.80,11.44,599.7,8.1
535.77,11.16,590.1,7.8
"""

with st.expander("Input format help", expanded=False):
    st.markdown(
        "- Paste CSV/TSV with alternating columns: `age, sigma, age, sigma, ...`\n"
        "- Column pairs are treated as separate samples.\n"
        "- Use discordancy-filtered grains before pasting (e.g., 10% filter)."
    )
    st.code(example)

left, right = st.columns(2)
with left:
    in_mode = st.selectbox("Input uncertainty mode", INPUT_MODES, index=1)
with right:
    out_mode = st.selectbox("Output uncertainty mode", OUTPUT_MODES, index=1)

text = st.text_area("Paste table", height=220, value=example)
run = st.button("Run MDA calculations", type="primary")

if run:
    try:
        df = parse_table(text)
    except Exception as exc:
        st.error(f"Could not parse table: {exc}")
        st.stop()

    samples = split_samples(df, in_mode)
    if not samples:
        st.warning("No valid sample pairs found. Ensure your columns are age/sigma pairs.")
        st.stop()

    for sample_name, ages, sigma2_abs in samples:
        st.header(f"Sample: {sample_name}")
        results = compute_all_metrics(ages, sigma2_abs)

        rows = []
        for mname, res in results.items():
            s = summarize_metric(mname, res, out_mode)
            rows.append(
                {
                    "Method": mname,
                    "Age (Ma)": None if s["age"] is None else round(float(s["age"]), 4),
                    f"Uncertainty ({out_mode})": None if s.get("uncertainty") is None else round(float(s["uncertainty"]), 4),
                    "Details": s.get("aux", ""),
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with st.expander(f"Plots for {sample_name}", expanded=False):
            methods = list(results.keys())
            selected = st.multiselect(
                f"Select methods to plot ({sample_name})",
                methods,
                default=["YSG", "YC2σ", "YSP", "MLA", "YPP", "τ"],
                key=f"plot_{sample_name}",
            )
            for method in selected:
                st.plotly_chart(make_plot(ages, sigma2_abs, method, results[method]), use_container_width=True)

st.markdown("---")
st.caption("Note: MLA is implemented as a Gaussian-mixture/BIC approximation to IsoplotR youngest-component unmixing.")
