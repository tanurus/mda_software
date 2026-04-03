"""
MDA Web Calculator — Maximum Depositional Age analysis from detrital zircon U-Pb data.

Streamlit application entry point.  UI components live in ui_components.py,
plotting in plot_components.py, and scientific methods in mda_methods.py.
"""

import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from mda_methods import compute_all_metrics
from plot_components import (
    make_forest_plot,
    make_publication_plot,
    method_interval,
    export_plot_pdf,
    export_plot_png,
)
from ui_components import (
    CUSTOM_CSS,
    INPUT_MODES,
    OUTPUT_MODES,
    render_footer,
    render_hero,
    render_sidebar,
    render_spreadsheet,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MDA Web Calculator — Detrital Zircon",
    page_icon="\U0001f48e",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Unit conversion helpers (kept here to avoid circular imports)
# ---------------------------------------------------------------------------

def to_internal_sigma2_abs(ages: np.ndarray, sigma: np.ndarray, mode: str) -> np.ndarray:
    if mode == "1\u03c3 absolute":
        return sigma * 2.0
    if mode == "2\u03c3 absolute":
        return sigma
    if mode == "1\u03c3 percent":
        return (sigma / 100.0) * ages * 2.0
    if mode == "2\u03c3 percent":
        return (sigma / 100.0) * ages
    raise ValueError(f"Unknown mode: {mode}")


def from_internal_sigma2_abs(ages: np.ndarray, sigma2_abs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "1\u03c3 absolute":
        return sigma2_abs / 2.0
    if mode == "2\u03c3 absolute":
        return sigma2_abs
    if mode == "1\u03c3 percent":
        return (sigma2_abs / 2.0) / ages * 100.0
    if mode == "2\u03c3 percent":
        return sigma2_abs / ages * 100.0
    raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Sample extraction
# ---------------------------------------------------------------------------

def split_samples(
    df: pd.DataFrame, input_mode: str, first_col_idx: int = 0, pair_stride: int = 2,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out: list = []
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
        ages, sigma2_abs = ages[mask], sigma2_abs[mask]
        if len(ages) == 0:
            continue
        out.append((str(age_col), ages, sigma2_abs))
    return out


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def summarize_metric(metric_name: str, result: Dict, output_mode: str) -> Dict:
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
        "aux": (
            f"N={result.get('N', '-')}, MSWD={result.get('MSWD', np.nan):.3f}"
            if result.get("MSWD") is not None
            else f"N={result.get('N', '-')}"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

# Hero banner
render_hero()

# Sidebar settings
cfg = render_sidebar()

# Spreadsheet input + toolbar
run_clicked, working_df = render_spreadsheet()

# ---------------------------------------------------------------------------
# Computation & results
# ---------------------------------------------------------------------------

summary_rows: list = []
forest_fig = None

if run_clicked:
    if working_df is None or working_df.empty:
        st.error("The spreadsheet is empty. Paste data or upload a file first.")
        st.stop()

    first_col_idx = st.session_state.get("first_col_idx", 0)
    samples = split_samples(working_df, cfg["in_mode"], first_col_idx, cfg["pair_stride"])
    if not samples:
        st.warning("No valid age/sigma pairs found. Check your first-age-column setting and data layout.")
        st.stop()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ---- Parsed preview ----
    with st.expander("Parsed input preview", expanded=False):
        st.dataframe(working_df.head(40), use_container_width=True, height=260)

    # ---- Per-sample results ----
    for sample_name, ages, sigma2_abs in samples:
        st.markdown(f"### Sample: **{sample_name}**")

        with st.spinner(f"Computing MDA methods for {sample_name}\u2026"):
            results = compute_all_metrics(ages, sigma2_abs)

        # Build results table
        rows = []
        for mname, res in results.items():
            s = summarize_metric(mname, res, cfg["out_mode"])
            mid, lo, hi = method_interval(res, mname)
            if mid is not None:
                summary_rows.append({
                    "Sample": sample_name,
                    "Method": mname,
                    "Age (Ma)": float(mid),
                    "Lower (Ma)": float(lo) if lo is not None else float(mid),
                    "Upper (Ma)": float(hi) if hi is not None else float(mid),
                })
            rows.append({
                "Method": mname,
                "Age (Ma)": None if s["age"] is None else round(float(s["age"]), 4),
                f"Uncertainty ({cfg['out_mode']})": (
                    None if s.get("uncertainty") is None else round(float(s["uncertainty"]), 4)
                ),
                "Details": s.get("aux", ""),
            })

        table_df = pd.DataFrame(rows)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # Export buttons for this sample
        exp_a, exp_b = st.columns(2)
        with exp_a:
            st.download_button(
                f"Export {sample_name} results (CSV)",
                data=table_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{sample_name}_mda_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_b:
            # Excel export
            buf = io.BytesIO()
            table_df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                f"Export {sample_name} results (Excel)",
                data=buf.getvalue(),
                file_name=f"{sample_name}_mda_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # ---- Individual plots ----
        with st.expander(f"Individual grain plots \u2014 {sample_name}", expanded=True):
            methods = list(results.keys())
            selected = st.multiselect(
                f"Methods to plot ({sample_name})",
                methods,
                default=["YSG", "YC2\u03c3", "YSP", "MLA", "YPP", "\u03c4"],
                key=f"plot_{sample_name}",
            )
            for method in selected:
                fig = make_publication_plot(ages, sigma2_abs, method, results[method], sample_name)
                st.plotly_chart(fig, use_container_width=True)
                pa, pb, pc = st.columns(3)
                with pa:
                    st.download_button(
                        f"HTML",
                        data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                        file_name=f"{sample_name}_{method}_plot.html",
                        mime="text/html",
                        use_container_width=True,
                    )
                with pb:
                    try:
                        st.download_button(
                            f"PNG",
                            data=export_plot_png(fig),
                            file_name=f"{sample_name}_{method}_plot.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption("PNG export requires kaleido package")
                with pc:
                    try:
                        st.download_button(
                            f"PDF",
                            data=export_plot_pdf(fig),
                            file_name=f"{sample_name}_{method}_plot.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption("PDF export requires kaleido package")

    # ---- Forest plot (multi-sample summary) ----
    if summary_rows:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Multi-sample summary (forest plot)")

        summary_df = pd.DataFrame(summary_rows)
        forest_fig = make_forest_plot(
            summary_df,
            ref_age=cfg["ref_age"],
            ref_label=cfg["ref_label"],
            ref_age_2=cfg["ref_age_2"],
            ref_label_2=cfg["ref_label_2"],
        )
        st.plotly_chart(forest_fig, use_container_width=True)

        # Export row
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.download_button(
                "Summary CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="mda_summary_all_samples.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e2:
            buf = io.BytesIO()
            summary_df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button(
                "Summary Excel",
                data=buf.getvalue(),
                file_name="mda_summary_all_samples.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with e3:
            st.download_button(
                "Forest plot HTML",
                data=forest_fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                file_name="mda_summary_plot.html",
                mime="text/html",
                use_container_width=True,
            )
        with e4:
            try:
                st.download_button(
                    "Forest plot PDF",
                    data=export_plot_pdf(forest_fig, width=1600, height=max(600, 100 + 80 * len(summary_df["Sample"].unique()))),
                    file_name="mda_summary_plot.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception:
                st.caption("PDF export requires kaleido package")

        # Handle deferred PDF button from toolbar
        if st.session_state.get("pdf_requested") and forest_fig is not None:
            try:
                pdf_bytes = export_plot_pdf(
                    forest_fig,
                    width=1600,
                    height=max(600, 100 + 80 * len(summary_df["Sample"].unique())),
                )
                st.download_button(
                    "Download forest plot PDF",
                    data=pdf_bytes,
                    file_name="mda_forest_plot.pdf",
                    mime="application/pdf",
                )
            except Exception:
                st.warning("PDF export requires the `kaleido` package. Install it with: `pip install kaleido`")
            st.session_state["pdf_requested"] = False

# ---------------------------------------------------------------------------
# Method reference (collapsible)
# ---------------------------------------------------------------------------

with st.expander("MDA method descriptions", expanded=False):
    st.markdown("""
| Method | Full Name | Description |
|--------|-----------|-------------|
| **YSG** | Youngest Single Grain | Age \u00b1 uncertainty of the single youngest grain |
| **YDZ** | Youngest Detrital Zircon | Monte Carlo simulation of the youngest grain distribution |
| **YC1\u03c3** | Youngest Cluster (1\u03c3) | Weighted mean of the youngest cluster overlapping at 1\u03c3 |
| **YC2\u03c3** | Youngest Cluster (2\u03c3) | Weighted mean of the youngest cluster overlapping at 2\u03c3 |
| **Y3Za** | Youngest 3 Grains (absolute) | Weighted mean of the 3 youngest grains |
| **Y3Zo_2\u03c3** | Youngest 3 Overlapping (2\u03c3) | Weighted mean of the youngest 3 grains with 2\u03c3 overlap |
| **YSP** | Youngest Statistical Population | Largest youngest population passing MSWD \u03c7\u00b2 test |
| **MLA** | Maximum Likelihood Age | Youngest Gaussian mixture component (GMM-BIC) |
| **YPP** | Youngest Probability Peak | Youngest KDE peak age |
| **\u03c4** | Tau Method | Weighted mean of youngest PDP peak cluster |
""")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

render_footer()
