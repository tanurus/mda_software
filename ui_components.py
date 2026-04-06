"""UI components for the MDA Web Calculator – spreadsheet grid, toolbar, hero, sidebar."""

import io
import string
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from handsontable_grid import handsontable_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_MODES = ["1\u03c3 absolute", "2\u03c3 absolute", "1\u03c3 percent", "2\u03c3 percent"]
OUTPUT_MODES = INPUT_MODES

DEFAULT_EXAMPLE = """\
sample1_age,sample1_sigma,sample2_age,sample2_sigma
518.37,11.03,610.1,8.2
525.66,11.10,603.4,7.9
529.80,11.44,599.7,8.1
535.77,11.16,590.1,7.8
"""

_EMPTY_ROWS = 100
_EMPTY_COLS = 26

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_letter(idx: int) -> str:
    """Return Excel-style column letter(s): 0->A, 25->Z, 26->AA, ..."""
    result = ""
    while True:
        result = string.ascii_uppercase[idx % 26] + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


def _make_empty_grid_data(n_rows: int = _EMPTY_ROWS, n_cols: int = _EMPTY_COLS) -> list:
    """Create an empty 2D list for Handsontable (all None)."""
    return [[None] * n_cols for _ in range(n_rows)]


def _df_to_grid_data(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a 2D list for Handsontable.
    Row 0 = column headers, rows 1+ = data values.
    Pads with empty columns/rows to fill at least _EMPTY_COLS × _EMPTY_ROWS."""
    n_cols = max(len(df.columns), _EMPTY_COLS)
    header = list(df.columns) + [None] * (n_cols - len(df.columns))
    rows = [header]
    for _, row in df.iterrows():
        vals = row.tolist()
        # Convert NaN/NaT to None for JSON serialisation
        vals = [None if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
        vals += [None] * (n_cols - len(vals))
        rows.append(vals)
    # Pad to minimum row count
    while len(rows) < _EMPTY_ROWS:
        rows.append([None] * n_cols)
    return rows


def _grid_data_to_df(data: list) -> pd.DataFrame:
    """Convert 2D list (from Handsontable) back to a DataFrame.
    Row 0 = headers, rest = data.  Drops fully-empty columns and rows."""
    if not data or len(data) < 2:
        return pd.DataFrame()
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(data[0])]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    # Drop fully-empty columns and rows
    df = df.replace("", np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    return df


@st.cache_data
def parse_table(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text.strip()), sep=None, engine="python")


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f766e 0%, #1e40af 100%);
    padding: 2.2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    color: white;
}
.hero-banner h1 {
    margin: 0 0 0.3rem 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.hero-banner p {
    margin: 0;
    font-size: 1.02rem;
    opacity: 0.92;
    line-height: 1.5;
}
.hero-badges {
    margin-top: 0.7rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: white;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    color: #94a3b8;
    font-size: 0.82rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 2rem;
}
.app-footer a { color: #0f766e; text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }

/* Section dividers */
.section-divider {
    border: 0; height: 1px;
    background: linear-gradient(to right, transparent, #cbd5e1, transparent);
    margin: 1.5rem 0;
}
</style>
"""


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>MDA Web Calculator</h1>
            <p>
                Calculate <b>10 Maximum Depositional Age methods</b> from your detrital
                zircon U-Pb data. Paste from Excel, upload a file, and export
                publication-ready plots and tables instantly.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">Free &amp; Open Source</span>
                <span class="hero-badge">No Installation</span>
                <span class="hero-badge">10 MDA Methods</span>
                <span class="hero-badge">Publication-Ready Plots</span>
                <span class="hero-badge">PDF / PNG / CSV Export</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the settings sidebar and return configuration dict."""
    with st.sidebar:
        st.header("Settings")

        in_mode = st.selectbox(
            "Input uncertainty",
            INPUT_MODES,
            index=1,
            help="How your sigma values are expressed in the pasted data.",
        )
        out_mode = st.selectbox(
            "Output uncertainty",
            OUTPUT_MODES,
            index=1,
            help="How uncertainties should be reported in results.",
        )
        pair_stride = st.selectbox(
            "Column stride",
            [2, 1],
            index=0,
            help="2 = age/sigma column pairs; 1 = every adjacent column pair.",
        )

        st.markdown("---")
        st.subheader("Reference lines")
        ref_age = st.number_input(
            "Reference age (Ma)",
            value=0.0,
            min_value=0.0,
            step=0.1,
            help="Add a vertical reference line on the forest plot. Set to 0 to disable.",
        )
        ref_label = st.text_input(
            "Reference label",
            value="",
            placeholder="e.g. 538.8 Ma",
        )
        ref_age_2 = st.number_input(
            "Second reference age (Ma)",
            value=0.0,
            min_value=0.0,
            step=0.1,
            help="Optional second reference with arrow annotation. Set to 0 to disable.",
        )
        ref_label_2 = st.text_input(
            "Second reference label",
            value="",
            placeholder="e.g. 553.2 Ma",
        )

        st.markdown("---")
        st.subheader("About")
        st.markdown(
            "**MDA Web Calculator** v2.0  \n"
            "A free, open-source tool for detrital zircon geochronology.  \n\n"
            "**Methods**: YSG, YDZ, YC1\u03c3, YC2\u03c3, Y3Za, Y3Zo_2\u03c3, YSP, MLA, YPP, \u03c4  \n\n"
            "**Citation**: If you use this tool in a publication, please cite the "
            "underlying methods and acknowledge this calculator."
        )

    return {
        "in_mode": in_mode,
        "out_mode": out_mode,
        "pair_stride": pair_stride,
        "ref_age": ref_age if ref_age > 0 else None,
        "ref_label": ref_label or None,
        "ref_age_2": ref_age_2 if ref_age_2 > 0 else None,
        "ref_label_2": ref_label_2 or None,
    }


# ---------------------------------------------------------------------------
# Spreadsheet grid
# ---------------------------------------------------------------------------

def _init_grid():
    """Ensure session_state has grid data (2D list)."""
    if "grid_data" not in st.session_state or st.session_state.grid_data is None:
        try:
            st.session_state.grid_data = _df_to_grid_data(parse_table(DEFAULT_EXAMPLE))
        except Exception:
            st.session_state.grid_data = _make_empty_grid_data()
    if "grid_version" not in st.session_state:
        st.session_state.grid_version = 0


def render_spreadsheet() -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Render the Handsontable spreadsheet input area with toolbar.

    Returns (run_clicked, working_df) where working_df is a pandas DataFrame
    with the original column headers restored (or None if grid is empty).
    """
    _init_grid()

    # ---- Input format help ----
    with st.expander("How to use the spreadsheet", expanded=False):
        st.markdown(
            "1. **Paste from Excel**: Copy your data in Excel, click any cell "
            "in the grid below, and press **Ctrl+V**.  \n"
            "2. **Upload a file**: Click **Open** to upload CSV, TSV, or Excel files.  \n"
            "3. **Column layout**: Arrange data as alternating **age, sigma** column pairs. "
            "Row 1 should contain column header names.  \n"
            "4. **First age column**: Use the selector below the grid to indicate which "
            "column your age/sigma pairs start from.  \n"
            "5. **Filtering**: Apply filters (e.g., U<1000 ppm, exclude values with "
            "\u00b2\u2070\u2076Pb/\u00b2\u2070\u2074Pb between 0 and 2000) *before* pasting.  \n"
            "6. **Right-click** any cell for options: insert/remove rows and columns, "
            "undo, redo, copy, cut, alignment, read-only."
        )

    # ---- File uploader (hidden by default, shown via Open button) ----
    if "show_uploader" not in st.session_state:
        st.session_state.show_uploader = False

    # ---- Handsontable grid ----
    st.caption("Right-click for context menu \u2022 Paste from Excel \u2022 Columns A\u2013Z")
    grid_key = f"hot_grid_v{st.session_state.grid_version}"
    returned_data = handsontable_grid(
        data=st.session_state.grid_data,
        height=480,
        key=grid_key,
    )
    if returned_data is not None:
        st.session_state.grid_data = returned_data

    # ---- File uploader (conditionally shown) ----
    if st.session_state.show_uploader:
        upload_key = f"file_upload_v{st.session_state.grid_version}"
        uploaded = st.file_uploader(
            "Upload data file",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            key=upload_key,
        )
        if uploaded is not None:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    raw = pd.read_excel(uploaded, engine="openpyxl")
                else:
                    raw = pd.read_csv(uploaded, sep=None, engine="python")
                st.session_state.grid_data = _df_to_grid_data(raw)
                st.session_state.show_uploader = False
                st.session_state.grid_version += 1
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    # ---- Toolbar ----
    run_clicked = False
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        if st.button("Defaults", use_container_width=True, help="Load example data"):
            st.session_state.grid_data = _df_to_grid_data(parse_table(DEFAULT_EXAMPLE))
            st.session_state.grid_version += 1
            st.rerun()

    with c2:
        if st.button("Clear", use_container_width=True, help="Clear all data"):
            st.session_state.grid_data = _make_empty_grid_data()
            st.session_state.grid_version += 1
            st.rerun()

    with c3:
        if st.button("Open", use_container_width=True, help="Upload CSV / Excel file"):
            st.session_state.show_uploader = not st.session_state.show_uploader
            st.rerun()

    with c4:
        working = _grid_data_to_df(st.session_state.grid_data)
        csv_bytes = working.to_csv(index=False).encode("utf-8") if not working.empty else b""
        st.download_button(
            "Save",
            data=csv_bytes,
            file_name="mda_input_data.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download current data as CSV",
        )

    with c5:
        run_clicked = st.button(
            "\u25B6  PLOT",
            type="primary",
            use_container_width=True,
            help="Compute all MDA methods and generate plots",
        )

    with c6:
        if st.button("PDF", use_container_width=True, help="Export summary plot as PDF (after plotting)"):
            st.session_state["pdf_requested"] = True

    # ---- First-column selector ----
    if not working.empty and len(working.columns) > 0:
        col_options = list(range(len(working.columns)))
        saved_idx = st.session_state.get("first_col_idx", 0)
        if saved_idx >= len(col_options):
            saved_idx = 0
        first_col_idx = st.select_slider(
            "First age column",
            options=col_options,
            value=saved_idx,
            format_func=lambda i: f"{_col_letter(i)} \u2014 {working.columns[i]}",
            help="Select which column contains the first age data. The next column is treated as sigma.",
            key="first_col_slider",
        )
    else:
        first_col_idx = 0

    st.session_state["first_col_idx"] = first_col_idx

    return run_clicked, working if not working.empty else None


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <b>MDA Web Calculator</b> v2.0 &mdash; Free, open-source tool for detrital zircon geochronology<br>
            MLA is implemented as a Gaussian-mixture / BIC approximation to IsoplotR youngest-component unmixing.<br>
            If you use this tool in a publication, please acknowledge it and cite the underlying method references.<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
