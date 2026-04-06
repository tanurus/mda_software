"""Custom Streamlit component wrapping Handsontable for Excel-like grid input."""

import os
from typing import List, Optional

import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
_component_func = components.declare_component("handsontable_grid", path=_COMPONENT_DIR)


def handsontable_grid(
    data: Optional[List[List]] = None,
    height: int = 500,
    key: Optional[str] = None,
) -> Optional[List[List]]:
    """
    Render a Handsontable spreadsheet grid.

    Parameters
    ----------
    data : list of lists
        2D data to display.  Row 0 is typically column header names.
    height : int
        Pixel height of the grid container.
    key : str, optional
        Streamlit widget key for state management.

    Returns
    -------
    list of lists or None
        The current grid data (updated by the user), or *data* on first render.
    """
    if data is None:
        data = [[]]
    component_value = _component_func(data=data, height=height, key=key, default=data)
    return component_value
