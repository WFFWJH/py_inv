"""Lazy checkerboard parameters and synthetic observation grid."""
from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace


@lru_cache(maxsize=1)
def _context() -> SimpleNamespace:
    import numpy as np

    ny = 700
    x1d_m = np.unique(
        np.concatenate(
            (
                np.linspace(-50, -20, 50),
                np.linspace(-20, 20, 200),
                np.linspace(20, 50, 50),
            )
        )
    )
    y1d_m = np.linspace(-20, 120, ny)
    X, Y = np.meshgrid(x1d_m, y1d_m)
    xe, yn = X.ravel() * 1000.0, Y.ravel() * 1000.0
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layers, n_layer = 2, 10
    slip_matrix = np.zeros((layers, n_layer))
    one_layer_slip = np.arange(n_layer) % 2 == 0
    slip_matrix[0] = one_layer_slip
    slip_matrix[1] = ~one_layer_slip

    return SimpleNamespace(
        ny=ny,
        x1d_m=x1d_m,
        y1d_m=y1d_m,
        X=X,
        Y=Y,
        xe=xe,
        yn=yn,
        fault_file=os.path.join(project_root, "checkerboard.txt"),
        ref=dict(ref_lon=95, lonc=95.33, latc=19.61),
        axis_range=[0, 20, 0, 100, -20, 0],
        dip=[80],
        width=20e3,
        length=100e3,
        l_ratio=1,
        w_ratio=1,
        len_top=10e3,
        layers=layers,
        n_layer=n_layer,
        wid_top=10e3,
        n_patch=n_layer * layers,
        slip_matrix=slip_matrix,
        top0_model="top0_most_refine_extend.mph",
        top1_model="top1_most_refine_extend.mph",
    )


_CONTEXT_NAMES = {
    "ny", "x1d_m", "y1d_m", "X", "Y", "xe", "yn", "fault_file", "ref",
    "axis_range", "dip", "width", "length", "l_ratio", "w_ratio", "len_top",
    "layers", "n_layer", "wid_top", "n_patch", "slip_matrix", "top0_model",
    "top1_model",
}


def __getattr__(name: str):
    if name not in _CONTEXT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_context(), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _CONTEXT_NAMES)
