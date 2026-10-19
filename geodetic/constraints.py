"""Bounds, zero-slip boundary, and ramp column helpers."""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

_ADD_COL_LOOKUP = {"bi_ramp": 4, "qu_ramp_7": 7, "qu_ramp_5": 5}


def add_col_of(ramp_choice: str) -> int:
    return _ADD_COL_LOOKUP.get((ramp_choice or "").lower(), 0)


def ramp_zero_cols(h1: int, ramp_choice: str) -> np.ndarray:
    k = add_col_of(ramp_choice)
    return np.zeros((h1, k), dtype=np.float64)


def bounds_new(
    NS: int,
    NT: int,
    tSm: Sequence[int],
    slip_max: float,
    add_col: int,
    Con: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct lb, ub vectors (port of ``bounds_new.m``)."""
    tSm = np.asarray(tSm, dtype=int)
    Npatch = int(tSm.sum())

    lb = -slip_max * np.ones(NT * Npatch, dtype=np.float64)
    ub = slip_max * np.ones(NT * Npatch, dtype=np.float64)

    NS1 = NS
    Con = list(Con)

    for i in range(1, NS1 + 1):
        k1 = int(tSm[:i].sum())
        k2 = int(tSm[: i + 1].sum())
        for k in range(k1, k2):
            if Con[0] > 0:
                lb[k] = 0.0
            if Con[1] > 0:
                lb[k + Npatch] = 0.0
            if Con[0] < 0:
                ub[k] = 0.0
            if Con[1] < 0:
                ub[k + Npatch] = 0.0

    if add_col > 0:
        ramp_lb = -np.inf * np.ones(add_col, dtype=np.float64)
        ramp_ub = np.inf * np.ones(add_col, dtype=np.float64)
        lb = np.concatenate([lb, ramp_lb])
        ub = np.concatenate([ub, ramp_ub])

    return lb, ub


def zero_slip_boundary(
    slip_model: np.ndarray,
    segment_ID: List[int],
    top_layer_no: Union[int, str],
    ratio: float,
    ramp_choice: str,
    dip_smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of ``zero_slip_boundary.m``."""
    from geodetic.smoothing import compute_patch_each_layer

    Np = slip_model.shape[0]
    all_fault_id = slip_model[:, 0].astype(int)
    all_patch_id = slip_model[:, 1].astype(int)
    all_layer_id = slip_model[:, 2].astype(int)
    nL = compute_patch_each_layer(slip_model)

    if dip_smooth:
        V = np.zeros(2 * Np, dtype=np.float64)
        d = np.zeros(2 * Np, dtype=np.float64)
    else:
        V = np.zeros(Np, dtype=np.float64)
        d = np.zeros(Np, dtype=np.float64)

    for this_seg in segment_ID:
        mask = all_fault_id == this_seg
        patch_this = all_patch_id[mask]
        layer_this = all_layer_id[mask]
        nL_this = nL[this_seg - 1, :]
        patch_before = int(nL[: this_seg - 1, :].sum())

        if isinstance(top_layer_no, (int, np.integer)):
            patch_top = patch_this[layer_this == int(top_layer_no)]
        elif top_layer_no == "left":
            Nlayer = int(layer_this.max())
            patch_top = np.array(
                [patch_before + 1 + int(nL_this[:jj].sum()) for jj in range(Nlayer)],
                dtype=int,
            )
        elif top_layer_no == "right":
            Nlayer = int(layer_this.max())
            patch_top = np.array(
                [patch_before + int(nL_this[: jj + 1].sum()) for jj in range(Nlayer)],
                dtype=int,
            )
        else:
            raise ValueError(f"Unsupported top_layer_no: {top_layer_no!r}")

        strike_indx = patch_top.astype(int) - 1
        dip_indx = strike_indx + Np

        if dip_smooth:
            zero_slip_indx = np.concatenate([strike_indx, dip_indx])
        else:
            zero_slip_indx = strike_indx

        V[zero_slip_indx] = ratio

    W = np.diag(V)
    h1 = W.shape[0]
    rmp = ramp_zero_cols(h1, ramp_choice)
    W = np.concatenate([W, rmp], axis=1)
    return W, d
