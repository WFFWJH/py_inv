"""Port of zero_slip_boundary.m."""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from build_smooth_function import compute_patch_each_layer


def _ramp_zero_cols(h1: int, ramp_choice: str) -> np.ndarray:
    r = (ramp_choice or "").lower()
    if r == "bi_ramp":
        k = 4
    elif r == "qu_ramp_7":
        k = 7
    elif r == "qu_ramp_5":
        k = 5
    else:
        k = 0
    return np.zeros((h1, k), dtype=np.float64)


def zero_slip_boundary(slip_model: np.ndarray,
                       segment_ID: List[int],
                       top_layer_no: Union[int, str],
                       ratio: float,
                       ramp_choice: str,
                       dip_smooth: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
        patch_before = int(nL[:this_seg - 1, :].sum())

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
                [patch_before + int(nL_this[:jj + 1].sum()) for jj in range(Nlayer)],
                dtype=int,
            )
        else:
            raise ValueError(f"Unsupported top_layer_no: {top_layer_no!r}")

        # MATLAB patch indices are 1-based; convert to 0-based for numpy
        strike_indx = patch_top.astype(int) - 1
        dip_indx = strike_indx + Np

        if dip_smooth:
            zero_slip_indx = np.concatenate([strike_indx, dip_indx])
        else:
            zero_slip_indx = strike_indx

        V[zero_slip_indx] = ratio

    W = np.diag(V)
    h1 = W.shape[0]
    rmp = _ramp_zero_cols(h1, ramp_choice)
    W = np.concatenate([W, rmp], axis=1)
    return W, d
