"""Port of build_smooth_function.m (+ smoo1_each_plane.m, smoo1_segments.m).

Tikhonov first-derivative smoothing for a multi-segment, multi-layer fault.

Not ported (intentionally skipped, same as when intersect_file=[] in MATLAB):
  - smoo1_dip_vertical_intersection.m
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def compute_patch_each_layer(slip_model: np.ndarray) -> np.ndarray:
    """Return nL[Nf, Nl]: patches per (segment, layer)."""
    fault_id = slip_model[:, 0].astype(int)
    layer_id = slip_model[:, 2].astype(int)
    Nf = int(fault_id.max())
    Nl = int(layer_id.max())
    nL = np.zeros((Nf, Nl), dtype=int)
    for i in range(1, Nf + 1):
        mask = fault_id == i
        layers = layer_id[mask]
        for j in range(1, Nl + 1):
            nL[i - 1, j - 1] = int(np.sum(layers == j))
    return nL


def _build_segment_geometry(slip_model: np.ndarray, width_col: int):
    """Compute Wp[i], nL[i,j], d[i,j] (patch length *or* width depending on width_col)."""
    fault_id = slip_model[:, 0].astype(int)
    layer_id = slip_model[:, 2].astype(int)
    Nf = int(fault_id.max())
    Nl = int(layer_id.max())

    Wp = np.zeros(Nf, dtype=int)
    for i in range(1, Nf + 1):
        mask = fault_id == i
        if np.any(mask):
            Wp[i - 1] = int(layer_id[mask].max())

    nL = np.zeros((Nf, Nl), dtype=int)
    dL = np.zeros((Nf, Nl), dtype=np.float64)
    for i in range(1, Nf + 1):
        mask = fault_id == i
        layers = layer_id[mask]
        lengths = slip_model[mask, width_col]
        for j in range(1, Wp[i - 1] + 1):
            sel = layers == j
            nL[i - 1, j - 1] = int(np.sum(sel))
            if np.any(sel):
                dL[i - 1, j - 1] = float(lengths[sel][0])
    return Nf, Nl, Wp, nL, dL


def smoo1_each_plane(slip_model: np.ndarray,
                     Fdip: float = 5.0,
                     dip_smooth: bool = True) -> np.ndarray:
    """Plane-internal first-derivative smoothing.

    Returns dense H (rows vary, cols = 2*Np).
    """
    Np = slip_model.shape[0]
    Nf, Nl, Wp, nL, dL = _build_segment_geometry(slip_model, width_col=6)  # use lp

    tSm = np.concatenate([[0], nL.sum(axis=1)])  # length Nf+1, cumulative helper

    rows: List[np.ndarray] = []

    # along-strike smoothing within each layer
    for i in range(Nf):
        for j in range(Wp[i]):
            for k in range(nL[i, j] - 1):
                base = int(tSm[:i + 1].sum()) + int(nL[i, :j].sum())
                idx_minus_s = base + k
                idx_plus_s = base + (k + 1)
                idx_minus_d = idx_minus_s + Np
                idx_plus_d = idx_plus_s + Np
                if dip_smooth:
                    row_s = np.zeros(2 * Np, dtype=np.float64)
                    row_d = np.zeros(2 * Np, dtype=np.float64)
                    row_s[idx_minus_s] = -1.0
                    row_s[idx_plus_s] = 1.0
                    row_d[idx_minus_d] = -Fdip
                    row_d[idx_plus_d] = Fdip
                    rows.append(row_s)
                    rows.append(row_d)
                else:
                    row = np.zeros(2 * Np, dtype=np.float64)
                    row[idx_minus_s] = -1.0
                    row[idx_plus_s] = 1.0
                    rows.append(row)

    # between-layer smoothing within the same segment
    for i in range(Nf):
        for j in range(Wp[i] - 1):
            d_top = dL[i, j]
            d_bot = dL[i, j + 1]
            for k in range(nL[i, j]):
                Lp_top_l = k * d_top
                Lp_top_r = (k + 1) * d_top
                for k_d in range(nL[i, j + 1]):
                    Lp_bot_l = k_d * d_bot
                    Lp_bot_r = (k_d + 1) * d_bot
                    contact = (
                        abs(Lp_top_l - Lp_bot_l) < 1e-3
                        or abs(Lp_top_r - Lp_bot_r) < 1e-3
                        or (Lp_top_l < Lp_bot_l < Lp_top_r)
                        or (Lp_top_l < Lp_bot_r < Lp_top_r)
                    )
                    if not contact:
                        continue
                    base = int(tSm[:i + 1].sum())
                    idx_minus_s = base + int(nL[i, :j].sum()) + k
                    idx_plus_s = base + int(nL[i, :j + 1].sum()) + k_d
                    idx_minus_d = idx_minus_s + Np
                    idx_plus_d = idx_plus_s + Np
                    if dip_smooth:
                        row_s = np.zeros(2 * Np, dtype=np.float64)
                        row_d = np.zeros(2 * Np, dtype=np.float64)
                        row_s[idx_minus_s] = -1.0
                        row_s[idx_plus_s] = 1.0
                        row_d[idx_minus_d] = -Fdip
                        row_d[idx_plus_d] = Fdip
                        rows.append(row_s)
                        rows.append(row_d)
                    else:
                        row = np.zeros(2 * Np, dtype=np.float64)
                        row[idx_minus_s] = -1.0
                        row[idx_plus_s] = 1.0
                        rows.append(row)

    if not rows:
        return np.zeros((0, 2 * Np), dtype=np.float64)
    return np.vstack(rows)


def smoo1_segments(slip_model: np.ndarray,
                   segment_file: str,
                   Fdip: float = 5.0,
                   dip_smooth: bool = True) -> np.ndarray:
    """Cross-segment smoothing. segment_file format:
       <id1> <left|right> <id2> <left|right>
    """
    Np = slip_model.shape[0]
    fault_id = slip_model[:, 0].astype(int)
    dip_all = slip_model[:, 9]

    Nf, Nl, Wp, nL, dW = _build_segment_geometry(slip_model, width_col=7)  # use wp
    tSm = np.concatenate([[0], nL.sum(axis=1)])

    connects = []
    with open(segment_file, "r", encoding="utf-8") as fid:
        for line in fid:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            connects.append((int(parts[0]), parts[1].lower(), int(parts[2]), parts[3].lower()))

    rows: List[np.ndarray] = []
    for id1, s1, id2, s2 in connects:
        # first-matching dip for each segment
        idx_first_1 = np.argmax(fault_id == id1)
        idx_first_2 = np.argmax(fault_id == id2)
        dip_id1 = float(dip_all[idx_first_1])
        dip_id2 = float(dip_all[idx_first_2])
        sin_d1 = np.sin(np.deg2rad(dip_id1))
        sin_d2 = np.sin(np.deg2rad(dip_id2))

        for jj1 in range(1, Wp[id1 - 1] + 1):
            if s1 == "right":
                idx_minus = int(tSm[:id1].sum()) + int(nL[id1 - 1, :jj1].sum())
            else:  # 'left'
                idx_minus = int(tSm[:id1].sum()) + int(nL[id1 - 1, :jj1 - 1].sum()) + 1
            idx_minus -= 1  # convert MATLAB 1-based index to 0-based
            wp_top1 = dW[id1 - 1, :jj1 - 1].sum()
            wp_bot1 = dW[id1 - 1, :jj1].sum()

            for jj2 in range(1, Wp[id2 - 1] + 1):
                if s2 == "right":
                    idx_plus = int(tSm[:id2].sum()) + int(nL[id2 - 1, :jj2].sum())
                else:
                    idx_plus = int(tSm[:id2].sum()) + int(nL[id2 - 1, :jj2 - 1].sum()) + 1
                idx_plus -= 1
                wp_top2 = dW[id2 - 1, :jj2 - 1].sum()
                wp_bot2 = dW[id2 - 1, :jj2].sum()

                contact = (
                    abs(wp_top1 - wp_top2) < 1e-3
                    or abs(wp_bot1 - wp_bot2) < 1e-3
                    or (wp_top1 < wp_top2 < wp_bot1)
                    or (wp_top1 < wp_bot2 < wp_bot1)
                )
                if not contact:
                    continue

                idx_minus_s = idx_minus
                idx_minus_d = idx_minus_s + Np
                idx_plus_s = idx_plus
                idx_plus_d = idx_plus_s + Np

                row_s = np.zeros(2 * Np, dtype=np.float64)
                row_d = np.zeros(2 * Np, dtype=np.float64)
                row_s[idx_minus_s] = -1.0
                row_s[idx_plus_s] = 1.0
                row_d[idx_minus_d] = -Fdip * sin_d1
                row_d[idx_plus_d] = Fdip * sin_d2

                if dip_smooth:
                    rows.append(row_s)
                    rows.append(row_d)
                else:
                    rows.append(row_s)

    if not rows:
        return np.zeros((0, 2 * Np), dtype=np.float64)
    return np.vstack(rows)


def _ramp_zero_cols(h1: int, ramp_choice: str) -> np.ndarray:
    r = ramp_choice.lower() if ramp_choice else "none"
    if r == "bi_ramp":
        k = 4
    elif r == "qu_ramp_7":
        k = 7
    elif r == "qu_ramp_5":
        k = 5
    else:
        k = 0
    return np.zeros((h1, k), dtype=np.float64)


def build_smooth_function(slip_model_vs: np.ndarray,
                          slip_model_ds: Optional[np.ndarray],
                          segment_file: Optional[str],
                          intersect_file: Optional[str],
                          ramp_choice: str,
                          dip_id: Optional[List[int]] = None,
                          dip_smooth: bool = True,
                          Fdip: float = 5.0) -> Tuple[np.ndarray, int, np.ndarray]:
    """Returns H, h1, indx_less_smooth_patch."""
    if slip_model_ds is None or len(slip_model_ds) == 0:
        slip_model = slip_model_vs.copy()
    else:
        slip_model = np.vstack([slip_model_vs, slip_model_ds])
    slip_model = slip_model.copy()
    slip_model[:, 1] = np.arange(1, slip_model.shape[0] + 1)

    H_plane = smoo1_each_plane(slip_model, Fdip=Fdip, dip_smooth=dip_smooth)

    if segment_file:
        H_segment = smoo1_segments(slip_model, segment_file, Fdip=Fdip, dip_smooth=dip_smooth)
    else:
        H_segment = np.zeros((0, 2 * slip_model.shape[0]), dtype=np.float64)

    if intersect_file:
        raise NotImplementedError(
            "smoo1_dip_vertical_intersection is not ported; current inversion "
            "uses intersect_file=[] in MATLAB."
        )

    H = np.vstack([H_plane, H_segment])
    h1 = H.shape[0]
    Np = H.shape[1] // 2

    indx_less_smooth_patch = np.array([], dtype=int)
    if dip_id:
        all_fault_id = slip_model[:, 0].astype(int)
        all_patch_id = slip_model[:, 1].astype(int)
        idx_dip_patch: List[int] = []
        for sid in dip_id:
            idx_dip_patch.extend(all_patch_id[all_fault_id == sid].tolist())
        indx_less_smooth_patch = np.asarray(idx_dip_patch, dtype=int)

        for ii in range(h1):
            nz = np.flatnonzero(H[ii, :])
            if nz.size != 2:
                raise RuntimeError("There are more than 2 elements in each row of smoothing matrix")
            if nz.max() < Np:
                continue  # skip strike-only rows
            left_nz = nz[0] - Np
            right_nz = nz[1] - Np
            # MATLAB patch IDs are 1-based; indx_less_smooth_patch is 1-based
            if (left_nz + 1) in indx_less_smooth_patch and (right_nz + 1) in indx_less_smooth_patch:
                H[ii, nz] /= 2.0

    rmp = _ramp_zero_cols(h1, ramp_choice)
    H = np.concatenate([H, rmp], axis=1)
    return H, h1, indx_less_smooth_patch
