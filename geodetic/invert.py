"""Port of make_fault_from_insar1.m.

Pipeline (high-level):
  1. Load downsampled InSAR/AZO data for every track, build its Green's matrix
     (Okada) and append ramp columns if requested.
  2. Re-group ramp columns so that every orbit (ascending/descending track number)
     has its own block (expand_last_columns).
  3. Build Tikhonov first-derivative smoothing H, and zero-slip boundary
     constraints (bottom, left edge of segment 4, right edge of segment 1).
  4. Stack everything -> Greens, bdata_sm and solve a bounded linear least
     squares problem (scipy.optimize.least_squares with TRF; MATLAB lsqlin).
  5. Split u into strike-slip, dip-slip and ramp coefficients; compute M0 & Mw.

This port assumes model_type='okada' (EDCMP layered model is skipped).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import savemat

from geodetic.constraints import add_col_of
from geodetic.constraints import bounds_new, zero_slip_boundary
from geodetic.green_function import build_green_function
from geodetic.plot_insar_model_resampled import plot_insar_model_resampled
from geodetic.smoothing import build_smooth_function
from geodetic.solver import solve_bounded_lsq


def _track_orbit_number(track: str) -> Optional[int]:
    """Extract orbit number from track path (pattern /[AD]\\d+/)."""
    path = track.replace("\\", "/")
    m = re.search(r"/[AD](\d+)", path)
    if m is None:
        return None
    return int(m.group(1))


def _expand_last_columns(matrices: List[np.ndarray],
                         add_col: int,
                         class_map: Sequence[int]) -> List[np.ndarray]:
    """Port of expand_last_columns.m.

    Replace the last `add_col` columns of each matrix with a block of
    `n_classes*add_col` columns where only the class's slot is non-zero.
    """
    n_m = len(matrices)
    class_map = np.asarray(class_map, dtype=int)
    if class_map.size != n_m:
        raise ValueError("class_map length must equal number of matrices")
    n_classes = int(class_map.max())

    result = []
    for i, A in enumerate(matrices):
        if A.shape[1] < add_col:
            raise ValueError(f"Matrix {i} has fewer columns than add_col={add_col}")
        last_cols = A[:, -add_col:]
        B = np.zeros((A.shape[0], n_classes * add_col), dtype=np.float64)
        cls = int(class_map[i])
        pos_start = (cls - 1) * add_col
        B[:, pos_start:pos_start + add_col] = last_cols

        C = np.zeros((A.shape[0], A.shape[1] + (n_classes - 1) * add_col), dtype=np.float64)
        C[:, : A.shape[1] - add_col] = A[:, : A.shape[1] - add_col]
        C[:, A.shape[1] - add_col:] = B
        result.append(C)
    return result


def _pad_zero_cols(A: np.ndarray, extra: int) -> np.ndarray:
    if extra <= 0:
        return A
    return np.concatenate([A, np.zeros((A.shape[0], extra), dtype=A.dtype)], axis=1)


def _matlab_col(v: np.ndarray) -> np.ndarray:
    """1D / (n,1) -> MATLAB 列向量 (n, 1), float64."""
    return np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1, 1))


def _matlab_scalar_double(v) -> np.ndarray:
    """标量 -> MATLAB double 标量."""
    return np.array(float(v), dtype=np.float64)


def build_greens_mat_dict(
    *,
    G_last: np.ndarray,
    Bdata: np.ndarray,
    bd_last: np.ndarray,
    slip_model: np.ndarray,
    bdata_sm: np.ndarray,
    GrF: np.ndarray,
    H: np.ndarray,
    h1,
    Wb: np.ndarray,
    Wl: np.ndarray,
    Wr: np.ndarray,
    ramp_choice: str,
    u: np.ndarray,
    return_var: Optional[np.ndarray] = None,
    RMS_misfit: Optional[float] = None,
    model_roughness: Optional[float] = None,
    class_map: Optional[np.ndarray] = None,
) -> dict:
    """组装与 MATLAB ``save('greens_okada.mat', ...)`` 兼容的变量 dict."""
    out = {
        "G_last": np.ascontiguousarray(G_last, dtype=np.float64),
        "Bdata": _matlab_col(Bdata),
        "bd_last": _matlab_col(bd_last),
        "slip_model": np.ascontiguousarray(slip_model, dtype=np.float64),
        "bdata_sm": _matlab_col(bdata_sm),
        "GrF": np.ascontiguousarray(GrF, dtype=np.float64),
        "H": np.ascontiguousarray(H, dtype=np.float64),
        "h1": _matlab_scalar_double(h1),
        "Wb": np.ascontiguousarray(Wb, dtype=np.float64),
        "Wl": np.ascontiguousarray(Wl, dtype=np.float64),
        "Wr": np.ascontiguousarray(Wr, dtype=np.float64),
        "ramp_choice": ramp_choice,
        "u": _matlab_col(u),
    }
    if return_var is not None:
        out["return_var"] = _matlab_col(return_var)
    if RMS_misfit is not None:
        out["RMS_misfit"] = _matlab_scalar_double(RMS_misfit)
    if model_roughness is not None:
        out["model_roughness"] = _matlab_scalar_double(model_roughness)
    if class_map is not None:
        out["class_map"] = np.ascontiguousarray(class_map, dtype=np.float64).reshape(-1, 1)
    return out


def make_fault_from_insar1(slip_model_vs: np.ndarray,
                           slip_model_ds: Optional[np.ndarray],
                           iter_step: int,
                           tracks: Sequence[str],
                           paths_type: Sequence[str],
                           *,
                           smoothness: float = 1.0,
                           alos_ratio: float = 1.0,   # beta -- weight for AZI
                           rng_ratio: float = 0.8,    # alpha -- weight for LOS
                           segment_smooth_file: Optional[str] = None,
                           intersect_smooth_file: Optional[str] = None,
                           shallow_dip_id: Optional[List[int]] = None,
                           model_type: str = "okada",
                           fault_file: Optional[str] = None,
                           ref_lon: Optional[float] = None,
                           lonc: Optional[float] = None,
                           latc: Optional[float] = None,
                           Con: Sequence[int] = (0, 0, 0),
                           ramp_choice: str = "none",
                           nu: float = 0.25,
                           backend: str = "auto",
               save_greens_path: Optional[str] = None,
               max_nfev: int = 100,
               verbose: bool = True,
               plot_resampled_fits: bool = True,
               ) -> Tuple[np.ndarray, float, float, np.ndarray, dict]:
    """Run a single linear inversion step.

    Returns
    -------
    slip_model : ndarray
        Input geometry augmented with u in columns 11/12 (0-indexed), matching
        MATLAB's `slip_model(:, 12/13)`.
    rms_val : float
        Root-mean-square residual of the data rows.
    model_roughness : float
    return_var : ndarray (5,)
        [redu_perc, exitflag, rms, Mw, M0/1e20]
    extras : dict
        Intermediate matrices useful for verification against MATLAB.
    plot_resampled_fits
        若 True 且 ``iter_step==1`` (与 ``make_fault_from_insar1.m`` 中 ``iint==1`` 一致), 对每道分别调用
        ``plot_insar_model_resampled`` 写出三合一图与 ``los_model.mat``。
    """
    alpha = float(rng_ratio)
    beta = float(alos_ratio)
    lam = float(smoothness)
    ramp_choice = (ramp_choice or "none").lower()

    iint = int(iter_step)

    if slip_model_ds is None or len(slip_model_ds) == 0:
        slip_model = slip_model_vs.copy()
    else:
        slip_model = np.vstack([slip_model_vs, slip_model_ds])
    slip_model = slip_model.astype(np.float64, copy=True)
    slip_model[:, 1] = np.arange(1, slip_model.shape[0] + 1)
    if verbose:
        print(f"There are total {int(slip_model[:, 0].max())} segments")

    # ------------------------------------------------------------------
    # Load sampled data and build per-track Green's functions
    # ------------------------------------------------------------------
    G_raw_list: List[np.ndarray] = []
    G_list: List[np.ndarray] = []
    bd_raw_list: List[np.ndarray] = []
    bd_list: List[np.ndarray] = []
    labels_tracks = np.zeros(len(tracks), dtype=int)

    for i, trk in enumerate(tracks):
        fname = os.path.join(trk, f"los_samp{iint}.mat")
        dtype = paths_type[i]

        G_raw, G, bd_raw, bd = build_green_function(
            slip_model, fname, dtype, ramp_choice, model_type=model_type, nu=nu, backend=backend)

        if dtype == "azo":
            G_raw *= beta
            G *= beta
            bd_raw *= beta
            bd *= beta
        elif dtype == "insar":
            G_raw *= alpha
            G *= alpha
            bd_raw *= alpha
            bd *= alpha
        else:
            raise ValueError(f"Unknown path type: {dtype}")

        G_raw_list.append(G_raw)
        G_list.append(G)
        bd_raw_list.append(bd_raw)
        bd_list.append(bd)

        num = _track_orbit_number(trk)
        if num is not None:
            labels_tracks[i] = num

    # Same orbit shares a ramp -> class_map from unique orbit numbers
    unique_orbits = np.unique(labels_tracks)
    class_map = np.zeros(len(tracks), dtype=int)
    for idx, orb in enumerate(unique_orbits, start=1):
        class_map[labels_tracks == orb] = idx
    n_classes = int(class_map.max()) if class_map.size else 1

    # ------------------------------------------------------------------
    # Smoothing and zero-slip boundary constraints
    # ------------------------------------------------------------------
    dip_smooth = True
    H, h1, _ = build_smooth_function(
        slip_model_vs, slip_model_ds, segment_smooth_file, intersect_smooth_file,
        ramp_choice, dip_id=shallow_dip_id, dip_smooth=dip_smooth)

    plane_fault = list(range(1, int(slip_model[:, 0].max()) + 1))
    bottom_layer_no = int(slip_model[:, 2].max())
    Wb, db = zero_slip_boundary(slip_model, plane_fault, bottom_layer_no,
                                ratio=3e-4, ramp_choice=ramp_choice, dip_smooth=dip_smooth)
    # left / right boundary constraints follow MATLAB: right_fault=[1] uses 'right',
    # left_fault=[4] uses 'left' (the naming is MATLAB's internal convention).
    Wr, dr = zero_slip_boundary(slip_model, [4], "left",
                                ratio=3e-4, ramp_choice=ramp_choice, dip_smooth=dip_smooth)
    Wl, dl = zero_slip_boundary(slip_model, [1], "right",
                                ratio=3e-4, ramp_choice=ramp_choice, dip_smooth=dip_smooth)

    add_col = add_col_of(ramp_choice)

    if add_col > 0 and n_classes > 1:
        G_raw_ramp = _expand_last_columns(G_raw_list, add_col, class_map)
        G_ramp = _expand_last_columns(G_list, add_col, class_map)
        extra = (n_classes - 1) * add_col
        H = _pad_zero_cols(H, extra)
        Wb = _pad_zero_cols(Wb, extra)
        Wl = _pad_zero_cols(Wl, extra)
        Wr = _pad_zero_cols(Wr, extra)
    else:
        G_raw_ramp = G_raw_list
        G_ramp = G_list

    # ------------------------------------------------------------------
    # Bounds (lb, ub)
    # ------------------------------------------------------------------
    nflt = int(slip_model[:, 0].max())
    fault_id = slip_model[:, 0].astype(int)
    tSm = np.zeros(nflt + 1, dtype=int)
    for i in range(1, nflt + 1):
        tSm[i] = int(np.sum(fault_id == i))

    NT = 2
    NS = nflt
    total_ramp_cols = add_col * max(n_classes, 1)  # matches expanded Greens width
    lb, ub = bounds_new(NS, NT, tSm,10, total_ramp_cols, Con)

    # ------------------------------------------------------------------
    # Assemble final system
    # ------------------------------------------------------------------
    G_last = np.vstack(G_ramp) if len(G_ramp) else np.zeros((0, 0))
    bd_last = np.concatenate(bd_list) if len(bd_list) else np.zeros(0)

    Greens = np.vstack([G_last, H * (lam / h1), Wb, Wl, Wr])
    bdata_sm = np.concatenate([bd_last, np.zeros(h1), db, dl, dr])

    GrF = np.vstack(G_raw_ramp) if len(G_raw_ramp) else np.zeros((0, 0))
    Bdata = np.concatenate(bd_raw_list) if len(bd_raw_list) else np.zeros(0)

    # Bounded LSQ: see geodetic.solver.solve_bounded_lsq
    sol = solve_bounded_lsq(Greens, bdata_sm, lb, ub, max_iter=int(max_nfev), verbose=verbose)
    u = sol.u
    residual = sol.residual
    exitflag = sol.exitflag
    resnorm = sol.resnorm

    rms0 = float(np.sum(Bdata * Bdata))
    rms1 = float(np.sum((GrF @ u - Bdata) ** 2))
    redu_perc = 100.0 * (rms0 - rms1) / rms0 if rms0 else 0.0
    rms_val = float(np.sqrt(np.mean((GrF @ u - Bdata) ** 2)))

    if verbose:
        print(f"rms misfit (dat., res.) = {rms0:.6e} {rms1:.6e} ({redu_perc:.6f}%)")
        print(f"RMS: {rms_val:.6f}")
        print(f"resnorm, resid. = {np.sqrt(max(resnorm, 0.0)):.6e} {np.mean(residual):.6e}")
        print(f"exitflag is {exitflag}")
        if sol.name == "least_squares" and not sol.success and sol.message:
            print(f"  optimizer: {sol.message}")

    rough_matrix = H @ u
    model_roughness = float(np.sqrt(np.mean(rough_matrix * rough_matrix))) \
        if rough_matrix.size else 0.0

    Npatch = int(tSm.sum())
    slip_model[:, 11] = u[:Npatch]
    slip_model[:, 12] = u[Npatch:2 * Npatch]

    mu = 30e9
    Apatch = slip_model[:, 6] * slip_model[:, 7]
    strike_u = slip_model[:, 11]
    strike_d = slip_model[:, 12]
    M0 = float(np.sum(mu * np.sqrt(strike_u ** 2 + strike_d ** 2) * Apatch))
    Mw = 2.0 / 3.0 * (np.log10(M0) - 9.1) if M0 > 0 else float("nan")
    if verbose:
        print(f"The moment magnitude is Mw = {Mw:.6f}\nM0 = {M0:.6e}")

    return_var = np.array([redu_perc, exitflag, rms_val, Mw, M0 / 1e20])

    extras = dict(
        G_last=G_last, Bdata=Bdata, bd_last=bd_last, bdata_sm=bdata_sm,
        GrF=GrF, H=H, h1=h1, Wb=Wb, Wl=Wl, Wr=Wr,
        ramp_choice=ramp_choice, u=u, class_map=class_map, total_ramp_cols=total_ramp_cols,
    )

    if  plot_resampled_fits and len(G_raw_ramp) > 0:
        rlon = float(ref_lon) if ref_lon is not None else 95.0
        lnc = float(lonc) if lonc is not None else 95.33
        ltc = float(latc) if latc is not None else 19.61
        for i, trk in enumerate(tracks):
            fname = os.path.join(str(trk), f"los_samp{iint}.mat")
            insar_model_i = np.asarray(G_raw_ramp[i] @ u, dtype=np.float64).ravel()
            try:
                out_png = plot_insar_model_resampled(
                    fname,
                    insar_model_i,
                    iter_step=iint,
                    fault=fault_file,
                    model_type=model_type,
                    misfit_range=1.0,
                    defo_max=3.0,
                    ref_lon=rlon,
                    lonc=lnc,
                    latc=ltc,
                    show=False,
                )
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print("plot_insar_model_resampled 失败(%s): %s" % (fname, e), flush=True)
                raise
            if verbose:
                print("plot_resampled -> %s" % os.path.normpath(str(out_png)), flush=True)

    if save_greens_path:
        savemat(
            save_greens_path,
            build_greens_mat_dict(
                G_last=G_last, Bdata=Bdata, bd_last=bd_last,
                slip_model=slip_model, bdata_sm=bdata_sm, GrF=GrF,
                H=H, h1=h1, Wb=Wb, Wl=Wl, Wr=Wr,
                ramp_choice=ramp_choice, u=u,
            ),
            do_compression=True,
            oned_as="column",
        )
        if verbose:
            print(f"Saved Greens snapshot -> {save_greens_path}")

    return slip_model, rms_val, model_roughness, return_var, extras
