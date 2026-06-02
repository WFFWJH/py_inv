"""Okada-based Green's functions for InSAR LOS and azimuth offsets.

Ports `calc_green_insar_okada.m` and `calc_green_AZO_okada.m`.

slip_model column convention (0-indexed here, matches load_fault_one_plane.py):
    0: fault_id   1: patch_id   2: layer_id
    3: xp  4: yp  5: zp  (km? no -- meters, matching MATLAB)
    6: lp  7: wp          (patch length / down-dip width, meters)
    8: strike (deg)  9: dip (deg)
"""
from __future__ import annotations

import numpy as np

from calc_okada import calc_okada


def _xy2xy(x1: np.ndarray, y1: np.ndarray, phi: float):
    """MATLAB xy2XY.m: 2-D rotation by angle phi (radians)."""
    c = np.cos(phi)
    s = np.sin(phi)
    return c * x1 + s * y1, -s * x1 + c * y1


def _build_green_patch_loop(data_slip_model: np.ndarray,
                            data_insar: np.ndarray,
                            projection: str,
                            nu: float = 0.25,
                            backend: str = "auto") -> np.ndarray:
    """Shared inner loop for LOS (insar) and azimuth offsets (AZO)."""
    d2r = np.pi / 180.0
    xp = data_slip_model[:, 3]
    yp = data_slip_model[:, 4]
    zp = data_slip_model[:, 5]
    lp = data_slip_model[:, 6]
    wp = data_slip_model[:, 7]
    strkp = data_slip_model[:, 8]
    dip0 = data_slip_model[:, 9]

    Npatch = zp.size
    Npara = 2 * Npatch

    xe = np.asarray(data_insar[:, 0], dtype=np.float64)
    yn = np.asarray(data_insar[:, 1], dtype=np.float64)
    ve = np.asarray(data_insar[:, 3], dtype=np.float64)
    vn = np.asarray(data_insar[:, 4], dtype=np.float64)
    if projection == "insar":
        vz = np.asarray(data_insar[:, 5], dtype=np.float64)
    else:  # AZO
        theta_az = -np.degrees(np.arctan2(vn, ve)) - 180.0
        sin_az = np.sin(np.deg2rad(theta_az))
        cos_az = np.cos(np.deg2rad(theta_az))

    Nobs = xe.size
    G = np.zeros((Nobs, Npara), dtype=np.float64)

    HF = 1.0
    tp = np.zeros(Nobs, dtype=np.float64)

    for k in range(Npatch):
        strike_k = strkp[k] * d2r
        theta_k = (90.0 - strkp[k]) * d2r
        dxf = lp[k] * 0.5
        dx, dy = _xy2xy(np.asarray(dxf), np.asarray(0.0), -theta_k)
        xxo = xp[k] + float(dx)
        yyo = yp[k] + float(dy)
        zzo = zp[k]

        xpt = xe - xxo
        ypt = yn - yyo
        delta = dip0[k] * d2r
        d = -zzo
        L = lp[k]
        W = wp[k]

        ue1, un1, uz1 = calc_okada(HF, 1.0, xpt, ypt, nu, delta, d, L, W, 1, strike_k, tp, backend=backend)
        ue2, un2, uz2 = calc_okada(HF, 1.0, xpt, ypt, nu, delta, d, L, W, 2, strike_k, tp, backend=backend)

        if projection == "insar":
            u1 = ue1 * ve + un1 * vn + uz1 * vz
            u2 = ue2 * ve + un2 * vn + uz2 * vz
        else:  # AZO
            u1 = ue1 * sin_az + un1 * cos_az
            u2 = ue2 * sin_az + un2 * cos_az

        G[:, k] = u1
        G[:, k + Npatch] = u2

    return G


def calc_green_insar_okada(data_slip_model: np.ndarray,
                           data_insar: np.ndarray,
                           nu: float = 0.25,
                           backend: str = "auto") -> np.ndarray:
    """data_insar columns: [xe, yn, los, ve, vn, vz]."""
    return _build_green_patch_loop(data_slip_model, data_insar, "insar", nu, backend)


def calc_green_AZO_okada(data_slip_model: np.ndarray,
                         data_insar: np.ndarray,
                         nu: float = 0.25,
                         backend: str = "auto") -> np.ndarray:
    """data_insar columns: [xe, yn, azo, ve, vn, vz]. Only ve, vn are used
    to reconstruct heading; the azo value itself lives in column 2.
    """
    return _build_green_patch_loop(data_slip_model, data_insar, "AZO", nu, backend)
