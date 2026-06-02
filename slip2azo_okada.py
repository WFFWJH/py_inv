r"""前向方位向偏移 (AZO) 预测 — 与 ``slip2AZO_okada.m`` 一致 (仅用水平分量与走航角)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

from calc_green import _xy2xy
from calc_okada import calc_okada


def _azo_patches_contrib(
    k0: int,
    k1: int,
    xinsar: np.ndarray,
    yinsar: np.ndarray,
    slip_model_in: np.ndarray,
    nu: float,
    backend: str,
) -> Tuple[np.ndarray, np.ndarray]:
    d2r = np.pi / 180.0
    HF = 1.0
    ng = int(xinsar.size)
    uE = np.zeros(ng, dtype=np.float64)
    uN = np.zeros(ng, dtype=np.float64)
    tp = np.zeros(ng, dtype=np.float64)
    xp = slip_model_in[:, 3]
    yp = slip_model_in[:, 4]
    zp = slip_model_in[:, 5]
    lp = slip_model_in[:, 6]
    wp = slip_model_in[:, 7]
    strkp = slip_model_in[:, 8]
    dip0 = slip_model_in[:, 9]
    s1 = slip_model_in[:, 11]
    s2 = slip_model_in[:, 12]
    for k in range(int(k0), int(k1)):
        theta = (90.0 - strkp[k]) * d2r
        dxf = lp[k] * 0.5
        dx, dy = _xy2xy(np.array(dxf), np.array(0.0), -theta)
        xxo = float(xp[k] + dx)
        yyo = float(yp[k] + dy)
        zzo = float(zp[k])
        u1, u2 = float(s1[k]), float(s2[k])
        xpt = xinsar - xxo
        ypt = yinsar - yyo
        delta = float(dip0[k] * d2r)
        d = float(-zzo)
        L, W = float(lp[k]), float(wp[k])
        strike_k = float(strkp[k] * d2r)
        ue1, un1, _ = calc_okada(HF, u1, xpt, ypt, nu, delta, d, L, W, 1, strike_k, tp, backend=backend)
        ue2, un2, _ = calc_okada(HF, u2, xpt, ypt, nu, delta, d, L, W, 2, strike_k, tp, backend=backend)
        uE += np.asarray(ue1, dtype=np.float64) + np.asarray(ue2, dtype=np.float64)
        uN += np.asarray(un1, dtype=np.float64) + np.asarray(un2, dtype=np.float64)
    return uE, uN


def slip2azo_okada(
    xin: np.ndarray,
    yin: np.ndarray,
    zin: np.ndarray,
    look_e: np.ndarray,
    look_n: np.ndarray,
    slip_model_in: np.ndarray,
    *,
    nu: float = 0.25,
    backend: str = "auto",
    n_patch_workers: Optional[int] = None,
) -> np.ndarray:
    r"""与 .m: ``\theta_{az} = -atan2(vn,ve) - 180°``, ``u_e``, ``u_n`` 叠加后按方位投影.

    ``n_patch_workers`` / 环境 ``RESAMP_PATCH_WORKERS`` 同 ``slip2insar_okada``.
    """
    npt = int(slip_model_in.shape[0])
    if n_patch_workers is None:
        s = (os.environ.get("RESAMP_PATCH_WORKERS", "") or "").strip().lower()
        if s in ("0", "1", "none", "off", "false", "no"):
            n_patch_workers = 1
        elif s in ("", "auto"):
            n_patch_workers = max(1, min(8, (os.cpu_count() or 4)))
        else:
            try:
                n_patch_workers = max(1, int(s))
            except ValueError:
                n_patch_workers = max(1, min(8, (os.cpu_count() or 4)))
    n_patch_workers = max(1, int(n_patch_workers))
    if n_patch_workers > 1 and npt < 2:
        n_patch_workers = 1

    zout = np.full(xin.shape, np.nan, dtype=np.float64)
    good = ~np.isnan(zin)
    if not np.any(good):
        return zout

    xinsar = xin[good].ravel()
    yinsar = yin[good].ravel()
    ve = look_e[good].ravel()
    vn = look_n[good].ravel()
    theta_az = -np.degrees(np.arctan2(vn, ve)) - 180.0
    tr = np.deg2rad(theta_az)
    sin_az = np.sin(tr)
    cos_az = np.cos(tr)

    if n_patch_workers <= 1:
        uE, uN = _azo_patches_contrib(0, npt, xinsar, yinsar, slip_model_in, float(nu), backend)
    else:
        n_w = int(min(n_patch_workers, npt))
        idx_parts = np.array_split(np.arange(npt, dtype=int), n_w)
        ranges: list[Tuple[int, int]] = []
        for c in idx_parts:
            if c.size == 0:
                continue
            ranges.append((int(c[0]), int(c[-1]) + 1))
        with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
            futs = [
                ex.submit(_azo_patches_contrib, lo, hi, xinsar, yinsar, slip_model_in, float(nu), backend)
                for lo, hi in ranges
            ]
        uE, uN = futs[0].result()
        uE = uE.copy()
        uN = uN.copy()
        for fut in futs[1:]:
            a, b = fut.result()
            uE += a
            uN += b

    z_good = uE * sin_az + uN * cos_az
    zout[good] = z_good
    return zout
