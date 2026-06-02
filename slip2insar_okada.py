r"""前向 InSAR LOS 预测 — 与 ``slip2insar_okada.m`` 一致 ( Okada, cm 单位与 Green 构阵一致)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from calc_green import _xy2xy
from calc_okada import calc_okada


def _insar_patches_contrib(
    k0: int,
    k1: int,
    xinsar: np.ndarray,
    yinsar: np.ndarray,
    ve: np.ndarray,
    vn: np.ndarray,
    vz: np.ndarray,
    slip_model_in: np.ndarray,
    nu: float,
    backend: str,
) -> np.ndarray:
    d2r = np.pi / 180.0
    HF = 1.0
    ng = int(xinsar.size)
    z_acc = np.zeros(ng, dtype=np.float64)
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
        ue1, un1, uz1 = calc_okada(HF, u1, xpt, ypt, nu, delta, d, L, W, 1, strike_k, tp, backend=backend)
        ue2, un2, uz2 = calc_okada(HF, u2, xpt, ypt, nu, delta, d, L, W, 2, strike_k, tp, backend=backend)
        ue = np.asarray(ue1, dtype=np.float64) + np.asarray(ue2, dtype=np.float64)
        un = np.asarray(un1, dtype=np.float64) + np.asarray(un2, dtype=np.float64)
        uz = np.asarray(uz1, dtype=np.float64) + np.asarray(uz2, dtype=np.float64)
        z_acc += ue * ve + un * vn + uz * vz
    return z_acc


def slip2insar_okada(
    xin: np.ndarray,
    yin: np.ndarray,
    zin: np.ndarray,
    look_e: np.ndarray,
    look_n: np.ndarray,
    look_z: np.ndarray,
    slip_model_in: np.ndarray,
    *,
    nu: float = 0.25,
    backend: str = "auto",
    n_patch_workers: Optional[int] = None,
) -> np.ndarray:
    """在观测格点上计算合成 LOS: ``u·n = ue*ve+un*vn+uz*vz`` (``zin`` 里 NaN 处输出 NaN).

    ``n_patch_workers`` 为 1: 与原先串行一致. 大于 1 时按断层块区间并行累加 (依赖 Numba ``calc_okada`` 释放 GIL). \
    显式传入时优先. 为 ``None`` 时读环境 ``RESAMP_PATCH_WORKERS``: \
    未设或 ``auto`` = 多线程(``min(8, CPU)``); ``1``/``0``/``off``/``none`` = 串行; 正整数=线程数.
    """
    xp = slip_model_in[:, 3]
    npt = int(xp.size)
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

    good = ~np.isnan(zin)
    if not np.any(good):
        return np.full_like(zin, np.nan, dtype=np.float64)

    xinsar = xin[good].ravel()
    yinsar = yin[good].ravel()
    ve = look_e[good].ravel()
    vn = look_n[good].ravel()
    vz = look_z[good].ravel()
    ng = int(xinsar.size)

    if n_patch_workers <= 1:
        z_good = _insar_patches_contrib(0, npt, xinsar, yinsar, ve, vn, vz, slip_model_in, float(nu), backend)
    else:
        n_workers = int(min(n_patch_workers, npt))
        idx_parts = np.array_split(np.arange(npt, dtype=int), n_workers)
        ranges: list[tuple[int, int]] = []
        for c in idx_parts:
            if c.size == 0:
                continue
            ranges.append((int(c[0]), int(c[-1]) + 1))
        with ThreadPoolExecutor(max_workers=len(ranges)) as ex:
            futs = [
                ex.submit(
                    _insar_patches_contrib,
                    lo, hi, xinsar, yinsar, ve, vn, vz, slip_model_in, float(nu), backend
                )
                for lo, hi in ranges
            ]
        z_good = futs[0].result().copy()
        for fut in futs[1:]:
            z_good += fut.result()

    zout = np.full(xin.shape, np.nan, dtype=np.float64)
    zout[good] = z_good
    return zout
