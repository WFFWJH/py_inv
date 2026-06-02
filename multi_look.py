"""多视平均 — 与 ``multi_look.m`` (Kang Wang) 行为一致."""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from numba import njit

    _NUMBA = True
except Exception:
    _NUMBA = False
    njit = None  # type: ignore[misc, assignment]


if _NUMBA:
    @njit(cache=True)
    def _multi_look_njit(
        x_in: np.ndarray,
        y_in: np.ndarray,
        data_in: np.ndarray,
        nx: int,
        ny: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ny_in, nx_in = int(data_in.shape[0]), int(data_in.shape[1])
        nxe = int(np.ceil(nx_in / nx))
        nye = int(np.ceil(ny_in / ny))
        x_out = np.empty(nxe, dtype=np.float64)
        y_out = np.empty(nye, dtype=np.float64)
        data_out = np.empty((nye, nxe), dtype=np.float64)
        for jy in range(nye):
            j1 = jy * ny
            j2 = j1 + ny + 1
            if j2 > ny_in:
                j2 = ny_in
            s = 0.0
            for jjj in range(j1, j2):
                s += y_in[jjj]
            y_out[jy] = s / (j2 - j1) if j2 > j1 else 0.0
        for i in range(nxe):
            i1 = i * nx
            i2 = i1 + nx + 1
            if i2 > nx_in:
                i2 = nx_in
            s = 0.0
            for ii in range(i1, i2):
                s += x_in[ii]
            x_out[i] = s / (i2 - i1) if i2 > i1 else 0.0
        for jy in range(nye):
            j1 = jy * ny
            j2 = j1 + ny + 1
            if j2 > ny_in:
                j2 = ny_in
            for i in range(nxe):
                i1 = i * nx
                i2 = i1 + nx + 1
                if i2 > nx_in:
                    i2 = nx_in
                s = 0.0
                cnt = 0
                for jjj in range(j1, j2):
                    for ii in range(i1, i2):
                        v = data_in[jjj, ii]
                        if v == v:
                            s += v
                            cnt += 1
                if cnt > 0:
                    data_out[jy, i] = s / cnt
                else:
                    data_out[jy, i] = np.nan
        return x_out, y_out, data_out


def multi_look(
    x_in: np.ndarray,
    y_in: np.ndarray,
    data_in: np.ndarray,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿快、慢方向以 ``nx``×``ny`` 为步长对 ``data_in`` 分块, 与 ``multi_look.m`` 一致.

    MATLAB: ``indx2 = indx1 + nx`` 且下标**闭区间**, 每块(未截断时)取 **nx+1** 个样点(见原 .m);
    0-based 下对应 ``i2 = min(i1 + nx + 1, N)``.

    返回 ``x_out``/``y_out`` 为各块 1D 坐标的平均, ``data_out`` 形状 ``(Ny_out, Nx_out)``.
    """
    x_in = np.asarray(x_in, dtype=np.float64).ravel()
    y_in = np.asarray(y_in, dtype=np.float64).ravel()
    data_in = np.ascontiguousarray(data_in, dtype=np.float64)
    if data_in.ndim != 2:
        raise ValueError("data_in 须为二维 (Ny, Nx)")

    ny_in, nx_in = data_in.shape
    if _NUMBA:
        return _multi_look_njit(x_in, y_in, data_in, int(nx), int(ny))
    nxe = int(np.ceil(nx_in / nx))
    nye = int(np.ceil(ny_in / ny))
    x_out = np.zeros(nxe, dtype=np.float64)
    y_out = np.zeros(nye, dtype=np.float64)
    data_out = np.full((nye, nxe), np.nan, dtype=np.float64)

    for j in range(nye):
        j1 = j * ny
        j2 = min(j1 + ny + 1, ny_in)
        y_out[j] = float(np.mean(y_in[j1:j2]))
    for i in range(nxe):
        i1 = i * nx
        i2 = min(i1 + nx + 1, nx_in)
        x_out[i] = float(np.mean(x_in[i1:i2]))
        for j in range(nye):
            j1 = j * ny
            j2 = min(j1 + ny + 1, ny_in)
            data_tmp = data_in[j1:j2, i1:i2]
            good = data_tmp[~np.isnan(data_tmp)]
            if good.size:
                data_out[j, i] = float(np.mean(good))
    return x_out, y_out, data_out
