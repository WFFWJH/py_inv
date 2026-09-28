r"""按模型重采样 InSAR — 与 ``resamp_insar_data.m`` (Wang & Fialko, GRL 2015) 一致.

对合成 LOS/AZO 做与 ``make_insar_data`` 相同的 quadtree 下采样, 写出 ``los_samp{iter_step}.mat``.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Union

import numpy as np
from scipy.io import savemat

from load_fault_one_plane import _ll2xy
from make_insar_data import make_insar_downsample, make_look_downsample, read_grd, write_grd
from multi_look import multi_look
from slip2azo_okada import slip2azo_okada
from slip2insar_okada import slip2insar_okada


def resamp_insar_data(
    slip_model_in: np.ndarray,
    track: Sequence[str],
    npt: Sequence[int],
    nmin: Sequence[int],
    nmax: Sequence[int],
    data_types: Sequence[Union[str, bytes]],
    iter_step: int,
    *,
    lonc: float,
    latc: float,
    ref_lon: float,
    fault_file: Optional[str] = None,  # MATLAB 接口兼容, 未使用
    dec: int = 1,
    output_path: Optional[str] = None,
    verbose: bool = True,
    quad_verbose: Optional[bool] = None,
    patch_workers: Optional[int] = None,
) -> None:
    """用 ``slip_model_in`` 前向算模型场再 quad 重采样.

    各道须已有 ``*_low.grd``. ``output_path`` 仅在 ``len(track)==1`` 时可用.
    ``patch_workers`` / 环境 ``RESAMP_PATCH_WORKERS`` 由 slip2* 内部解析.
    """
    _ = fault_file
    xo, yo = _ll2xy(lonc, latc, ref_lon)
    iint = int(iter_step)
    nlook = max(1, int(dec or 1))
    if output_path is not None and len(track) != 1:
        raise ValueError("output_path 仅在与单道 track 同用时有效 (len(track)==1)")
    if quad_verbose is None:
        quad_verbose = os.environ.get("RESAMP_QUAD_VERBOSE", "").strip().lower() in (
            "1", "true", "yes", "y",
        )

    def _p(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    n_tr = len(track)
    for k, this_track in enumerate(track):
        this_track = os.path.normpath(str(this_track))
        dt0 = data_types[k]
        data_type = (dt0.decode() if isinstance(dt0, (bytes, bytearray)) else str(dt0)).lower()
        this_npt = int(npt[k])
        _p(f"working on {this_track}  type: {data_type}")

        t0 = time.perf_counter()
        demin = read_grd(os.path.join(this_track, "dem_low.grd"))[2]
        x1, y1, losin = read_grd(os.path.join(this_track, "los_clean_detrend.grd"))
        ze = read_grd(os.path.join(this_track, "look_e.grd"))[2]
        zn = read_grd(os.path.join(this_track, "look_n.grd"))[2]
        zu = read_grd(os.path.join(this_track, "look_u.grd"))[2]
        _p(f"  [{k + 1}/{n_tr}] grd 读入 los={losin.shape} ({time.perf_counter() - t0:.2f}s)")

        grids = [demin, losin, ze, zn, zu]
        if nlook > 1:
            t1 = time.perf_counter()
            looked = [multi_look(x1, y1, g, nlook, nlook) for g in grids]
            lon1, lat1 = looked[-1][0], looked[-1][1]
            deml, losl, zel, znl, zul = (r[2] for r in looked)
            _p(f"  multi_look dec={nlook} losl={losl.shape} ({time.perf_counter() - t1:.2f}s)")
        else:
            lon1, lat1 = x1, y1
            deml, losl, zel, znl, zul = grids
            _p(f"  未 multi_look, losl={losl.shape}")

        xm1, ym1 = np.meshgrid(lon1, lat1, indexing="xy")
        xutm, yutm = _ll2xy(xm1.ravel(), ym1.ravel(), ref_lon)
        xin = (np.asarray(xutm) - xo).reshape(xm1.shape)
        yin = (np.asarray(yutm) - yo).reshape(ym1.shape)

        t2 = time.perf_counter()
        _p(f"  {data_type}: slip2 网格 {xin.size} 点, {slip_model_in.shape[0]} 块 ...")
        if data_type in ("insar", "rng"):
            los_model = slip2insar_okada(
                xin, yin, losl, zel, znl, zul, slip_model_in,
                n_patch_workers=patch_workers,
            )
        elif data_type == "azo":
            los_model = slip2azo_okada(
                xin, yin, losl, zel, znl, slip_model_in,
                n_patch_workers=patch_workers,
            )
        else:
            raise ValueError(f"data_type 须为 insar, rng 或 azo, 得到 {data_type!r}")
        _p(f"  slip2 完成 ({time.perf_counter() - t2:.1f}s)")

        los_model_grd = os.path.join(this_track, "los_model.grd")
        write_grd(lon1, lat1, los_model, los_model_grd)

        t3 = time.perf_counter()
        lon_model, lat_model, _, _, rms_out, xx1, xx2, yy1, yy2 = make_insar_downsample(
            lon1, lat1, los_model, this_npt, int(nmin[k]), int(nmax[k]),
            method="mean", verbose=bool(quad_verbose),
        )
        _p(f"  quad npt={this_npt} 块数={len(lon_model)} ({time.perf_counter() - t3:.2f}s)")

        t4 = time.perf_counter()
        bbox = (lon_model, lat_model, xx1, xx2, yy1, yy2)
        zout, dem_out, ve, vn, vz = (
            make_look_downsample(lon1, lat1, g, *bbox)[2]
            for g in (losl, deml, zel, znl, zul)
        )
        _p(f"  look 下采样完成 ({time.perf_counter() - t4:.2f}s)")

        xutm, yutm = _ll2xy(lon_model, lat_model, ref_lon)
        good = ~np.isnan(zout)
        sampled_insar_data = np.ascontiguousarray(
            np.column_stack([
                np.asarray(xutm, dtype=np.float64)[good] - xo,
                np.asarray(yutm, dtype=np.float64)[good] - yo,
                zout[good], ve[good], vn[good], vz[good],
            ]),
            dtype=np.float64,
        )
        out_mat = (
            str(output_path)
            if output_path is not None
            else os.path.join(this_track, f"los_samp{iint}.mat")
        )
        savemat(
            out_mat,
            {
                "sampled_insar_data": sampled_insar_data,
                "rms_out": rms_out[good],
                "dem_out": dem_out[good],
            },
            do_compression=True,
        )
        print(f"  saved {out_mat}", flush=True)
