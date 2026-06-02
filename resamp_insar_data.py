r"""按模型重采样 InSAR — 与 ``resamp_insar_data.m`` (Wang & Fialko, GRL 2015 思路) 一致.

在各道的 ``los_ll_low`` / 视线 / ``dem`` 上算合成 LOS/AZO, 再对**合成场**做与 ``make_insar_data`` 相同的 quadtree 下采样,
写出 ``los_samp{iter_step}.mat``.
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
    fault_file: Optional[str] = None,
    dec: int = 1,
    output_path: Optional[str] = None,
    verbose: bool = True,
    quad_verbose: Optional[bool] = None,
    patch_workers: Optional[int] = None,
) -> None:
    """Wang & Fialko: 用 ``slip_model_in`` 前向算模型场, 再 quad 重采样; 与 .m 同名/参数序一致(关键字 ref_lon, fault, dec).

    ``fault_file`` 仅与 MATLAB 接口兼容(原 .m 中注释未绘图), 此处可不传.

    各道须已有 ``*\_low.grd`` (与 ``make_insar_data`` 第 1 次下采样后相同命名).

    ``output_path`` 若给定(且 ``len(track)==1``), 将 ``.mat`` 写到此路径, 不覆盖道目录下原文件(便于与 MATLAB 参考对比).

    ``verbose`` 为真时按阶段打印(类似 ``resamp_insar_data.m`` 中 ``disp``). \
    ``quad_verbose`` 控制 ``make_insar_downsample`` 内迭代输出; 未给定时由环境变量 ``RESAMP_QUAD_VERBOSE=1`` 开启.

    ``patch_workers`` 传给 ``slip2insar_okada`` / ``slip2azo_okada`` 的 ``n_patch_workers``(按断层块并行). \
    未给定时: 环境 **未设或** ``auto`` → 多线程 ``min(8, CPU 核数)``; ``1``/``0``/``off`` → 串行; 正整数 → 指定线程数.
    """
    xo, yo = _ll2xy(lonc, latc, ref_lon)
    iint = int(iter_step)
    nlook = int(dec) if dec and int(dec) > 0 else 1
    if output_path is not None and len(track) != 1:
        raise ValueError("output_path 仅在与单道 track 同用时有效 (len(track)==1)")
    if quad_verbose is None:
        quad_verbose = os.environ.get("RESAMP_QUAD_VERBOSE", "").strip() in ("1", "true", "True", "yes", "Y")

    def _p(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    def _resolved_slip2_workers() -> int:
        if patch_workers is not None:
            return max(1, int(patch_workers))
        s = (os.environ.get("RESAMP_PATCH_WORKERS", "") or "").strip().lower()
        if s in ("0", "1", "none", "off", "false", "no"):
            return 1
        if s in ("", "auto") or (not s):
            return max(1, min(8, (os.cpu_count() or 4)))
        try:
            return max(1, int(s))
        except ValueError:
            return max(1, min(8, (os.cpu_count() or 4)))

    _rpw = _resolved_slip2_workers()
    _p("  slip2 断层块线程数 = %d  (全串行请设 环境 RESAMP_PATCH_WORKERS=1 或 patch_workers=1)" % _rpw)

    n_tr = len(track)
    for k, this_track in enumerate(track):
        this_track = os.path.normpath(str(this_track))
        dt0 = data_types[k]
        data_type = dt0.decode().lower() if isinstance(dt0, (bytes, bytearray)) else str(dt0).lower()
        # 与 .m: disp(['working on ', this_track, 'type: ', data_type]);
        _p("working on %s  type: %s" % (this_track, data_type))
        this_npt = int(npt[k])

        t0 = time.perf_counter()
        _p("  [%d/%d] read grd: dem, los_ll, look_e/n/u" % (k + 1, n_tr))
        demin = read_grd(os.path.join(this_track, "dem_low.grd"))[2]
        x1, y1, losin = read_grd(os.path.join(this_track, "los_clean_detrend.grd"))
        ze = read_grd(os.path.join(this_track, "look_e.grd"))[2]
        zn = read_grd(os.path.join(this_track, "look_n.grd"))[2]
        zu = read_grd(os.path.join(this_track, "look_u.grd"))[2]
        _p("  grd 读入, los shape = %s  (%.2fs)" % (repr(losin.shape), time.perf_counter() - t0))

        if nlook > 1:
            t1 = time.perf_counter()
            _p("  multi_look  dec=%d" % nlook)
            _, _, deml = multi_look(x1, y1, demin, nlook, nlook)
            _, _, losl = multi_look(x1, y1, losin, nlook, nlook)
            _, _, zel = multi_look(x1, y1, ze, nlook, nlook)
            _, _, znl = multi_look(x1, y1, zn, nlook, nlook)
            lon1, lat1, zul = multi_look(x1, y1, zu, nlook, nlook)
            _p("  multi_look 后 losl shape = %s  (%.2fs)" % (repr(losl.shape), time.perf_counter() - t1))
        else:
            lon1, lat1 = x1, y1
            deml, losl, zel, znl, zul = demin, losin, ze, zn, zu
            _p("  未 multi_look (dec=1), losl shape = %s" % (repr(losl.shape),))

        xm1, ym1 = np.meshgrid(lon1, lat1, indexing="xy")
        xutm, yutm = _ll2xy(xm1.ravel(), ym1.ravel(), ref_lon)
        xin = (np.asarray(xutm) - xo).reshape(xm1.shape)
        yin = (np.asarray(yutm) - yo).reshape(ym1.shape)

        if data_type == "insar" or data_type == "rng":
            _p("  " + data_type)  # .m: disp('insar') / disp('rng')
            t2 = time.perf_counter()
            _p(
                "  slip2insar_okada: 网格 %d 点, %d 块, 可能较久..."
                % (xin.size, int(slip_model_in.shape[0]))
            )
            los_model = slip2insar_okada(
                xin, yin, losl, zel, znl, zul, slip_model_in,
                n_patch_workers=patch_workers,
            )
            _p("  slip2insar_okada 完成 (%.1fs)" % (time.perf_counter() - t2,))
        elif data_type == "azo":
            _p("  azi")  # .m: disp('azi');
            t2 = time.perf_counter()
            _p(
                "  slip2azo_okada: 网格 %d 点, %d 块, 可能较久..."
                % (xin.size, int(slip_model_in.shape[0]))
            )
            los_model = slip2azo_okada(
                xin, yin, losl, zel, znl, slip_model_in,
                n_patch_workers=patch_workers,
            )
            _p("  slip2azo_okada 完成 (%.1fs)" % (time.perf_counter() - t2,))
        else:
            raise ValueError("data_type 须为 insar, rng 或 azo, 得到 %r" % (data_type,))

        los_model_grd = os.path.join(this_track, "los_model.grd")
        write_grd(lon1, lat1, los_model, los_model_grd)
        _p("  wrote %s" % los_model_grd)

        t3 = time.perf_counter()
        _p("  make_insar_downsample(quad) npt=%d Nmin=%d Nmax=%d ..." % (this_npt, int(nmin[k]), int(nmax[k])))
        lon_model, lat_model, _, _, rms_out, xx1, xx2, yy1, yy2 = make_insar_downsample(
            lon1, lat1, los_model, this_npt, int(nmin[k]), int(nmax[k]),
            method="mean", verbose=bool(quad_verbose),
        )
        _p("  make_insar_downsample 块数 = %d (%.2fs)" % (len(lon_model), time.perf_counter() - t3))
        t4 = time.perf_counter()
        _p("  make_look_downsample (los, dem, e,n,u) ...")
        _, _, zout = make_look_downsample(lon1, lat1, losl, lon_model, lat_model, xx1, xx2, yy1, yy2)
        _, _, dem_out = make_look_downsample(lon1, lat1, deml, lon_model, lat_model, xx1, xx2, yy1, yy2)
        _, _, ve = make_look_downsample(lon1, lat1, zel, lon_model, lat_model, xx1, xx2, yy1, yy2)
        _, _, vn = make_look_downsample(lon1, lat1, znl, lon_model, lat_model, xx1, xx2, yy1, yy2)
        _, _, vz = make_look_downsample(lon1, lat1, zul, lon_model, lat_model, xx1, xx2, yy1, yy2)
        _p("  make_look_downsample 完成 (%.2fs)" % (time.perf_counter() - t4,))

        xutm, yutm = _ll2xy(lon_model, lat_model, ref_lon)
        xpt = np.asarray(xutm, dtype=np.float64) - xo
        ypt = np.asarray(yutm, dtype=np.float64) - yo

        indx_good = ~np.isnan(zout)
        xpt = xpt[indx_good]
        ypt = ypt[indx_good]
        zout = zout[indx_good]
        dem_out = dem_out[indx_good]
        rms_out = rms_out[indx_good]
        ve = ve[indx_good]
        vn = vn[indx_good]
        vz = vz[indx_good]
        xx1 = np.asarray(xx1, dtype=np.float64)[indx_good]
        yy1 = np.asarray(yy1, dtype=np.float64)[indx_good]
        xx2 = np.asarray(xx2, dtype=np.float64)[indx_good]
        yy2 = np.asarray(yy2, dtype=np.float64)[indx_good]

        sampled_insar_data = np.ascontiguousarray(
            np.column_stack([xpt, ypt, zout, ve, vn, vz]), dtype=np.float64
        )
        if output_path is not None:
            out_mat = str(output_path)
        else:
            out_mat = os.path.join(this_track, "los_samp%d.mat" % iint)
        savemat(
            out_mat,
            {
                "sampled_insar_data": sampled_insar_data,
                "rms_out": rms_out,
                "dem_out": dem_out,
            },
            do_compression=True,
        )
        print("  saved %s" % out_mat, flush=True)
