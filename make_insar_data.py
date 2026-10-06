"""
InSAR LOS quadtree / uniform downsampling.

Python port of MATLAB files:
    - make_insar_data.m
    - make_insar_downsample.m        (quadtree with 4 inner functions)
    - make_look_downsample.m         (downsample look vectors by boxes)
    - plot_insar_sample_new.m        (plotting)

Key simplifications vs. MATLAB:
    * quad_decomp_mean / quad_decomp_trend merged into one recursive routine
      parametrised by `method`.
    * rms_block_demean / rms_block_detrend merged the same way.
    * the four nearly-identical "quadrant" blocks become a `for` loop.
    * GMT netCDF `.grd` I/O via `xarray` (replaces grdread2 / grdwrite2).
    * coordinate transform re-uses the standard Transverse-Mercator `_ll2xy`
      from `load_fault_one_plane`.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Re-use the already-tested Transverse-Mercator projection.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from load_fault_one_plane import _ll2xy  # noqa: E402


# =====================================================================
# GMT .grd  I/O   (xarray-based replacement for grdread2 / grdwrite2)
# =====================================================================
_TEXT_GRID_SUFFIXES = frozenset({".txt", ".xyz", ".dat", ".asc", ".xy"})


def _read_grd_xyz(filename):
    """Read whitespace-separated x y z text into a regular grid."""
    data = np.loadtxt(filename, comments=("#", "%", "@"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(
            "%r: expected at least 3 columns (x y z), got %d"
            % (filename, data.shape[1])
        )

    xc = np.asarray(data[:, 0], dtype=np.float64)
    yc = np.asarray(data[:, 1], dtype=np.float64)
    zc = np.asarray(data[:, 2], dtype=np.float64)
    x = np.unique(xc)
    y = np.unique(yc)
    nx, ny = int(x.size), int(y.size)
    ix = np.searchsorted(x, xc)
    iy = np.searchsorted(y, yc)
    if not (np.all(x[ix] == xc) and np.all(y[iy] == yc)):
        raise ValueError(
            "%r: x/y coordinates are not on a regular grid" % (filename,)
        )

    z = np.full((ny, nx), np.nan, dtype=np.float64)
    z[iy, ix] = zc
    return x, y, z


def read_grd(filename, engine: Optional[str] = None):
    """Read a GMT netCDF grid or xyz text file. Returns (x, y, z) with z.shape == (ny, nx).

    NetCDF: accepts either ``x/y`` (GMT4) or ``lon/lat`` (CF-style) coordinate names.
    Text: whitespace-separated three columns ``x y z`` on a regular grid
    (extensions ``.txt``, ``.xyz``, ``.dat``, ``.asc``, ``.xy``).

    依次尝试 ``h5netcdf``、默认引擎、``netcdf4``, 避免部分环境下 ``netCDF4`` DLL 失败.

    Parameters
    ----------
    engine
        若给定 (如 ``"h5netcdf"`` / ``"netcdf4"`` / ``None`` 为 scipy 类默认),
        仅使用该引擎; 未安装则抛错. 供测试或强制复现; 一般调用勿传.
    """
    if Path(filename).suffix.lower() in _TEXT_GRID_SUFFIXES:
        return _read_grd_xyz(filename)

    import xarray as xr  # lazy import: users may only need the math

    if engine is not None:
        engines = [engine]
    else:
        engines = []
        try:
            import h5netcdf  # noqa: F401

            engines.append("h5netcdf")
        except ImportError:
            pass
        engines.append(None)
        try:
            import netcdf4  # noqa: F401

            engines.append("netcdf4")
        except ImportError:
            pass

    last_err: Optional[BaseException] = None
    ds = None
    for eng in engines:
        try:
            ds = xr.open_dataset(filename, engine=eng)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if ds is None:
        raise last_err if last_err else OSError("cannot open %r" % (filename,))
    try:
        if "z" in ds.data_vars:
            zvar = ds["z"]
        else:
            zvar = next(iter(ds.data_vars.values()))

        coord_names = list(ds.coords)
        xname = next((c for c in ("x", "lon", "longitude") if c in coord_names), None)
        yname = next((c for c in ("y", "lat", "latitude") if c in coord_names), None)
        if xname is None or yname is None:
            # fall back: declared dim order (last = x, first = y)
            xname = coord_names[-1]
            yname = coord_names[0]
        x = np.asarray(ds[xname].values).ravel()
        y = np.asarray(ds[yname].values).ravel()
        ny, nx = int(y.size), int(x.size)
        if zvar.ndim == 2:
            s0, s1 = int(zvar.shape[0]), int(zvar.shape[1])
            if (s0, s1) == (ny, nx):
                z = np.asarray(zvar.values, dtype=np.float64)
            elif (s0, s1) == (nx, ny):
                # netCDF 中常为 (nx,ny) 与 MATLAB grdread2' 的维序, 行沿 y: (ny,nx)
                z = np.asarray(zvar.values, dtype=np.float64).T
            else:
                z = np.asarray(zvar.values, dtype=np.float64)
                if z.shape != (ny, nx) and z.T.shape == (ny, nx):
                    z = z.T
        else:
            z = np.asarray(zvar.values, dtype=np.float64)
            if z.ndim == 2 and z.shape == (nx, ny):
                z = z.T
            elif z.ndim == 2 and z.shape != (ny, nx) and z.T.shape == (ny, nx):
                z = z.T
    finally:
        ds.close()
    return x, y, z


def write_grd(x, y, z, filename):
    """Write (x, y, z) as a GMT-compatible netCDF grid."""
    import xarray as xr

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if z.shape != (y.size, x.size):
        z = z.T
    da = xr.DataArray(
        z, coords={"y": y, "x": x}, dims=("y", "x"), name="z",
        attrs={"long_name": "z"},
    )
    ds = da.to_dataset()
    ds["x"].attrs["long_name"] = "x"
    ds["x"].attrs["actual_range"] = [float(np.min(x)), float(np.max(x))]
    ds["y"].attrs["long_name"] = "y"
    ds["y"].attrs["actual_range"] = [float(np.min(y)), float(np.max(y))]
    ds.attrs["Conventions"] = "COARDS/CF-1.0"
    ds.attrs["title"] = os.path.basename(filename)
    ds.to_netcdf(filename)


# =====================================================================
# Quadtree downsampling
# =====================================================================
_R_GOOD_DEFAULT = 0.4


def _results_to_arrays(out):
    if not out:
        empty = np.array([], dtype=np.float64)
        return (empty,) * 9
    arr = np.asarray(out, dtype=np.float64)
    return (arr[:, 0], arr[:, 1], arr[:, 2],
            arr[:, 3].astype(np.int64), arr[:, 4],
            arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8])


def _block_at_min_size(lx, ly, nres_min):
    """Match sample.py's default 'or' stopping rule."""
    return lx <= nres_min or ly <= nres_min


def _quad_decomp_sample(x, y, z, threshold, nres_min, nres_max, method):
    """Stack-based, non-overlapping quadtree pass used by sample.py."""
    ny, nx = z.shape
    stack = [(np.arange(nx), np.arange(ny))]
    out = []

    while stack:
        x_idx, y_idx = stack.pop()
        zsub = z[np.ix_(y_idx, x_idx)]
        valid = ~np.isnan(zsub)
        ngood = int(valid.sum())
        if ngood == 0:
            continue

        lx, ly = len(x_idx), len(y_idx)
        r_good = ngood / (lx * ly)
        zdata = zsub[valid]
        rows, cols = np.nonzero(valid)
        xdata = x[x_idx[cols]]
        ydata = y[y_idx[rows]]
        xmean = float(xdata.mean())
        ymean = float(ydata.mean())
        zmean = float(zdata.mean())

        if ngood == 1:
            rms_true = 0.0
            rms_for_split = 1.0e30
        elif method == "trend":
            design = np.column_stack([xdata, ydata, np.ones(ngood)])
            coeffs, *_ = np.linalg.lstsq(design, zdata, rcond=None)
            residual = zdata - design @ coeffs
            rms_true = float(np.sqrt(np.mean((zdata - zmean) ** 2)))
            rms_for_split = float(np.sqrt(np.mean(residual ** 2)))
        else:
            rms_true = float(np.sqrt(np.mean((zdata - zmean) ** 2)))
            rms_for_split = rms_true

        can_split_x = lx >= 2
        can_split_y = ly >= 2
        need_split = False

        if _block_at_min_size(lx, ly, nres_min) or not (
            can_split_x or can_split_y
        ):
            if r_good <= _R_GOOD_DEFAULT:
                continue
        elif lx > 2 and lx < nres_max and ly > 2 and ly < nres_max:
            if rms_for_split > threshold:
                need_split = True
            elif r_good <= _R_GOOD_DEFAULT:
                continue
        else:
            need_split = True

        if need_split and (can_split_x or can_split_y):
            nx_mid, ny_mid = lx // 2, ly // 2
            if can_split_x and can_split_y:
                for ys_part in (y_idx[:ny_mid], y_idx[ny_mid:]):
                    for xs_part in (x_idx[:nx_mid], x_idx[nx_mid:]):
                        if xs_part.size and ys_part.size:
                            stack.append((xs_part, ys_part))
            elif can_split_x:
                stack.extend(
                    (xs_part, y_idx)
                    for xs_part in (x_idx[:nx_mid], x_idx[nx_mid:])
                    if xs_part.size
                )
            else:
                stack.extend(
                    (x_idx, ys_part)
                    for ys_part in (y_idx[:ny_mid], y_idx[ny_mid:])
                    if ys_part.size
                )
            continue

        out.append((
            xmean, ymean, zmean, ngood, rms_true,
            float(x[x_idx[0]]), float(x[x_idx[-1]]),
            float(y[y_idx[0]]), float(y[y_idx[-1]]),
        ))

    return out


def make_insar_downsample(xinsar, yinsar, zinsar, nmin, nres_min, nres_max,
                          method="mean", max_iter=100, verbose=True,
                          initial_threshold=None):
    """Quadtree downsample an InSAR LOS matrix.

    Parameters
    ----------
    xinsar, yinsar : 1D arrays  (x[i] varies along columns, y[j] along rows)
    zinsar         : 2D array of shape (ny, nx) with NaNs allowed
    nmin           : target number of output points
    nres_min, nres_max : min/max block edge length (in pixels) allowed
    method         : 'mean' or 'trend' (trend fit is used only for split decisions)

    Returns
    -------
    xout, yout, zout : 1D arrays  (center of each accepted block)
    npts             : per-block valid-pixel count
    rms_out          : per-block rms of zgood vs block mean
    xx1, xx2, yy1, yy2 : per-block bounding box
    """
    x = np.asarray(xinsar, dtype=np.float64)
    y = np.asarray(yinsar, dtype=np.float64)
    z = np.asarray(zinsar, dtype=np.float64)
    if z.shape != (y.size, x.size):
        raise ValueError(f"zinsar shape {z.shape} does not match (ny={y.size}, nx={x.size})")
    if method not in ("mean", "trend"):
        raise ValueError(f"method must be 'mean' or 'trend', got {method!r}")
    valid = z[~np.isnan(z)]
    if valid.size == 0:
        raise ValueError("quad 下采样未产生任何有效块 (全为 NaN)")
    if nres_min < 1 or nres_max < 1:
        raise ValueError("nres_min 和 nres_max 必须为正整数")
    points_num = max(1, int(nmin))

    if initial_threshold is None:
        mean = float(valid.mean())
        threshold = float(np.sqrt(np.mean((valid - mean) ** 2)))
        if not np.isfinite(threshold) or threshold <= 0.0:
            threshold = float(np.max(valid) - np.min(valid))
        if not np.isfinite(threshold) or threshold <= 0.0:
            threshold = 1.0
    else:
        threshold = float(initial_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("initial_threshold 必须为有限正数")

    n_lo = 0.99 * points_num
    n_hi = 1.1 * points_num

    def run(current_threshold):
        return _quad_decomp_sample(
            x, y, z, current_threshold, int(nres_min), int(nres_max), method
        )

    out = run(threshold)
    if not out:
        raise ValueError("quad 下采样未产生任何有效块 (检查 NaN 掩膜或 Nres 参数)")
    best_out = out
    best_n = len(out)
    ndata = best_n
    best_diff = abs(best_n - points_num)
    best_threshold = threshold
    if verbose:
        rms_values = np.asarray(out, dtype=np.float64)[:, 4]
        print("max_rms_out:", float(rms_values.max()), "min_rms_out:", float(rms_values.min()))
        print("threshold:", threshold, "NUM:", best_n)
    if n_lo <= best_n <= n_hi:
        return _results_to_arrays(out)

    thr_lo = threshold if best_n > points_num else None
    thr_hi = threshold if best_n <= points_num else None
    n_prev = best_n
    stagnant = 0

    for it in range(1, max_iter + 1):
        if thr_lo is not None and thr_hi is not None and thr_hi > thr_lo:
            threshold = 0.5 * (thr_lo + thr_hi)
        elif ndata > points_num:
            threshold *= 1.1
        else:
            threshold *= 0.9

        if not np.isfinite(threshold) or threshold <= 0.0:
            if verbose:
                print("threshold 无效, 停止迭代")
            break

        out = run(threshold)
        ndata = len(out)
        if ndata == 0:
            raise ValueError("quad 下采样迭代后块数为 0 (threshold=%.4g)" % threshold)
        diff = abs(ndata - points_num)
        if diff < best_diff or (diff == best_diff and ndata >= best_n):
            best_out = out
            best_n = ndata
            best_diff = diff
            best_threshold = threshold

        if verbose:
            rms_values = np.asarray(out, dtype=np.float64)[:, 4]
            print(f"strain: {threshold:.4f} NUM: {ndata} (iter {it}, was {n_prev})")
            print("max_rms_out:", float(rms_values.max()), "min_rms_out:", float(rms_values.min()))
        if n_lo <= ndata <= n_hi:
            break

        if ndata == n_prev:
            stagnant += 1
            if stagnant >= 2 and (threshold > rms_values.max() or threshold < rms_values.min()):
                if verbose:
                    print("块数连续不变, 提前停止")
                break
        else:
            stagnant = 0

        if ndata > points_num:
            thr_lo = threshold if thr_lo is None else max(thr_lo, threshold)
        else:
            thr_hi = threshold if thr_hi is None else min(thr_hi, threshold)
        if thr_lo is not None and thr_hi is not None and thr_hi <= thr_lo * 1.0001:
            if verbose:
                print("阈值搜索区间已收窄完毕")
            break
        n_prev = ndata
    else:
        if verbose:
            print("Reached max iteration, 点数可能仍未达到目标")

    if verbose:
        print(
            f"Nint done, blocks = {best_n} (target {points_num}, "
            f"thr={best_threshold:.4g}, in_window={n_lo <= best_n <= n_hi})"
        )
    return _results_to_arrays(best_out)


# =====================================================================
# Downsample look vectors / DEM at the same boxes
# =====================================================================
def make_look_downsample(xlook, ylook, zlook, xin, yin, xx1, xx2, yy1, yy2):
    """Mean-aggregate `zlook` inside each box (xx1..xx2, yy1..yy2)."""
    xlook = np.asarray(xlook)
    ylook = np.asarray(ylook)
    zlook = np.asarray(zlook)
    n = len(xx1)
    xout = np.asarray(xin, dtype=np.float64).copy()
    yout = np.asarray(yin, dtype=np.float64).copy()
    zout = np.full(n, np.nan, dtype=np.float64)
    for k in range(n):
        ix = np.where((xlook >= xx1[k]) & (xlook <= xx2[k]))[0]
        iy = np.where((ylook >= yy1[k]) & (ylook <= yy2[k]))[0]
        if ix.size == 0 or iy.size == 0:
            continue
        blk = zlook[np.ix_(iy, ix)]
        good = blk[~np.isnan(blk)]
        if good.size > 0:
            zout[k] = float(good.mean())
    return xout, yout, zout


# =====================================================================
# Plotting  (replaces plot_insar_sample_new.m)
# =====================================================================
def plot_insar_sample(xinsar, yinsar, zinsar, zout, xx1, xx2, yy1, yy2,
                     fault_file=None, title=None, show=False, savepath=None):
    """Side-by-side plot: original LOS + downsampled points, with optional fault trace."""
    import matplotlib.pyplot as plt

    zinsar = np.asarray(zinsar)
    cmean = float(np.nanmean(zinsar))
    cstd = float(np.nanstd(zinsar))
    cmin = cmean - 8 * cstd
    cmax = cmean + 8 * cstd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(zinsar, extent=[xinsar.min(), xinsar.max(),
                                     yinsar.min(), yinsar.max()],
                     origin="lower", cmap="jet", vmin=cmin, vmax=cmax,
                     aspect="equal", interpolation="nearest")
    fig.colorbar(im1, ax=ax1, orientation="horizontal", fraction=0.04, pad=0.08)
    ax1.set_title("Original LOS")

    pxc = (np.asarray(xx1) + np.asarray(xx2)) / 2.0
    pyc = (np.asarray(yy1) + np.asarray(yy2)) / 2.0
    sc = ax2.scatter(pxc, pyc, c=zout, s=30, cmap="jet", vmin=cmin, vmax=cmax)
    fig.colorbar(sc, ax=ax2, orientation="horizontal", fraction=0.04, pad=0.08)
    ax2.set_aspect("equal")
    ax2.set_xlim(xinsar.min(), xinsar.max())
    ax2.set_ylim(yinsar.min(), yinsar.max())
    ax2.set_title(f"#pt = {len(zout)}" if title is None else title)

    if fault_file is not None and Path(fault_file).is_file():
        ft = np.loadtxt(fault_file)
        if ft.ndim == 1:
            ft = ft.reshape(1, -1)
        for row in ft:
            lon1, lat1, lon2, lat2 = row[0], row[1], row[2], row[3]
            ax1.plot([lon1, lon2], [lat1, lat2], "k-", lw=1.5)
            ax2.plot([lon1, lon2], [lat1, lat2], "k-", lw=1.5)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =====================================================================
# Top-level driver  (replaces make_insar_data.m)
# =====================================================================
def make_insar_data(tracks: Sequence[str],
                    npt: Sequence[int],
                    region: np.ndarray,
                    nmin: Sequence[int],
                    nmax: Sequence[int],
                    method: str = "quadtree",
                    lonc: float = 0.0,
                    latc: float = 0.0,
                    ref_lon: Optional[float] = None,
                    fault_file: Optional[str] = None,
                    sample_area: Optional[Sequence[float]] = None,
                    save_mat: bool = True,
                    save_plot: bool = True):
    """End-to-end InSAR downsampling driver.

    Parameters mirror the MATLAB make_insar_data(track, npt, region, Nmin, Nmax, ...)
    signature. `method` is 'quadtree' or 'uniform'.
    """
    from scipy.io import savemat

    if ref_lon is None:
        ref_lon = lonc
    region = np.asarray(region, dtype=np.float64).reshape(-1, 4)
    nmin = np.asarray(nmin).ravel()
    nmax = np.asarray(nmax).ravel()

    if method == "quadtree":
        grd_file = "los_clean_detrend.grd"
        file_suffix = "low"
        iint = 0
        downsample_method = "mean"
    elif method == "uniform":
        grd_file = "unwrap_clean_sample.grd"
        nmax = nmin.copy()  # force uniform
        file_suffix = "uniform"
        iint = "_uniform"
        downsample_method = "mean"
    else:
        raise ValueError(f"method must be 'quadtree' or 'uniform', got {method!r}")

    xo, yo = _ll2xy(lonc, latc, ref_lon)

    for k, track in enumerate(tracks):
        track = str(track)
        x1, y1, z1 = read_grd(os.path.join(track, grd_file))
        xmin, xmax, ymin, ymax = region[k]
        ix = np.where((x1 >= xmin) & (x1 <= xmax))[0]
        iy = np.where((y1 >= ymin) & (y1 <= ymax))[0]
        xin = x1[ix]
        yin = y1[iy]
        losin = z1[np.ix_(iy, ix)]

        _, _, zdem = read_grd(os.path.join(track, "dem.grd"))
        demin = zdem[np.ix_(iy, ix)]

        _, _, ze = read_grd(os.path.join(track, "look_e.grd"))
        _, _, zn = read_grd(os.path.join(track, "look_n.grd"))
        _, _, zu = read_grd(os.path.join(track, "look_u.grd"))
        ein = ze[np.ix_(iy, ix)]
        nin = zn[np.ix_(iy, ix)]
        uin = zu[np.ix_(iy, ix)]

        # Write cropped grids back
        for arr, name in ((losin, "los_ll"), (ein, "look_e"), (nin, "look_n"),
                          (uin, "look_u"), (demin, "dem")):
            write_grd(xin, yin, arr, os.path.join(track, f"{name}_{file_suffix}.grd"))

        xout, yout, zout, npts, rms_out, xx1, xx2, yy1, yy2 = make_insar_downsample(
            xin, yin, losin, int(npt[k]), int(nmin[k]), int(nmax[k]),
            method=downsample_method,
        )

        xutm, yutm = _ll2xy(xout, yout, ref_lon)
        xsar = xutm - xo
        ysar = yutm - yo

        _, _, ve = make_look_downsample(xin, yin, ein, xout, yout, xx1, xx2, yy1, yy2)
        _, _, vn = make_look_downsample(xin, yin, nin, xout, yout, xx1, xx2, yy1, yy2)
        _, _, vz = make_look_downsample(xin, yin, uin, xout, yout, xx1, xx2, yy1, yy2)
        _, _, dem_out = make_look_downsample(xin, yin, demin, xout, yout, xx1, xx2, yy1, yy2)

        sampled = np.column_stack([xsar, ysar, zout, ve, vn, vz])
        if save_mat:
            savemat(os.path.join(track, f"los_samp{iint}.mat"),
                    {"sampled_insar_data": sampled,
                     "rms_out": rms_out,
                     "dem_out": dem_out})
        if save_plot:
            plot_insar_sample(xin, yin, losin, zout, xx1, xx2, yy1, yy2,
                              fault_file=fault_file,
                              savepath=os.path.join(track, f"los_samp{iint}.png"))
