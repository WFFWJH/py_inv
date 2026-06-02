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
def read_grd(filename, engine: Optional[str] = None):
    """Read a GMT netCDF grid. Returns (x, y, z) with z.shape == (ny, nx).

    Accepts either ``x/y`` (GMT4) or ``lon/lat`` (CF-style) coordinate names.

    依次尝试 ``h5netcdf``、默认引擎、``netcdf4``, 避免部分环境下 ``netCDF4`` DLL 失败.

    Parameters
    ----------
    engine
        若给定 (如 ``"h5netcdf"`` / ``"netcdf4"`` / ``None`` 为 scipy 类默认),
        仅使用该引擎; 未安装则抛错. 供测试或强制复现; 一般调用勿传.
    """
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


def _rms_block(x, y, z, nres_min, nres_max, method):
    """Evaluate a candidate block. Returns (xo, yo, zo, ngood, r_good, rms).

    Matches MATLAB's rms_block_demean / rms_block_detrend semantics:
        * too-small block  -> rms = 0       (will be accepted)
        * appropriately-sized block -> real rms from mean/plane fit
        * too-large block  -> rms = 1000    (forces subdivision)
        * empty block      -> ngood == 0    (caller should skip)
    """
    nx = x.size
    ny = y.size
    mask = ~np.isnan(z)
    ngood = int(mask.sum())
    n_block = nx * ny
    if ngood == 0:
        return (np.nan, np.nan, np.nan, 0, 0.0, 0.0)

    xx, yy = np.meshgrid(x, y)
    xdata = xx[mask]
    ydata = yy[mask]
    zdata = z[mask]
    r_good = ngood / n_block

    xo = float(xdata.mean())
    yo = float(ydata.mean())
    zo = float(zdata.mean())

    lx = int(np.unique(x).size)
    ly = int(np.unique(y).size)

    if method == "mean":
        if ngood <= 3 or lx <= nres_min or ly <= nres_min:
            rms = 0.0
        elif ngood > 5 and 2 < lx < nres_max and 2 < ly < nres_max:
            rms = float(np.sqrt(np.sum((zdata - zo) ** 2) / ngood))
        else:
            rms = 1000.0
    elif method == "trend":
        if ngood <= 3 or lx <= nres_min or ly <= nres_min:
            rms = 0.0
        elif ngood > 3 and 2 < lx <= nres_max and 2 < ly <= nres_max:
            # plane fit z = a*x + b*y + c
            A = np.column_stack([xdata, ydata, np.ones(ngood)])
            C, *_ = np.linalg.lstsq(A, zdata, rcond=None)
            zfit = A @ C
            rms = float(np.sqrt(np.sum((zdata - zfit) ** 2) / ngood))
        else:
            rms = 1000.0
    else:
        raise ValueError(f"unknown method: {method!r}")

    return xo, yo, zo, ngood, r_good, rms


def _quad_decomp(x, y, z, threshold, nres_min, nres_max, method, out):
    """Recursive 4-quadrant decomposition. Appends accepted blocks to `out`.

    `out` is a list of tuples (xc, yc, zc, ngood, rms, x1, x2, y1, y2)
    whose order follows a Q1,Q2,Q3,Q4 pre-order traversal (matches MATLAB).
    """
    nx = x.size
    ny = y.size

    # --- base case: block already small enough ----------------------
    if nx <= nres_min or ny <= nres_min:
        mask = ~np.isnan(z)
        ngood = int(mask.sum())
        if ngood == 0:
            return
        r_good = ngood / (nx * ny)
        if r_good <= _R_GOOD_DEFAULT:
            return
        zgood = z[mask]
        zbar = float(zgood.mean())
        rms = float(np.sqrt(np.sum((zgood - zbar) ** 2) / ngood))
        if rms < 1e-6:
            # MATLAB mean base-case uses 10, trend base-case uses 1
            rms = 10.0 if method == "mean" else 1.0
        out.append((float(x.mean()), float(y.mean()), zbar, ngood, rms,
                    float(x[0]), float(x[-1]), float(y[0]), float(y[-1])))
        return

    # --- split into 4 quadrants -------------------------------------
    # MATLAB 1-based midpoints  nx2=floor(nx/2)+1, nx3=nx2, nx4=nx
    nx_mid = nx // 2 + 1
    ny_mid = ny // 2 + 1
    x_left, x_right = x[:nx_mid], x[nx_mid - 1:]
    y_low, y_high = y[:ny_mid], y[ny_mid - 1:]

    # quadrant ordering must match MATLAB: Q1, Q2, Q3, Q4
    quadrants = (
        (x_left, y_low, z[:ny_mid, :nx_mid]),         # Q1  y-low , x-left
        (x_right, y_low, z[:ny_mid, nx_mid - 1:]),    # Q2  y-low , x-right
        (x_left, y_high, z[ny_mid - 1:, :nx_mid]),    # Q3  y-high, x-left
        (x_right, y_high, z[ny_mid - 1:, nx_mid - 1:]),  # Q4  y-high, x-right
    )

    for xs, ys, zs in quadrants:
        xo, yo, zo, ngood, r_good, rms = _rms_block(
            xs, ys, zs, nres_min, nres_max, method
        )
        if ngood == 0:
            continue

        if rms <= threshold and r_good > _R_GOOD_DEFAULT:
            # accept this sub-block
            zgood = zs[~np.isnan(zs)]
            rms_acc = float(np.sqrt(np.sum((zgood - zo) ** 2) / ngood))
            if rms_acc < 1e-6:
                rms_acc = 1.0  # rms_default (same for mean & trend here)
            out.append((xo, yo, zo, ngood, rms_acc,
                        float(xs[0]), float(xs[-1]),
                        float(ys[0]), float(ys[-1])))
        elif rms > threshold:
            _quad_decomp(xs, ys, zs, threshold, nres_min, nres_max, method, out)
        # else: dropped (rms <= threshold but r_good too low)


def _results_to_arrays(out):
    if not out:
        empty = np.array([], dtype=np.float64)
        return (empty,) * 9
    arr = np.asarray(out, dtype=np.float64)
    return (arr[:, 0], arr[:, 1], arr[:, 2],
            arr[:, 3].astype(np.int64), arr[:, 4],
            arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8])


def make_insar_downsample(xinsar, yinsar, zinsar, nmin, nres_min, nres_max,
                          method="mean", max_iter=100, verbose=True):
    """Quadtree downsample an InSAR LOS matrix.

    Parameters
    ----------
    xinsar, yinsar : 1D arrays  (x[i] varies along columns, y[j] along rows)
    zinsar         : 2D array of shape (ny, nx) with NaNs allowed
    nmin           : target number of output points  (accept if nmin <= N <= 1.3*nmin)
    nres_min, nres_max : min/max block edge length (in pixels) allowed
    method         : 'mean' or 'trend'

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

    # Allow Python recursion for tall/wide grids.
    _old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(_old, max(x.size, y.size) * 4))
    try:
        # initial threshold from z-range (matches MATLAB r1_norm*z_range, r1_norm=1)
        z_range = float(np.nanmax(z) - np.nanmin(z))
        threshold = z_range

        out = []
        _quad_decomp(x, y, z, threshold, nres_min, nres_max, method, out)
        ndata = len(out)

        for it in range(1, max_iter + 1):
            if nmin <= ndata <= 1.3 * nmin:
                break
            n1 = ndata
            threshold *= 1.05 if ndata > 1.3 * nmin else 0.95
            out = []
            _quad_decomp(x, y, z, threshold, nres_min, nres_max, method, out)
            n2 = len(out)
            ndata = n2
            if verbose:
                print(f"Iter {it}: threshold={threshold:.4f}, N1={n1}, N2={n2}")
            # stopping criterion (matches MATLAB)
            if n2 > 0.9 * nmin and n2 < 2 * nmin and (n2 - n1) < 0.005 * max(n1, 1):
                break
        else:
            if verbose:
                print("Reached max iteration, Ndata may still be < Nmin")
        if verbose:
            print(f"Nint: {it}")
    finally:
        sys.setrecursionlimit(_old)

    return _results_to_arrays(out)


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
