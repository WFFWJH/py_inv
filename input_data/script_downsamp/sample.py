from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from numba import njit


def read_grd(filename, engine: Optional[str] = None):
    """Read a GMT netCDF grid. Returns (x, y, z) with z.shape == (ny, nx).

    Accepts either ``x/y`` (GMT4) or ``lon/lat`` (CF-style) coordinate names.

    依次尝试 ``h5netcdf``、默认引擎、``netcdf4``, 避免部分环境下 ``netCDF4`` DLL 失败.
    """
    import xarray as xr

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


_N_BLOCK_COLS = 9


def _alloc_block_buf(z):
    """预分配块结果缓冲 ``(nx*ny, 9)``; 列: x,y,z,npt,rms,xx1,xx2,yy1,yy2."""
    z = np.asarray(z)
    ny, nx = z.shape
    return np.empty((int(nx) * int(ny), _N_BLOCK_COLS), dtype=np.float64)


def _blocks_view(buf, n):
    """已写入 ``n`` 行后的 ``(n, 9)`` 视图."""
    return buf[:n]


def _append_blocks(buf, n, xo, yo, zo, npt, rms, xx1, xx2, yy1, yy2):
    m = int(np.asarray(xo).size)
    if m == 0:
        return n
    e = n + m
    buf[n:e, 0] = np.asarray(xo, dtype=np.float64).ravel()
    buf[n:e, 1] = np.asarray(yo, dtype=np.float64).ravel()
    buf[n:e, 2] = np.asarray(zo, dtype=np.float64).ravel()
    buf[n:e, 3] = np.asarray(npt, dtype=np.float64).ravel()
    buf[n:e, 4] = np.asarray(rms, dtype=np.float64).ravel()
    buf[n:e, 5] = np.asarray(xx1, dtype=np.float64).ravel()
    buf[n:e, 6] = np.asarray(xx2, dtype=np.float64).ravel()
    buf[n:e, 7] = np.asarray(yy1, dtype=np.float64).ravel()
    buf[n:e, 8] = np.asarray(yy2, dtype=np.float64).ravel()
    return e


def _append_one_block(buf, n, xmean, ymean, zmean, ngood, rms_block, xx1, xx2, yy1, yy2):
    buf[n, 0] = xmean
    buf[n, 1] = ymean
    buf[n, 2] = zmean
    buf[n, 3] = float(ngood)
    buf[n, 4] = rms_block
    buf[n, 5] = xx1
    buf[n, 6] = xx2
    buf[n, 7] = yy1
    buf[n, 8] = yy2
    return n + 1


def _results_to_arrays(out):
    """将 ``(n, 9)`` 块数组拆成与 quad 函数相同的 9 个 1D 数组."""
    arr = np.asarray(out, dtype=np.float64)
    if arr.size == 0:
        empty = np.array([], dtype=np.float64)
        return (empty, empty, empty, np.array([], dtype=np.int64), empty,
                empty, empty, empty, empty)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return (
        arr[:, 0], arr[:, 1], arr[:, 2],
        arr[:, 3].astype(np.int64), arr[:, 4],
        arr[:, 5], arr[:, 6], arr[:, 7], arr[:, 8],
    )


def _quad_decomp_fill_out(x, y, z, threshold, nres_min, nres_max, buf, n, type_samp):
    """四叉树分解; 块写入 ``buf`` 从行 ``n`` 起, 返回新行数."""
    if type_samp == 0:
        xo, yo, zo, npt, rms, xx1, xx2, yy1, yy2 = quad_decomp_mean2(
            x, y, z, threshold, nres_min, nres_max,
            np.empty(0), np.empty(0), np.empty(0),
            np.empty(0), np.empty(0),
            np.empty(0), np.empty(0), np.empty(0), np.empty(0),
        )
        return _append_blocks(buf, n, xo, yo, zo, npt, rms, xx1, xx2, yy1, yy2)
    if type_samp == 1:
        return _quad_decomp_mean2_fast(x, y, z, threshold, nres_min, nres_max, buf, n)
    if type_samp == 2:
        xo, yo, zo, npt, rms, xx1, xx2, yy1, yy2 = quad_decomp_mean2_numba(
            x, y, z, threshold, nres_min, nres_max,
        )
        return _append_blocks(buf, n, xo, yo, zo, npt, rms, xx1, xx2, yy1, yy2)
    raise ValueError(f"type_samp 须为 0, 1 或 2, 得到 {type_samp!r}")


def iter_quad_downsample(
    x, y, z, points_num, nres_min, nres_max, type_samp,
    *,
    initial_threshold: Optional[float] = None,
    max_iter: int = 100,
    verbose: bool = True,
):
    """迭代 quad 下采样: 每轮写入 ``(n, 9)`` 数组并重算 threshold (与 ``make_insar_downsample`` 相同结构).

    ``initial_threshold`` 为 ``None`` 时, 用 ``nanmax(z)-nanmin(z)`` (无效则用 ``1.5*std`` 或 ``1.0``).
    """
    z = np.asarray(z, dtype=np.float64)
    if initial_threshold is None:
        threshold = float(np.nanmax(z) - np.nanmin(z))
        if not np.isfinite(threshold) or threshold <= 0.0:
            valid = z[~np.isnan(z)]
            threshold = float(1.5 * np.std(valid)) if valid.size else 1.0
    else:
        threshold = float(initial_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                f"initial_threshold 须为有限正数, 得到 {initial_threshold!r}"
            )

    buf = _alloc_block_buf(z)
    n = _quad_decomp_fill_out(
        x, y, z, threshold, nres_min, nres_max, buf, 0, type_samp,
    )
    out = _blocks_view(buf, n)
    ndata = out.shape[0]
    max_rms = max(out[:, 4])
    min_rms = min(out[:, 4])
    print("max_rms_out:",max_rms,"min_rms_out:",min_rms)
    threshold = 2*max_rms if ndata > points_num else 0.9*max_rms
    print("threshold:",threshold)

    for it in range(1, max_iter + 1):
        if points_num <= ndata <= 1.1 * points_num:
            break
        n1 = ndata
        threshold *= 1.1 if ndata > points_num else 0.9
        n = _quad_decomp_fill_out(
            x, y, z, threshold, nres_min, nres_max, buf, 0, type_samp,
        )
        out = _blocks_view(buf, n)
        n2 = out.shape[0]
        ndata = n2
        if verbose:
            print(f"strain: {threshold:.4f} NUM: {n2} (iter {it}, was {n1})")
            print("max_rms_out:",max(out[:, 4]),"min_rms_out:",min(out[:, 4]))
        # if n2 > 0.9 * points_num and n2 <1.1*points_num and n2 != n1 and abs(n2 - n1) < 0.005 * max(n1, 1):
        #     break
    else:
        if verbose:
            print("Reached max iteration, 点数可能仍未达到目标")

    if verbose and ndata:
        print(f"Nint done, blocks = {ndata}")
    return _results_to_arrays(out)


@njit
def quad_decomp_mean2_numba(x, y, z, threshold, Nres_min, Nres_max):
    ny, nx = z.shape

    # ===== 预分配（最大block数量粗略上限）=====
    max_blocks = nx * ny

    xout = np.zeros(max_blocks)
    yout = np.zeros(max_blocks)
    zout = np.zeros(max_blocks)
    rms_out = np.zeros(max_blocks)
    Ndata = np.zeros(max_blocks, dtype=np.int64)
    xx1 = np.zeros(max_blocks)
    xx2 = np.zeros(max_blocks)
    yy1 = np.zeros(max_blocks)
    yy2 = np.zeros(max_blocks)

    out_count = 0

    # ===== 手动stack（每行: x_start, x_end, y_start, y_end）=====
    stack = np.zeros((max_blocks, 4), dtype=np.int64)
    sp = 0

    stack[sp] = np.array([0, nx, 0, ny])
    sp += 1

    r_good_default = 0.4

    while sp > 0:
        sp -= 1
        xs, xe, ys, ye = stack[sp]

        nx_sub = xe - xs
        ny_sub = ye - ys

        Ngood = 0
        sumz = 0.0
        sumx = 0.0
        sumy = 0.0

        # ===== 遍历 block =====
        for j in range(ys, ye):
            for i in range(xs, xe):
                val = z[j, i]
                if not np.isnan(val):
                    Ngood += 1
                    sumz += val
                    sumx += x[i]
                    sumy += y[j]

        if Ngood == 0:
            continue

        n_block = nx_sub * ny_sub
        r_good = Ngood / n_block

        zmean = sumz / Ngood
        xmean = sumx / Ngood
        ymean = sumy / Ngood

        # ===== RMS =====
        if Ngood == 1:
            rms_block = 10.0
        else:
            s = 0.0
            for j in range(ys, ye):
                for i in range(xs, xe):
                    val = z[j, i]
                    if not np.isnan(val):
                        d = val - zmean
                        s += d * d
            rms_block = np.sqrt(s / Ngood)

        # ===== 判定是否分裂 =====
        need_split = False

        if (nx_sub <= Nres_min) or (ny_sub <= Nres_min):
            if not (r_good > r_good_default):
                continue

        elif (nx_sub > 2 and nx_sub < Nres_max and ny_sub > 2 and ny_sub < Nres_max):
            if rms_block > threshold:
                need_split = True
            elif not (r_good > r_good_default):
                continue
        else:
            need_split = True

        # ===== 分裂 =====
        if need_split:
            xm = xs + nx_sub // 2
            ym = ys + ny_sub // 2

            # 4 blocks
            stack[sp] = np.array([xs, xm+1, ys, ym+1]); sp += 1
            stack[sp] = np.array([xm, xe, ys, ym+1]); sp += 1
            stack[sp] = np.array([xs, xm+1, ym, ye]); sp += 1
            stack[sp] = np.array([xm, xe, ym, ye]); sp += 1

        else:
            xout[out_count] = xmean
            yout[out_count] = ymean
            zout[out_count] = zmean
            rms_out[out_count] = rms_block
            Ndata[out_count] = Ngood

            xx1[out_count] = x[xs]
            xx2[out_count] = x[xe-1]
            yy1[out_count] = y[ys]
            yy2[out_count] = y[ye-1]

            out_count += 1

    # ===== 截断有效长度 =====
    return (
        xout[:out_count],
        yout[:out_count],
        zout[:out_count],
        Ndata[:out_count],
        rms_out[:out_count],
        xx1[:out_count],
        xx2[:out_count],
        yy1[:out_count],
        yy2[:out_count],
    )

def _quad_decomp_mean2_fast(x, y, z, threshold, Nres_min, Nres_max, buf, n):
    """四叉树 mean 下采样; 接受的块写入 ``buf`` 从行 ``n`` 起, 返回新行数."""
    ny, nx = z.shape

    xx_full, yy_full = np.meshgrid(x, y)
    stack = [(np.arange(nx), np.arange(ny))]
    r_good_default = 0.4

    while stack:
        x_idx, y_idx = stack.pop()

        xsub = x[x_idx]
        ysub = y[y_idx]
        zsub = z[np.ix_(y_idx, x_idx)]

        xx = xx_full[np.ix_(y_idx, x_idx)]
        yy = yy_full[np.ix_(y_idx, x_idx)]

        indx_good = ~np.isnan(zsub)
        Ngood = int(np.sum(indx_good))

        nx_sub = len(x_idx)
        ny_sub = len(y_idx)
        n_block = nx_sub * ny_sub

        if Ngood == 0:
            continue

        r_good = Ngood / n_block

        zdata = zsub[indx_good]
        if zdata.size == 0:
            continue

        zmean = float(np.mean(zdata))
        xmean = float(np.mean(xx[indx_good]))
        ymean = float(np.mean(yy[indx_good]))

        if zdata.size == 1:
            rms_block = 10.0
        else:
            rms_block = float(np.sqrt(np.mean((zdata - zmean) ** 2)))

        lx = len(np.unique(xsub))
        ly = len(np.unique(ysub))

        need_split = False

        if (lx <= Nres_min) or (ly <= Nres_min):
            if not (Ngood > 0 and r_good > r_good_default):
                continue
        elif (lx > 2 and lx < Nres_max) and (ly > 2 and ly < Nres_max):
            if rms_block > threshold:
                need_split = True
            elif not (r_good > r_good_default):
                continue
        else:
            need_split = True

        if need_split:
            nx_mid = nx_sub // 2
            ny_mid = ny_sub // 2
            x_splits = [x_idx[:nx_mid + 1], x_idx[nx_mid:]]
            y_splits = [y_idx[:ny_mid + 1], y_idx[ny_mid:]]
            for ys in y_splits:
                for xs in x_splits:
                    if len(xs) > 0 and len(ys) > 0:
                        stack.append((xs, ys))
        else:
            n = _append_one_block(
                buf, n, xmean, ymean, zmean, Ngood, rms_block,
                float(xsub[0]), float(xsub[-1]), float(ysub[0]), float(ysub[-1]),
            )

    return n


def quad_decomp_mean2_fast(x, y, z, threshold, Nres_min, Nres_max):
    """兼容旧接口: 内部分配缓冲再转数组."""
    buf = _alloc_block_buf(z)
    n = _quad_decomp_mean2_fast(x, y, z, threshold, Nres_min, Nres_max, buf, 0)
    return _results_to_arrays(_blocks_view(buf, n))

def quad_decomp_mean2(x, y, z, threshold, Nres_min, Nres_max,
                     xout_in, yout_in, zout_in,
                     Ndata_in, rms_in,
                     xx1_in, xx2_in, yy1_in, yy2_in):

    # 初始化输出（复制输入）
    xout = xout_in.copy()
    yout = yout_in.copy()
    zout = zout_in.copy()
    Ndata = Ndata_in.copy()
    rms_out = rms_in.copy()
    xx1 = xx1_in.copy()
    xx2 = xx2_in.copy()
    yy1 = yy1_in.copy()
    yy2 = yy2_in.copy()

    # meshgrid（注意：MATLAB默认是xy顺序）
    xx, yy = np.meshgrid(x, y)

    # 过滤非 NaN 数据
    indx_good = ~np.isnan(z)
    ny, nx = z.shape
    n_block = nx * ny

    xdata = xx[indx_good]
    ydata = yy[indx_good]
    zdata = z[indx_good]

    Ngood = len(zdata)
    r_good = Ngood / n_block if n_block > 0 else 0
    r_good_default = 0.4

    # block 均值
    if Ngood > 0:
        xout_block = np.mean(xdata)
        yout_block = np.mean(ydata)
        zout_block = np.mean(zdata)
    else:
        xout_block = yout_block = zout_block = np.nan

    # unique 数量
    lx = len(np.unique(x))
    ly = len(np.unique(y))

    need_split = False

    # --- 情况1 ---
    if (lx <= Nres_min or ly <= Nres_min):
        if (Ngood > 0 and r_good > r_good_default):
            zz = zdata
            if zz.size == 1:
                rms_block = 10
            else:
                zzfit = np.mean(zz)
                dz = zz - zzfit
                rms_block = np.sqrt(np.sum(dz**2) / Ngood)
        else:
            return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2

    # --- 情况2 ---
    elif ((lx > 2 and lx < Nres_max) and (ly > 2 and ly < Nres_max)):
        zz = zdata
        zzfit = np.mean(zz)
        dz = zz - zzfit
        rms_block = np.sqrt(np.sum(dz**2) / Ngood) if Ngood > 0 else np.nan

        if (rms_block > threshold and Ngood > 0):
            need_split = True
        elif (rms_block <= threshold and Ngood > 0 and r_good > r_good_default):
            need_split = False
        else:
            return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2

    # --- 情况3 ---
    else:
        need_split = True

    # =========================
    # 递归拆分
    # =========================
    if need_split:
        nx_mid = nx // 2
        ny_mid = ny // 2

        nx_idx = [slice(0, nx_mid + 1), slice(nx_mid, nx)]
        ny_idx = [slice(0, ny_mid + 1), slice(ny_mid, ny)]

        for iy in range(2):
            for ix in range(2):

                xsub = x[nx_idx[ix]]
                ysub = y[ny_idx[iy]]
                zsub = z[ny_idx[iy], nx_idx[ix]]

                xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2 = \
                    quad_decomp_mean2(
                        xsub, ysub, zsub,
                        threshold, Nres_min, Nres_max,
                        xout, yout, zout,
                        Ndata, rms_out,
                        xx1, xx2, yy1, yy2
                    )

    # =========================
    # 不拆分：记录结果
    # =========================
    else:
        xout = np.append(xout, xout_block)
        yout = np.append(yout, yout_block)
        zout = np.append(zout, zout_block)
        rms_out = np.append(rms_out, rms_block)
        Ndata = np.append(Ndata, Ngood)

        xx1_this_block = x[0]
        xx2_this_block = x[-1]
        yy1_this_block = y[0]
        yy2_this_block = y[-1]

        xx1 = np.append(xx1, xx1_this_block)
        yy1 = np.append(yy1, yy1_this_block)
        xx2 = np.append(xx2, xx2_this_block)
        yy2 = np.append(yy2, yy2_this_block)

    return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2

def subsample(
    file, points_num, uplimit, downlimit, write_or_not, plot_or_not,
    type_samp=0, initial_threshold=None,
):

    # ==== 读取 grd 文件 ====
    xvec, yvec, zz = read_grd(file)

    # ==== 统计信息 ====
    valid = ~np.isnan(zz)
    vals = zz[valid]

    global_mean = np.mean(vals)
    global_std = np.std(vals)
    global_rms = np.sqrt(np.mean((vals - global_mean) ** 2))
    n_total = len(vals)

    print(f"有效点数={n_total}, mean={global_mean:.4g}, std={global_std:.4g}, rms={global_rms:.4g}")

    thr = global_rms if initial_threshold is None else initial_threshold
    x, y, z, Npt, rms_out, xx1, xx2, yy1, yy2 = iter_quad_downsample(
        xvec, yvec, zz, points_num, downlimit, uplimit, type_samp,
        initial_threshold=thr,
    )

    # ==== 权重 (沿用原脚本: 1/sqrt(块内 RMS)) ====
    n = 1.0 / np.sqrt(rms_out)

    # ==== 绘图 ====
    if plot_or_not == 1:
        plt.figure()

        numcol = 200
        cmap = plt.get_cmap('jet', numcol)

        clim = [-np.max(np.abs(z)), np.max(np.abs(z))]

        norm = plt.Normalize(vmin=clim[0], vmax=clim[1])

        plt.scatter(x, y, c=z, cmap=cmap, s=10)

        plt.colorbar(label='z')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.gca().set_aspect('equal')

        plt.title("Subsampled Points")
#        plt.show(block=False)
        plt.show()

    print(f"number of subsampled data: {len(x)}")

    print(zz.dtype)
    # ==== 写文件 ====
    if write_or_not == 1:
        from pathlib import Path
        file = Path(file)
        path_ = file.with_suffix(".grd")
        outfile = file.parent / f"{file.stem}_py.llde"
        print(outfile)
        with open(outfile, 'w') as fid:
            for j in range(len(x)):
                fid.write(f"{x[j]:.9f}\t{y[j]:.9f}\t{z[j]:.9f}\t{n[j]:.9f}\n")

    return x, y, z, n

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

def subsample1(
    file, file1, points_num, uplimit, downlimit, write_or_not, plot_or_not,
    type_samp=0, initial_threshold=None,
):

    # ==== 读取 grd 文件 ====
    xvec, yvec, zz = read_grd(file)
    xvec1, yvec1, zz1 = read_grd(file1)
    assert( xvec.shape == xvec1.shape and yvec.shape == yvec1.shape and zz.shape == zz1.shape)
    # ==== 统计信息 ====
    valid = ~np.isnan(zz)
    vals = zz[valid]

    global_mean = np.mean(vals)
    global_std = np.std(vals)
    global_rms = np.sqrt(np.mean((vals - global_mean) ** 2))
    n_total = len(vals)

    print(f"有效点数={n_total}, mean={global_mean:.4g}, std={global_std:.4g}, rms={global_rms:.4g}")

    thr = global_rms if initial_threshold is None else initial_threshold
    x, y, z, Npt, rms_out, xx1, xx2, yy1, yy2 = iter_quad_downsample(
        xvec, yvec, zz, points_num, downlimit, uplimit, type_samp,
        initial_threshold=thr,
    )
    x1, y1, z1 = make_look_downsample(xvec1, yvec1, zz1, x, y, xx1, xx2, yy1, yy2)
    assert( x1.shape == x.shape and y1.shape == y.shape and z1.shape == z.shape)
    # ==== 权重 (沿用原脚本: 1/sqrt(块内 RMS)) ====
    n = 1.0 / np.sqrt(rms_out)

    # ==== 绘图 ====
    if plot_or_not == 1:
        plt.figure()

        numcol = 200
        cmap = plt.get_cmap('jet', numcol)

        clim = [-np.max(np.abs(z)), np.max(np.abs(z))]

        norm = plt.Normalize(vmin=clim[0], vmax=clim[1])

        plt.scatter(x, y, c=z, cmap=cmap, s=10)

        plt.colorbar(label='z')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.gca().set_aspect('equal')

        plt.title("Subsampled Points")
#        plt.show(block=False)
        plt.show()

    print(f"number of subsampled data: {len(x)}")

    print(zz.dtype)
    # ==== 写文件 ====
    if write_or_not == 1:
        from pathlib import Path
        file = Path(file)
        outfile = file.parent / f"{file.stem}_py.llde"
        print(outfile)
        with open(outfile, 'w') as fid:
            for j in range(len(x)):
                fid.write(f"{x1[j]:.9f}\t{y1[j]:.9f}\t{z1[j]:.9f}\t{n[j]:.9f}\n")

    return x1, y1, z1, n




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("file", type=str)
    parser.add_argument("points_num", type=int)
    parser.add_argument("uplimit", type=float)
    parser.add_argument("downlimit", type=float)
    parser.add_argument("write_or_not", type=int)
    parser.add_argument("plot_or_not", type=int)
    parser.add_argument("type_samp", type=int)

    # 可选参数
    parser.add_argument(
        "file1",
        nargs="?",
        default=None,
        type=str
    )
    parser.add_argument(
        "--initial-threshold",
        dest="initial_threshold",
        default=None,
        type=float,
        help="quad 下采样初始 threshold; 未指定时由数据 z 范围自动估算",
    )

    args = parser.parse_args()

    if args.file1 is None:
        subsample(
            file=args.file,
            points_num=args.points_num,
            uplimit=args.uplimit,
            downlimit=args.downlimit,
            write_or_not=args.write_or_not,
            plot_or_not=args.plot_or_not,
            type_samp=args.type_samp,
            initial_threshold=args.initial_threshold,
        )
    else:
        subsample1(
            file=args.file,
            file1=args.file1,
            points_num=args.points_num,
            uplimit=args.uplimit,
            downlimit=args.downlimit,
            write_or_not=args.write_or_not,
            plot_or_not=args.plot_or_not,
            type_samp=args.type_samp,
            initial_threshold=args.initial_threshold,
        )

