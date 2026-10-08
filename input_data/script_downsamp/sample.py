"""InSAR 四叉树下采样 (quadtree downsample)。

用途
----
对规则网格上的 LOS（视线）位移做基于 RMS 的四叉树分解，输出稀疏采样点，
供后续反演使用。可选地对伴随栅格 (file1) 在同一批盒子内取均值。

典型调用 (与 mask_sample.sh / make_los_samp1.sh 一致)
----------------------------------------------------
    python sample.py <file> <points_num> <uplimit> <downlimit> \\
                     <write> <plot> <type_samp> [file1]

位置参数含义
------------
file         : LOS 输入 —— GMT/netCDF ``.grd`` / ``.nc``
points_num   : 目标采样点数 (迭代调节 threshold 逼近)
uplimit      : Nres_max —— 块边长超过此像素数必须继续分裂
downlimit    : Nres_min —— 块边长达到此下限后不再按 RMS 分裂
write_or_not : 1 则写结果文件
plot_or_not  : 1 则弹窗画采样点
type_samp    : 0 = 纯 Python 栈式四叉树; 1 = numba 加速 (推荐)
file1        : (可选) 伴随栅格，在 LOS 同一批盒子内取均值后写出

输出文件
--------
``{stem}_py.llde``  列: x y z weight

权重 (方案 A, 供 GM=d 行加权)
----------------------------
σ = max(块内 RMS, 0.001 m);  w = 1/σ;  再除以 max(w) 归一化到 (0, 1]。
写入的 RMS 为真实标准差 (单点为 0), 不再使用旧版魔法数 10。

可选开关
--------
--initial-threshold  初始 RMS 阈值; 默认用数据 global_rms
--min-stop-mode      and | or  (最小块停止条件，默认 and；见四叉树分区注释)
--representative     mean | median_pixel  (块代表点取法，默认 mean)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# =============================================================================
# 1. 输入 / 输出 —— 读 .grd，写 llde
# =============================================================================


def read_grd(filename, engine: Optional[str] = None):
    """读 GMT/netCDF 栅格为 (x, y, z[ny,nx])。

    engine 选择顺序 (未指定时): h5netcdf → 默认 → netcdf4。
    坐标名优先识别 x/lon/longitude 与 y/lat/latitude；
    若 z 的二维形状与 (ny,nx) 颠倒则自动转置。
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
            # 常见 GMT 约定: 最后一维是 x，第一维是 y
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


def _write_llde(outfile, x, y, z, weights):
    """写 4 列: x y z weight。"""
    print(outfile)
    with open(outfile, "w", encoding="utf-8") as fid:
        for j in range(len(x)):
            fid.write(
                f"{x[j]:.9f}\t{y[j]:.9f}\t{z[j]:.9f}\t{weights[j]:.9f}\n"
            )


def _plot_subsample(x, y, z, title="Subsampled Points"):
    """散点图预览采样结果 (对称 jet 色标)。"""
    plt.figure()
    cmap = plt.get_cmap("jet", 200)
    vmax = float(np.max(np.abs(z)))
    clim = [-vmax, vmax]
    norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
    plt.scatter(x, y, c=z, cmap=cmap, norm=norm, s=10)
    plt.colorbar(label="z")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.show()


# =============================================================================
# 2. 四叉树核心 —— 块缓冲、单次分解、阈值迭代
# =============================================================================
#
# 每个接受的块写成一行，列含义 (_N_BLOCK_COLS = 9):
#   0:x  1:y  2:z  3:npt  4:rms  5:xx1  6:xx2  7:yy1  8:yy2
#   其中 (xx1,xx2,yy1,yy2) 是该块在物理坐标上的包围盒，供 file1 伴随采样用。
#
# ---------- 单个块的判定逻辑 (Python / numba 两套实现一致) ----------
# 对当前矩形块:
#   ngood   = 非 NaN 像素数
#   r_good  = ngood / (nx_sub * ny_sub)   # 有效覆盖率
#   zmean   = 有效 z 的均值
#   rms_true = sqrt(mean((z-zmean)^2)); 单点为 0 (写入结果)
#   分裂时单点用极大 RMS, 以便在仍可切时继续切
#   r_good_default = 0.4                 # 覆盖率太低则丢弃该块
#
# 三种结局:
#   A. 已达最小尺寸 (由 min_stop_mode 决定):
#        and → lx<=Nres_min 且 ly<=Nres_min   (本脚本默认)
#        or  → lx 或 ly 任一侧 <=Nres_min     (MATLAB 原版)
#      → 若 r_good > 0.4 则接受，否则丢弃；不再看 RMS。
#   B. 处于中间尺寸 (2 < lx < Nres_max 且 2 < ly < Nres_max):
#      → rms > threshold 则分裂; 否则若 r_good>0.4 接受, 否则丢弃。
#   C. 其它 (过大等) → 继续分裂; 若某一边已 <2 无法再切, 则按覆盖率接受/丢弃。
#
# 分裂方式: 在索引中点做不重叠划分 (子块严格小于父块, 避免死循环):
#   两边都 ≥2 → 切成 2×2; 仅一边 ≥2 → 沿该边一分为二。
#
# ---------- 块代表点 (representative) ----------
#   mean          : (x,y)=有效点坐标质心, z=块内均值  (默认, 同 MATLAB)
#   median_pixel  : 取 z 最接近块内中位数的那一格像素;
#                   并列时选距质心最近者。
#
# ---------- 阈值迭代 (iter_quad_downsample) ----------
# 目标: 让块数落在 [0.9*points_num, 1.1*points_num]。
# 阈值越大 → 通常块越少。用二分夹逼 (两侧都试探过之后),
# 尚未建立上下界时仍用 ×1.1 / ×0.9。块数连续不变则早停;
# 若达不到窗口, 返回最接近目标点数的那一轮结果。
#
# ---------- 反演权重 (方案 A) ----------
# σ = max(RMS, SIGMA_FLOOR_M), w = 1/σ, 再归一化到 (0, 1]。

_N_BLOCK_COLS = 9

# 位移单位下的 σ 地板 (米)
SIGMA_FLOOR_M = 0.001
# 单点块仅用于分裂判据的假 RMS (不写入结果)
_RMS_SPLIT_SINGLE = 1.0e30


def _weights_from_rms(rms, sigma_floor: float = SIGMA_FLOOR_M, normalize: bool = True):
    """方案 A: σ=max(rms, sigma_floor), w=1/σ; 默认再归一化到 (0, 1]。"""
    rms = np.asarray(rms, dtype=np.float64)
    sigma = np.maximum(rms, float(sigma_floor))
    w = 1.0 / sigma
    if normalize and w.size > 0:
        wmax = float(np.max(w))
        if wmax > 0.0:
            w = w / wmax
    return w


def _parse_min_stop_mode(min_stop_mode: str) -> bool:
    """解析最小块停止模式。返回 True=and, False=or。"""
    mode = str(min_stop_mode).lower()
    if mode not in ("and", "or"):
        raise ValueError(
            "min_stop_mode 须为 'and' 或 'or', 得到 %r" % (min_stop_mode,)
        )
    return mode == "and"


def _parse_representative(representative: str) -> bool:
    """解析代表点模式。返回 True=median_pixel, False=mean。"""
    mode = str(representative).lower()
    if mode in ("mean", "block_mean", "centroid"):
        return False
    if mode in ("median", "median_pixel", "median-pixel"):
        return True
    raise ValueError(
        "representative 须为 'mean' 或 'median_pixel', 得到 %r" % (representative,)
    )


def _block_representative(
    x, y, x_idx, y_idx, rows, cols, zdata, xmean, ymean, zmean, use_median_pixel,
):
    """决定一个接受块输出的 (x, y, z)。

    mean 路径直接返回质心/均值；
    median_pixel 路径在有效像素中找 z 最近中位数者，并列按距质心距离破平。
    """
    if not use_median_pixel or zdata.size == 1:
        return xmean, ymean, zmean

    zmed = float(np.median(zdata))
    dz = np.abs(zdata - zmed)
    tol = 1e-12 * max(abs(zmed), 1.0)
    cand = np.flatnonzero(dz <= np.min(dz) + tol)
    xc = x[x_idx[cols[cand]]]
    yc = y[y_idx[rows[cand]]]
    dxy2 = (xc - xmean) ** 2 + (yc - ymean) ** 2
    pick = int(cand[int(np.argmin(dxy2))])
    return (
        float(x[x_idx[cols[pick]]]),
        float(y[y_idx[rows[pick]]]),
        float(zdata[pick]),
    )


def _block_at_min_size(lx, ly, nres_min, min_stop_and: bool) -> bool:
    """当前块是否已到达最小尺寸停止条件。"""
    if min_stop_and:
        return lx <= nres_min and ly <= nres_min
    return lx <= nres_min or ly <= nres_min


def _alloc_block_buf(z):
    """预分配块结果缓冲 (nx*ny, 9)，足够容纳最坏情况下每像素一块。"""
    z = np.asarray(z)
    ny, nx = z.shape
    return np.empty((int(nx) * int(ny), _N_BLOCK_COLS), dtype=np.float64)


def _blocks_view(buf, n):
    """已写入 n 行后的 (n, 9) 视图。"""
    return buf[:n]


def _append_one_block(buf, n, xmean, ymean, zmean, ngood, rms_block, xx1, xx2, yy1, yy2):
    """把一个接受块写入 buf[n]，返回新的行数 n+1。"""
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
    """把 (n, 9) 块数组拆成 9 个 1D 数组 (与历史 quad 返回值一致)。"""
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


def _quad_decomp_fill_out(
    x, y, z, threshold, nres_min, nres_max, buf, n, type_samp,
    min_stop_and=True, use_median_pixel=False,
):
    """一次完整四叉树分解；接受块写入 buf[n:]，返回新行数。

    type_samp: 0 → Python 栈式; 1 → numba。其它值报错。
    """
    ts = int(type_samp)
    if ts == 1:
        return _quad_decomp_mean2_numba_fill(
            buf, n, x, y, z, threshold, nres_min, nres_max, min_stop_and,
            use_median_pixel,
        )
    if ts == 0:
        return _quad_decomp_mean2(
            x, y, z, threshold, nres_min, nres_max, buf, n,
            min_stop_and=min_stop_and, use_median_pixel=use_median_pixel,
        )
    raise ValueError("type_samp 须为 0 (Python) 或 1 (numba), 得到 %r" % (type_samp,))


def iter_quad_downsample(
    x, y, z, points_num, nres_min, nres_max, type_samp,
    *,
    initial_threshold: Optional[float] = None,
    max_iter: int = 100,
    verbose: bool = True,
    min_stop_mode: str = "or",
    representative: str = "mean",
):
    """迭代调节 threshold，使四叉树块数逼近 points_num。

    参数
    ----
    x, y, z
        规则网格坐标与 LOS 值 (z 形状 ny×nx，NaN 为掩膜)。
    points_num
        目标块数。停止窗口: [0.9*points_num, 1.1*points_num]。
    nres_min, nres_max
        对应 CLI 的 downlimit / uplimit (像素)。
    type_samp
        0=Python, 1=numba。
    initial_threshold
        初始 RMS 阈值。None 时用 nanmax(z)-nanmin(z)；
        若无效再用 1.5*std 或 1.0。
        注意: 高层入口通常会传入 global_rms。
    min_stop_mode
        ``"and"`` / ``"or"``，见分区头注释。默认 ``"or"``。
    representative
        ``"mean"`` / ``"median_pixel"``，见分区头注释。

    返回
    ----
    x, y, z, npt, rms, xx1, xx2, yy1, yy2  (各为 1D 数组)
    """
    min_stop_and = _parse_min_stop_mode(min_stop_mode)
    use_median_pixel = _parse_representative(representative)
    nres_min = int(nres_min)
    nres_max = int(nres_max)
    points_num = max(1, int(points_num))
    z = np.asarray(z, dtype=np.float64)

    n_lo = 0.99 * points_num
    n_hi = 1.1 * points_num

    # ----- 确定初始 threshold -----
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

    def _run(thr: float):
        nn = _quad_decomp_fill_out(
            x, y, z, thr, nres_min, nres_max, buf, 0, type_samp,
            min_stop_and=min_stop_and, use_median_pixel=use_median_pixel,
        )
        return nn, _blocks_view(buf, nn).copy()

    # ----- 首轮 -----
    n, out = _run(threshold)
    ndata = int(out.shape[0])
    if ndata == 0:
        raise ValueError("quad 下采样未产生任何有效块 (检查 NaN 掩膜或 Nres 参数)")

    best_out = out
    best_n = ndata
    best_thr = threshold
    best_diff = abs(ndata - points_num)

    if verbose:
        print(
            "max_rms_out:",
            float(np.max(out[:, 4])),
            "min_rms_out:",
            float(np.min(out[:, 4])),
        )
        print("threshold:", threshold, "NUM:", ndata)

    if n_lo <= ndata <= n_hi:
        if verbose:
            print(f"Nint done, blocks = {ndata}")
        return _results_to_arrays(out)

    # thr_lo: 块数仍偏多时的下界 (阈值偏小); thr_hi: 块数仍偏少时的上界 (阈值偏大)
    thr_lo = None
    thr_hi = None
    if ndata > points_num:
        thr_lo = threshold
    else:
        thr_hi = threshold

    stagnant = 0
    n_prev = ndata

    for it in range(1, max_iter + 1):
        # 更新阈值: 有双侧边界则二分, 否则按比例试探
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

        n, out = _run(threshold)
        ndata = int(out.shape[0])
        if ndata == 0:
            raise ValueError("quad 下采样迭代后块数为 0 (threshold=%.4g)" % threshold)

        diff = abs(ndata - points_num)
        if diff < best_diff or (diff == best_diff and ndata >= best_n):
            best_diff = diff
            best_n = ndata
            best_thr = threshold
            best_out = out

        if verbose:
            print(f"threshold: {threshold:.4f} NUM: {ndata} (iter {it}, was {n_prev})")
            print(
                "max_rms_out:",
                float(np.max(out[:, 4])),
                "min_rms_out:",
                float(np.min(out[:, 4])),
            )

        if n_lo <= ndata <= n_hi:
            break

        # 块数连续两轮不变 → 阈值已不影响结果, 早停
        if ndata == n_prev:
            stagnant += 1
            if stagnant >= 2 and (threshold > np.max(out[:, 4]) or threshold < np.min(out[:, 4])):
                if verbose:
                    print("块数连续不变, 提前停止")
                break
        else:
            stagnant = 0

        # 收紧二分区间 (阈值↑ → 块数↓ 的单调假设)
        if ndata > points_num:
            thr_lo = threshold if thr_lo is None else max(thr_lo, threshold)
        else:
            thr_hi = threshold if thr_hi is None else min(thr_hi, threshold)

        # 区间塌缩或交叉 → 无法再进步
        if thr_lo is not None and thr_hi is not None and thr_hi <= thr_lo * 1.0001:
            if verbose:
                print("阈值搜索区间已收窄完毕")
            break

        n_prev = ndata
    else:
        if verbose:
            print("Reached max iteration, 点数可能仍未达到目标")

    if verbose:
        in_win = n_lo <= best_n <= n_hi
        print(
            f"Nint done, blocks = {best_n} (target {points_num}, "
            f"thr={best_thr:.4g}, in_window={in_win})"
        )
    return _results_to_arrays(best_out)


@njit
def _pick_median_pixel_numba(x, y, z, xs, xe, ys, ye, xmean, ymean, ngood):
    """numba 版: 选 z 最近块中位数的格点，并列按距质心距离。"""
    zbuf = np.empty(ngood, dtype=np.float64)
    k = 0
    for j in range(ys, ye):
        for i in range(xs, xe):
            val = z[j, i]
            if not np.isnan(val):
                zbuf[k] = val
                k += 1
    zmed = np.median(zbuf)
    best_dz = 1.0e300
    best_dxy2 = 1.0e300
    out_x = xmean
    out_y = ymean
    out_z = zmed
    for j in range(ys, ye):
        for i in range(xs, xe):
            val = z[j, i]
            if not np.isnan(val):
                dz = abs(val - zmed)
                dx = x[i] - xmean
                dy = y[j] - ymean
                dxy2 = dx * dx + dy * dy
                if dz < best_dz - 1.0e-15 or (
                    abs(dz - best_dz) <= 1.0e-15 and dxy2 < best_dxy2
                ):
                    best_dz = dz
                    best_dxy2 = dxy2
                    out_x = x[i]
                    out_y = y[j]
                    out_z = val
    return out_x, out_y, out_z


@njit
def _quad_decomp_mean2_numba_fill(
    buf, n, x, y, z, threshold, Nres_min, Nres_max, min_stop_and,
    use_median_pixel=False,
):
    """numba 四叉树: 用显式索引栈遍历，接受块写入 buf。

    栈元素为半开区间 [xs,xe) × [ys,ye) 的像素下标。
    判定逻辑见本文件「四叉树核心」分区头注释。
    """
    ny, nx = z.shape
    max_blocks = nx * ny
    stack = np.empty((max_blocks, 4), dtype=np.int64)
    # 初始: 整幅图
    stack[0, 0] = 0
    stack[0, 1] = nx
    stack[0, 2] = 0
    stack[0, 3] = ny
    sp = 1
    r_good_default = 0.4

    while sp > 0:
        sp -= 1
        xs = stack[sp, 0]
        xe = stack[sp, 1]
        ys = stack[sp, 2]
        ye = stack[sp, 3]

        nx_sub = xe - xs
        ny_sub = ye - ys

        # 一次扫描累计有效统计量 (避免临时数组)
        ngood = 0
        sumz = 0.0
        sumx = 0.0
        sumy = 0.0
        sumsq = 0.0
        for j in range(ys, ye):
            for i in range(xs, xe):
                val = z[j, i]
                if not np.isnan(val):
                    ngood += 1
                    sumz += val
                    sumx += x[i]
                    sumy += y[j]
                    sumsq += val * val

        if ngood == 0:
            continue

        n_block = nx_sub * ny_sub
        r_good = ngood / n_block
        zmean = sumz / ngood
        xmean = sumx / ngood
        ymean = sumy / ngood

        if ngood == 1:
            rms_true = 0.0
            rms_for_split = _RMS_SPLIT_SINGLE
        else:
            var = sumsq / ngood - zmean * zmean
            if var < 0.0:
                var = 0.0
            rms_true = np.sqrt(var)
            rms_for_split = rms_true

        need_split = False
        can_split_x = nx_sub >= 2
        can_split_y = ny_sub >= 2

        if min_stop_and:
            at_min = (nx_sub <= Nres_min) and (ny_sub <= Nres_min)
        else:
            at_min = (nx_sub <= Nres_min) or (ny_sub <= Nres_min)

        if at_min or not (can_split_x or can_split_y):
            # 最小块, 或已无法再切: 只看覆盖率
            if not (r_good > r_good_default):
                continue
        elif (
            nx_sub > 2 and nx_sub < Nres_max
            and ny_sub > 2 and ny_sub < Nres_max
        ):
            if rms_for_split > threshold:
                need_split = True
            elif not (r_good > r_good_default):
                continue
        else:
            # 过大: 继续切 (能切的方向才会真正入栈)
            need_split = True

        if need_split and (can_split_x or can_split_y):
            xm = xs + nx_sub // 2
            ym = ys + ny_sub // 2
            # 不重叠半开区间: [xs,xm)×[ys,ym) 等, 子块严格小于父块
            if can_split_x and can_split_y:
                stack[sp, 0] = xs
                stack[sp, 1] = xm
                stack[sp, 2] = ys
                stack[sp, 3] = ym
                sp += 1
                stack[sp, 0] = xm
                stack[sp, 1] = xe
                stack[sp, 2] = ys
                stack[sp, 3] = ym
                sp += 1
                stack[sp, 0] = xs
                stack[sp, 1] = xm
                stack[sp, 2] = ym
                stack[sp, 3] = ye
                sp += 1
                stack[sp, 0] = xm
                stack[sp, 1] = xe
                stack[sp, 2] = ym
                stack[sp, 3] = ye
                sp += 1
            elif can_split_x:
                stack[sp, 0] = xs
                stack[sp, 1] = xm
                stack[sp, 2] = ys
                stack[sp, 3] = ye
                sp += 1
                stack[sp, 0] = xm
                stack[sp, 1] = xe
                stack[sp, 2] = ys
                stack[sp, 3] = ye
                sp += 1
            else:
                stack[sp, 0] = xs
                stack[sp, 1] = xe
                stack[sp, 2] = ys
                stack[sp, 3] = ym
                sp += 1
                stack[sp, 0] = xs
                stack[sp, 1] = xe
                stack[sp, 2] = ym
                stack[sp, 3] = ye
                sp += 1
        else:
            out_x = xmean
            out_y = ymean
            out_z = zmean
            if use_median_pixel and ngood > 1:
                out_x, out_y, out_z = _pick_median_pixel_numba(
                    x, y, z, xs, xe, ys, ye, xmean, ymean, ngood,
                )
            buf[n, 0] = out_x
            buf[n, 1] = out_y
            buf[n, 2] = out_z
            buf[n, 3] = float(ngood)
            buf[n, 4] = rms_true
            buf[n, 5] = x[xs]
            buf[n, 6] = x[xe - 1]
            buf[n, 7] = y[ys]
            buf[n, 8] = y[ye - 1]
            n += 1

    return n


def quad_decomp_mean2_numba(
    x, y, z, threshold, Nres_min, Nres_max, min_stop_and=True,
    use_median_pixel=False,
):
    """兼容旧接口: 内部分配缓冲，一次 numba 分解后转成 9 个数组。"""
    buf = _alloc_block_buf(z)
    n = _quad_decomp_mean2_numba_fill(
        buf, 0, x, y, z, threshold, Nres_min, Nres_max, min_stop_and,
        use_median_pixel,
    )
    return _results_to_arrays(_blocks_view(buf, n))


def _quad_decomp_mean2(
    x, y, z, threshold, Nres_min, Nres_max, buf, n, min_stop_and=True,
    use_median_pixel=False,
):
    """纯 Python 栈式四叉树 (type_samp=0)。逻辑与 numba 版一致，便于对照调试。"""
    ny, nx = z.shape
    stack = [(np.arange(nx), np.arange(ny))]
    r_good_default = 0.4

    while stack:
        x_idx, y_idx = stack.pop()

        zsub = z[np.ix_(y_idx, x_idx)]
        indx_good = ~np.isnan(zsub)
        ngood = int(np.sum(indx_good))

        nx_sub = len(x_idx)
        ny_sub = len(y_idx)
        n_block = nx_sub * ny_sub

        if ngood == 0:
            continue

        r_good = ngood / n_block
        zdata = zsub[indx_good]
        rows, cols = np.nonzero(indx_good)
        xmean = float(np.mean(x[x_idx[cols]]))
        ymean = float(np.mean(y[y_idx[rows]]))
        zmean = float(np.mean(zdata))

        if ngood == 1:
            rms_true = 0.0
            rms_for_split = _RMS_SPLIT_SINGLE
        else:
            rms_true = float(np.sqrt(np.mean((zdata - zmean) ** 2)))
            rms_for_split = rms_true

        lx = nx_sub
        ly = ny_sub
        need_split = False
        can_split_x = lx >= 2
        can_split_y = ly >= 2

        if _block_at_min_size(lx, ly, Nres_min, min_stop_and) or not (
            can_split_x or can_split_y
        ):
            if not (ngood > 0 and r_good > r_good_default):
                continue
        elif (lx > 2 and lx < Nres_max) and (ly > 2 and ly < Nres_max):
            if rms_for_split > threshold:
                need_split = True
            elif not (r_good > r_good_default):
                continue
        else:
            need_split = True

        if need_split and (can_split_x or can_split_y):
            nx_mid = nx_sub // 2
            ny_mid = ny_sub // 2
            # 不重叠划分: 子块严格小于父块
            if can_split_x and can_split_y:
                x_splits = [x_idx[:nx_mid], x_idx[nx_mid:]]
                y_splits = [y_idx[:ny_mid], y_idx[ny_mid:]]
                for ys_part in y_splits:
                    for xs_part in x_splits:
                        if len(xs_part) > 0 and len(ys_part) > 0:
                            stack.append((xs_part, ys_part))
            elif can_split_x:
                for xs_part in (x_idx[:nx_mid], x_idx[nx_mid:]):
                    if len(xs_part) > 0:
                        stack.append((xs_part, y_idx))
            else:
                for ys_part in (y_idx[:ny_mid], y_idx[ny_mid:]):
                    if len(ys_part) > 0:
                        stack.append((x_idx, ys_part))
        else:
            xout, yout, zout = _block_representative(
                x, y, x_idx, y_idx, rows, cols, zdata,
                xmean, ymean, zmean, use_median_pixel,
            )
            n = _append_one_block(
                buf, n, xout, yout, zout, ngood, rms_true,
                float(x[x_idx[0]]), float(x[x_idx[-1]]),
                float(y[y_idx[0]]), float(y[y_idx[-1]]),
            )

    return n


def quad_decomp_mean2(x, y, z, threshold, Nres_min, Nres_max, min_stop_and=True):
    """兼容旧接口: Python 四叉树一次分解，返回 9 个数组。"""
    buf = _alloc_block_buf(z)
    n = _quad_decomp_mean2(
        x, y, z, threshold, Nres_min, Nres_max, buf, 0, min_stop_and=min_stop_and,
    )
    return _results_to_arrays(_blocks_view(buf, n))


# =============================================================================
# 3. 伴随栅格采样 —— file1 在 LOS 同一批盒子 / 代表点上取值
# =============================================================================


def make_look_downsample(xlook, ylook, zlook, xin, yin, xx1, xx2, yy1, yy2):
    """在每个盒子 [xx1,xx2]×[yy1,yy2] 内对 zlook 取有效均值。

    返回的 xout/yout 直接复制 xin/yin (通常是 LOS 块代表点坐标)。
    盒子内全 NaN 时对应 zout 为 NaN。
    """
    xlook = np.asarray(xlook, dtype=np.float64)
    ylook = np.asarray(ylook, dtype=np.float64)
    zlook = np.asarray(zlook, dtype=np.float64)
    n = len(xx1)
    xout = np.asarray(xin, dtype=np.float64).copy()
    yout = np.asarray(yin, dtype=np.float64).copy()
    zout = np.full(n, np.nan, dtype=np.float64)
    for k in range(n):
        i0 = int(np.searchsorted(xlook, xx1[k], side="left"))
        i1 = int(np.searchsorted(xlook, xx2[k], side="right"))
        j0 = int(np.searchsorted(ylook, yy1[k], side="left"))
        j1 = int(np.searchsorted(ylook, yy2[k], side="right"))
        if i0 >= i1 or j0 >= j1:
            continue
        blk = zlook[j0:j1, i0:i1]
        good = blk[~np.isnan(blk)]
        if good.size > 0:
            zout[k] = float(good.mean())
    return xout, yout, zout


def sample_grid_at_points(xgrid, ygrid, zgrid, xq, yq):
    """在规则网格上对查询点 (xq,yq) 做最近邻取值。

    ``median_pixel`` 模式下 LOS 代表点已落在真实格点上，
    file1 用此函数与代表像素对齐。
    """
    xgrid = np.asarray(xgrid, dtype=np.float64).ravel()
    ygrid = np.asarray(ygrid, dtype=np.float64).ravel()
    zgrid = np.asarray(zgrid, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64).ravel()
    yq = np.asarray(yq, dtype=np.float64).ravel()

    def _nn_index(grid, q):
        if grid.size == 1:
            return np.zeros(q.shape, dtype=np.int64)
        idx = np.searchsorted(grid, q, side="left")
        idx = np.clip(idx, 1, grid.size - 1)
        left = idx - 1
        use_right = np.abs(grid[idx] - q) <= np.abs(grid[left] - q)
        return np.where(use_right, idx, left).astype(np.int64)

    ix = _nn_index(xgrid, xq)
    iy = _nn_index(ygrid, yq)
    return zgrid[iy, ix].astype(np.float64, copy=False)


# =============================================================================
# 4. 高层入口 —— subsample / subsample1
# =============================================================================


def _load_grid_stats(file):
    """读栅格并打印全局统计，返回 (x, y, z, global_rms)。

    global_rms 常用作默认 initial_threshold。
    """
    xvec, yvec, zz = read_grd(file)
    valid = ~np.isnan(zz)
    vals = zz[valid]
    if vals.size == 0:
        raise ValueError("%r 无有效数据点" % (file,))
    global_mean = float(np.mean(vals))
    global_std = float(np.std(vals))
    global_rms = float(np.sqrt(np.mean((vals - global_mean) ** 2)))
    n_total = int(vals.size)
    print(
        "有效点数=%d, mean=%.4g, std=%.4g, rms=%.4g"
        % (n_total, global_mean, global_std, global_rms)
    )
    return xvec, yvec, zz, global_rms


def _run_quad_from_grid(
    xvec, yvec, zz, points_num, uplimit, downlimit, type_samp,
    initial_threshold, min_stop_mode, global_rms, representative="mean",
):
    """把 CLI 的 uplimit/downlimit 映射到 nres_max/nres_min 并跑迭代。

    未指定 initial_threshold 时用 global_rms。
    """
    thr = global_rms if initial_threshold is None else initial_threshold
    return iter_quad_downsample(
        xvec, yvec, zz, points_num, int(downlimit), int(uplimit), type_samp,
        initial_threshold=thr,
        min_stop_mode=min_stop_mode,
        representative=representative,
    )


def subsample(
    file, points_num, uplimit, downlimit, write_or_not, plot_or_not,
    type_samp=1, initial_threshold=None, min_stop_mode="and",
    representative="mean",
):
    """主入口: 对 LOS .grd 做四叉树下采样，写 ``*_py.llde``。"""
    xvec, yvec, zz, global_rms = _load_grid_stats(file)
    x, y, z, Npt, rms_out, xx1, xx2, yy1, yy2 = _run_quad_from_grid(
        xvec, yvec, zz, points_num, uplimit, downlimit, type_samp,
        initial_threshold, min_stop_mode, global_rms, representative,
    )
    weights = _weights_from_rms(rms_out)

    if plot_or_not == 1:
        _plot_subsample(x, y, z)

    print(f"number of subsampled data: {len(x)}")

    if write_or_not == 1:
        infile = Path(file)
        outfile = infile.parent / f"{infile.stem}_py.llde"
        _write_llde(outfile, x, y, z, weights)

    return x, y, z, weights


def subsample1(
    file, file1, points_num, uplimit, downlimit, write_or_not, plot_or_not,
    type_samp=1, initial_threshold=None, min_stop_mode="and",
    representative="mean",
):
    """用 file 的 LOS 决定四叉树块，在同一位置对 file1 取值。

    写出仍是 ``*_py.llde``，但 z 列来自 file1:
    - ``mean``: 盒子内均值
    - ``median_pixel``: 在 LOS 代表点处最近邻采样
    make_los_samp1.sh 依赖此路径。
    """
    xvec, yvec, zz, global_rms = _load_grid_stats(file)
    xvec1, yvec1, zz1 = read_grd(file1)
    if not (
        xvec.shape == xvec1.shape
        and yvec.shape == yvec1.shape
        and zz.shape == zz1.shape
    ):
        raise ValueError("file 与 file1 的网格尺寸不一致")

    x, y, z, Npt, rms_out, xx1, xx2, yy1, yy2 = _run_quad_from_grid(
        xvec, yvec, zz, points_num, uplimit, downlimit, type_samp,
        initial_threshold, min_stop_mode, global_rms, representative,
    )
    if _parse_representative(representative):
        z1 = sample_grid_at_points(xvec1, yvec1, zz1, x, y)
        x1, y1 = x, y
    else:
        x1, y1, z1 = make_look_downsample(
            xvec1, yvec1, zz1, x, y, xx1, xx2, yy1, yy2,
        )
    weights = _weights_from_rms(rms_out)

    if plot_or_not == 1:
        _plot_subsample(x, y, z)

    print(f"number of subsampled data: {len(x)}")

    if write_or_not == 1:
        infile = Path(file)
        outfile = infile.parent / f"{infile.stem}_py.llde"
        _write_llde(outfile, x1, y1, z1, weights)

    return x1, y1, z1, weights


# =============================================================================
# 5. 命令行入口
# =============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="InSAR quadtree downsample (LOS .grd)",
    )

    parser.add_argument("file", type=str, help="LOS .grd / .nc")
    parser.add_argument("points_num", type=int)
    parser.add_argument("uplimit", type=int, help="Nres_max (max block size in pixels)")
    parser.add_argument("downlimit", type=int, help="Nres_min (min block size in pixels)")
    parser.add_argument("write_or_not", type=int)
    parser.add_argument("plot_or_not", type=int)
    parser.add_argument(
        "type_samp",
        type=int,
        choices=(0, 1),
        help="0=Python 栈式 quad; 1=numba (推荐)",
    )
    parser.add_argument(
        "file1",
        nargs="?",
        default=None,
        type=str,
        help="(可选) 伴随栅格, 在 LOS 同一批盒子内取均值",
    )
    parser.add_argument(
        "--initial-threshold",
        dest="initial_threshold",
        default=None,
        type=float,
        help="quad 下采样初始 threshold; 未指定时用数据 global_rms",
    )
    parser.add_argument(
        "--min-stop-mode",
        dest="min_stop_mode",
        default="or",
        choices=("and", "or"),
        help="最小块停止条件: and=xy均达Nres_min(默认); or=MATLAB原版",
    )
    parser.add_argument(
        "--representative",
        dest="representative",
        default="mean",
        choices=("mean", "median_pixel"),
        help="块代表点: mean=块均值+质心; median_pixel=z近中位数的格点",
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
            min_stop_mode=args.min_stop_mode,
            representative=args.representative,
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
            min_stop_mode=args.min_stop_mode,
            representative=args.representative,
        )
