import math
from statistics import linear_regression
from typing import Optional

import numpy as np

from geodetic.coords import ll2xy, xy2xy


def load_fault_one_plane(
    fault_segment_file,
    lonc=-117.5,
    latc=35.5,
    fault_id=0,
    width=30e3,
    layers=5,
    len_top=2e3,
    l_ratio=1.3,
    w_ratio=1.3,
    depth_start=0.0,
    dip: Optional[np.ndarray] = None,
    ref_lon=None,
    coord_mode="lonlat",
):
    lon_eq = lonc
    lat_eq = latc
    W = float(width)
    n_layer = int(layers)
    lp_top = float(len_top)
    bias_lp = float(l_ratio)
    bias_wp = float(w_ratio)
    zstart = float(depth_start)
    if ref_lon is None:
        ref_lon = lon_eq

    if dip is None:
        dips = np.array([], dtype=np.float64)
    else:
        dips = np.asarray(dip, dtype=np.float64).reshape(-1)

    fault_data = np.loadtxt(fault_segment_file, dtype=np.float64)
    if fault_data.ndim == 1:
        fault_data = fault_data.reshape(1, -1)
    if fault_data.shape[1] < 4:
        raise ValueError("fault_segment_file must contain at least 4 columns")

    nflt = fault_data.shape[0]
    if dips.size != nflt:
        raise ValueError(f"'dip' length ({dips.size}) must equal fault segment count ({nflt})")

    if coord_mode == "lonlat":
        xo, yo = ll2xy(lon_eq, lat_eq, ref_lon)
        lon_pt = np.concatenate([fault_data[:, 0], fault_data[:, 2]])
        lat_pt = np.concatenate([fault_data[:, 1], fault_data[:, 3]])
        xutm_pt, yutm_pt = ll2xy(lon_pt, lat_pt, ref_lon)
        xpt = xutm_pt - xo
        ypt = yutm_pt - yo
    elif coord_mode == "local_xy":
        # 4 columns: x1 y1 x2 y2 (m), relative to the same origin as InSAR / slip_model.
        xpt = np.concatenate([fault_data[:, 0], fault_data[:, 2]])
        ypt = np.concatenate([fault_data[:, 1], fault_data[:, 3]])
    else:
        raise ValueError("coord_mode must be 'lonlat' or 'local_xy'")

    wp_factor = np.array([bias_wp ** k for k in range(n_layer)], dtype=np.float64)
    wp_top = W / np.sum(wp_factor)
    wp_layer = np.array([wp_top * (bias_wp ** j) for j in range(n_layer)], dtype=np.float64)

    d2r = np.pi / 180.0
    strikes = np.zeros(nflt, dtype=np.float64)
    thetas = np.zeros(nflt, dtype=np.float64)
    xo_segments = np.zeros(nflt, dtype=np.float64)
    yo_segments = np.zeros(nflt, dtype=np.float64)
    n_per_layer = np.zeros((nflt, n_layer), dtype=np.int64)
    lp_this_layer = np.zeros((nflt, n_layer), dtype=np.float64)

    for i in range(nflt):
        xstart = xpt[i + nflt]
        ystart = ypt[i + nflt]
        xend = xpt[i]
        yend = ypt[i]
        dx = xend - xstart
        dy = yend - ystart
        theta = np.arctan2(dy, dx)
        xseg, yseg = xy2xy(xstart, ystart, theta)
        xo_segments[i] = xseg
        yo_segments[i] = yseg

        strike1 = 90.0 - theta / d2r
        if strike1 < 0.0:
            strike1 += 360.0
        thetas[i] = theta
        strikes[i] = strike1

        L = math.sqrt(dx * dx + dy * dy)
        for j in range(n_layer):
            lp_rough = lp_top * (bias_lp ** j)
            n_this_layer = int(np.round(L / lp_rough))
            if n_this_layer == 0:
                n_this_layer = 1
            n_per_layer[i, j] = n_this_layer
            lp_this_layer[i, j] = L / n_this_layer

    total_patches = int(np.sum(n_per_layer))
    slip_model = np.zeros((total_patches, 13), dtype=np.float64)
    index_all_patches = 0
    current_fault_id = int(fault_id)

    wp_cum = np.cumsum(wp_layer)
    for i in range(nflt):
        current_fault_id += 1
        dip_i = float(dips[i])
        theta = thetas[i]
        xo_segment = xo_segments[i]
        yo_segment = yo_segments[i]
        indx_patch = 0

        for j in range(n_layer):
            n_this_layer = int(n_per_layer[i, j])
            depth_offset = wp_cum[j] - wp_layer[j]
            for k in range(n_this_layer):
                indx_patch += 1
                xpatch_tmp = xo_segment + k * lp_this_layer[i, j]
                ypatch_tmp = yo_segment - np.cos(np.deg2rad(dip_i)) * depth_offset
                xp, yp = xy2xy(xpatch_tmp, ypatch_tmp, -theta)
                zp = (-depth_offset) * np.sin(np.deg2rad(dip_i)) + zstart

                slip_model[index_all_patches, 0:10] = [
                    current_fault_id,
                    indx_patch,
                    j + 1,
                    xp,
                    yp,
                    zp,
                    lp_this_layer[i, j],
                    wp_layer[j],
                    strikes[i],
                    dip_i,
                ]
                index_all_patches += 1

    return slip_model

if __name__ == "__main__":
    import os
    # 项目根目录（本包位于 <root>/geodetic/ 下）
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fault_file = os.path.join(_PROJECT_ROOT, "checkerboard.txt")

    width = 20e3
    length = 50e3
    len_top = 4e3
    n_layer = int(length/len_top)
    layers = 5

    slip_model = load_fault_one_plane(fault_file,dip=[80],
    lonc=95.33,
    latc=19.61,
    ref_lon=95,
    l_ratio=1,
    w_ratio=1,
    width=width,
    len_top=len_top,
    layers=layers,
    coord_mode="local_xy"
    );

    slip_matrix = np.zeros((layers, n_layer))
    one_layer_slip = np.zeros(n_layer)
    next_layer_slip = np.zeros(n_layer)
    for i in range(n_layer):
        if i%2 == 0:
            one_layer_slip[i] = 1
            next_layer_slip[i] = 0
        else:
            one_layer_slip[i] = 0
            next_layer_slip[i] = 1
    for i in range(layers):
        if i%2 == 0:
            slip_matrix[i, :] = one_layer_slip
        else:
            slip_matrix[i, :] = next_layer_slip
    for i in range(layers):
        for j in range(n_layer):
            slip_model[i*n_layer+j, 11] = slip_matrix[i, j]

    import matplotlib
    print("backend =", matplotlib.get_backend())
    print("SHOW_SLIP =", __import__("os").environ.get("SHOW_SLIP"))
    from geodetic.show_slip_model import show_slip_model

    show_slip_model(
        slip_model,
        ref_lon=95, lonc=95.33, latc=19.61,
        axis_range=[0, 20, 0, 100, -20, 0],
        apply_axis_range=True,
        out_path="fault_one_plane.png",
        block=False,  # 立刻返回; 脚本结束前会自动等你关掉图窗
    )
    # 后面可以继续写代码; 图窗会一直开着, 直到进程退出前由 show_slip_model 挂起等待
    print("继续运行 ... (关掉图窗后进程才会结束)", flush=True)