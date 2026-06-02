import math
from typing import Optional

import numpy as np


def _ll2xy(xi, yi, lon_c):
    """Transverse Mercator (WGS84, k0=0.9996, x_0=5e5, y_0=0).

    Southern hemisphere returns negative y directly (matches pyproj's
    `+proj=tmerc +y_0=0` convention, no false-northing folding).
    """
    xi_arr = np.asarray(xi, dtype=np.float64)
    yi_arr = np.asarray(yi, dtype=np.float64)
    scalar_input = (xi_arr.ndim == 0) and (yi_arr.ndim == 0)
    xi_arr = np.atleast_1d(xi_arr)
    yi_arr = np.atleast_1d(yi_arr)

    r_dtor = np.pi / 180.0
    r_a = 6378137.0
    r_e2 = 0.006694379990141
    r_k0 = 0.9996
    r_lat0 = 0.0
    r_fe = 5e5
    r_fn = 0.0

    r_ep2 = r_e2 / (1.0 - r_e2)
    r_e4 = r_e2 * r_e2
    r_e6 = r_e4 * r_e2

    xi_rad = xi_arr * r_dtor
    yi_rad = yi_arr * r_dtor
    r_lon0 = lon_c * r_dtor

    sin_yi = np.sin(yi_rad)
    cos_yi = np.cos(yi_rad)
    tan_yi = np.tan(yi_rad)

    r_n = r_a / np.sqrt(1.0 - r_e2 * sin_yi * sin_yi)
    r_t = tan_yi * tan_yi
    r_t2 = r_t * r_t
    r_c = r_ep2 * cos_yi * cos_yi
    r_ba = (xi_rad - r_lon0) * cos_yi

    r_a2 = r_ba * r_ba
    r_a3 = r_ba * r_a2
    r_a4 = r_ba * r_a3
    r_a5 = r_ba * r_a4
    r_a6 = r_ba * r_a5

    r_m = r_a * (
        (1.0 - r_e2 / 4.0 - 3.0 * r_e4 / 64.0 - 5.0 * r_e6 / 256.0) * yi_rad
        - (3.0 * r_e2 / 8.0 + 3.0 * r_e4 / 32.0 + 45.0 * r_e6 / 1024.0) * np.sin(2.0 * yi_rad)
        + (15.0 * r_e4 / 256.0 + 45.0 * r_e6 / 1024.0) * np.sin(4.0 * yi_rad)
        - (35.0 * r_e6 / 3072.0) * np.sin(6.0 * yi_rad)
    )

    r_m0 = r_a * (
        (1.0 - r_e2 / 4.0 - 3.0 * r_e4 / 64.0 - 5.0 * r_e6 / 256.0) * r_lat0
        - (3.0 * r_e2 / 8.0 + 3.0 * r_e4 / 32.0 + 45.0 * r_e6 / 1024.0) * np.sin(2.0 * r_lat0)
        + (15.0 * r_e4 / 256.0 + 45.0 * r_e6 / 1024.0) * np.sin(4.0 * r_lat0)
        - (35.0 * r_e6 / 3072.0) * np.sin(6.0 * r_lat0)
    )

    xo = r_k0 * r_n * (
        r_ba
        + (1.0 - r_t + r_c) * r_a3 / 6.0
        + (5.0 - 18.0 * r_t + r_t2 + 72.0 * r_c - 58.0 * r_ep2) * r_a5 / 120.0
    )
    xo = xo + r_fe

    yo = r_k0 * (
        r_m
        - r_m0
        + r_n
        * tan_yi
        * (
            r_a2 / 2.0
            + (5.0 - r_t + 9.0 * r_c + 4.0 * r_c * r_c) * (r_a4 / 24.0)
            + (61.0 - 58.0 * r_t + r_t2 + 600.0 * r_c - 330.0 * r_ep2) * (r_a6 / 720.0)
        )
    )
    yo = yo + r_fn

    if scalar_input:
        return float(xo[0]), float(yo[0])
    return xo, yo


def _xy2xy(x1, y1, phi):
    x2 = np.cos(phi) * x1 + np.sin(phi) * y1
    y2 = -np.sin(phi) * x1 + np.cos(phi) * y1
    return x2, y2


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
        raise ValueError("fault_segment_file must contain at least 4 columns: lon1 lat1 lon2 lat2")

    nflt = fault_data.shape[0]
    if dips.size != nflt:
        raise ValueError(f"'dip' length ({dips.size}) must equal fault segment count ({nflt})")

    xo, yo = _ll2xy(lon_eq, lat_eq, ref_lon)
    lon_pt = np.concatenate([fault_data[:, 0], fault_data[:, 2]])
    lat_pt = np.concatenate([fault_data[:, 1], fault_data[:, 3]])
    xutm_pt, yutm_pt = _ll2xy(lon_pt, lat_pt, ref_lon)
    xpt = xutm_pt - xo
    ypt = yutm_pt - yo

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
        xseg, yseg = _xy2xy(xstart, ystart, theta)
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
                xp, yp = _xy2xy(xpatch_tmp, ypatch_tmp, -theta)
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
    fault_file = os.path.join(os.path.dirname(__file__), "fault_trace.txt")
    slip_model = load_fault_one_plane(fault_file,dip=[60,60,60,70],
    lonc=95.33,
    latc=19.61,
    ref_lon=95,
    w_ratio=1.2,
    width=20e3,
    len_top=2e3,
    layers=5);


    print(slip_model)