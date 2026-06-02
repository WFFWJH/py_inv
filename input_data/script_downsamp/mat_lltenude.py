#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat, savemat
import argparse
import os
import glob

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
def _xy2ll(xi, yi, lon_c):
    """Inverse of `_ll2xy`: projected TM coords -> lon/lat (degrees).

    Uses the same WGS84 ellipsoid, k0=0.9996, false easting 5e5, and
    y_0=0 convention as `_ll2xy` (no southern false northing).
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
    r_fe = 5e5

    r_ep2 = r_e2 / (1.0 - r_e2)
    r_e4 = r_e2 * r_e2
    r_e6 = r_e4 * r_e2

    r_vu_x = xi_arr - r_fe
    r_vu_y = yi_arr

    r_lon0 = lon_c * r_dtor

    r_et = np.sqrt(1.0 - r_e2)
    r_e1 = (1.0 - r_et) / (1.0 + r_et)
    r_e12 = r_e1 * r_e1
    r_e13 = r_e1 * r_e12
    r_e14 = r_e1 * r_e13

    r_m = r_vu_y / r_k0
    r_mu = r_m / (r_a * (1.0 - r_e2 / 4.0 - 3.0 * r_e4 / 64.0 - 5.0 * r_e6 / 256.0))
    r_lat1 = (
        r_mu
        + (3.0 * r_e1 / 2.0 - 27.0 * r_e13 / 32.0) * np.sin(2.0 * r_mu)
        + (21.0 * r_e12 / 16.0 - 55.0 * r_e14 / 32.0) * np.sin(4.0 * r_mu)
        + (51.0 * r_e13 / 96.0) * np.sin(6.0 * r_mu)
        + (1097.0 * r_e14 / 512.0) * np.sin(8.0 * r_mu)
    )

    sin_lat1 = np.sin(r_lat1)
    cos_lat1 = np.cos(r_lat1)
    tan_lat1 = np.tan(r_lat1)

    denom = np.sqrt(1.0 - r_e2 * sin_lat1 * sin_lat1)
    r_n = r_a / denom
    r_r = (r_a * (1.0 - r_e2)) / (denom**3)
    r_t = tan_lat1 * tan_lat1
    r_t2 = r_t * r_t
    r_c = r_ep2 * cos_lat1 * cos_lat1
    r_c2 = r_c * r_c

    r_d = r_vu_x / (r_n * r_k0)
    r_d2 = r_d * r_d
    r_d3 = r_d2 * r_d
    r_d4 = r_d3 * r_d
    r_d5 = r_d4 * r_d
    r_d6 = r_d5 * r_d

    yo = r_lat1 - (r_n * tan_lat1 / r_r) * (
        r_d2 / 2.0
        - (5.0 + 3.0 * r_t + 10.0 * r_c - 4.0 * r_c2 - 9.0 * r_ep2) * r_d4 / 24.0
        + (61.0 + 90.0 * r_t + 298.0 * r_c + 45.0 * r_t2 - 252.0 * r_ep2 - 3.0 * r_c2)
        * r_d6
        / 720.0
    )
    xo = r_lon0 + (
        r_d
        - (1.0 + 2.0 * r_t + r_c) * r_d3 / 6.0
        + (5.0 - 2.0 * r_c + 28.0 * r_t - 3.0 * r_c2 + 8.0 * r_ep2 + 24.0 * r_t2)
        * r_d5
        / 120.0
    ) / cos_lat1

    xo = xo / r_dtor
    yo = yo / r_dtor

    if scalar_input:
        return float(xo[0]), float(yo[0])
    return xo, yo

def convert_xy2ll(data, ref):
    x = data[:, 0]
    y = data[:, 1]

    
    x0,y0 = _ll2xy(95.33,19.61,ref)
    # 平移到局部坐标
    x += x0
    y += y0

    lon,lat = _xy2ll(x, y, ref)
    data[:, 0] = lon
    data[:, 1] = lat
#    data[:, 6] = data[:,6]/100

    return data


def mat_to_lltenude(mat_file, out_file=None, precision=10):
    """MAT → .lltenude"""

    data = loadmat(mat_file)

    try:
        sampled = convert_xy2ll(data['sampled_insar_data'],95)   # Nx6
        dem_out = data['dem_out'].reshape(-1, 1)
        rms_out = data['rms_out'].reshape(-1, 1)
    except KeyError as e:
        raise KeyError(f"{mat_file} missing variable: {e}")

    # 拼接 Nx8
    out = np.hstack((sampled, dem_out, rms_out))
    out = np.hstack((
     sampled[:,0:2],
     dem_out,
     sampled[:,3:6],
     sampled[:,2:3],
     rms_out
     ))


    if out_file is None:
        out_file = os.path.splitext(mat_file)[0] + '.lltenude'

    fmt = f'%.{precision}f'

    np.savetxt(
        out_file,
        out,
        fmt=fmt,
        header='xsar ysar zout ve vn vz dem_out rms_out'
    )

    print(f"[MAT → TXT] Saved: {out_file}")


def convert_lonlat_to_xy(data, ref):
    lon = data[:, 0]
    lat = data[:, 1]

    x, y = _ll2xy(lon, lat, ref)
    x0,y0 = _ll2xy(95.33,19.61,ref)
    # 平移到局部坐标
    x -= x0
    y -= y0

    data[:, 0] = x
    data[:, 1] = y
#    data[:, 6] = data[:,6]/100

    return data

def lltenude_to_mat(txt_file, out_mat=None):
    """.lltenude → MAT"""

    data = np.loadtxt(txt_file)

    if data.shape[1] != 8:
        raise ValueError(f"{txt_file} must have 8 columns")

    data = convert_lonlat_to_xy(data,95)
    sampled = np.hstack([
     data[:,0:2],
     data[:,6:7],
     data[:,3:6]
    ])
    dem_out = np.ascontiguousarray(data[:, 2:3], dtype=np.float64)
    rms_out = np.ascontiguousarray(data[:, 7:8], dtype=np.float64)

    if out_mat is None:
        out_mat = os.path.splitext(txt_file)[0] + '.mat'

    savemat(out_mat, {
        'sampled_insar_data': sampled,
        'dem_out': dem_out,
        'rms_out': rms_out,
    }, oned_as='column')

    print(f"[TXT → MAT] Saved: {out_mat}")

def extract_track(filename):
    """
    从文件名提取轨道号
    A143_azi.lltenude → A143
    """
    base = os.path.basename(filename)
    return base.split('_')[0]


def main():
    input_dir = "azi_best_1"
    work_dir = "WORK"

    files = glob.glob(os.path.join(input_dir, "*.lltenude"))

    if not files:
        print("No .lltenude files found.")
        return

    for f in files:
        track = extract_track(f)

        # 构造目标目录
        target_dir = os.path.join(work_dir, track, "AZI")

        if not os.path.isdir(target_dir):
            print(f"[WARNING] Skip {f}, target dir not exist: {target_dir}")
            continue

        # 输出文件名
        out_mat = os.path.join(
            target_dir,"los_samp0.mat"
           # os.path.basename(f).replace(".lltenude", ".mat")
        )

        print(out_mat)
        try:
            lltenude_to_mat(f, out_mat)
            print(f"[OK] {f} → {out_mat}")
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

def main1():
    import shutil

    shutil.copy("./WORK/A143/AZI/los_samp0.mat", "./a143_los.mat")
    shutil.copy("./bak_WORK/A143/AZI/los_samp0.mat", "./a143_los1.mat")
#    input = "./los_samp0.mat"
#    input = "./a.mat"
#    mat_to_lltenude(input)
    mat_to_lltenude("./a143_los.mat")
    mat_to_lltenude("./a143_los1.mat")

def main2():
    input = "./D106_azi.lltenude"
    lltenude_to_mat(input)
def main3():
    import argparse
    parser = argparse.ArgumentParser()

    # 位置参数（按顺序）
    parser.add_argument("input_file", type=str)

    parser.add_argument("output_file", type=str)

    args = parser.parse_args()

    lltenude_to_mat(args.input_file,args.output_file)

def main4():
    import argparse
    parser = argparse.ArgumentParser()

    # 位置参数（按顺序）
    parser.add_argument("input_file", type=str)

    parser.add_argument("output_file", type=str)

    args = parser.parse_args()

    mat_to_lltenude(args.input_file,args.output_file)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--mat2txt", action="store_true",
                        help="convert mat to lltenude")

    args = parser.parse_args()

    (mat_to_lltenude if args.mat2txt else lltenude_to_mat)(
        args.input_file, args.output_file
    )
