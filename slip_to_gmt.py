r"""把 slip_model 转成 GMT ``plot3d -C -L`` / ``plot -C -L`` 能直接画的多段多边形数据.

输出 1: slip_3dgrid.gmt  — 三维有限断层模型
    每段 = 一个片元的 4 个角点, 经度/纬度/深度(km, 向下为负), 段头 ``> -Z<滑移量>``.

    # 数据头段
    >  -Z5.900000e+00
    95.9319  22.0100  -0.0000
    95.9500  22.0300  -0.0000
    95.9502  22.0298  -3.2430
    95.9321  22.0098  -3.2430

输出 2: slip_2dgrid.gmt  — 二维展开剖面
    段内 4 个角点为 (沿走向里程 km, 沿倾向距离 km, 0), x/y 与
    ``show_slip_model_2d.unfold_patches`` 完全一致 (段序倒排, 段间无空隙),
    所以 GMT 出的二维图和 matplotlib 出的二维图可以直接对照.

运行:
    python slip_to_gmt.py                 # 默认 iint0, 滑移量单位 m
    python slip_to_gmt.py --unit cm       # Z 值换成 cm
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from geodetic.config import resolve_geo_origin
from geodetic.coords import ll2xy, xy2ll, xy2xy
from geodetic.show_slip_model import load_slip_model_from_file
from geodetic.show_slip_model_2d import unfold_patches

HERE = os.path.dirname(os.path.abspath(__file__))
D2R = np.pi / 180.0


# --------------------------------------------------------------------------- #
def patch_corners_ll(slip: np.ndarray, ref_lon: float, lonc: float, latc: float):
    """每个片元的 4 个角点 -> (lon, lat, depth_km), 形状 (N, 4, 3).

    角点顺序 (与 GMT 示例一致, 顺/逆时针均可):
        p1 上边界起点 -> p2 上边界终点 -> p3 下边界终点 -> p4 下边界起点
    """
    data = np.asarray(slip, dtype=np.float64)
    xe, yn, zr = data[:, 3], data[:, 4], data[:, 5]      # m, 相对 (lonc,latc); zr 向上为正、地表为 0
    lp, wp = data[:, 6], data[:, 7]                      # 沿走向长 / 沿倾向宽, m
    strike, dip = data[:, 8], data[:, 9] * D2R

    n = data.shape[0]
    out = np.empty((n, 4, 3), dtype=np.float64)

    x0, y0 = ll2xy(lonc, latc, ref_lon)
    x0 = float(np.ravel(x0)[0])
    y0 = float(np.ravel(y0)[0])

    for i in range(n):
        theta = (90.0 - strike[i]) * D2R
        # 断层局部系: x' 沿走向, y' 沿走向法向
        x1f, y1f = xy2xy(xe[i], yn[i], theta)
        x2f, y2f = x1f + lp[i], y1f
        x3f, y3f = x2f, y2f - wp[i] * np.cos(dip[i])
        x4f, y4f = x1f, y3f

        z1 = z2 = zr[i]
        z3 = z4 = zr[i] - wp[i] * np.sin(dip[i])          # 向下为负

        corners_f = ((x1f, y1f), (x2f, y2f), (x3f, y3f), (x4f, y4f))
        xy = np.array([xy2xy(x, y, -theta) for x, y in corners_f], dtype=np.float64)
        lon, lat = xy2ll(xy[:, 0] + x0, xy[:, 1] + y0, ref_lon)
        out[i, :, 0] = np.ravel(lon)
        out[i, :, 1] = np.ravel(lat)
        out[i, :, 2] = np.array([z1, z2, z3, z4]) / 1000.0   # m -> km
    return out


# --------------------------------------------------------------------------- #
def write_gmt_poly(path: str, verts: np.ndarray, zvals: np.ndarray,
                   header: str = "", ndim: int = 3) -> None:
    """按 GMT 多段格式写出: ``> -Z<val>`` + 每行一个顶点."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if header:
            fh.write("# %s\n" % header)
        for i in range(verts.shape[0]):
            fh.write(">  -Z%.6e\n" % zvals[i])
            for v in verts[i]:
                if ndim == 3:
                    fh.write("%.5f  %.5f  %.4f\n" % (v[0], v[1], v[2]))
                else:
                    fh.write("%.5f  %.5f\n" % (v[0], v[1]))


def write_slip_model_gmt(
    slip: np.ndarray,
    *,
    prefix: str,
    ref_lon: float,
    lonc: float,
    latc: float,
    unit: str = "m",
    out3d: Optional[str] = None,
    out2d: Optional[str] = None,
) -> Tuple[str, str]:
    """Write 3-D and unfolded 2-D GMT polygons for an existing slip solution."""
    if unit not in ("m", "cm"):
        raise ValueError("unit must be 'm' or 'cm'")
    data = np.asarray(slip, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 13:
        raise ValueError("slip_model 期望形状 (N, >=13)")

    scale = 100.0 if unit == "cm" else 1.0
    zvals = np.hypot(data[:, 11], data[:, 12]) * scale
    vertices_3d = patch_corners_ll(data, ref_lon, lonc, latc)
    unfolded = unfold_patches(data, seg_gap_km=0.0, along="dip", seg_order="desc")
    vertices_2d = np.asarray(unfolded["rects"], dtype=np.float64)

    path_3d = out3d or prefix + "_3dgrid.gmt"
    path_2d = out2d or prefix + "_2dgrid.gmt"
    write_gmt_poly(
        path_3d,
        vertices_3d,
        zvals,
        header="lon  lat  depth(km, negative down)  -Z slip(%s)" % unit,
        ndim=3,
    )
    write_gmt_poly(
        path_2d,
        vertices_2d,
        zvals,
        header="along-strike(km)  along-dip(km)  -Z slip(%s)" % unit,
        ndim=2,
    )
    return path_3d, path_2d


def plot_gmt_files(prefix: str, *, project_root: Optional[str] = None) -> None:
    """Render the GMT files for ``prefix`` using the installed GMT executable."""
    root = Path(project_root or HERE)
    stem = Path(prefix)
    if not stem.is_absolute():
        stem = root / stem
    inputs = (Path(str(stem) + "_3dgrid.gmt"), Path(str(stem) + "_2dgrid.gmt"))
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    def run_gmt(*args: str) -> None:
        subprocess.run(["gmt", *args], cwd=root, check=True)

    run_gmt("begin", str(stem) + "_gmt_3d", "png")
    run_gmt("set", "MAP_FRAME_TYPE", "plain")
    run_gmt("set", "MAP_GRID_PEN", "0p,gray,-")
    run_gmt("set", "FONT_ANNOT_PRIMARY", "8p", "FONT_LABEL", "9p")
    run_gmt("set", "MAP_ANNOT_OFFSET_PRIMARY", "2p", "MAP_FRAME_PEN", "thinner", "MAP_TICK_LENGTH", "2p")
    run_gmt(
        "basemap",
        "-R95.90/96.50/18.35/22.85/-20/0",
        "-JX6c/16c",
        "-JZ2c",
        '-Bxa0.2fg+lLongitude (deg)',
        '-Bya0.5fg+lLatitude (deg)',
        '-Bza5f1g5+lDepth (km)',
        "-BwSEnZ",
        "-p100/20",
    )
    run_gmt("makecpt", "-Cwhite,lightblue,yellow,red", "-T0/6.5/0.2", "-Z")
    run_gmt("plot3d", str(inputs[0]), "-C", "-L", "-W0p,gray", "-p100/20")
    run_gmt(
        "colorbar",
        "-C",
        "-DjBR+w3c/0.3c+h+o1.5c/1.3c+ml",
        "-Ba1",
        '-Bx+lSlip (m)',
    )
    run_gmt("end", "show")

    run_gmt("begin", str(stem) + "_gmt_2d", "png")
    run_gmt("set", "MAP_FRAME_TYPE", "plain")
    run_gmt("set", "FONT_ANNOT_PRIMARY", "5p", "FONT_LABEL", "6p")
    run_gmt("set", "MAP_ANNOT_OFFSET_PRIMARY", "2p", "MAP_FRAME_PEN", "thinner", "MAP_TICK_LENGTH", "2p")
    run_gmt(
        "basemap",
        "-R0/478/0/20",
        "-JX20c/-2c",
        "-Bxa50f25+lAlong-strike distance (km)",
        "-Bya5f2.5+lAlong-dip distance (km)",
        "-BWS",
    )
    run_gmt("makecpt", "-Cwhite,lightblue,yellow,red", "-T0/6.5/0.2", "-Z")
    run_gmt("plot", str(inputs[1]), "-C", "-L", "-W0p,gray")
    run_gmt(
        "colorbar",
        "-C",
        "-DJMR+w2c/0.2c+v+o0.2c/0c+ml",
        "-Ba1",
        "-Bx+lSlip (m)",
    )
    run_gmt("end", "show")
    print("done: %s_gmt_3d.png  %s_gmt_2d.png" % (stem, stem))


def main() -> None:
    ap = argparse.ArgumentParser(description="slip_model -> GMT 多边形数据")
    ap.add_argument("--mat", default=os.path.join(HERE, "py_inversion_iint0.mat"))
    ap.add_argument("--prefix", default=os.path.join(HERE, "slip"))
    ap.add_argument("--plot-prefix", help="只绘制此前生成的 GMT 文件")
    ap.add_argument("--out3d")
    ap.add_argument("--out2d")
    ap.add_argument("--unit", default="m", choices=("m", "cm"),
                    help="Z 值 (滑移量) 单位, 默认 m")
    ap.add_argument("--ref-lon", type=float, default=None)
    ap.add_argument("--lonc", type=float, default=None)
    ap.add_argument("--latc", type=float, default=None)
    args = ap.parse_args()

    if args.plot_prefix:
        plot_gmt_files(args.plot_prefix)
        return

    ref_lon, lonc, latc = resolve_geo_origin(args.ref_lon, args.lonc, args.latc)
    slip = load_slip_model_from_file(args.mat)
    print("slip_model %s | origin ref_lon=%.4f lonc=%.4f latc=%.4f"
          % (slip.shape, ref_lon, lonc, latc))

    out3d, out2d = write_slip_model_gmt(
        slip,
        prefix=args.prefix,
        ref_lon=ref_lon,
        lonc=lonc,
        latc=latc,
        unit=args.unit,
        out3d=args.out3d,
        out2d=args.out2d,
    )
    zvals = np.hypot(slip[:, 11], slip[:, 12]) * (100.0 if args.unit == "cm" else 1.0)

    for path in (out3d, out2d):
        print("写出 %s" % path)
    print(
        "%d 段 | slip:[%.3f, %.3f] %s"
        % (slip.shape[0], zvals.min(), zvals.max(), args.unit)
    )


if __name__ == "__main__":
    main()
