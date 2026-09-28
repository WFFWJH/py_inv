"""Regular-grid uniform downsampling for comparison with quad ``sample.py``.

Reads the same ``.grd`` / xyz inputs, picks nodes on a regular sub-grid with
(approximately) equal x/y coordinate spacing, and writes ``*_uniform.llde``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sample import read_grd  # noqa: E402


def _median_spacing(coord: np.ndarray) -> float:
    coord = np.asarray(coord, dtype=np.float64).ravel()
    d = np.diff(coord)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return 1.0
    return float(np.median(np.abs(d)))


def _count_valid(z: np.ndarray, stride_x: int, stride_y: int) -> int:
    return int(np.sum(~np.isnan(z[::stride_y, ::stride_x])))


def _pick_strides_for_target(
    z: np.ndarray,
    dx: float,
    dy: float,
    points_num: int,
    equal_coord_spacing: bool,
) -> Tuple[int, int]:
    """Return (stride_x, stride_y) with count close to ``points_num``."""
    ny, nx = z.shape
    points_num = max(1, int(points_num))

    if equal_coord_spacing:
        lo_s, hi_s = 0.0, max(nx * dx, ny * dy) * 2.0
        best = (1, 1)
        best_diff = abs(_count_valid(z, 1, 1) - points_num)
        for _ in range(64):
            s = 0.5 * (lo_s + hi_s)
            sx = max(1, int(round(s / dx)))
            sy = max(1, int(round(s / dy)))
            n = _count_valid(z, sx, sy)
            diff = abs(n - points_num)
            if diff < best_diff:
                best = (sx, sy)
                best_diff = diff
            if n > points_num:
                lo_s = s
            else:
                hi_s = s
        return best

    lo, hi = 1, max(nx, ny)
    best = 1
    best_diff = abs(_count_valid(z, 1, 1) - points_num)
    for _ in range(48):
        mid = (lo + hi) // 2
        n = _count_valid(z, mid, mid)
        diff = abs(n - points_num)
        if diff < best_diff:
            best = mid
            best_diff = diff
        if n > points_num:
            lo = mid + 1
        else:
            hi = mid
    return best, best


def uniform_downsample(
    x,
    y,
    z,
    points_num: int,
    *,
    stride_x: Optional[int] = None,
    stride_y: Optional[int] = None,
    equal_coord_spacing: bool = True,
):
    """Uniform sub-grid sampling on a regular ``(ny, nx)`` grid.

    Parameters
    ----------
    points_num
        Target number of valid output points (used when strides are not given).
    stride_x, stride_y
        If both set, use these strides directly and ignore ``points_num``.
    equal_coord_spacing
        If True (default), choose strides so ``stride_x * dx ≈ stride_y * dy``.
        If False, use the same index stride in x and y.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (y.size, x.size):
        raise ValueError(
            "z.shape %s 与 (len(y), len(x))=(%d, %d) 不一致"
            % (z.shape, y.size, x.size)
        )

    dx = _median_spacing(x)
    dy = _median_spacing(y)

    if stride_x is not None and stride_y is not None:
        sx = max(1, int(stride_x))
        sy = max(1, int(stride_y))
    else:
        sx, sy = _pick_strides_for_target(
            z, dx, dy, points_num, equal_coord_spacing,
        )

    xs, ys, zs, ws = [], [], [], []
    for j in range(0, y.size, sy):
        for i in range(0, x.size, sx):
            val = z[j, i]
            if np.isnan(val):
                continue
            xs.append(x[i])
            ys.append(y[j])
            zs.append(val)
            ws.append(1.0)

    x_out = np.asarray(xs, dtype=np.float64)
    y_out = np.asarray(ys, dtype=np.float64)
    z_out = np.asarray(zs, dtype=np.float64)
    w_out = np.asarray(ws, dtype=np.float64)

    spacing_x = sx * dx
    spacing_y = sy * dy
    info = {
        "nx": x.size,
        "ny": y.size,
        "dx": dx,
        "dy": dy,
        "stride_x": sx,
        "stride_y": sy,
        "spacing_x": spacing_x,
        "spacing_y": spacing_y,
        "n_out": x_out.size,
    }
    return x_out, y_out, z_out, w_out, info


def _print_spacing_report(info: dict) -> None:
    print(
        "输入网格: nx=%d, ny=%d, dx=%.6g, dy=%.6g"
        % (info["nx"], info["ny"], info["dx"], info["dy"])
    )
    print(
        "均匀采样: stride_x=%d, stride_y=%d, 坐标间距 x=%.6g, y=%.6g"
        % (
            info["stride_x"],
            info["stride_y"],
            info["spacing_x"],
            info["spacing_y"],
        )
    )
    ratio = info["spacing_x"] / info["spacing_y"] if info["spacing_y"] else np.nan
    print("行列坐标间距比 (x/y): %.4f" % ratio)
    print("输出点数: %d" % info["n_out"])


def subsample_uniform(
    file,
    points_num,
    write_or_not,
    plot_or_not,
    *,
    stride_x: Optional[int] = None,
    stride_y: Optional[int] = None,
    equal_coord_spacing: bool = True,
):
    xvec, yvec, zz = read_grd(file)

    valid = ~np.isnan(zz)
    vals = zz[valid]
    n_total = int(valid.sum())
    print(
        "有效点数=%d, mean=%.4g, std=%.4g"
        % (n_total, np.mean(vals), np.std(vals))
    )

    x, y, z, w, info = uniform_downsample(
        xvec,
        yvec,
        zz,
        points_num,
        stride_x=stride_x,
        stride_y=stride_y,
        equal_coord_spacing=equal_coord_spacing,
    )
    _print_spacing_report(info)

    if plot_or_not == 1:
        plt.figure()
        cmap = plt.get_cmap("jet", 200)
        plt.scatter(x, y, c=z, cmap=cmap, s=10)
        plt.colorbar(label="z")
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.gca().set_aspect("equal")
        plt.title(
            "Uniform Subsample  (stride %d x %d)"
            % (info["stride_x"], info["stride_y"])
        )
        plt.show()

    print("number of subsampled data: %d" % len(x))

    if write_or_not == 1:
        infile = Path(file)
        outfile = infile.parent / f"{infile.stem}_uniform.llde"
        print(outfile)
        with open(outfile, "w", encoding="utf-8") as fid:
            for j in range(len(x)):
                fid.write(
                    f"{x[j]:.9f}\t{y[j]:.9f}\t{z[j]:.9f}\t{w[j]:.9f}\n"
                )

    return x, y, z, w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regular uniform downsampling (compare spacing with sample.py)",
    )
    parser.add_argument("file", type=str, help="input .grd or xyz text")
    parser.add_argument("points_num", type=int, help="target number of valid points")
    parser.add_argument("write_or_not", type=int, help="1=write *_uniform.llde")
    parser.add_argument("plot_or_not", type=int, help="1=show scatter plot")
    parser.add_argument(
        "--stride-x",
        type=int,
        default=None,
        help="fixed x stride (grid columns); with --stride-y skips auto search",
    )
    parser.add_argument(
        "--stride-y",
        type=int,
        default=None,
        help="fixed y stride (grid rows)",
    )
    parser.add_argument(
        "--index-stride",
        action="store_true",
        help="use same index stride in x/y (ignore equal coordinate spacing)",
    )

    args = parser.parse_args()
    if (args.stride_x is None) ^ (args.stride_y is None):
        parser.error("--stride-x and --stride-y must be given together")

    subsample_uniform(
        file=args.file,
        points_num=args.points_num,
        write_or_not=args.write_or_not,
        plot_or_not=args.plot_or_not,
        stride_x=args.stride_x,
        stride_y=args.stride_y,
        equal_coord_spacing=not args.index_stride,
    )
