r"""3D 断层滑移模型可视化 — 对应 show_slip_model.m.

``slip_model`` 与 MATLAB 一致: 行=patch, 列 1–10(1-based) 为 idx 与几何,
  列 11(1-based) 为 ``tp`` 等, 列 12–13 为走滑/倾滑 (m, 代码内 /100) — Python 下标 11, 12.

**与 InversionExample 一致的重要约定**

- `InversionExample.m` 中调用 ``show_slip_model(..., 'axis_range', [50 150 -180 350 -20 0])``;
  原 .m 里 ``axis(axis_range)`` **被注释**（不裁剪坐标轴）; 本函数默认
  **不应用** ``axis_range``，否则会把断层格点裁到框外、看起来像「没有断层」.
  若需固定视窗, 设 ``apply_axis_range=True``。
- 地表迹线: MATLAB 原 .m 在 ``fault_file`` 非空时用工作区里 ``lonf``/``latf``/``LS``
  绘图, 与 ``load`` 的断层列向量约定一致, 与 ``plot_insar_*.m`` 相同:
  ``lonf = [lon1; lon3]``, ``latf = [lat2; lat4]`` (见 ``load_fault_one_plane.m``).
  此处改为直接读取与 ``load_fault_one_plane`` 相同的 **4 列文件**
  (每行 ``lon1 lat1 lon2 lat2``) 在 ``z=0`` 上画黑线, 等效于逐段
  ``[lonf(ii) lonf(ii+LS)]`` 连线。

``ref_lon`` / ``(lonc, latc)`` 用于 ``_ll2xy`` 与震中/迹线(相对参考点的 km) — 与 InversionExample 里 ``ref_lon, lonc, latc`` 一致。

直接运行本文件::

  无参: 若存在 ``tests/inversion/py_inversion_iint0.mat``(与 ``tests/inversion/run_inversion.py`` 输出一致),
  则加载之并以 ``ref_lon=95, lonc=95.33, latc=19.61`` 绘图, 输出 ``py_inversion_iint0_show.png``;
  并自动叠画 ``fault_trace.txt``(若同目录存在). 否则提示后改为内置自测(见 ``--self-test``).

  显式指定文件::

    python -u show_slip_model.py path/to/slip_model.mat --ref-lon 95.0 --lonc 95.33 --latc 19.61

  可选: ``--fault my_fault_segments.txt`` (4 列, 同 ``load_fault_one_plane``),
  ``-o out.png``, ``--no-show``, 以及 ``--axis-range x0 x1 y0 y1 z0 z1`` 与
  ``--apply-axis-range`` 配合使用.

- 无显示器: ``SHOW_SLIP=0`` 或 ``--no-show`` 仅保存图.
- 自动化流水线(需接着跑反演/重采样): 设 ``SHOW_SLIP_BLOCK=0`` 使 ``plt.show(block=False)``, 图窗可保留且脚本不等待关图.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from load_fault_one_plane import _ll2xy


def _xy2xy(x1: float, y1: float, phi: float) -> Tuple[float, float]:
    c, s = np.cos(phi), np.sin(phi)
    return c * x1 + s * y1, -s * x1 + c * y1


def _out_of_polygon_2d(
    xs: np.ndarray, ys: np.ndarray, xv: np.ndarray, yv: np.ndarray
) -> np.ndarray:
    """与 ``out = ~inpolygon(xs,ys,xv,yv)`` 一致(相对 ref 的 km 平面)。"""
    from matplotlib.path import Path as MPath

    pth = MPath(np.column_stack([np.asarray(xv, dtype=float), np.asarray(yv, dtype=float)]))
    pts = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    return ~pth.contains_points(pts)


def _cmap_jet() -> Any:
    """Matplotlib 3.7+ 兼容的 jet colormap (避免 get_cmap 弃用警告)."""
    import matplotlib

    try:
        return matplotlib.colormaps["jet"]
    except (AttributeError, TypeError, KeyError):
        from matplotlib import cm

        return cm.get_cmap("jet")


def _set_3d_limits_from_data(
    ax: Any,
    polys: list,
    extras: List[Tuple[Sequence[float], Sequence[float], Sequence[float]]],
    *,
    margin: float = 0.04,
) -> None:
    """由块体顶点 + 震源/地震/迹线等点集设置轴范围(对应 .m 中 ``axis(axis_range)`` 被注释后的自动视窗)。"""
    parts: list = []
    if polys:
        parts.append(np.vstack(polys))
    for ex, why, zee in extras:
        ex = np.ravel(np.asarray(ex, dtype=np.float64))
        why = np.ravel(np.asarray(why, dtype=np.float64))
        zee = np.ravel(np.asarray(zee, dtype=np.float64))
        n = ex.size
        if n == 0:
            continue
        if why.size != n or zee.size != n:
            raise ValueError("extras 中 x,y,z 须同长度")
        parts.append(np.column_stack([ex, why, zee]))
    if not parts:
        return
    v = np.vstack(parts)
    lo, hi = v.min(axis=0), v.max(axis=0)
    ptp = np.maximum(hi - lo, 1e-9)
    m = margin * ptp
    ax.set_xlim(float(lo[0] - m[0]), float(hi[0] + m[0]))
    ax.set_ylim(float(lo[1] - m[1]), float(hi[1] + m[1]))
    ax.set_zlim(float(lo[2] - m[2]), float(hi[2] + m[2]))


def _resolve_fault_path(
    fault: Optional[Union[str, os.PathLike]], fault_file: Optional[Union[str, os.PathLike]]
) -> Optional[str]:
    p = fault if fault is not None else fault_file
    if p is None or (isinstance(p, str) and not p.strip()):
        return None
    return os.path.abspath(str(p))


def load_slip_model_from_file(path: Union[str, os.PathLike]) -> np.ndarray:
    """从已保存结果加载 ``(N, >=13)`` 的 ``slip_model`` 数组.

    支持:

    - ``.mat`` (``scipy.io.loadmat``): 优先变量 ``slip_model``; 否则任一双精度二维数组, 且列数 :math:`\\ge 13`.
    - ``.npy``: 二维数组, 与内存中 ``slip_model`` 同形状.
    - ``.npz``: 键 ``slip_model``; 否则第一个 ``Nx(>=13)`` 的数组.
    """
    p = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".npy":
        a = np.load(p, allow_pickle=False)
    elif ext == ".npz":
        z = np.load(p)
        if "slip_model" in z.files:
            a = z["slip_model"]
        else:
            a = None
            for k in z.files:
                t = np.asarray(z[k], dtype=np.float64)
                if t.ndim == 2 and t.shape[1] >= 13:
                    a = t
                    break
            if a is None:
                raise ValueError("npz 中需有 slip_model 或任一 Nx(>=13) 数组, 键: %r" % (z.files,))
    elif ext == ".mat":
        from scipy.io import loadmat

        d = loadmat(p, squeeze_me=True, struct_as_record=False)
        a = d.get("slip_model")
        if a is not None:
            a = np.asarray(a, dtype=np.float64)
            if a.ndim != 2 or a.shape[1] < 13:
                a = None
        if a is None:
            for k, v in d.items():
                if k.startswith("_"):
                    continue
                t = np.asarray(v, dtype=np.float64)
                if t.ndim == 2 and t.shape[1] >= 13:
                    a = t
                    break
        if a is None:
            raise ValueError("未在 %s 中找到 slip_model 或 Nx(>=13) 的二维数组" % p)
    else:
        raise ValueError("不支持的文件类型 %r，请使用 .mat / .npy / .npz" % (ext,))
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 13:
        raise ValueError("slip_model 期望形状 (N, >=13), 得到 %r" % (a.shape,))
    return a


def show_slip_model(
    slip_model: np.ndarray,
    *,
    ref_lon: float,
    lonc: float,
    latc: float,
    axis_range: Optional[Sequence[float]] = None,
    apply_axis_range: bool = False,
    fault: Optional[Union[str, os.PathLike]] = None,
    fault_file: Optional[Union[str, os.PathLike]] = None,
    seismic: Optional[Union[str, os.PathLike]] = None,
    slon: float = 95.936,
    slat: float = 22.011,
    sdepth: float = -10.0,
    title: str = "slip model",
    out_path: Optional[Union[str, os.PathLike]] = None,
    show: bool = True,
    block: Optional[bool] = None,
) -> Any:
    """绘制 3D 断面块体上色 + 归一化滑移箭矢, 以及可选震中与地震/断层迹线.

    Parameters
    ----------
    axis_range
        ``[xmin, xmax, ymin, ymax, zmin, zmax]`` (km, 见 show_slip_model.m). 仅当
        **apply_axis_range** 为 True 时用于 ``set_xlim/ylim/zlim``; 与 .m 一致时保持 **apply_axis_range=False**
        (InversionExample 里传入的 **axis_range 在 .m 中被注释, 不生效**).
    apply_axis_range
        False(默认) 不裁剪轴, 由数据+迹线定范围, 与 MATLAB 当前 .m 行为一致.
    fault / fault_file
        与 ``InversionExample`` 中 ``'fault'``/断层列表文件相同含义的 4 列文件路径
        (每行 ``lon1 lat1 lon2 lat2``), 在 z=0 画黑线(相对 (lonc,latc) 的 km 平面) .
        两个名字等价, 任填其一; 与 ``load_fault_one_plane`` 输入格式一致.
    seismic
        ``.mat`` 文件, 内为 N×3 或 3 列的 ``lon, lat, depth_km(浅源为正)``; depth 会乘 -1 与 MATLAB 一致.
    out_path
        若给路径则 ``savefig``; 不显示时也应给 ``out_path`` 以生成文件.
    show
        若 False 且给了 ``out_path`` 则只保存(需要 GUI 的 backend 可能仍需 Agg).
    block
        传给 `matplotlib.pyplot.show` 的 **block** 参数. ``True``(默认) 时保持 MATLAB 式行为: \
        在关闭图窗前**不返回**, 故后续 Step6/重采样等不会执行. ``False`` 时**立即返回**、\
        图窗可继续留在屏幕上, 脚本会接着跑(长时间纯计算时窗体可能暂时不刷新, 属正常现象).

        若为 ``None``, 由环境变量 ``SHOW_SLIP_BLOCK`` 决定: 未设或 ``1``/``true`` 为阻塞, \
        ``0``/``no``/``false`` 为非阻塞(与 ``SHOW_SLIP=0`` 不同: 后者是不弹窗、只存图).
    """
    # 须在任何 pyplot 子模块加载前定 backend；无界面只保存时用 Agg
    import matplotlib

    if out_path and not show:
        try:
            matplotlib.use("Agg", force=True)  # matplotlib >= 3.5
        except TypeError:
            matplotlib.use("Agg")
    if block is None:
        _b = os.environ.get("SHOW_SLIP_BLOCK", "1").strip()
        block = _b not in ("0", "false", "False", "no", "No")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.io import loadmat

    d2r = np.pi / 180.0
    data = np.asarray(slip_model, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 13:
        raise ValueError("slip_model 期望形状 (N, 13)")

    xe = data[:, 3]
    yn = data[:, 4]
    zr = data[:, 5]
    lp = data[:, 6]
    wp = data[:, 7]
    p_strk = data[:, 8]
    p_dip = data[:, 9] * d2r
    slip1 = data[:, 11] 
    slip2 = data[:, 12] 
    N = int(xe.size)

    polys: list = []
    slip_mag = np.empty(N, dtype=np.float64)
    xv_list: list = []
    yv_list: list = []
    zv_list: list = []
    xo_m: list = []
    yo_m: list = []
    zo_m: list = []

    for i in range(N):
        p_theta = (90.0 - p_strk[i]) * d2r
        x1f, y1f = _xy2xy(xe[i], yn[i], p_theta)
        z1f = zr[i]
        x2f, y2f = x1f + lp[i], y1f
        z2f = z1f
        x3f, y3f = x2f, y2f - wp[i] * np.cos(p_dip[i])
        z3f = z2f - wp[i] * np.sin(p_dip[i])
        x4f, y4f = x1f, y3f
        z4f = z3f

        x1, y1 = _xy2xy(x1f, y1f, -p_theta)
        x2, y2 = _xy2xy(x2f, y2f, -p_theta)
        x3, y3 = _xy2xy(x3f, y3f, -p_theta)
        x4, y4 = _xy2xy(x4f, y4f, -p_theta)

        X = np.array([x1, x2, x3, x4]) / 1000.0
        Y = np.array([y1, y2, y3, y4]) / 1000.0
        Z = np.array([z1f, z2f, z3f, z4f]) / 1000.0
        polys.append(np.column_stack([X, Y, Z]))

        xo = float(np.mean(X))
        yo = float(np.mean(Y))
        zo = float(np.mean(Z))
        xo_m.append(xo)
        yo_m.append(yo)
        zo_m.append(zo)

        xuf = float(slip1[i])
        yvf = float(slip2[i] * np.cos(p_dip[i]))
        zwf = float(slip2[i] * np.sin(p_dip[i]))
        xu, yu = _xy2xy(xuf, yvf, -p_theta)
        xv_list.append(xu)
        yv_list.append(yu)
        zv_list.append(zwf)
        slip0 = float(np.sqrt(xu * xu + yu * yu + zwf * zwf))
        slip_mag[i] = slip0

    x0_ref, y0_ref = _ll2xy(lonc, latc, ref_lon)

    XO = np.asarray(xo_m)
    YO = np.asarray(yo_m)
    ZO = np.asarray(zo_m)
    XV = np.asarray(xv_list, dtype=np.float64)
    YV = np.asarray(yv_list, dtype=np.float64)
    ZV = np.asarray(zv_list, dtype=np.float64)
    slipmax = float(np.max(slip_mag)) if N else 1.0
    if slipmax < 1e-20:
        slipmax = 1.0
    nrm = 1.0 / slipmax

    cvals = np.sqrt(np.asarray(slip1) ** 2 + np.asarray(slip2) ** 2)
    cmax = float(np.max(cvals)) if N else 1.0
    if cmax < 1e-20:
        cmax = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    normc = colors.Normalize(vmin=0.0, vmax=cmax)
    cmap = _cmap_jet()
    facecolors = [cmap(normc(cvals[i])) for i in range(N)]
    coll = Poly3DCollection(
        [p for p in polys], facecolors=facecolors, edgecolors="k", linewidths=0.1, alpha=1.0
    )
    ax.add_collection3d(coll)

    if np.any(np.abs(XV) + np.abs(YV) + np.abs(ZV) > 1e-30):
        ax.quiver(
            XO, YO, ZO, XV * nrm, YV * nrm, ZV * nrm,
            length=1.5, normalize=True, color="k", linewidth=0.5, arrow_length_ratio=0.15,
        )

    m = ScalarMappable(cmap=cmap, norm=normc)
    m.set_array(cvals)
    cb = fig.colorbar(m, ax=ax, shrink=0.5, aspect=20)
    cb.set_label("slip (m)")

    xx, yy = _ll2xy(slon, slat, ref_lon)
    xs = (xx - x0_ref) / 1000.0
    ys = (yy - y0_ref) / 1000.0
    # 与 .m 中 ``scatter3(..., 'p', 'filled')`` 接近: 红色五角形标记
    ax.scatter(
        [xs], [ys], [sdepth], c="r", s=100, marker="*", edgecolors="k", zorder=5,
    )

    limit_extras: List[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = [
        ([float(xs)], [float(ys)], [float(sdepth)]),
    ]

    if seismic is not None:
        d = loadmat(str(seismic))
        dseis = None
        for k, v in d.items():
            if k.startswith("_"):
                continue
            a = np.asarray(v, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 3:
                dseis = a
                break
        if dseis is None:
            raise ValueError("seismic .mat 中需至少一变量为 Nx3( lon, lat, depth )")
        dseis = dseis[:, :3].reshape(-1, 3)
        slon2 = dseis[:, 0]
        slat2 = dseis[:, 1]
        sdep = -dseis[:, 2]

        # xv, yv 与 MATLAB 一致(固定裁切多边形, km, 与相对 ref 的 xs,ys 同坐标系)
        xv_poly = np.array([-3, -5, 20, 20, -2], dtype=np.float64)
        yv_poly = np.array([33, 51, 51, 30, 32], dtype=np.float64)
        xx2, yy2 = _ll2xy(slon2, slat2, ref_lon)
        xss = (xx2 - x0_ref) / 1000.0
        yss = (yy2 - y0_ref) / 1000.0
        in_mask = _out_of_polygon_2d(xss, yss, xv_poly, yv_poly)
        xin = xss[in_mask]
        yin = yss[in_mask]
        din = sdep[in_mask]
        if xin.size:
            ax.scatter(
                xin, yin, din, c="black", s=5, marker="o", zorder=4,
            )
            limit_extras.append(
                (xin.tolist(), yin.tolist(), din.tolist()),
            )

    fault_path = _resolve_fault_path(fault, fault_file)
    if fault_path and os.path.isfile(fault_path):
        seg = np.loadtxt(fault_path, dtype=np.float64)
        if seg.ndim == 1:
            seg = seg.reshape(1, -1)
        if seg.shape[1] < 4:
            raise ValueError("fault 文件每行需至少 4 列: lon1 lat1 lon2 lat2 (与 load_fault_one_plane 一致)")
        for r in range(seg.shape[0]):
            lon_a, lat_a, lon_b, lat_b = seg[r, 0], seg[r, 1], seg[r, 2], seg[r, 3]
            xa, ya = _ll2xy(np.array([lon_a, lon_b]), np.array([lat_a, lat_b]), ref_lon)
            xsa = (np.asarray(xa) - x0_ref) / 1000.0
            ysa = (np.asarray(ya) - y0_ref) / 1000.0
            xsa = np.ravel(xsa, order="C")
            ysa = np.ravel(ysa, order="C")
            limit_extras.append((xsa.tolist(), ysa.tolist(), [0.0, 0.0]))
            ax.plot(xsa, ysa, [0.0, 0.0], c="k", linewidth=1.5)

    ax.set_zlabel("Depth (km)")
    if apply_axis_range and axis_range is not None and len(axis_range) >= 6:
        ax.set_xlim(axis_range[0], axis_range[1])
        ax.set_ylim(axis_range[2], axis_range[3])
        ax.set_zlim(axis_range[4], axis_range[5])
    else:
        _set_3d_limits_from_data(ax, polys, limit_extras, margin=0.04)
    try:
        xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        ax.set_box_aspect(
            (
                float(xl[1] - xl[0]) + 1e-9,
                float(yl[1] - yl[0]) + 1e-9,
                float(zl[1] - zl[0]) + 1e-9,
            )
        )
    except Exception:  # pragma: no cover
        pass
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title(title)
    ax.grid(True)

    try:
        plt.tight_layout()
    except Exception:  # pragma: no cover
        pass
    if out_path:
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="w")
    if show:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if isinstance(fig.canvas, FigureCanvasAgg):
            # 全局已落在 Agg(无头 / MPLBACKEND=Agg 等); 不调用 plt.show, 避免 “cannot be shown”
            if out_path:
                print(
                    "show_slip_model: 无界面后端(Agg), 不弹窗; 图已存: %s" % (out_path,),
                    flush=True,
                )
            else:
                print(
                    "show_slip_model: 无界面后端(Agg), 不弹窗; 请设 out_path 或 MPLBACKEND=TkAgg. ",
                    flush=True,
                )
            plt.close(fig)
        else:
            plt.show(block=block)
            if not block:
                # 部分后端(尤其 Windows)下给事件循环一瞬, 窗体先完成首次绘制
                try:
                    plt.pause(0.05)
                except Exception:  # pragma: no cover
                    pass
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    import sys
    import tempfile

    # ----- 可移植配置: 只改本段, 在 IDE 中直接运行, 不依赖 PowerShell/命令行参数 -----
    # slip_path: 指定 .mat/.npy/.npz; 为 None 时若存在默认反演 mat 则加载, 否则转 self_test
    slip_path: Optional[str] = None
    self_test: bool = False
    ref_lon: Optional[float] = None
    lonc: Optional[float] = None
    latc: Optional[float] = None
    fault: Optional[str] = None
    out: Optional[str] = None
    title: Optional[str] = None
    apply_axis_range: bool = False
    axis_range: Optional[Sequence[float]] = None
    no_show: bool = False
    # ---------------------------------------------------------------------------

    if apply_axis_range and (axis_range is None or len(list(axis_range)) < 6):
        print("apply_axis_range 为 True 时请在 axis_range 中提供 6 个 float(km).", file=sys.stderr, flush=True)
        sys.exit(2)
    ar_axis: Optional[Sequence[float]] = list(axis_range) if axis_range is not None else None

    _here = os.path.dirname(os.path.abspath(__file__))
    _DEFAULT_INVERSION_REF_LON = 95.0
    _DEFAULT_INVERSION_LONC = 95.33
    _DEFAULT_INVERSION_LATC = 19.61
    _DEFAULT_INVERSION_SLIP_MAT = os.path.join(_here, "tests", "inversion", "py_inversion_iint0.mat")
    _DEFAULT_INVERSION_FAULT = os.path.join(_here, "fault_trace.txt")

    do_show = os.environ.get("SHOW_SLIP", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
    )
    if no_show:
        do_show = False
    if not do_show:
        os.environ.setdefault("MPLBACKEND", "Agg")

    if not slip_path and not self_test:
        if os.path.isfile(_DEFAULT_INVERSION_SLIP_MAT):
            slip_p = _DEFAULT_INVERSION_SLIP_MAT
            out_png = out
            if not out_png:
                base, _ = os.path.splitext(slip_p)
                out_png = base + "_show.png"
            fault_p = _DEFAULT_INVERSION_FAULT if os.path.isfile(_DEFAULT_INVERSION_FAULT) else None
            try:
                sm = load_slip_model_from_file(slip_p)
            except Exception as e:  # noqa: BLE001
                print("加载失败:", e, file=sys.stderr, flush=True)
                sys.exit(1)
            ar = list(ar_axis) if ar_axis is not None else None
            print(
                "slip_model (default tests/inversion/py_inversion_iint0.mat):",
                sm.shape,
                "->",
                out_png,
                flush=True,
            )
            try:
                show_slip_model(
                    sm,
                    ref_lon=_DEFAULT_INVERSION_REF_LON,
                    lonc=_DEFAULT_INVERSION_LONC,
                    latc=_DEFAULT_INVERSION_LATC,
                    fault=fault_p,
                    out_path=out_png,
                    show=do_show,
                    title=(title or "py_inversion iint0"),
                    apply_axis_range=apply_axis_range,
                    axis_range=ar,
                )
            except Exception:
                import traceback

                traceback.print_exc()
                sys.exit(1)
            print("Done. Figure saved:", os.path.abspath(out_png), flush=True)
            if do_show:
                print("If no window, open the png above or set SHOW_SLIP=0 to save only.", flush=True)
            sys.exit(0)
        if not out:
            print(
                "未找到", _DEFAULT_INVERSION_SLIP_MAT, "，改为内置自测(或先运行 tests/inversion/run_inversion.py)。",
                flush=True,
            )
        self_test = True

    if not slip_path and self_test:
        out_png = out or os.path.join(_here, "show_slip_model_selftest.png")
        print("show_slip_model: self-test ->", out_png, flush=True)
        sm = np.zeros((2, 13), dtype=np.float64)
        sm[:, 3:10] = [
            [6.5e4, 1.5e5, 0.0, 2e3, 2.5e3, 90.0, 80.0],
            [6.5e4 + 2e3, 1.5e5, 0.0, 2e3, 2.5e3, 90.0, 80.0],
        ]
        sm[:, 11:13] = [[50.0, 0.0], [0.0, 50.0]]
        _fd, fault_tmp = tempfile.mkstemp(suffix="_fault4.txt", text=True)
        os.close(_fd)
        with open(fault_tmp, "w", encoding="utf-8") as f:
            f.write("95.0 19.0 95.4 20.0\n")
        ar = list(ar_axis) if ar_axis is not None else None
        try:
            show_slip_model(
                sm,
                ref_lon=95.0,
                lonc=95.33,
                latc=19.61,
                fault=fault_tmp,
                out_path=out_png,
                show=do_show,
                title=(title or "self-test"),
                apply_axis_range=apply_axis_range,
                axis_range=ar,
            )
        except Exception:
            import traceback

            traceback.print_exc()
            sys.exit(1)
        finally:
            try:
                os.remove(fault_tmp)
            except OSError:
                pass
        print("Done. Figure saved:", os.path.abspath(out_png), flush=True)
        if do_show:
            print("If no window, open the png above or set SHOW_SLIP=0 to save only.", flush=True)
        sys.exit(0)

    if ref_lon is None or lonc is None or latc is None:
        print(
            "指定 slip_path 时须同时设置 ref_lon, lonc, latc(与反演 configpara 一致).",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    slip_p = slip_path
    if out:
        out_png = out
    else:
        base, _ = os.path.splitext(str(slip_p))
        out_png = base + "_show.png"
    try:
        sm = load_slip_model_from_file(slip_p)
    except Exception as e:  # noqa: BLE001
        print("加载失败:", e, file=sys.stderr, flush=True)
        sys.exit(1)
    ar = list(ar_axis) if ar_axis is not None else None
    print("slip_model:", sm.shape, "->", out_png, flush=True)
    try:
        show_slip_model(
            sm,
            ref_lon=float(ref_lon),
            lonc=float(lonc),
            latc=float(latc),
            fault=fault,
            out_path=out_png,
            show=do_show,
            title=(title or "slip model"),
            apply_axis_range=apply_axis_range,
            axis_range=ar,
        )
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("Done. Figure saved:", os.path.abspath(out_png), flush=True)
