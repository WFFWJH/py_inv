r"""3D 断层滑移模型可视化 — 对应 show_slip_model.m.

``slip_model`` 列 3–9 为几何 (m), 11–12 为走滑/倾滑 (m).
默认不裁剪 ``axis_range`` (与 .m 中 ``axis(axis_range)`` 被注释一致); 需固定视窗时设 ``apply_axis_range=True``.
"""
from __future__ import annotations

import atexit
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from load_fault_one_plane import _ll2xy

# 非阻塞弹窗: 脚本结束时若还有未关图窗, 自动挂起等用户关掉 (否则进程一退窗口就闪没)
_KEEP_FIGURES_ATEXIT = False


def _wait_open_figures_at_exit() -> None:
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():
            print("show_slip_model: 图窗仍打开, 关闭全部窗口后脚本才会退出 ...", flush=True)
            plt.show(block=True)
    except Exception:
        pass


def _xy2xy(x1: float, y1: float, phi: float) -> Tuple[float, float]:
    c, s = np.cos(phi), np.sin(phi)
    return c * x1 + s * y1, -s * x1 + c * y1


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v not in ("0", "false", "no")


def _ll_to_km(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    ref_lon: float,
    x0: float,
    y0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _ll2xy(lon, lat, ref_lon)
    return (np.asarray(x, dtype=np.float64) - x0) / 1000.0, (np.asarray(y, dtype=np.float64) - y0) / 1000.0


def _set_3d_limits(ax: Any, polys: list, extras: list, *, margin: float = 0.04) -> None:
    """按数据自动定轴范围; margin 为各轴外延比例 (默认 4%)."""
    parts = list(polys)
    for ex, ey, ez in extras:
        parts.append(np.column_stack([np.ravel(ex), np.ravel(ey), np.ravel(ez)]))
    if not parts:
        return
    v = np.vstack(parts)
    lo, hi = v.min(axis=0), v.max(axis=0)
    m = margin * np.maximum(hi - lo, 1e-9)
    ax.set_xlim(float(lo[0] - m[0]), float(hi[0] + m[0]))
    ax.set_ylim(float(lo[1] - m[1]), float(hi[1] + m[1]))
    ax.set_zlim(float(lo[2] - m[2]), float(hi[2] + m[2]))


# 纯无界面后端 (注意: QtAgg/TkAgg 名字里也带 agg, 不能用 "agg" in name 判断)
_HEADLESS_BACKENDS = frozenset({"agg", "svg", "pdf", "ps", "cairo", "template"})


def _is_headless_backend() -> bool:
    import matplotlib

    return matplotlib.get_backend().lower().split()[-1] in _HEADLESS_BACKENDS


def _ensure_interactive_backend(show: bool) -> None:
    """``show=True`` 时尽量用可弹窗的后端 (Cursor/IDE 里默认常为 Agg)."""
    if not show or os.environ.get("MPLBACKEND"):
        return
    import matplotlib

    if not _is_headless_backend():
        return  # 已是 QtAgg / TkAgg 等, 勿误切
    for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "WXAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


def _finalize_figure(fig: Any, out_path: Optional[str], show: bool, block: bool) -> None:
    """显示图窗.

    ``block=False``(默认): 立刻返回, 后续代码可继续; 脚本真正退出前会挂起直到关掉图窗
    (类似 MATLAB: figure 一直开着, 命令还能接着敲). ``block=True``: 当场等关窗再返回.
    """
    global _KEEP_FIGURES_ATEXIT
    import matplotlib.pyplot as plt

    if out_path:
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="w")
    if not show:
        plt.close(fig)
        return
    # 不能用 isinstance(canvas, FigureCanvasAgg): QtAgg/TkAgg 的 canvas 都继承它
    if _is_headless_backend():
        saved = out_path
        if not saved:
            saved = os.path.abspath("slip_model_show.png")
            fig.savefig(str(saved), dpi=200, bbox_inches="tight", facecolor="w")
        print(
            "show_slip_model: 当前为无界面后端, 无法弹窗; 图已保存:",
            saved,
            "\n  若要交互窗口: 运行前设 MPLBACKEND=TkAgg",
            sep="\n",
            flush=True,
        )
        plt.close(fig)
        return
    if block:
        plt.show(block=True)
        return

    # 非阻塞: 交互模式 + 刷一帧后立刻返回; 注册 atexit, 防止脚本结束瞬间窗口被关掉
    plt.ion()
    fig.show()
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.1)
    except Exception:
        pass
    if not _KEEP_FIGURES_ATEXIT:
        atexit.register(_wait_open_figures_at_exit)
        _KEEP_FIGURES_ATEXIT = True


def load_slip_model_from_file(path: Union[str, os.PathLike]) -> np.ndarray:
    """从 .mat / .npy / .npz 加载 ``(N, >=13)`` 的 ``slip_model``."""
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
                raise ValueError("npz 中需有 slip_model 或任一 Nx(>=13) 数组")
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
    show_slip_arrows: bool = True,
    slip_arrow_toggle: bool = True,
) -> Any:
    """绘制 3D 断面块体 + 滑移箭矢; 可选震中、地震目录与断层迹线.

    axis_range
        固定视窗, 6 个数 [xmin, xmax, ymin, ymax, zmin, zmax], 单位 km.
        坐标系: 以 (lonc, latc) 为原点的 Easting/Northing, 深度向下为负.
        须配合 apply_axis_range=True 才生效.
    apply_axis_range
        True 时使用 axis_range; False(默认) 由数据自动定范围.
    show_slip_arrows
        Initial visibility of slip-direction arrows (center origin, strike+dip vector,
        length scaled to plot size by slip magnitude).
    slip_arrow_toggle
        If True, add a Show/Hide Arrows button in interactive windows.
    """
    import matplotlib

    if not _env_flag("SHOW_SLIP", default=True):
        show = False
    if show:
        _ensure_interactive_backend(True)

    if out_path and not show:
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")
    if block is None:
        # 默认 False: 弹窗不阻塞后续代码 (MATLAB 行为); 设 SHOW_SLIP_BLOCK=1 则关窗后才继续
        block = _env_flag("SHOW_SLIP_BLOCK", default=False)

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.io import loadmat

    d2r = np.pi / 180.0
    data = np.asarray(slip_model, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 13:
        raise ValueError("slip_model 期望形状 (N, 13)")

    xe, yn, zr = data[:, 3], data[:, 4], data[:, 5]
    lp, wp = data[:, 6], data[:, 7]
    p_strk, p_dip = data[:, 8], data[:, 9] * d2r
    slip1, slip2 = data[:, 11], data[:, 12]
    n = int(xe.size)

    polys: list = []
    slip_mag = np.empty(n, dtype=np.float64)
    xo_m, yo_m, zo_m = [], [], []
    xv_list, yv_list, zv_list = [], [], []

    for i in range(n):
        theta = (90.0 - p_strk[i]) * d2r
        x1f, y1f = _xy2xy(xe[i], yn[i], theta)
        z1f = zr[i]
        x2f, y2f = x1f + lp[i], y1f
        z2f = z1f
        x3f, y3f = x2f, y2f - wp[i] * np.cos(p_dip[i])
        z3f = z2f - wp[i] * np.sin(p_dip[i])
        x4f, y4f = x1f, y3f

        corners = [_xy2xy(x, y, -theta) for x, y in ((x1f, y1f), (x2f, y2f), (x3f, y3f), (x4f, y4f))]
        xyz = np.column_stack([np.array(corners)[:, 0] / 1000.0, np.array(corners)[:, 1] / 1000.0,
                               np.array([z1f, z2f, z3f, z3f]) / 1000.0])
        polys.append(xyz)
        xo_m.append(float(np.mean(xyz[:, 0])))
        yo_m.append(float(np.mean(xyz[:, 1])))
        zo_m.append(float(np.mean(xyz[:, 2])))

        xu, yu = _xy2xy(float(slip1[i]), float(slip2[i] * np.cos(p_dip[i])), -theta)
        zv = float(slip2[i] * np.sin(p_dip[i]))
        xv_list.append(xu)
        yv_list.append(yu)
        zv_list.append(zv)
        slip_mag[i] = float(np.sqrt(xu * xu + yu * yu + zv * zv))

    x0_ref, y0_ref = _ll2xy(lonc, latc, ref_lon)
    xo = np.asarray(xo_m)
    yo = np.asarray(yo_m)
    zo = np.asarray(zo_m)
    xv = np.asarray(xv_list)
    yv = np.asarray(yv_list)
    zv = np.asarray(zv_list)
    slipmax = max(float(np.max(slip_mag)) if n else 1.0, 1e-20)
    cvals = np.sqrt(slip1 ** 2 + slip2 ** 2)
    cmax = max(float(np.max(cvals)) if n else 1.0, 1e-20)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    normc = colors.Normalize(vmin=0.0, vmax=cmax)
    try:
        cmap = matplotlib.colormaps["jet"]
    except (AttributeError, TypeError, KeyError):
        from matplotlib import cm
        cmap = cm.get_cmap("jet")

    # Semi-transparent faces when arrows are on: mplot3d has no z-buffer, so opaque
    # coplanar patches hide in-plane slip arrows.
    face_alpha = 0.72 if show_slip_arrows else 1.0
    coll = Poly3DCollection(
        polys,
        facecolors=[cmap(normc(cvals[i])) for i in range(n)],
        edgecolors="k",
        linewidths=0.1,
        alpha=face_alpha,
    )
    ax.add_collection3d(coll)

    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=normc), ax=ax, shrink=0.5, aspect=20)
    cb.set_label("slip (m)")

    xs, ys = _ll_to_km(slon, slat, ref_lon, x0_ref, y0_ref)
    ax.scatter([float(xs)], [float(ys)], [sdepth], c="r", s=100, marker="*", edgecolors="k", zorder=5)
    limit_extras = [(xs, ys, np.array([sdepth]))]

    if seismic is not None:
        d = loadmat(str(seismic))
        dseis = next(
            np.asarray(v, dtype=np.float64)
            for k, v in d.items()
            if not k.startswith("_") and np.asarray(v).ndim == 2 and np.asarray(v).shape[1] >= 3
        )[:, :3]
        xss, yss = _ll_to_km(dseis[:, 0], dseis[:, 1], ref_lon, x0_ref, y0_ref)
        from matplotlib.path import Path as MPath

        poly = MPath(np.column_stack([[-3, -5, 20, 20, -2], [33, 51, 51, 30, 32]]))
        mask = ~poly.contains_points(np.column_stack([xss, yss]))
        if np.any(mask):
            ax.scatter(xss[mask], yss[mask], -dseis[:, 2][mask], c="black", s=5, zorder=4)
            limit_extras.append((xss[mask], yss[mask], -dseis[:, 2][mask]))

    fault_path = fault if fault is not None else fault_file
    if fault_path and os.path.isfile(str(fault_path)):
        seg = np.loadtxt(str(fault_path), dtype=np.float64)
        if seg.ndim == 1:
            seg = seg.reshape(1, -1)
        if seg.shape[1] < 4:
            raise ValueError("fault 文件每行需 4 列: lon1 lat1 lon2 lat2")
        for row in seg:
            xsa, ysa = _ll_to_km([row[0], row[2]], [row[1], row[3]], ref_lon, x0_ref, y0_ref)
            ax.plot(xsa, ysa, [0.0, 0.0], c="k", linewidth=1.5)
            limit_extras.append((xsa, ysa, np.array([0.0, 0.0])))

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_zlabel("Depth (km)")
    ax.set_title(title)
    ax.grid(True)

    # ----- 坐标轴范围 (改绘图视窗主要改这里) -----
    # 方式 A (推荐手动调): apply_axis_range=True, 填 axis_range 六个 km 值
    #   例 (InversionExample.m): [50, 150, -180, 350, -20, 0]
    #       x: 50~150 km (Easting),  y: -180~350 km (Northing),  z: -20~0 km (深度)
    # 方式 B (默认): apply_axis_range=False, 由块体顶点 + 震中 + 断层迹线自动算范围
    #   想稍微放大/缩小自动框: 改 _set_3d_limits(..., margin=0.04) 里的 margin
    if apply_axis_range and axis_range is not None and len(axis_range) >= 6:
        ax.set_xlim(axis_range[0], axis_range[1])   # Easting (km)
        ax.set_ylim(axis_range[2], axis_range[3])   # Northing (km)
        ax.set_zlim(axis_range[4], axis_range[5])   # Depth (km), 负值=地下
    else:
        _set_3d_limits(ax, polys, limit_extras)
    try:
        xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        ax.set_box_aspect((xl[1] - xl[0] + 1e-9, yl[1] - yl[0] + 1e-9, zl[1] - zl[0] + 1e-9))
    except Exception:
        pass
    try:
        plt.tight_layout()
    except Exception:
        pass

    # Slip arrows: center origin, strike+dip direction, length ~ (slip/slipmax)*plot size.
    # mplot3d has no z-buffer: use lifted arrows + Line3DCollection + triangular heads
    # (Poly3DCollection) so they depth-sort with fault faces. Batched quiver3 is unreliable.
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    slip_artists: List[Any] = []
    slip_mask = slip_mag > 1e-30
    if np.any(slip_mask):
        xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        box_diag = float(np.sqrt((xl[1] - xl[0]) ** 2 + (yl[1] - yl[0]) ** 2 + (zl[1] - zl[0]) ** 2))
        max_arrow_len = 0.06 * max(box_diag, 1e-9)
        # lift ~ patch size so arrows sit clearly above the fault plane
        med_patch = float(np.median(np.minimum(lp, wp))) / 1000.0
        lift = max(0.35 * med_patch, 0.02 * box_diag)
        head_frac = 0.32
        idx = np.flatnonzero(slip_mask)

        shaft_segs: List[np.ndarray] = []
        head_tris: List[np.ndarray] = []
        tip_xyz: List[np.ndarray] = []

        for i in idx:
            corners = polys[int(i)]
            nvec = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            nn = float(np.linalg.norm(nvec))
            nvec = nvec / nn if nn > 1e-15 else np.array([0.0, 0.0, 1.0])
            if nvec[2] < 0.0:
                nvec = -nvec

            # xv,yv,zv are in meters (same as slip); scale to km display length
            raw = np.array([xv[i], yv[i], zv[i]], dtype=np.float64)
            rlen = float(np.linalg.norm(raw))
            if rlen < 1e-30:
                continue
            vec = raw * (max_arrow_len / slipmax)
            vlen = float(np.linalg.norm(vec))
            p0 = np.array([xo[i], yo[i], zo[i]], dtype=np.float64) + lift * nvec
            p1 = p0 + vec
            shaft_segs.append(np.vstack([p0, p1]))
            tip_xyz.append(p1)

            uhat = vec / vlen
            side = np.cross(uhat, nvec)
            sn = float(np.linalg.norm(side))
            if sn < 1e-15:
                side = np.cross(uhat, np.array([0.0, 0.0, 1.0]))
                sn = float(np.linalg.norm(side))
            if sn < 1e-15:
                continue
            side = side / sn
            hlen = head_frac * vlen
            hwid = 0.5 * hlen
            left = p1 - hlen * uhat + hwid * side
            right = p1 - hlen * uhat - hwid * side
            head_tris.append(np.vstack([p1, left, right]))

        if shaft_segs:
            lc_bg = Line3DCollection(shaft_segs, colors="#111111", linewidths=4.0)
            lc_fg = Line3DCollection(shaft_segs, colors="#FFEA00", linewidths=2.4)
            ax.add_collection3d(lc_bg)
            ax.add_collection3d(lc_fg)
            lc_bg.set_visible(show_slip_arrows)
            lc_fg.set_visible(show_slip_arrows)
            slip_artists.extend([lc_bg, lc_fg])

        if head_tris:
            heads = Poly3DCollection(
                head_tris,
                facecolors="#FFEA00",
                edgecolors="#111111",
                linewidths=1.2,
                alpha=1.0,
            )
            ax.add_collection3d(heads)
            heads.set_visible(show_slip_arrows)
            slip_artists.append(heads)

        if tip_xyz:
            tips = np.asarray(tip_xyz)
            sc = ax.scatter(
                tips[:, 0], tips[:, 1], tips[:, 2],
                c="#FF1744", s=28, depthshade=False, edgecolors="#111111", linewidths=0.4,
            )
            sc.set_visible(show_slip_arrows)
            slip_artists.append(sc)

        if show and slip_arrow_toggle and not _is_headless_backend() and slip_artists:
            from matplotlib.widgets import Button

            btn_ax = fig.add_axes([0.02, 0.02, 0.16, 0.05])
            slip_btn = Button(
                btn_ax,
                "Hide Arrows" if show_slip_arrows else "Show Arrows",
                color="#FFF9C4",
                hovercolor="#FDD835",
            )

            def _toggle_slip_arrows(_event: Any) -> None:
                vis = not slip_artists[0].get_visible()
                for art in slip_artists:
                    art.set_visible(vis)
                # restore solid faces when arrows hidden
                try:
                    coll.set_alpha(0.72 if vis else 1.0)
                except Exception:
                    pass
                slip_btn.label.set_text("Hide Arrows" if vis else "Show Arrows")
                fig.canvas.draw_idle()

            slip_btn.on_clicked(_toggle_slip_arrows)
            fig._slip_arrow_btn = slip_btn
            fig._slip_arrow_artists = slip_artists
            fig._slip_face_coll = coll

    _finalize_figure(fig, str(out_path) if out_path else None, show, block)
    return fig


if __name__ == "__main__":
    import sys

    # =========================================================================
    # 直接运行本文件时改下面配置 (IDE 里 Run 即可, 不必命令行参数)
    # =========================================================================
    here = os.path.dirname(os.path.abspath(__file__))

    # --- 输入 ---
    slip_path = os.path.join(here, "tests", "inversion", "py_inversion_iint0.mat")
    ref_lon, lonc, latc = 95.0, 95.33, 19.61   # 与反演 configpara 一致, 用于震中/迹线 km 换算
    fault_path = os.path.join(here, "fault_trace.txt")  # 4 列 lon1 lat1 lon2 lat2; 无则 None
    out_png = None  # None -> 与 slip 同目录 *_show.png

    # --- 绘图范围 (重点) ---
    # apply_axis_range=False: 自动包住所有块体 (默认, 一般不用改)
    # apply_axis_range=True : 使用下面 axis_range 六个数, 单位 km
    apply_axis_range = False
    axis_range = [50, 150, -180, 350, -20, 0]  # [xmin,xmax, ymin,ymax, zmin,zmax]

    # --- 显示 ---
    show = _env_flag("SHOW_SLIP")   # 环境变量 SHOW_SLIP=0 则不弹窗
    title = "py_inversion iint0"
    # =========================================================================

    if not os.path.isfile(slip_path):
        print("找不到 slip 文件:", slip_path, file=sys.stderr)
        sys.exit(1)
    if not show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    if out_png is None:
        out_png = os.path.splitext(slip_path)[0] + "_show.png"

    sm = load_slip_model_from_file(slip_path)
    show_slip_model(
        sm,
        ref_lon=ref_lon,
        lonc=lonc,
        latc=latc,
        fault=fault_path if os.path.isfile(fault_path) else None,
        out_path=out_png,
        show=show,
        title=title,
        apply_axis_range=apply_axis_range,
        axis_range=axis_range if apply_axis_range else None,
    )
    print("Done:", os.path.abspath(out_png))
