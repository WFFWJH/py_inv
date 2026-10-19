r"""3D 断层滑移模型可视化 — 对应 show_slip_model.m.

``slip_model`` 列 3–9 为几何 (m), 11–12 为走滑/倾滑 (m).
默认不裁剪 ``axis_range`` (与 .m 中 ``axis(axis_range)`` 被注释一致); 需固定视窗时设 ``apply_axis_range=True``.
"""
from __future__ import annotations

import atexit
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from geodetic.coords import ll2xy, xy2xy

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
    x, y = ll2xy(lon, lat, ref_lon)
    return (np.asarray(x, dtype=np.float64) - x0) / 1000.0, (np.asarray(y, dtype=np.float64) - y0) / 1000.0


def _set_3d_limits(ax: Any, polys: list, extras: list, *, margin: float = 0.02) -> None:
    """按数据自动定轴范围; margin 为各轴外延比例 (默认 2%)."""
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


def _crop_whitespace(path: str, pad_px: int = 14) -> None:
    """按墨迹包围盒裁掉四周白边 (只处理 png).

    ``bbox_inches="tight"`` 只能裁到 axes 矩形, 裁不掉 axes **内部**的空白;
    真实比例下块体是细长条, 那圈空白比数据本身还大, 所以出图后再按非白像素裁一次.
    """
    if not str(path).lower().endswith(".png"):
        return
    try:
        from PIL import Image  # matplotlib 的硬依赖, 无需额外安装
    except Exception:
        return
    try:
        im = Image.open(path).convert("RGB")
        a = np.asarray(im)
        ink = a.min(axis=2) < 248
        ys, xs = np.nonzero(ink)
        if xs.size == 0:
            return
        box = (
            max(int(xs.min()) - pad_px, 0),
            max(int(ys.min()) - pad_px, 0),
            min(int(xs.max()) + pad_px, a.shape[1]),
            min(int(ys.max()) + pad_px, a.shape[0]),
        )
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            return
        im.crop(box).save(path)
    except Exception:
        return


def _add_toolbar_toggle(fig: Any, label: str, initial: bool, on_change: Any) -> bool:
    """把开关挂到窗口工具栏上, 而不是在图里画一个按钮.

    图里画按钮有两个副作用: 占掉画布一角, 而且按钮底色是高饱和色, 会被
    "彩色块体"检测和裁白逻辑当成数据 (见 ``_place_title_above_band``).
    工具栏属于窗口 chrome, 既不占画布, 也不会进 ``savefig``.

    各后端的工具栏结构不同, 依次尝试; 全都不支持就返回 False, 由调用方退回快捷键.

    * Qt: ``NavigationToolbar2QT`` 本身就是 QToolBar, 直接 ``addWidget``;
    * Tk: ``NavigationToolbar2Tk`` 是 ``tk.Frame``, 里面按钮都是 pack 到左侧的.
    """
    tb = getattr(fig.canvas, "toolbar", None)
    if tb is None:
        return False

    # --- Qt: 工具栏是 QToolBar ---
    if hasattr(tb, "addWidget"):
        try:
            from matplotlib.backends.qt_compat import QtWidgets

            btn = QtWidgets.QPushButton(label, tb)
            btn.setCheckable(True)
            btn.setChecked(bool(initial))
            btn.setFlat(True)
            btn.setToolTip("show / hide slip-direction arrows (shortcut: a)")
            btn.toggled.connect(lambda on: on_change(bool(on)))
            tb.addWidget(btn)
            fig._slip_arrow_toolbar_btn = btn
            return True
        except Exception:
            pass

    # --- Tk: 工具栏是 tk.Frame ---
    try:
        import tkinter as tk

        if isinstance(tb, tk.Frame):
            var = tk.IntVar(master=tb, value=1 if initial else 0)

            def _flip() -> None:
                on_change(bool(var.get()))

            btn = tk.Checkbutton(
                tb,
                text=label,
                variable=var,
                command=_flip,
                indicatoron=False,      # 变成普通按钮外观, 按下即高亮
                relief="flat",
                overrelief="raised",
                bd=1,
                padx=6,
                pady=0,
            )
            btn.pack(side=tk.LEFT, padx=2)
            fig._slip_arrow_toolbar_btn = btn
            fig._slip_arrow_toolbar_var = var
            return True
    except Exception:
        pass
    return False


def _fit_axes_rect(fig: Any, ax: Any, off: dict) -> None:
    """把 3D 正方形放大到"块体刚好占满可用宽度", 复现 MATLAB ``axis equal`` 的填充行为.

    这是对 mplot3d 一个硬性设计缺陷的绕法. MATLAB 的 ``axis equal`` 保持 axes 矩形不变,
    把**绘图盒**按真实比例塞进矩形里, 所以窗口越宽盒子越大; 而 mplot3d 的
    ``apply_aspect()`` 是把 **axes 本身**收缩成正方形, 边长 = min(rect宽, rect高) ——
    横向拉宽窗口时边长不变, 多出来的宽度全成留白.

    这里反过来利用它: 既然边长 = min(rect宽, rect高), 那就把 rect 给到"宽高都 >= 目标边长",
    正方形自然就变成目标边长. 正方形在竖直方向超出画布也无所谓 —— 块体又扁又居中,
    超出的部分本来就是空白.
    """
    try:
        W, H = fig.get_size_inches()
    except Exception:
        return
    bw = max(off["fx1"] - off["fx0"], 1e-6)     # 块体占正方形宽度的比例
    bh = max(off["fy_top"] - off["fy_bot"], 1e-6)
    # 左右留白按英寸给: 左边放 Easting 刻度+竖排标签, 右边放 z 刻度 + Depth 标签 + 色标
    m_left, m_right = 1.35, 3.05
    m_top, m_bot = 0.25, 0.60
    side = min(max(W - m_left - m_right, 1.0) / bw, max(H - m_top - m_bot, 1.0) / bh)
    if not np.isfinite(side) or side <= 0:
        return
    # 块体在正方形里的水平中心, 用它把块体摆到 [m_left, W - m_right] 之间
    bc = 0.5 * (off["fx0"] + off["fx1"])
    band_cx = (m_left + (W - m_right)) / 2.0
    sq_x0 = (band_cx - bc * side) / W
    rw = side / W
    rh = side / H
    ax.set_position([sq_x0, 0.5 - rh / 2.0, rw, rh])


def _apply_band_layout(fig: Any, ax: Any, cax: Any) -> None:
    """按当前 3D 正方形的位置, 把标题/轴标签/色标重新贴到块体上.

    ``Axes3D.apply_aspect()`` 会把 axes 收缩成边长 = min(宽, 高) 的正方形并居中,
    所以窗口一变, 正方形的位置和大小都会变. 如果色标用固定的 figure 坐标、标题用
    一次性算出的坐标, 横向拉宽窗口时它们会被留在左右两端, 中间撑出一大片空白,
    看起来像"三维区域没放大, 只是两边被拉开了".

    这里所有间距都按**英寸**折算成 figure 比例, 这样正方形放大后字号/间距不会被一起放大.
    """
    off = getattr(fig, "_slip_band_layout", None)
    if not off:
        return
    try:
        W, H = fig.get_size_inches()
    except Exception:
        return
    W = max(float(W), 1e-6)
    H = max(float(H), 1e-6)
    sq = ax.get_position()
    x0 = sq.x0 + off["fx0"] * sq.width
    x1 = sq.x0 + off["fx1"] * sq.width
    y_top = sq.y0 + off["fy_top"] * sq.height
    y_bot = sq.y0 + off["fy_bot"] * sq.height

    title_txt, ylab_txt = getattr(fig, "_slip_band_texts", (None, None))
    if title_txt is not None:
        title_txt.set_position((x0, y_top - 0.06 / H))
    if ylab_txt is not None:
        ylab_txt.set_position(((x0 + x1) / 2.0, max(y_bot - 0.42 / H, 0.008)))

    # 色标贴着块体右端 (让开 Depth 轴标签), 尺寸用英寸, 不随正方形放大
    try:
        side_in = sq.width * W
        cx = min(x1 + 1.55 / W, 0.985)
        cw = max(min(0.09 / W, 0.02), 0.004)
        ch = max(min(0.30 * side_in / H, 0.55), 0.10)
        cy = (y_top + y_bot) / 2.0 - ch / 2.0
        cax.set_position([cx, max(cy, 0.01), cw, ch])
    except Exception:
        pass


def _place_title_above_band(fig: Any, ax: Any, cax: Any, title: str) -> None:
    """把标题放进块体上方空白处, 并把 Northing 轴标签放到块体正下方.

    两个都是为了绕开 mplot3d 的排版缺陷:

    * 方形 axes 里块体居中, ``ax.set_title`` 会把标题顶到画布最上方, 裁白后
      标题与块体之间留一大段空白; 这里改为贴着块体最高点往下放, 不再额外占一行高度;
    * 走向轴几乎水平, 它自带的标签被钉在轴线中点, 正好压住中间那个刻度标签;
      而 ``labelpad`` 对 3D 轴几乎无效 (实测 7→90 只挪了 17 px), 调不开.

    做法: 先画一帧, 用"高饱和像素"(彩色块体)的包围盒定位块体; 把块体相对 3D 正方形
    的偏移量记下来, 之后窗口缩放时由 ``_apply_band_layout`` 按同一比例跟随.
    定位失败则退回 matplotlib 默认排版.
    """
    try:
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.int16)
        sat = (buf.max(axis=2) - buf.min(axis=2)) > 40
        bb = cax.get_window_extent()
        sat[:, max(int(bb.x0) - 2, 0):] = False        # 排除右侧色标
        ys, xs = np.nonzero(sat)
        if xs.size == 0:
            raise RuntimeError("no colored pixels")
        w, h = fig.canvas.get_width_height()
        x0 = float(xs.min()) / float(w)
        x1 = float(xs.max()) / float(w)
        y_top = 1.0 - float(ys.min()) / float(h)
        y_bot = 1.0 - float(ys.max()) / float(h)

        # 记下块体相对 3D 正方形的位置 (正方形会随窗口变化, 块体在其中的相对位置不变)
        sq = ax.get_position()
        off = {
            "fx0": (x0 - sq.x0) / sq.width,
            "fx1": (x1 - sq.x0) / sq.width,
            "fy_top": (y_top - sq.y0) / sq.height,
            "fy_bot": (y_bot - sq.y0) / sq.height,
        }
        try:
            zb = ax.zaxis.label.get_window_extent()
            off["fz"] = (zb.x1 / float(w) - sq.x0) / sq.width
        except Exception:
            off["fz"] = 1.02
        fig._slip_band_layout = off
        # 先把 3D 正方形放大到填满可用宽度 (绕开 mplot3d 的正方形限制)
        _fit_axes_rect(fig, ax, off)

        title_txt = None
        if str(title or "").strip():
            # va="top": 标题顶端与块体最高点对齐, 不再额外占一行画布高度
            title_txt = fig.text(x0, y_top, str(title), fontsize=12.5, fontweight="bold",
                                 color="#1a1a1a", ha="left", va="top")

        ylab_txt = None
        ylab = ax.get_ylabel()
        if str(ylab or "").strip():
            ylab_txt = fig.text((x0 + x1) / 2.0, y_bot, str(ylab), fontsize=9.5,
                                color="black", ha="center", va="top")
            ax.set_ylabel("")       # 手动放好之后再清掉自带标签, 避免两者都不见了

        fig._slip_band_texts = (title_txt, ylab_txt)
        _apply_band_layout(fig, ax, cax)
    except Exception:
        if str(title or "").strip():
            try:
                ax.set_title(str(title), fontsize=12, pad=10)
            except Exception:
                pass


def _finalize_figure(fig: Any, out_path: Optional[str], show: bool, block: bool,
                     crop: bool = True) -> None:
    """显示图窗.

    ``block=False``(默认): 立刻返回, 后续代码可继续; 脚本真正退出前会挂起直到关掉图窗
    (类似 MATLAB: figure 一直开着, 命令还能接着敲). ``block=True``: 当场等关窗再返回.
    ``crop=True``: 存盘后按墨迹包围盒裁掉白边 (只对 .png 生效).
    """
    global _KEEP_FIGURES_ATEXIT
    import matplotlib.pyplot as plt

    def _save(target: str) -> None:
        # 不用 bbox_inches="tight": 它会重新排版, 把 3D 轴标签挤出画布外裁掉;
        # 空白交给下面的 _crop_whitespace 按墨迹裁, 效果更好也更稳.
        # (箭头开关在窗口工具栏里, 不属于画布, 不会进图.)
        fig.savefig(str(target), dpi=200, facecolor="w")
        if crop:
            _crop_whitespace(str(target))

    if out_path:
        _save(str(out_path))
    if not show:
        plt.close(fig)
        return
    # 不能用 isinstance(canvas, FigureCanvasAgg): QtAgg/TkAgg 的 canvas 都继承它
    if _is_headless_backend():
        saved = out_path
        if not saved:
            saved = os.path.abspath("slip_model_show.png")
            _save(str(saved))
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
    ref_lon: Optional[float] = None,
    lonc: Optional[float] = None,
    latc: Optional[float] = None,
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
    show_slip_arrows: bool = False,
    slip_arrow_toggle: bool = True,
    vmax: Optional[float] = None,
    view_elev: float = 24.0,
    view_azim: float = -14.0,
    crop_whitespace: bool = True,
    color_values: Optional[Sequence[float]] = None,
    cmap_name: str = "jet",
    color_label: Optional[str] = None,
    symmetric: bool = False,
) -> Any:
    """绘制 3D 断面块体 + 滑移箭矢; 可选震中、地震目录与断层迹线.

    ref_lon, lonc, latc
        投影原点. 三者都不给时自动从 ``configpara.txt`` 读取
        (见 :func:`geodetic.config.resolve_geo_origin`), 不必在脚本里重复硬编码.
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
        If True, add an ``Arrows`` toggle to the window toolbar (Qt / Tk) instead of
        drawing a button inside the figure; ``a`` on the keyboard always works.
        A button in the canvas would eat plot area and pollute the whitespace crop.
    vmax
        色标上限 (m). 默认 None = 按本图滑移量自适应. 多个模型要横向比较时,
        传入同一个 ``vmax`` 才能保证颜色含义一致.
    view_elev, view_azim
        观察仰角/方位角 (度). 默认 (24, -14): 让长轴接近横向铺开. 只换观察方向,
        不改变 xyz 比例. 想要原来的斜视图可传 (30, -60).
    crop_whitespace
        存盘后按墨迹包围盒裁掉四周白边 (仅对 .png 生效). 真实比例下块体是细长条,
        axes 内部那圈空白比数据本身还大, 必须裁掉才不显得空.
    color_values
        逐片元着色量; None(默认) = 滑移量大小 |slip| (保持旧行为). 传等长数组即可
        画别的量, 例如两个模型的逐片元差值.
    cmap_name
        色标名称; 默认 ``"jet"``. 差值图建议 ``"RdBu_r"`` 或 ``"coolwarm"``.
    color_label
        色标轴标签; None = ``"slip (m)"``.
    symmetric
        True 时用以 0 为中心的对称色标 (vmin=-lim, vmax=+lim, lim = max|color_values|),
        适合差值等有正负的量; 此时忽略 ``vmax``.
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

    # 投影原点: 缺省时从 configpara.txt 读 (避免把 95.0/95.33/19.61 抄进代码)
    from geodetic.config import resolve_geo_origin

    ref_lon, lonc, latc = resolve_geo_origin(ref_lon, lonc, latc)

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
        x1f, y1f = xy2xy(xe[i], yn[i], theta)
        z1f = zr[i]
        x2f, y2f = x1f + lp[i], y1f
        z2f = z1f
        x3f, y3f = x2f, y2f - wp[i] * np.cos(p_dip[i])
        z3f = z2f - wp[i] * np.sin(p_dip[i])
        x4f, y4f = x1f, y3f

        corners = [xy2xy(x, y, -theta) for x, y in ((x1f, y1f), (x2f, y2f), (x3f, y3f), (x4f, y4f))]
        xyz = np.column_stack([np.array(corners)[:, 0] / 1000.0, np.array(corners)[:, 1] / 1000.0,
                               np.array([z1f, z2f, z3f, z3f]) / 1000.0])
        polys.append(xyz)
        xo_m.append(float(np.mean(xyz[:, 0])))
        yo_m.append(float(np.mean(xyz[:, 1])))
        zo_m.append(float(np.mean(xyz[:, 2])))

        xu, yu = xy2xy(float(slip1[i]), float(slip2[i] * np.cos(p_dip[i])), -theta)
        zv = float(slip2[i] * np.sin(p_dip[i]))
        xv_list.append(xu)
        yv_list.append(yu)
        zv_list.append(zv)
        slip_mag[i] = float(np.sqrt(xu * xu + yu * yu + zv * zv))

    x0_ref, y0_ref = ll2xy(lonc, latc, ref_lon)
    xo = np.asarray(xo_m)
    yo = np.asarray(yo_m)
    zo = np.asarray(zo_m)
    xv = np.asarray(xv_list)
    yv = np.asarray(yv_list)
    zv = np.asarray(zv_list)
    slipmax = max(float(np.max(slip_mag)) if n else 1.0, 1e-20)
    if color_values is None:
        cvals = np.sqrt(slip1 ** 2 + slip2 ** 2)
    else:
        cvals = np.asarray(color_values, dtype=np.float64).reshape(-1)
        if cvals.size != n:
            raise ValueError(
                "color_values 有 %d 个值, 与片元数 %d 不一致"
                % (cvals.size, n)
            )
    cmax = max(float(np.max(cvals)) if n else 1.0, 1e-20)
    if vmax is not None:
        # 显式色标上限: 多个模型共用同一刻度, 便于横向比较
        cmax = max(float(vmax), 1e-20)

    # 方形画布: Axes3D.apply_aspect() 会把 3D axes 收缩成正方形
    # (axes3d.py:391 shrunk_to_aspect(box_aspect=1, ...)), 画布越宽, 两侧死区越大.
    fig = plt.figure(figsize=(12.2, 9.6), dpi=150)
    ax = fig.add_axes([0.13, 0.13, 0.80, 0.78], projection="3d")
    try:
        ax.set_proj_type("ortho")   # 正交投影: 473 km 长轴的远端不再被透视缩短
    except Exception:
        pass
    normc = colors.Normalize(vmin=0.0, vmax=cmax)
    if symmetric:
        # 差值等有正负的量: 以 0 为中心, 两极值对称 -> 颜色才表示"谁大谁小"
        lim = max(float(np.max(np.abs(cvals))) if n else 1.0, 1e-20)
        normc = colors.Normalize(vmin=-lim, vmax=lim)
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, TypeError, KeyError):
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)

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

    # 色标挪到 axes 外侧: 原来的 [0.93, 0.22, 0.018, 0.56] 会压住 Depth 轴标签
    cax = fig.add_axes([0.945, 0.40, 0.009, 0.20])
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=normc), cax=cax, aspect=30)
    cb.ax.tick_params(labelsize=8.5, length=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#cccccc")
    cb.set_label(color_label or "slip (m)", fontsize=9.5, labelpad=5)

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
            ax.plot(xsa, ysa, [0.0, 0.0], c="k", linewidth=0.8)
            limit_extras.append((xsa, ysa, np.array([0.0, 0.0])))

    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    # 深度轴只有 ~21 km 跨度, 刻度多了会挤在一起; 走向轴很长, 需要更密一些
    for axis, nbins in ((ax.xaxis, 3), (ax.yaxis, 5), (ax.zaxis, 3)):
        axis.set_major_locator(MaxNLocator(nbins=nbins))
        axis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis.get_major_formatter().set_scientific(False)
    ax.tick_params(axis="both", which="major", labelsize=8.5, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=8.5, pad=2)
    ax.set_xlabel("Easting (km)", fontsize=9.5, labelpad=7)
    ax.set_ylabel("Northing (km)", fontsize=9.5, labelpad=7)
    ax.set_zlabel("Depth (km)", fontsize=9.5, labelpad=10)
    # 标题不在这里放: 方形 axes 里块体居中, ax.set_title 会顶到画布最上方,
    # 裁白后留一大段空白. 改为画完后贴着块体上沿放 (见 _place_title_above_band).
    # 面板/网格: 3D 默认的灰色面板 + 三面网格线会横穿倾斜断层, 噪声比数据还大
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#e4e4e4")
        axis.pane.set_linewidth(0.6)
        axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis.line.set_color("#cccccc")

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
    # 保持 XYZ 等比例，但按实际轴长计算比值，避免 z 方向被人为放大。
    try:
        xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
        ax.set_box_aspect((max(xl[1] - xl[0], 1e-9), max(yl[1] - yl[0], 1e-9), max(zl[1] - zl[0], 1e-9)))
    except Exception:
        pass
    # 视角: 数据走向约 352°(近 N-S), 默认 elev=30/azim=-60 会把 473 km 长轴斜穿画面,
    # 投影后只占视口宽度的 40%; 改成让长轴横铺 (~78%). 只换观察方向, 不改比例.
    try:
        ax.view_init(elev=float(view_elev), azim=float(view_azim))
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
        max_arrow_len = 0.012 * max(box_diag, 1e-9)
        # keep arrows very close to the original fault plane; only slightly larger than the patch
        med_patch = float(np.median(np.minimum(lp, wp))) / 1000.0
        lift = max(0.05 * med_patch, 0.004 * box_diag)
        head_frac = 0.08
        idx = np.flatnonzero(slip_mask)

        shaft_segs: List[np.ndarray] = []
        head_tris: List[np.ndarray] = []

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
            lc_fg = Line3DCollection(shaft_segs, colors="#000000", linewidths=0.8, alpha=0.9)
            ax.add_collection3d(lc_fg)
            lc_fg.set_visible(show_slip_arrows)
            slip_artists.append(lc_fg)

        if head_tris:
            heads = Poly3DCollection(
                head_tris,
                facecolors="#000000",
                edgecolors="none",
                linewidths=0.0,
                alpha=0.9,
            )
            ax.add_collection3d(heads)
            heads.set_visible(show_slip_arrows)
            slip_artists.append(heads)

        fig._slip_arrow_artists = slip_artists
        fig._slip_face_coll = coll

        if slip_artists and show and slip_arrow_toggle and not _is_headless_backend():

            def _set_arrows(vis: bool) -> None:
                for art in slip_artists:
                    art.set_visible(vis)
                # 箭头藏起来时把面恢复不透明, 免得白看一层半透明
                try:
                    coll.set_alpha(0.72 if vis else 1.0)
                except Exception:
                    pass
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass

            # 开关放进窗口工具栏 (状态栏), 不占画布; 工具栏不支持时退回快捷键
            _add_toolbar_toggle(fig, "Arrows", show_slip_arrows, _set_arrows)

            def _on_key(event: Any) -> None:
                if (event.key or "").lower() != "a":
                    return
                vis = not slip_artists[0].get_visible()
                _set_arrows(vis)
                var = getattr(fig, "_slip_arrow_toolbar_var", None)
                if var is not None:
                    var.set(1 if vis else 0)

            fig.canvas.mpl_connect("key_press_event", _on_key)

    _place_title_above_band(fig, ax, cax, title)

    # 窗口缩放时重新贴一次: 3D 正方形随窗口变化, 色标/标题/标签要跟着块体走,
    # 否则横向拉宽会把它们甩到画布两端, 中间撑出空白.
    if show:
        def _on_resize(_event: Any) -> None:
            try:
                off = getattr(fig, "_slip_band_layout", None)
                if off:
                    _fit_axes_rect(fig, ax, off)   # 先按新窗口重算正方形大小
                ax.apply_aspect()                  # 再让 axes 收缩到该正方形
                _apply_band_layout(fig, ax, cax)
                fig.canvas.draw_idle()
            except Exception:
                pass

        try:
            fig._slip_resize_cid = fig.canvas.mpl_connect("resize_event", _on_resize)
        except Exception:
            pass

    _finalize_figure(fig, str(out_path) if out_path else None, show, block,
                     crop=crop_whitespace)
    return fig


if __name__ == "__main__":
    import sys

    # =========================================================================
    # 直接运行本文件时改下面配置 (IDE 里 Run 即可, 不必命令行参数)
    # =========================================================================
    # 项目根目录（本包位于 <root>/geodetic/ 下），.mat / 断层迹线都放在根目录。
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # --- 输入 ---
    slip_path = os.path.join(here, "py_inversion_iint1.mat")
    # 投影原点 (震中/迹线 km 换算用): 留 None 让它自己读 configpara.txt
    ref_lon = lonc = latc = None
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
