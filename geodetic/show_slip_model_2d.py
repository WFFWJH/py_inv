r"""2D 展开剖面 — 对应 ``show_slip_model_2d.m``.

把三维断层面沿走向摊平: 横轴 = 沿走向里程, 纵轴 = 沿倾向距离 (或深度).
片元在展开图里是真正的矩形, 不因倾角被压缩, 所以比 3D 斜视图更适合读数.

与 ``.m`` 的差异见模块末尾 ``NOTE``; 与 ``comsol_scripts/comsol_result.py`` 里
``_unfold_rects`` 的区别: 那个展开到「深度」, 本模块默认展开到「沿倾向距离」.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# --------------------------------------------------------------------------- #
# 展开
# --------------------------------------------------------------------------- #
def unfold_patches(
    slip_model: np.ndarray,
    *,
    seg_gap_km: float = 0.0,
    along: str = "dip",
    seg_order: str = "desc",
) -> Dict[str, Any]:
    """把片元表摊平到「沿走向里程 — 沿倾向距离」平面 (km).

    参数
    ----
    slip_model
        ``(N, >=13)`` 的片元表; 列 0/1/2 = 段号/片元号/层号, 6/7 = 长宽 (m),
        5 = 顶深 (m, 负值), 9 = 倾角 (度), 11/12 = 走滑/倾滑 (m).
    seg_gap_km
        段与段之间留的空白 (km), 默认 0 = 段首尾相接 (``.m`` 用 5 km).
    along
        ``"dip"``: 纵轴是沿倾向面内距离 (断层面摊平, 片元高度 = wp);
        ``"depth"``: 纵轴是深度 (片元高度 = wp·sin(dip)).
    seg_order
        ``"desc"``(默认, 同 ``.m`` 的 ``max:-1:1``): 段号从大到小排.
        这不是随手写的 —— 数据里每段内部纬度随走向递增、而段号从北往南编,
        只有倒序排布才能让纬度沿横轴单调, 横轴才切得成纬度.

    返回
    ----
    dict(rects, seg_bounds, seg_len, total_km, y_max_km, slip)
    """
    data = np.asarray(slip_model, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 13:
        raise ValueError("slip_model 期望形状 (N, >=13)")

    seg = data[:, 0].astype(int)
    pid = data[:, 1].astype(int)
    layer = data[:, 2].astype(int)
    lp = data[:, 6] / 1000.0
    wp = data[:, 7] / 1000.0
    zp = data[:, 5] / 1000.0                       # 顶深, 负值
    dip = np.deg2rad(data[:, 9])
    n = data.shape[0]

    # --- 层内沿走向起点: 同一 (段, 层) 内按片元号排序后累加长度 ---
    x0 = np.zeros(n, dtype=np.float64)
    for s in np.unique(seg):
        for lay in np.unique(layer[seg == s]):
            idx = np.flatnonzero((seg == s) & (layer == lay))
            idx = idx[np.argsort(pid[idx], kind="stable")]
            x0[idx] = np.concatenate(([0.0], np.cumsum(lp[idx])[:-1]))
    x1 = x0 + lp

    # --- 沿倾向起点: 用顶深换算, 比 .m 里"累加上一层 wp"稳 ---
    # .m 的 ddl(n-1) 在"层数>2 且各层 wp 相同"时会越界; 而且它假设各层严格首尾相接.
    sin_dip = np.maximum(np.sin(dip), 1e-6)
    if along == "dip":
        y0 = -zp / sin_dip                          # 深度 -> 沿倾向距离
        dy = wp
    elif along == "depth":
        y0 = -zp
        dy = wp * sin_dip
    else:
        raise ValueError("along 只能是 'dip' 或 'depth'")
    y1 = y0 + dy

    # --- 每段沿走向长度 (同段各层应一致, 取最大以免边界误差) ---
    seg_len: Dict[int, float] = {}
    for s in np.unique(seg):
        seg_len[int(s)] = max(
            float(np.sum(lp[(seg == s) & (layer == lay)]))
            for lay in np.unique(layer[seg == s])
        )

    # --- 段间留 gap; 默认倒序 (同 .m), 让纬度沿横轴单调 ---
    offset: Dict[int, float] = {}
    acc = 0.0
    for s in sorted(seg_len, reverse=(str(seg_order).lower() != "asc")):
        offset[s] = acc
        acc += seg_len[s] + seg_gap_km
    total = max(acc - seg_gap_km, 0.0) if seg_len else 0.0

    rects: List[List[Tuple[float, float]]] = []
    for i in range(n):
        xa = offset[int(seg[i])] + x0[i]
        xb = offset[int(seg[i])] + x1[i]
        rects.append([(xa, y0[i]), (xb, y0[i]), (xb, y1[i]), (xa, y1[i])])

    slip = np.hypot(data[:, 11], data[:, 12])
    bounds = {s: (offset[s], offset[s] + seg_len[s]) for s in seg_len}
    return {
        "rects": rects,
        "seg_bounds": bounds,
        "seg_len": seg_len,
        "total_km": total,
        "y_max_km": float(np.max(y1)) if n else 1.0,
        "slip": slip,
    }


# --------------------------------------------------------------------------- #
# 纬度映射
# --------------------------------------------------------------------------- #
def patch_latitudes(
    slip_model: np.ndarray,
    *,
    ref_lon: float,
    lonc: float,
    latc: float,
) -> np.ndarray:
    """每个片元中心的纬度 (度).

    片元表里的 ``xe/yn`` 是相对 ``(lonc, latc)`` 的横轴墨卡托坐标 (m), 先按走向旋到
    断层局部系、取片元中心, 再旋回来加上原点, 最后用 :func:`xy2ll` 反投影。
    """
    from geodetic.coords import ll2xy, xy2ll, xy2xy

    data = np.asarray(slip_model, dtype=np.float64)
    x0, y0 = ll2xy(lonc, latc, ref_lon)
    x0 = float(np.ravel(x0)[0])
    y0 = float(np.ravel(y0)[0])

    d2r = np.pi / 180.0
    theta = (90.0 - data[:, 8]) * d2r
    x1f, y1f = xy2xy(data[:, 3], data[:, 4], theta)
    xc_f = x1f + data[:, 6] / 2.0
    yc_f = y1f - data[:, 7] * np.cos(data[:, 9] * d2r) / 2.0
    xc, yc = xy2xy(xc_f, yc_f, -theta)

    _lon, lat = xy2ll(xc + x0, yc + y0, ref_lon)
    return np.asarray(lat, dtype=np.float64).ravel()


def _lat_axis_functions(rects: List[List[Tuple[float, float]]], lat: np.ndarray):
    """由片元中心建「沿走向里程 <-> 纬度」的双向插值 (都按单调化后处理)."""
    xc = np.array([0.5 * (r[0][0] + r[1][0]) for r in rects], dtype=np.float64)
    order = np.argsort(xc, kind="stable")
    xs = xc[order]
    ls = np.maximum.accumulate(lat[order])      # 抹掉几 km 级的回折, 保证可逆
    keep = np.concatenate(([True], np.diff(xs) > 1e-9))
    xs, ls = xs[keep], ls[keep]

    def km2lat(v: Any) -> Any:
        return np.interp(v, xs, ls)

    lu, idx = np.unique(ls, return_index=True)
    xu = xs[idx]

    def lat2km(v: Any) -> Any:
        return np.interp(v, lu, xu)

    return km2lat, lat2km, float(ls[0]), float(ls[-1])


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def show_slip_model_2d(
    slip_model: np.ndarray,
    *,
    title: str = "slip model",
    vmax: Optional[float] = None,
    seg_gap_km: float = 0.0,
    along: str = "dip",
    seg_order: str = "desc",
    xaxis: str = "km",
    ref_lon: Optional[float] = None,
    lonc: Optional[float] = None,
    latc: Optional[float] = None,
    config_dir: Optional[Union[str, os.PathLike]] = None,
    cmap_name: str = "jet",
    vertical_exaggeration: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
    out_path: Optional[Union[str, os.PathLike]] = None,
    show: bool = True,
    block: Optional[bool] = None,
    annotate_segments: bool = True,
    show_max_tick: bool = True,
    ylabel: Optional[str] = None,
) -> Any:
    """画 2D 展开剖面.

    Parameters
    ----------
    xaxis
        ``"km"``(默认): 横轴为沿走向里程; ``"lat"``: 横轴换成纬度 (°N);
        ``"both"``: 下面 km、上面再加一条纬度轴, 两套坐标同时可读.
        纬度用 :func:`geodetic.coords.xy2ll` 由片元中心反投影得到.
    ref_lon, lonc, latc
        投影原点. 选 ``"lat"`` / ``"both"`` 时需要, 但**不必手填**: 缺省时会从
        ``configpara.txt`` 的同名 key 读取 (见 :func:`geodetic.config.resolve_geo_origin`).
    config_dir
        ``configpara.txt`` 所在目录; None = 项目根目录.
    vertical_exaggeration
        垂向放大倍数. **默认 1.0 = 沿走向与沿倾向同一比例**, 片元保持真实形状;
        想看清薄层可传 2~5, 图上会在 y 轴标签里标出 ``VE ×N``.
    figsize
        None = 按数据比例自动定高 (1:1 时是一条长条, 不会留大片空白).
    vmax
        色标上限 (m); None = 按本图自适应. 多张图要横向比较时传同一个值.
    seg_gap_km
        段间空白 (km). ``.m`` 里硬编码 5000 m, 这里做成参数.
    along
        ``"dip"``(默认, 同 ``.m``): 纵轴为沿倾向面内距离;
        ``"depth"``: 纵轴为深度, 便于和地震剖面/深度约束对照.
    show_max_tick
        把最大滑移量强制加进色标刻度 (沿用 ``.m`` 的做法), 便于读峰值.
    """
    import matplotlib

    if not _env_flag("SHOW_SLIP", default=True):
        show = False
    if out_path and not show:
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")
    if block is None:
        block = _env_flag("SHOW_SLIP_BLOCK", default=False)

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, MaxNLocator

    data = np.asarray(slip_model, dtype=np.float64)
    u = unfold_patches(data, seg_gap_km=seg_gap_km, along=along)
    rects, slip = u["rects"], u["slip"]

    cmax = float(np.max(slip)) if slip.size else 1.0
    if vmax is not None:
        cmax = max(float(vmax), 1e-20)
    cmax = max(cmax, 1e-20)
    norm = colors.Normalize(vmin=0.0, vmax=cmax)
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, TypeError, KeyError):
        from matplotlib import cm as _cm
        cmap = _cm.get_cmap(cmap_name)

    # ---- 画布: 按数据真实比例定尺寸, 不留大片空白 ----
    # 默认 1:1 (vertical_exaggeration=1): 沿走向和沿倾向同一比例, 片元不被拉长/压扁.
    x_span = max(u["total_km"], 1e-6)
    y_span = max(u["y_max_km"], 1e-6)
    ve = float(vertical_exaggeration)
    if not np.isfinite(ve) or ve <= 0:
        ve = 1.0

    mode = str(xaxis).lower()
    both = mode == "both"
    m_left, m_right = 0.95, 1.60        # 英寸: 左给 y 标签+刻度, 右给色标+标签
    m_bot = 0.52                        # 下给 x 刻度+标签
    # 上: 标题 + 段号; 双轴时再给顶部纬度轴留刻度+标签的位置
    m_top = 0.72 if both else 0.36
    if figsize is None:
        fig_w = 14.0
        ax_w = max(fig_w - m_left - m_right, 2.0)
        ax_h = ax_w * y_span * ve / x_span      # ve 放大的是屏幕上的 y 长度
        figsize = (fig_w, max(m_bot + ax_h + m_top, 1.35))
    fig_w, fig_h = float(figsize[0]), float(figsize[1])
    ax_w = max(fig_w - m_left - m_right, 2.0)
    ax_h = min(max(fig_h - m_bot - m_top, 0.25), ax_w * y_span * ve / x_span)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([m_left / fig_w, m_bot / fig_h, ax_w / fig_w, ax_h / fig_h])
    ax.set_aspect(ve)                   # 1.0 = 真实比例; >1 才是垂向放大

    face = [cmap(norm(v)) for v in slip]
    # 描边用面色: 片元之间本来就有缝, 用黑边会在 858 个格子上糊成网
    ax.add_collection(PolyCollection(rects, facecolors=face, edgecolors=face,
                                     linewidths=0.3, antialiased=True))

    ax.set_xlim(0.0, x_span)
    ax.set_ylim(y_span, 0.0)            # 0 在上, 深度向下

    # 段界 + 段号
    if annotate_segments and u["seg_bounds"]:
        for k, (xa, xb) in enumerate(u["seg_bounds"].values(), start=1):
            for xx in (xa, xb):
                ax.axvline(xx, color="#666666", lw=0.6, ls=(0, (4, 3)), zorder=3)
            ax.text(0.5 * (xa + xb), 1.010, "S%d" % k, ha="center", va="bottom",
                    fontsize=8.5, color="#555555", zorder=4,
                    transform=ax.get_xaxis_transform())

    # 1:1 时纵轴很矮, 刻度要少给几个
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3 if ax_h < 1.6 else 6))
    ax.tick_params(labelsize=9, pad=3, colors="#333333")
    ax.grid(True, color="#e8e8e8", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- 横轴: 沿走向里程 / 纬度 / 两者 ----
    if mode in ("lat", "both"):
        from geodetic.config import resolve_geo_origin

        ref_lon, lonc, latc = resolve_geo_origin(ref_lon, lonc, latc, config_dir)
        lat = patch_latitudes(data, ref_lon=ref_lon, lonc=lonc, latc=latc)
        km2lat, lat2km, lat_lo, lat_hi = _lat_axis_functions(rects, lat)
        step = 0.5 if (lat_hi - lat_lo) > 3.0 else 0.25
        lat_ticks = np.arange(np.ceil(lat_lo / step) * step, lat_hi + 1e-9, step)

    if mode == "lat":
        ax.xaxis.set_major_locator(FixedLocator(lat2km(lat_ticks)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: "%.1f" % float(km2lat(v))))
        ax.set_xlabel("Latitude (°N)", fontsize=10)
    else:
        ax.set_xlabel("Along-strike distance (km)", fontsize=10)
        if mode == "both":
            # 顶部再加一条纬度轴: 下 km、上 lat, 论文里两套坐标都能读
            sec = ax.secondary_xaxis("top", functions=(km2lat, lat2km))
            sec.xaxis.set_major_locator(FixedLocator(lat_ticks))
            sec.xaxis.set_major_formatter(
                FuncFormatter(lambda v, _pos: "%.1f" % float(v)))
            # pad 放大: 让纬度刻度浮到段号 S1..Sn 上面, 两者不打架
            sec.tick_params(labelsize=8.5, pad=13, colors="#333333")
            sec.set_xlabel("Latitude (°N)", fontsize=9.5, labelpad=3)
            sec.spines["top"].set_color("#cccccc")
            sec.spines["top"].set_linewidth(0.8)
    if ylabel is None:
        # 1:1 时 axes 只有 ~0.5 in 高, 竖排标签再长就会被画布裁掉
        ylabel = "Along-dip (km)" if along == "dip" else "Depth (km)"
    # 只有真做了垂向放大才标注, 1:1 不用画蛇添足
    ax.set_ylabel(ylabel if abs(ve - 1.0) < 1e-9 else "%s   (VE ×%.1f)" % (ylabel, ve),
                  fontsize=10)

    # 色标: 高度跟着数据条走, 别比数据还高
    ch_in = min(max(ax_h * 1.5, 0.45), max(fig_h - m_bot - m_top, 0.45))
    cax = fig.add_axes([(fig_w - 1.05) / fig_w,
                        (m_bot + ax_h / 2.0 - ch_in / 2.0) / fig_h,
                        0.10 / fig_w, ch_in / fig_h])
    cb = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#cccccc")
    cb.set_label("slip (m)", fontsize=9.5, labelpad=5)
    if show_max_tick and slip.size:
        ticks = list(cb.get_ticks())
        smax = float(np.max(slip))
        if all(abs(smax - t) > 0.05 * cmax for t in ticks):
            ticks.append(smax)
        cb.set_ticks(sorted(t for t in ticks if -1e-9 <= t <= cmax + 1e-9))
    # 刻度格式统一, 否则补进来的最大值会显示成 7.238 而其它是 7
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f" if cmax < 20 else "%.0f"))

    fig.text(m_left / fig_w, 1.0 - 0.16 / fig_h, title, fontsize=12.5,
             fontweight="bold", color="#1a1a1a")

    if out_path:
        fig.savefig(str(out_path), dpi=dpi, facecolor="w")
    if not show:
        plt.close(fig)
        return fig
    _show_or_wait(fig, block)
    return fig


# --------------------------------------------------------------------------- #
# 小工具 (与 show_slip_model.py 保持同一套行为)
# --------------------------------------------------------------------------- #
def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name, "1" if default else "0").strip().lower()
    return v not in ("0", "false", "no")


def _show_or_wait(fig: Any, block: bool) -> None:
    import atexit
    import matplotlib.pyplot as plt

    if block:
        plt.show(block=True)
        return
    plt.ion()
    try:
        fig.show()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.1)
    except Exception:
        pass

    def _wait() -> None:
        try:
            if plt.get_fignums():
                plt.show(block=True)
        except Exception:
            pass

    atexit.register(_wait)


if __name__ == "__main__":
    import sys

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slip_path = os.path.join(here, "py_inversion_iint1.mat")
    out_png = os.path.splitext(slip_path)[0] + "_2d.png"

    if not os.path.isfile(slip_path):
        print("找不到 slip 文件:", slip_path, file=sys.stderr)
        sys.exit(1)

    from geodetic.show_slip_model import load_slip_model_from_file

    sm = load_slip_model_from_file(slip_path)
    show_slip_model_2d(
        sm,
        title="py_inversion iint0",
        out_path=out_png,
        show=_env_flag("SHOW_SLIP"),
        seg_gap_km=0.0,
        along="dip",
        # xaxis=_env_str("SLIP_XAXIS", "km"),
        xaxis="lat",
        ref_lon=95.0, lonc=95.33, latc=19.61,     # xaxis="lat" 时需要
    )
    print("Done:", os.path.abspath(out_png))


# NOTE 与 ``show_slip_model_2d.m`` 的差异
# -------------------------------------
# 采纳:
#   * 横轴沿走向里程、纵轴沿倾向距离, 片元画成矩形 (面内展开, 不做 sin(dip) 压缩);
#   * 段间留空、逐段排布;
#   * 色标刻度里补上最大滑移量 (便于直接读峰值).
# 不采纳:
#   * ``.m`` 用「累加上一层 wp」定位层起点 (``ddl(n-1)``), 当层数 > 2 且各层 wp
#     相同时会越界; 本实现改用顶深 ``zp`` 换算 ``-zp/sin(dip)``, 与数据自洽且不受层数影响.
#   * ``.m`` 段号倒序遍历 (``max:-1:1``); 这里正序并标 S1..Sn, 读起来更顺.
#   * ``.m`` 的 ``axis equal``: 保留「等比例」这个意图, 但画布尺寸不再写死 ——
#     按数据比例自动算图高 (493 km x 20 km -> 14 x 1.35 in), 所以既不失真也不留大片空白.
#     真要看清薄层时显式传 ``vertical_exaggeration=2~5``, y 轴标签会标出 ``VE ×N``.
#   * ``.m`` 里那段 ``scatter3`` 是 3D 版本的残留, 放在循环里且用了 3D API, 在 2D
#     图上无效, 故未移植.
