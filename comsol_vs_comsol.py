"""Per-model: Okada + uniform/taper COMSOL; dropdown A | B | A−B."""
import os

import matplotlib.pyplot as plt
import mph
import numpy as np
from matplotlib.backends.qt_compat import QtWidgets

from calc_green import _xy2xy
from calc_okada import calc_okada
from load_fault_one_plane import load_fault_one_plane
from paragram import (
    n_layer, len_top, layers, width, ref, axis_range, fault_file,
    l_ratio, w_ratio, dip, x1d_m, y1d_m, xe, yn,
)
from show_slip_model import show_slip_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
I, J = 0, 0
FAULT_TYPE = 1   # 1: strike-slip, 2: dip-slip
NU, U_SLIP = 0.25, 1.0

MPH_LIST = [  # (I==0, I>0)；每组单独开图，组间不混比
    ("top0_most_refine_extend.mph", "top1_most_refine_extend.mph"),
]

PLOT = dict(
    cmap="jet", figsize=(13, 12), dpi=120, digits=4, interpolation="bilinear",
    cbar_shrink=0.75, cbar_fraction=0.035, cbar_pad=0.02, cbar_labelpad=1,
)
FIGS, _SYNCING = [], False


def _bind_view_sync(fig):
    def apply(xlim, ylim):
        global _SYNCING
        if _SYNCING:
            return
        _SYNCING = True
        try:
            for f in FIGS:
                for ax in f.axes:
                    if ax.images:
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
                f.canvas.draw_idle()
        finally:
            _SYNCING = False

    def on_lim(ax):
        if not _SYNCING and ax.images:
            apply(ax.get_xlim(), ax.get_ylim())

    def on_key(event):
        if event.key != "r" or not FIGS:
            return
        for ax in FIGS[0].axes:
            if ax.images:
                apply(ax.images[0].get_extent()[:2], ax.images[0].get_extent()[2:])
                break

    for ax in fig.axes:
        if ax.images:
            ax.callbacks.connect("xlim_changed", on_lim)
            ax.callbacks.connect("ylim_changed", on_lim)
    fig.canvas.mpl_connect("key_press_event", on_key)
    FIGS.append(fig)


def _clim_ticks(cbar, vmin, vmax, symmetric=False):
    dig = PLOT["digits"]
    if symmetric:
        if vmax == 0.0:
            ticks, labels = [0.0], ["0"]
        else:
            ticks = [vmin, 0.0, vmax]
            labels = [f"min: {vmin:.{dig}f}", "0", f"max: {vmax:.{dig}f}"]
    elif vmin == vmax:
        ticks, labels = [vmin], [f"{vmin:.{dig}f}"]
    elif vmin < 0.0 < vmax:
        ticks = [vmin, 0.0, vmax]
        labels = [f"min: {vmin:.{dig}f}", "0", f"max: {vmax:.{dig}f}"]
    else:
        ticks = [vmin, vmax]
        labels = [f"min: {vmin:.{dig}f}", f"max: {vmax:.{dig}f}"]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)


def plot_field(ax, extent, z, title, cbar_label=None):
    """画场；cbar_label 为 None 时不建 colorbar（由外部统一创建）。"""
    vmin, vmax = float(z.min()), float(z.max())
    im = ax.imshow(
        z, origin="lower", extent=extent, aspect="equal",
        cmap=PLOT["cmap"], vmin=vmin, vmax=vmax,
        interpolation=PLOT["interpolation"], resample=True, rasterized=True,
    )
    ax.set(title=title, xlabel="x (km)", ylabel="y (km)")
    if cbar_label is None:
        return im
    cbar = plt.colorbar(
        im, ax=ax, shrink=PLOT["cbar_shrink"],
        fraction=PLOT["cbar_fraction"], pad=PLOT["cbar_pad"],
    )
    cbar.set_label(cbar_label, labelpad=PLOT["cbar_labelpad"])
    _clim_ticks(cbar, vmin, vmax)
    return im


def _make_cbar(fig, mappable, ax, label):
    cbar = fig.colorbar(
        mappable, ax=ax, shrink=PLOT["cbar_shrink"],
        fraction=PLOT["cbar_fraction"], pad=PLOT["cbar_pad"],
    )
    cbar.set_label(label, labelpad=PLOT["cbar_labelpad"])
    return cbar


def _apply_u0(model, fault_type, u0_pair):
    solid = model.component("comp1").physics("solid")
    direction = (
        [["free"], ["prescribed"], ["free"]] if fault_type == 1
        else [["prescribed"], ["free"], ["free"]]
    )
    for feat, u0 in zip(("disp1", "disp2"), u0_pair):
        solid.feature(feat).set("Direction", direction)
        solid.feature(feat).set("U0", u0)


def set_excluded_edges(model, mode):
    """mode: 'none' → .set() 清空；'all' → .all() 排除全部边。"""
    solid = model.component("comp1").physics("solid")
    for feat in ("disp1", "disp2"):
        sel = solid.feature(feat).selection("excludedEdges")
        if mode == "all":
            sel.all()
        else:
            sel.set()


def set_slip_bc(model, fault_type, *args, **kwargs):
    if fault_type == 1:
        u0 = ([[0], [0.5], [0]], [[0], [0.5], [0]])
    else:
        u0 = ([[-0.5], [0], [0]], [[0.5], [0], [0]])
    _apply_u0(model, fault_type, u0)


def _taper_y(a, b, edge_tol):
    """走向 (y) 一维 raised-cosine。"""
    return (
        f"if(min(y-{a}[m],{b}[m]-y)<{edge_tol}[m],"
        f"0.5*(1-cos(pi*min(y-{a}[m],{b}[m]-y)/{edge_tol}[m])),1)"
    )


def _taper_z(c, d, edge_tol, dip_deg):
    """倾向投影 (z/cos(dip)) 一维 raised-cosine。"""
    return (
        f"if(min(z/cos({dip_deg}*pi/180)-{c}[m],{d}[m]-z/cos({dip_deg}*pi/180))<{edge_tol}[m],"
        f"0.5*(1-cos(pi*min(z/cos({dip_deg}*pi/180)-{c}[m],{d}[m]-z/cos({dip_deg}*pi/180))/{edge_tol}[m])),1)"
    )


def _u0_from_taper(fault_type, taper):
    if fault_type == 1:
        return ([["0"], [f"0.5*({taper})"], ["0"]], [["0"], [f"0.5*({taper})"], ["0"]])
    return ([[f"-0.5*({taper})"], ["0"], ["0"]], [[f"0.5*({taper})"], ["0"], ["0"]])


def set_slip_bc_taper(model, fault_type, i, j, L, W, edge_tol, dip_deg=None):
    """二维 taper：y 约束 × z 约束。"""
    dip_deg = float(dip[0] if dip_deg is None else dip_deg)
    a, b, c, d = L * j, L * (j + 1), -W * (i + 1), -W * i
    taper = f"({_taper_y(a, b, edge_tol)})*({_taper_z(c, d, edge_tol, dip_deg)})"
    _apply_u0(model, fault_type, _u0_from_taper(fault_type, taper))


def set_slip_bc_taper1d(model, fault_type, i, j, L, W, edge_tol, dip_deg=None):
    """一维 taper：strike-slip 只用 y；dip-slip 只用 z。"""
    dip_deg = float(dip[0] if dip_deg is None else dip_deg)
    a, b, c, d = L * j, L * (j + 1), -W * (i + 1), -W * i
    taper = (
        _taper_y(a, b, edge_tol) if fault_type == 1
        else _taper_z(c, d, edge_tol, dip_deg)
    )
    _apply_u0(model, fault_type, _u0_from_taper(fault_type, taper))


def run_comsol(client, mph_path, L, W, i, j, fault_type, coords, set_bc,
               edge_tol=None, excl_mode="none"):
    mj = client.load(mph_path)
    model = mj.java
    r1 = model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")
    r1.set("size", [str(L), str(W)])
    r1.set("pos", [str(L * j), str(W * i)])
    set_bc(model, fault_type, i, j, L, W, edge_tol=edge_tol)
    set_excluded_edges(model, excl_mode)
    model.sol("sol1").runAll()
    model.result().numerical().create("uvw", "Interp")
    interp = model.result().numerical("uvw")
    interp.setInterpolationCoordinates(coords)
    interp.set("expr", ["u", "v", "w"])
    data = np.asarray(interp.getData())
    client.remove(mj)
    return data[0, 0], data[1, 0], data[2, 0]


def plot_source_picker(results, extent, title=""):
    names = list(results.keys())
    state = {"a": names[0], "b": names[1], "busy": False, "cbar": "individual"}
    comps = "ENU"

    fig, axes = plt.subplots(
        3, 3, figsize=PLOT["figsize"], dpi=PLOT["dpi"], sharex=True, sharey=True,
    )
    if title:
        fig.suptitle(title, fontsize=11)

    # 先建 imshow，colorbar 按模式重建
    images = []
    for row, comp in enumerate(comps):
        va, vb = results[state["a"]][row], results[state["b"]][row]
        images.append([
            plot_field(axes[row, 0], extent, va, f"{state['a']}: {comp}"),
            plot_field(axes[row, 1], extent, vb, f"{state['b']}: {comp}"),
            plot_field(axes[row, 2], extent, va - vb, f"{state['a']}−{state['b']}: {comp}"),
        ])

    colorbars = []

    def clear_cbars():
        for c in colorbars:
            c.remove()
        colorbars.clear()

    def create_cbars():
        """individual: 9 个；shareAB: 每行 前两列共用 1 个 + diff 1 个。"""
        clear_cbars()
        if state["cbar"] == "individual":
            for row, comp in enumerate(comps):
                for col, lab in enumerate((
                    f"u{comp.lower()} (m)", f"u{comp.lower()} (m)", "diff (m)",
                )):
                    colorbars.append(_make_cbar(fig, images[row][col], axes[row, col], lab))
        else:
            for row, comp in enumerate(comps):
                # 前两列共用色标（挂在两列右侧）
                colorbars.append(_make_cbar(
                    fig, images[row][0], axes[row, :2], f"u{comp.lower()} (m)",
                ))
                colorbars.append(_make_cbar(
                    fig, images[row][2], axes[row, 2], "diff (m)",
                ))
        fig.canvas.draw_idle()

    def apply_clim(im, cbar, z, vmin=None, vmax=None, symmetric=False):
        if symmetric:
            vmax = float(np.max(np.abs(z)))
            vmin = -vmax
        elif vmin is None:
            vmin, vmax = float(z.min()), float(z.max())
        im.set_data(z)
        im.set_clim(vmin, vmax)
        if cbar is not None:
            _clim_ticks(cbar, vmin, vmax, symmetric=symmetric)

    def refresh():
        share = state["cbar"] == "shareAB"
        for row, comp in enumerate(comps):
            va, vb = results[state["a"]][row], results[state["b"]][row]
            diff = va - vb
            axes[row, 0].set_title(f"{state['a']}: {comp}")
            axes[row, 1].set_title(f"{state['b']}: {comp}")
            axes[row, 2].set_title(f"{state['a']}−{state['b']}: {comp}")
            if share:
                vmin = float(min(va.min(), vb.min()))
                vmax = float(max(va.max(), vb.max()))
                apply_clim(images[row][0], None, va, vmin, vmax)
                apply_clim(images[row][1], None, vb, vmin, vmax)
                apply_clim(images[row][2], None, diff, symmetric=True)
                # colorbars: [ab0, diff0, ab1, diff1, ab2, diff2]
                _clim_ticks(colorbars[row * 2], vmin, vmax)
                vmax_e = float(np.max(np.abs(diff)))
                _clim_ticks(colorbars[row * 2 + 1], -vmax_e, vmax_e, symmetric=True)
            else:
                for col, z in enumerate((va, vb, diff)):
                    apply_clim(images[row][col], colorbars[row * 3 + col], z)
        fig.canvas.draw_idle()

    def fill_combo(combo, items, cur):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentText(cur if cur in items else items[0])
        combo.blockSignals(False)
        return combo.currentText()

    def make_combo(label, exclude):
        tb.addWidget(QtWidgets.QLabel(label))
        c = QtWidgets.QComboBox()
        c.setMinimumWidth(140)
        c.addItems([n for n in names if n != exclude])
        return c

    tb = fig.canvas.manager.toolbar
    tb.addSeparator()
    ca = make_combo(" A: ", state["b"])
    ca.setCurrentText(state["a"])
    tb.addWidget(ca)
    cb = make_combo(" B: ", state["a"])
    cb.setCurrentText(state["b"])
    tb.addWidget(cb)

    btn_cb = QtWidgets.QPushButton("CB: individual")
    btn_cb.setCheckable(True)
    tb.addSeparator()
    tb.addWidget(btn_cb)

    def on_a(text):
        if state["busy"] or not text:
            return
        state["busy"] = True
        state["a"] = text
        state["b"] = fill_combo(cb, [n for n in names if n != text], state["b"])
        refresh()
        state["busy"] = False

    def on_b(text):
        if state["busy"] or not text:
            return
        state["busy"] = True
        state["b"] = text
        state["a"] = fill_combo(ca, [n for n in names if n != text], state["a"])
        refresh()
        state["busy"] = False

    def on_cb(checked):
        state["cbar"] = "shareAB" if checked else "individual"
        btn_cb.setText(f"CB: {state['cbar']}")
        create_cbars()
        refresh()

    ca.currentTextChanged.connect(on_a)
    cb.currentTextChanged.connect(on_b)
    btn_cb.toggled.connect(on_cb)

    create_cbars()
    refresh()
    _bind_view_sync(fig)
    return fig


# ---------------------------------------------------------------------------
# Geometry + Okada
# ---------------------------------------------------------------------------
slip_model = load_fault_one_plane(
    fault_file, dip=dip, **ref, l_ratio=l_ratio, w_ratio=w_ratio,
    width=width, len_top=len_top, layers=layers, coord_mode="local_xy",
)
idx = I * n_layer + J
assert slip_model[idx, 1] == idx + 1 and slip_model[idx, 2] == I + 1
slip_model[idx, 11] = 1
show_slip_model(
    slip_model, **ref, axis_range=axis_range, apply_axis_range=True,
    out_path="fault_one_plane.png", block=False,
)

xp, yp, zp, L, W, strike_deg, dip_deg = slip_model[idx, 3:10]
d = max(-zp, 1e-10)
delta, strike = np.radians(dip_deg), np.radians(strike_deg)
nobs = xe.size
coords = np.array([xe, yn, np.zeros(nobs)])
shape = (y1d_m.size, x1d_m.size)
# paragram 中 x1d_m/y1d_m 已是 km（xe/yn 才是 ×1000 后的 m）；勿再 /1000
extent = [float(x1d_m[0]), float(x1d_m[-1]), float(y1d_m[0]), float(y1d_m[-1])]
to2d = lambda a: np.asarray(a, dtype=np.float64).reshape(shape)

# edge_tol（米）；此处已有 W，可写 W/50 等
# EDGE_TOLS = [W / 50, W / 100, W / 200]
EDGE_TOLS = [10]
# excludedEdges：与 BC 正交组合；"none"→.set()，"all"→.all()
EXCL_MODES = ["all", "none"]

theta = np.pi / 2 - strike
dx, dy = _xy2xy(np.array(L * 0.5), np.array(0.0), -theta)
okada = tuple(to2d(v) for v in calc_okada(
    1.0, U_SLIP, xe - (xp + float(dx)), yn - (yp + float(dy)),
    NU, delta, d, L, W, FAULT_TYPE, strike, np.zeros(nobs), backend="auto",
))

# ---------------------------------------------------------------------------
# 每模型：Okada × (uniform|taper) × EXCL_MODES → 一张下拉图
# ---------------------------------------------------------------------------
client = mph.start()
for mph0, mph1 in MPH_LIST:
    path = mph0 if I == 0 else mph1
    name = os.path.basename(path)
    print(f"=== {name}  I={I}  size=({L},{W})  pos=({L*J},{W*I}) ===")

    def solve(bc, et=None, excl="none"):
        return tuple(to2d(v) for v in run_comsol(
            client, path, L, W, I, J, FAULT_TYPE, coords, bc,
            edge_tol=et, excl_mode=excl))

    results = {"Okada": okada}
    for excl in EXCL_MODES:
        results[f"uniform/{excl}"] = solve(set_slip_bc, excl=excl)
        for et in EDGE_TOLS:
            results[f"taper2d_{et:g}m/{excl}"] = solve(set_slip_bc_taper, et, excl)
            results[f"taper1d_{et:g}m/{excl}"] = solve(set_slip_bc_taper1d, et, excl)

    plot_source_picker(results, extent, title=f"{name}: {' | '.join(results)}")

print("下拉选 A/B；工具栏 CB 切换前两列共用色标；缩放同步，按 r 恢复全图。")
plt.show()
