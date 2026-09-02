"""Per-model: Okada + uniform/taper COMSOL; dropdown A|B|A−B (models not mixed)."""
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
FAULT_TYPE = 1          # 1: strike-slip, 2: dip-slip
NU, U_SLIP = 0.25, 1.0

# 每组 (mph_I0, mph_Igt0)：按 I 选文件；每组单独开图，只对比
# Okada / uniform BC / taper BC（不同组模型互不对比）
MPH_LIST = [
    # ("top0.mph", "top1.mph"),
    ("top0_most_refine_extend.mph", "top1_most_refine_extend.mph"),
]

PLOT = dict(
    cmap="jet", figsize=(13, 12), dpi=120, digits=4, interpolation="bilinear",
    cbar_shrink=0.75, cbar_fraction=0.035, cbar_pad=0.02, cbar_labelpad=1,
)
FIGS = []
_SYNCING = False


def _apply_view(xlim, ylim):
    global _SYNCING
    if _SYNCING:
        return
    _SYNCING = True
    try:
        for fig in FIGS:
            for ax in fig.axes:
                if ax.images:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            fig.canvas.draw_idle()
    finally:
        _SYNCING = False


def _bind_view_sync(fig):
    def on_lim(ax):
        if not _SYNCING and ax.images:
            _apply_view(ax.get_xlim(), ax.get_ylim())

    def on_key(event):
        if event.key != "r" or not FIGS:
            return
        for ax in FIGS[0].axes:
            if ax.images:
                x0, x1, y0, y1 = ax.images[0].get_extent()
                _apply_view((x0, x1), (y0, y1))
                break

    for ax in fig.axes:
        if ax.images:
            ax.callbacks.connect("xlim_changed", on_lim)
            ax.callbacks.connect("ylim_changed", on_lim)
    fig.canvas.mpl_connect("key_press_event", on_key)
    FIGS.append(fig)


def plot_field(ax, extent, z, title, cbar_label):
    vmin, vmax = float(np.min(z)), float(np.max(z))
    im = ax.imshow(
        z, origin="lower", extent=extent, aspect="equal",
        cmap=PLOT["cmap"], vmin=vmin, vmax=vmax,
        interpolation=PLOT["interpolation"], resample=True, rasterized=True,
    )
    ax.set(title=title, xlabel="x (km)", ylabel="y (km)")
    cbar = plt.colorbar(
        im, ax=ax, shrink=PLOT["cbar_shrink"],
        fraction=PLOT["cbar_fraction"], pad=PLOT["cbar_pad"],
    )
    cbar.set_label(cbar_label, labelpad=PLOT["cbar_labelpad"])
    dig = PLOT["digits"]
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"min: {vmin:.{dig}f}", f"max: {vmax:.{dig}f}"])
    return im, cbar


def set_slip_bc(model, fault_type, *args, **kwargs):
    solid = model.component("comp1").physics("solid")
    if fault_type == 1:
        direction = [["free"], ["prescribed"], ["free"]]
        u0 = ([[0], [0.5], [0]], [[0], [0.5], [0]])
    else:
        direction = [["prescribed"], ["free"], ["free"]]
        u0 = ([[-0.5], [0], [0]], [[0.5], [0], [0]])
    for feat, u in zip(("disp1", "disp2"), u0):
        solid.feature(feat).set("Direction", direction)
        solid.feature(feat).set("U0", u)


def set_slip_bc_taper(model, fault_type, i, j, L, W, edge_tol):
    a, b, c, d = L * j, L * (j + 1), -W * (i + 1), -W * i
    taper = (
        f"if(min(y-{a}[m],{b}[m]-y)<{edge_tol}[m],"
        f"0.5*(1-cos(pi*min(y-{a}[m],{b}[m]-y)/{edge_tol}[m])),1)*"
        f"if(min(z-{c}[m],{d}[m]-z)<{edge_tol}[m],"
        f"0.5*(1-cos(pi*min(z-{c}[m],{d}[m]-z)/{edge_tol}[m])),1)"
    )
    solid = model.component("comp1").physics("solid")
    if fault_type == 1:
        direction = [["free"], ["prescribed"], ["free"]]
        u0 = ([["0"], [f"0.5*({taper})"], ["0"]], [["0"], [f"0.5*({taper})"], ["0"]])
    else:
        direction = [["prescribed"], ["free"], ["free"]]
        u0 = ([[f"-0.5*({taper})"], ["0"], ["0"]], [[f"0.5*({taper})"], ["0"], ["0"]])
    for feat, u in zip(("disp1", "disp2"), u0):
        solid.feature(feat).set("Direction", direction)
        solid.feature(feat).set("U0", u)


def run_comsol(client, mph_path, L, W, i, j, fault_type, coords, set_bc):
    model_java = client.load(mph_path)
    model = model_java.java
    r1 = model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")
    r1.set("size", [str(L), str(W)])
    r1.set("pos", [str(L * j), str(W * i)])
    set_bc(model, fault_type, i, j, L, W, edge_tol=W / 100)
    model.sol("sol1").runAll()
    model.result().numerical().create("uvw", "Interp")
    interp = model.result().numerical("uvw")
    interp.setInterpolationCoordinates(coords)
    interp.set("expr", ["u", "v", "w"])
    data = np.asarray(interp.getData())
    client.remove(model_java)
    return data[0, 0], data[1, 0], data[2, 0]


def plot_source_picker(results, extent):
    """工具栏下拉选 A/B（不可重复），第三列 A−B。"""
    names = list(results.keys())
    state = {"a": names[0], "b": names[1], "busy": False}

    fig, axes = plt.subplots(
        3, 3, figsize=PLOT["figsize"], dpi=PLOT["dpi"], sharex=True, sharey=True,
    )
    images, cbars = [], []
    for row, comp in enumerate("ENU"):
        va, vb = results[state["a"]][row], results[state["b"]][row]
        lab = f"u{comp.lower()} (m)"
        ims = [
            plot_field(axes[row, 0], extent, va, f"{state['a']}: {comp}", lab)[0],
            plot_field(axes[row, 1], extent, vb, f"{state['b']}: {comp}", lab)[0],
            plot_field(axes[row, 2], extent, va - vb, f"{state['a']}−{state['b']}: {comp}", "diff (m)")[0],
        ]
        images.append(ims)
        cbars.append([im.colorbar for im in ims])
    fig.tight_layout()

    def set_img(im, cbar, z):
        vmin, vmax = float(np.min(z)), float(np.max(z))
        im.set_data(z)
        im.set_clim(vmin, vmax)
        dig = PLOT["digits"]
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", f"max: {vmax:.{dig}f}"])

    def refresh():
        for row, comp in enumerate("ENU"):
            va, vb = results[state["a"]][row], results[state["b"]][row]
            set_img(images[row][0], cbars[row][0], va)
            set_img(images[row][1], cbars[row][1], vb)
            set_img(images[row][2], cbars[row][2], va - vb)
            axes[row, 0].set_title(f"{state['a']}: {comp}")
            axes[row, 1].set_title(f"{state['b']}: {comp}")
            axes[row, 2].set_title(f"{state['a']}−{state['b']}: {comp}")
        fig.canvas.draw_idle()

    def fill_combo(combo, items, cur):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentText(cur if cur in items else items[0])
        combo.blockSignals(False)
        return combo.currentText()

    tb = fig.canvas.manager.toolbar
    tb.addSeparator()
    tb.addWidget(QtWidgets.QLabel(" A: "))
    ca = QtWidgets.QComboBox()
    ca.setMinimumWidth(140)
    ca.addItems([n for n in names if n != state["b"]])
    ca.setCurrentText(state["a"])
    tb.addWidget(ca)
    tb.addWidget(QtWidgets.QLabel(" B: "))
    cb = QtWidgets.QComboBox()
    cb.setMinimumWidth(140)
    cb.addItems([n for n in names if n != state["a"]])
    cb.setCurrentText(state["b"])
    tb.addWidget(cb)

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

    ca.currentTextChanged.connect(on_a)
    cb.currentTextChanged.connect(on_b)
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
extent = [x1d_m[0] / 1000, x1d_m[-1] / 1000, y1d_m[0] / 1000, y1d_m[-1] / 1000]
to2d = lambda a: np.asarray(a, dtype=np.float64).reshape(shape)

theta = np.pi / 2 - strike
dx, dy = _xy2xy(np.array(L * 0.5), np.array(0.0), -theta)
okada = tuple(to2d(v) for v in calc_okada(
    1.0, U_SLIP, xe - (xp + float(dx)), yn - (yp + float(dy)),
    NU, delta, d, L, W, FAULT_TYPE, strike, np.zeros(nobs), backend="auto",
))

# ---------------------------------------------------------------------------
# 每个 mph：Okada + uniform BC + taper BC → 一张下拉对比图
# ---------------------------------------------------------------------------
client = mph.start()
for mph_i0, mph_igt0 in MPH_LIST:
    mph_path = mph_i0 if I == 0 else mph_igt0
    name = os.path.basename(mph_path)
    print(f"=== {name} (I={I}): size=({L},{W}) pos=({L*J},{W*I}) ===")
    results = {
        "Okada": okada,
        "uniform": tuple(to2d(v) for v in run_comsol(
            client, mph_path, L, W, I, J, FAULT_TYPE, coords, set_slip_bc)),
        "taper": tuple(to2d(v) for v in run_comsol(
            client, mph_path, L, W, I, J, FAULT_TYPE, coords, set_slip_bc_taper)),
    }
    plot_source_picker(results, extent)
    # 标题区分模型
    FIGS[-1].suptitle(f"{name}: Okada | uniform | taper", fontsize=12)
    FIGS[-1].tight_layout()

print("每幅图仅含该模型三类数据；下拉选 A/B。缩放同步，按 r 恢复全图。")
plt.show()
