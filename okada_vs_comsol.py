"""Compare Okada vs multiple COMSOL .mph models for one fault patch."""
import os
import sys

import matplotlib.pyplot as plt
import mph
import numpy as np

from okada_vs_comsol_copy import NX

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calc_green import _xy2xy
from calc_okada import calc_okada
from load_fault_one_plane import load_fault_one_plane
from show_slip_model import show_slip_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAULT_FILE = os.path.join(os.path.dirname(__file__), "checkerboard.txt")
WIDTH, LENGTH, LEN_TOP, LAYERS = 20e3, 100e3, 4e3, 5
N_LAYER = int(LENGTH / LEN_TOP)

I, J = 0, 1                          # patch layer / along-strike index
FAULT_TYPE = 2                       # 1: strike-slip, 2: dip-slip
NU, U_SLIP = 0.25, 1.0


# 每组 (mph_I0, mph_Igt0)：I==0 用前者，否则用后者；每组开一个图窗
MPH_GROUPS = [
    ("top0.mph", "top1.mph"),
    ("top0_infinit.mph","11"),
    ("top0_tri.mph","11"),
    ("top0_rec20.mph","11"),
    ("top0_most_refine.mph","11")
    # ("meshA0.mph", "meshA1.mph"),
]

REF = dict(ref_lon=95, lonc=95.33, latc=19.61)
AXIS_RANGE = [0, 20, 0, 100, -20, 0]

PLOT = dict(
    cmap="jet", figsize=(13, 12), dpi=120, digits=4,
    interpolation="bilinear",   # nearest 会呈色块；bilinear 更接近原 scatter 观感
    cbar_shrink=0.75, cbar_fraction=0.035, cbar_pad=0.02, cbar_labelpad=1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
COMPARE_FIGS = []  # 对比图窗；缩放后可同步视野，无需重算
_SYNCING_VIEW = False


def _apply_view(xlim, ylim):
    """把 (xlim, ylim) 应用到所有对比图的数据轴（不含 colorbar）。"""
    global _SYNCING_VIEW
    if _SYNCING_VIEW:
        return
    _SYNCING_VIEW = True
    try:
        for fig in COMPARE_FIGS:
            for ax in fig.axes:
                if ax.images:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            fig.canvas.draw_idle()
    finally:
        _SYNCING_VIEW = False


def _bind_view_sync(fig):
    """本图任一子图缩放后，同步到所有对比图窗。按 r 恢复全局范围。"""
    def on_lim(_ax):
        if _SYNCING_VIEW or _ax not in fig.axes or not _ax.images:
            return
        _apply_view(_ax.get_xlim(), _ax.get_ylim())

    def on_key(event):
        if event.key != "r" or not COMPARE_FIGS:
            return
        # 用第一幅图第一个 imshow 的 extent 恢复全图
        for ax in COMPARE_FIGS[0].axes:
            if ax.images:
                x0, x1, y0, y1 = ax.images[0].get_extent()
                _apply_view((x0, x1), (y0, y1))
                break

    for ax in fig.axes:
        if ax.images:
            ax.callbacks.connect("xlim_changed", on_lim)
            ax.callbacks.connect("ylim_changed", on_lim)
    fig.canvas.mpl_connect("key_press_event", on_key)


def plot_field(ax, extent, value_2d, title, cbar_label, symmetric=False):
    """Fast regular-grid plot via imshow."""
    if symmetric:
        vmax = float(np.max(np.abs(value_2d)))
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(value_2d)), float(np.max(value_2d))

    im = ax.imshow(
        value_2d,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=PLOT["cmap"],
        vmin=vmin,
        vmax=vmax,
        interpolation=PLOT["interpolation"],
        resample=True,
        rasterized=True,
    )
    ax.set(title=title, xlabel="x (km)", ylabel="y (km)")

    cbar = plt.colorbar(
        im, ax=ax,
        shrink=PLOT["cbar_shrink"], fraction=PLOT["cbar_fraction"], pad=PLOT["cbar_pad"],
    )
    cbar.set_label(cbar_label, labelpad=PLOT["cbar_labelpad"])
    dig = PLOT["digits"]
    if symmetric:
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", "0", f"max: {vmax:.{dig}f}"])
    else:
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", f"max: {vmax:.{dig}f}"])


def set_comsol_slip_bc(model, fault_type: int):
    solid = model.component("comp1").physics("solid")
    if fault_type == 1:  # strike-slip
        direction = [["free"], ["prescribed"], ["free"]]
        u0 = ([[0], [0.5], [0]], [[0], [0.5], [0]])
    else:                # dip-slip
        direction = [["prescribed"], ["free"], ["free"]]
        u0 = ([[-0.5], [0], [0]], [[0.5], [0], [0]])
    for feat, u in zip(("disp1", "disp2"), u0):
        solid.feature(feat).set("Direction", direction)
        solid.feature(feat).set("U0", u)


def run_comsol(client, mph_path, L, W, i, j, fault_type, coords):
    """Load mph, set patch, solve, return (u, v, w) at coords."""
    model_java = client.load(mph_path)
    model = model_java.java

    geom_r1 = model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")
    geom_r1.set("size", [str(L), str(W)])
    geom_r1.set("pos", [str(L * j), str(W * i)])

    set_comsol_slip_bc(model, fault_type)
    model.sol("sol1").runAll()

    model.result().numerical().create("uvw", "Interp")
    interp = model.result().numerical("uvw")
    interp.setInterpolationCoordinates(coords)
    interp.set("expr", ["u", "v", "w"])
    data = np.asarray(interp.getData())
    client.remove(model_java)  # 释放内存
    return data[0, 0], data[1, 0], data[2, 0]


def plot_okada_vs_comsol(extent, shape, okada_uvw, comsol_uvw, mph_name):
    """One figure window: Okada | COMSOL | Error for E/N/U."""
    to2d = lambda a: np.asarray(a).reshape(shape)
    ue, un, uz = map(to2d, okada_uvw)
    cu, cv, cw = map(to2d, comsol_uvw)
    panels = [("E", ue, cu), ("N", un, cv), ("U", uz, cw)]

    # sharex/sharey: 本图任一侧栏放大，9 个子图视野一起变
    fig, axes = plt.subplots(
        3, 3, figsize=PLOT["figsize"], dpi=PLOT["dpi"],
        sharex=True, sharey=True,
    )
    fig.suptitle(f"Okada vs {mph_name}", fontsize=14)
    for row, (name, oka, cos) in enumerate(panels):
        label = f"u{name.lower()} (m)"
        plot_field(axes[row, 0], extent, oka, f"Okada: {name}", label)
        plot_field(axes[row, 1], extent, cos, f"COMSOL: {name}", label)
        plot_field(axes[row, 2], extent, cos - oka, f"Error: {name}", "error (m)")
    fig.tight_layout()
    COMPARE_FIGS.append(fig)
    _bind_view_sync(fig)
    return fig


def _update_image(im, cbar, value_2d, symmetric=False):
    if symmetric:
        vmax = float(np.max(np.abs(value_2d)))
        vmin = -vmax
    else:
        vmin, vmax = float(np.min(value_2d)), float(np.max(value_2d))
    im.set_data(value_2d)
    im.set_clim(vmin, vmax)
    dig = PLOT["digits"]
    if symmetric:
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", "0", f"max: {vmax:.{dig}f}"])
    else:
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", f"max: {vmax:.{dig}f}"])


def _fill_combo(combo, items, current):
    """Rebuild QComboBox items without emitting signals; keep current if possible."""
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(items)
    if current in items:
        combo.setCurrentText(current)
    elif items:
        combo.setCurrentIndex(0)
    combo.blockSignals(False)
    return combo.currentText()


def plot_source_picker(results, extent):
    """交互图：下拉选择两列数据源（不可重复），第三列画 A−B 差异。

    results: {name: (E, N, U)}，每个分量为 shape=(NY, NX) 的 2D 数组。
    依赖 Qt 后端 (QtAgg)；下拉框挂在工具栏上。
    """
    from matplotlib.backends.qt_compat import QtWidgets

    names = list(results.keys())
    if len(names) < 2:
        raise ValueError("至少需要两个数据源才能对比")

    comps = ("E", "N", "U")
    state = {"a": names[0], "b": names[1], "busy": False}

    fig, axes = plt.subplots(
        3, 3, figsize=PLOT["figsize"], dpi=PLOT["dpi"],
        sharex=True, sharey=True,
    )
    fig.suptitle("Source picker: A | B | A−B", fontsize=14)

    images, cbars = [], []
    a_uvw, b_uvw = results[state["a"]], results[state["b"]]
    for row, comp in enumerate(comps):
        va, vb = a_uvw[row], b_uvw[row]
        label = f"u{comp.lower()} (m)"
        plot_field(axes[row, 0], extent, va, f"{state['a']}: {comp}", label)
        plot_field(axes[row, 1], extent, vb, f"{state['b']}: {comp}", label)
        plot_field(
            axes[row, 2], extent, va - vb,
            f"{state['a']} − {state['b']}: {comp}", "diff (m)",
        )
        row_ims = [axes[row, c].images[0] for c in range(3)]
        images.append(row_ims)
        cbars.append([im.colorbar for im in row_ims])

    fig.tight_layout()

    def refresh():
        a_uvw = results[state["a"]]
        b_uvw = results[state["b"]]
        for row, comp in enumerate(comps):
            va, vb = a_uvw[row], b_uvw[row]
            _update_image(images[row][0], cbars[row][0], va)
            _update_image(images[row][1], cbars[row][1], vb)
            _update_image(images[row][2], cbars[row][2], va - vb)
            axes[row, 0].set_title(f"{state['a']}: {comp}")
            axes[row, 1].set_title(f"{state['b']}: {comp}")
            axes[row, 2].set_title(f"{state['a']} − {state['b']}: {comp}")
        fig.canvas.draw_idle()

    # 工具栏上挂两个 QComboBox 作为下拉栏
    toolbar = fig.canvas.manager.toolbar
    toolbar.addSeparator()
    toolbar.addWidget(QtWidgets.QLabel(" A: "))
    combo_a = QtWidgets.QComboBox()
    combo_a.setMinimumWidth(160)
    combo_a.addItems([n for n in names if n != state["b"]])
    combo_a.setCurrentText(state["a"])
    toolbar.addWidget(combo_a)

    toolbar.addWidget(QtWidgets.QLabel(" B: "))
    combo_b = QtWidgets.QComboBox()
    combo_b.setMinimumWidth(160)
    combo_b.addItems([n for n in names if n != state["a"]])
    combo_b.setCurrentText(state["b"])
    toolbar.addWidget(combo_b)

    def on_a(text):
        if state["busy"] or not text:
            return
        state["busy"] = True
        try:
            state["a"] = text
            opts_b = [n for n in names if n != text]
            state["b"] = _fill_combo(combo_b, opts_b, state["b"])
            refresh()
        finally:
            state["busy"] = False

    def on_b(text):
        if state["busy"] or not text:
            return
        state["busy"] = True
        try:
            state["b"] = text
            opts_a = [n for n in names if n != text]
            state["a"] = _fill_combo(combo_a, opts_a, state["a"])
            refresh()
        finally:
            state["busy"] = False

    combo_a.currentTextChanged.connect(on_a)
    combo_b.currentTextChanged.connect(on_b)

    COMPARE_FIGS.append(fig)
    _bind_view_sync(fig)
    return fig


# ---------------------------------------------------------------------------
# Fault geometry & selected patch
# ---------------------------------------------------------------------------
slip_model = load_fault_one_plane(
    FAULT_FILE, dip=[80], **REF, l_ratio=1, w_ratio=1,
    width=WIDTH, len_top=LEN_TOP, layers=LAYERS, coord_mode="local_xy",
)

idx = I * N_LAYER + J
assert slip_model[idx, 1] == idx + 1 and slip_model[idx, 2] == I + 1
slip_model[idx, 11] = 1

show_slip_model(
    slip_model, **REF, axis_range=AXIS_RANGE, apply_axis_range=True,
    out_path="fault_one_plane.png", block=False,
)

xp, yp, zp, L, W, strike_deg, dip_deg = slip_model[idx, 3:10]
d = max(-zp, 1e-10)
delta = np.radians(dip_deg)
strike = np.radians(strike_deg)

# x1d_m = np.linspace(-50, 50, NX) * 1000.0
# y1d_m = np.linspace(-20, 120, NY) * 1000.0
# XE, YN = np.meshgrid(x1d_m, y1d_m)  # (NY, NX)
# xe, yn = XE.ravel(), YN.ravel()
ny = 700
x1d_m = np.unique(np.concatenate([
    np.linspace(-50, -20, 50),
    np.linspace(-20, 20, 200),
    np.linspace(20, 50, 50),
]))
y1d_m = np.linspace(-20, 120, ny)
X, Y = np.meshgrid(x1d_m, y1d_m)
NX = x1d_m.size
NY = y1d_m.size
xe = X.ravel() * 1000.0  # km -> m
yn = Y.ravel() * 1000.0
nobs = xe.size
coords = np.array([xe, yn, np.zeros(nobs)])
grid_shape = (NY, NX)
extent_km = [x1d_m[0] / 1000, x1d_m[-1] / 1000, y1d_m[0] / 1000, y1d_m[-1] / 1000]

theta = np.pi / 2 - strike
dx, dy = _xy2xy(np.array(L * 0.5), np.array(0.0), -theta)
xpt = xe - (xp + float(dx))
ypt = yn - (yp + float(dy))
tp = np.zeros(nobs)

ue, un, uz = calc_okada(
    1.0, U_SLIP, xpt, ypt, NU, delta, d, L, W, FAULT_TYPE, strike, tp, backend="auto",
)
okada_uvw = (ue, un, uz)

# ---------------------------------------------------------------------------
# Loop MPH_GROUPS: pick by I, each group -> one comparison figure
# ---------------------------------------------------------------------------
to2d = lambda a: np.asarray(a, dtype=np.float64).reshape(grid_shape)
results = {"Okada": tuple(to2d(a) for a in okada_uvw)}

client = mph.start()
for mph_i0, mph_igt0 in MPH_GROUPS:
    mph_path = mph_i0 if I == 0 else mph_igt0
    name = os.path.basename(mph_path)
    print(f"=== {name} (I={I}): patch size=({L}, {W})  pos=({L * J}, {W * I}) ===")
    comsol_uvw = run_comsol(client, mph_path, L, W, I, J, FAULT_TYPE, coords)
    results[name] = tuple(to2d(a) for a in comsol_uvw)
    plot_okada_vs_comsol(extent_km, grid_shape, okada_uvw, comsol_uvw, name)

plot_source_picker(results, extent_km)
print("缩放: 工具栏放大任一侧栏即可；视野会同步。按 r 恢复全图。")
print("Source picker: 顶部下拉选择 A/B（不可重复），第三列为 A−B。")
plt.show()
