"""Compare Okada vs multiple COMSOL .mph models for one fault patch."""
import os
import sys

import matplotlib.pyplot as plt
import mph
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calc_green import _xy2xy
from calc_okada import calc_okada
from load_fault_one_plane import load_fault_one_plane
from show_slip_model import show_slip_model



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


def _has_scatter(ax):
    return any(getattr(collection, "_data_scatter", False) for collection in ax.collections)


def _set_colorbar_range(collection, values, symmetric=False):
    values = np.ma.masked_invalid(np.asarray(values, dtype=float))
    vmin, vmax = _value_limits(values, symmetric)
    cbar = collection.colorbar
    collection.set_clim(vmin, vmax)
    cbar.update_normal(collection)
    if vmin == vmax:
        # Normalize expands equal limits internally; restore the data range
        # after colorbar synchronization so the reported range stays exact.
        collection.norm.vmin = vmin
        collection.norm.vmax = vmax
    dig = PLOT["digits"]
    if symmetric:
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", "0", f"max: {vmax:.{dig}f}"])
    else:
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"min: {vmin:.{dig}f}", f"max: {vmax:.{dig}f}"])


def _update_visible_colorbars(ax):
    """Update scatter colorbars from points currently inside the axes view."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for collection in ax.collections:
        if not getattr(collection, "_data_scatter", False):
            continue
        offsets = np.asarray(collection.get_offsets())
        values = np.ma.asarray(collection.get_array()).reshape(-1)
        visible = (
            (offsets[:, 0] >= min(xlim)) & (offsets[:, 0] <= max(xlim)) &
            (offsets[:, 1] >= min(ylim)) & (offsets[:, 1] <= max(ylim))
        )
        visible_values = values[visible]
        if np.ma.count(visible_values):
            _set_colorbar_range(collection, visible_values)


def _value_limits(values, symmetric=False):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return -1.0, 1.0
    if symmetric:
        vmax = float(np.max(np.abs(finite_values)))
        return -vmax, vmax
    return float(np.min(finite_values)), float(np.max(finite_values))


def _apply_view(xlim, ylim):
    """把 (xlim, ylim) 应用到所有对比图的数据轴（不含 colorbar）。"""
    global _SYNCING_VIEW
    if _SYNCING_VIEW:
        return
    _SYNCING_VIEW = True
    try:
        for fig in COMPARE_FIGS:
            for ax in fig.axes:
                if _has_scatter(ax):
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    _update_visible_colorbars(ax)
            fig.canvas.draw_idle()
    finally:
        _SYNCING_VIEW = False


def _bind_view_sync(fig):
    """本图任一子图缩放后，同步到所有对比图窗。按 r 恢复全局范围。"""
    def on_lim(_ax):
        if _SYNCING_VIEW or _ax not in fig.axes or not _has_scatter(_ax):
            return
        _apply_view(_ax.get_xlim(), _ax.get_ylim())

    def on_key(event):
        if event.key != "r" or not COMPARE_FIGS:
            return
        # 用第一幅散点图的坐标范围恢复全图
        for ax in COMPARE_FIGS[0].axes:
            if _has_scatter(ax):
                offsets = ax.collections[0].get_offsets()
                _apply_view(
                    (float(np.min(offsets[:, 0])), float(np.max(offsets[:, 0]))),
                    (float(np.min(offsets[:, 1])), float(np.max(offsets[:, 1]))),
                )
                break

    for ax in fig.axes:
        if _has_scatter(ax):
            ax.callbacks.connect("xlim_changed", on_lim)
            ax.callbacks.connect("ylim_changed", on_lim)
    fig.canvas.mpl_connect("key_press_event", on_key)


def plot_field(ax, x, y, values, title, cbar_label, symmetric=False):
    """Plot values at the observation coordinates as a scatter plot."""
    values = np.ma.masked_invalid(np.asarray(values, dtype=float))
    vmin, vmax = _value_limits(values, symmetric)

    points = ax.scatter(
        x, y, c=values, s=5,
        cmap=PLOT["cmap"],
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    points._data_scatter = True
    ax.set(title=title, xlabel="x (km)", ylabel="y (km)", aspect="equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    cbar = plt.colorbar(
        points, ax=ax,
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


def plot_okada_vs_comsol(x, y, okada_uvw, comsol_uvw, mph_name):
    """One figure window: Okada | COMSOL | Error for the Y displacement."""
    okada_v = np.asarray(okada_uvw[1])
    comsol_v = np.asarray(comsol_uvw[1])

    # sharex/sharey: 本图任一侧栏放大，9 个子图视野一起变
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.5), dpi=PLOT["dpi"],
        sharex=True, sharey=True,
    )
    fig.suptitle(f"Okada vs {mph_name}", fontsize=14)
    label = "v (m)"
    plot_field(axes[0], x, y, okada_v, "Okada: v", label)
    plot_field(axes[1], x, y, comsol_v, "COMSOL: v", label)
    plot_field(axes[2], x, y, comsol_v - okada_v, "Error: v", "error (m)")
    fig.tight_layout()
    COMPARE_FIGS.append(fig)
    _bind_view_sync(fig)
    return fig


def _update_image(im, cbar, values, symmetric=False):
    values = np.ma.masked_invalid(np.asarray(values, dtype=float))
    vmin, vmax = _value_limits(values, symmetric)
    im.set_array(values)
    im.set_clim(vmin, vmax)
    cbar.update_normal(im)
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


def plot_source_picker(results, x, y):
    """交互图：下拉选择两列数据源（不可重复），第三列画 A−B 差异。

    results: {name: (E, N, U)}，每个分量为与 v.txt 行数相同的一维数组。
    依赖 Qt 后端 (QtAgg)；下拉框挂在工具栏上。
    """
    from matplotlib.backends.qt_compat import QtWidgets

    names = list(results.keys())
    if len(names) < 2:
        raise ValueError("至少需要两个数据源才能对比")

    comps = (("v", 1),)
    state = {"a": names[0], "b": names[1], "busy": False}

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.5), dpi=PLOT["dpi"],
        sharex=True, sharey=True,
    )
    # fig.suptitle("Source picker: A | B | A−B", fontsize=14)

    images, cbars = [], []
    a_uvw, b_uvw = results[state["a"]], results[state["b"]]
    for row, (comp, component_index) in enumerate(comps):
        va, vb = a_uvw[component_index], b_uvw[component_index]
        label = "v (m)"
        plot_field(axes[0], x, y, va, f"{state['a']}: {comp}", label)
        plot_field(axes[1], x, y, vb, f"{state['b']}: {comp}", label)
        plot_field(
            axes[2], x, y, va - vb,
            f"{state['a']} − {state['b']}: {comp}", "diff (m)",
        )
        row_ims = [axes[c].collections[0] for c in range(3)]
        images.append(row_ims)
        cbars.append([im.colorbar for im in row_ims])

    fig.tight_layout()

    def refresh():
        a_uvw = results[state["a"]]
        b_uvw = results[state["b"]]
        for row, (comp, component_index) in enumerate(comps):
            va, vb = a_uvw[component_index], b_uvw[component_index]
            _update_image(images[row][0], cbars[row][0], va)
            _update_image(images[row][1], cbars[row][1], vb)
            _update_image(images[row][2], cbars[row][2], va - vb)
            axes[0].set_title(f"{state['a']}: {comp}")
            axes[1].set_title(f"{state['b']}: {comp}")
            axes[2].set_title(f"{state['a']} − {state['b']}: {comp}")
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
from paragram import n_layer, len_top, layers, width, ref, axis_range, fault_file, l_ratio, w_ratio, dip

I, J = 0, 0                         # patch layer / along-strike index
FAULT_TYPE = 1                     # 1: strike-slip, 2: dip-slip
NU, U_SLIP = 0.25, 1.0


# 每组 (mph_I0, mph_Igt0)：I==0 用前者，否则用后者；每组开一个图窗
MPH_GROUPS = [
    # ("top0.mph", "top1.mph"),
    # ("top0_infinit.mph","top1_infinit.mph"),
    # ("top0_tri.mph","11"),
    # ("top0_rec20.mph","11"),
    # ("top0_most_refine.mph","11"),
    ("top0_most_refine_extend.mph","top1_most_refine_extend.mph"),
    # ("top0_most_refine_extend_exclude.mph","top1_most_refine_extend.mph")
    # ("meshA0.mph", "meshA1.mph"),
]

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
delta = np.radians(dip_deg)
strike = np.radians(strike_deg)


v_data = np.loadtxt("v.txt", comments="%")
if v_data.ndim != 2 or v_data.shape[1] < 3:
    raise ValueError("v.txt must contain at least three columns")
v_data = v_data[:, :3]
v_data = v_data[np.all(np.isfinite(v_data), axis=1)]
if v_data.size == 0:
    raise ValueError("v.txt contains no finite data rows")

xe = v_data[:, 0]
yn = v_data[:, 1]
txt_v = v_data[:, 2]
nobs = xe.size
coords = np.array([xe, yn, np.zeros(nobs)])
x_plot = xe / 1000.0
y_plot = yn / 1000.0

theta = np.pi / 2 - strike
dx, dy = _xy2xy(np.array(L * 0.5), np.array(0.0), -theta)
xpt = xe - (xp + float(dx))
ypt = yn - (yp + float(dy))
tp = np.zeros(nobs)

ue, un, uz = calc_okada(
    1.0, U_SLIP, xpt, ypt, NU, delta, d, L, W, FAULT_TYPE, strike, tp, backend="numpy",
)
okada_uvw = (ue, un, uz)

# ---------------------------------------------------------------------------
# Loop MPH_GROUPS: pick by I, each group -> one comparison figure
# ---------------------------------------------------------------------------
results = {"Okada": tuple(np.asarray(a, dtype=np.float64) for a in okada_uvw)}
results["v.txt"] = (np.zeros(nobs), txt_v, np.zeros(nobs))

client = mph.start()
for mph_i0, mph_igt0 in MPH_GROUPS:
    mph_path = mph_i0 if I == 0 else mph_igt0
    name = os.path.basename(mph_path)
    print(f"=== {name} (I={I}): patch size=({L}, {W})  pos=({L * J}, {W * I}) ===")
    comsol_uvw = run_comsol(client, mph_path, L, W, I, J, FAULT_TYPE, coords)
    results[name] = tuple(np.asarray(a, dtype=np.float64) for a in comsol_uvw)
    plot_okada_vs_comsol(x_plot, y_plot, okada_uvw, comsol_uvw, name)

plot_source_picker(results, x_plot, y_plot)
print("缩放: 工具栏放大任一侧栏即可；视野会同步。按 r 恢复全图。")
print("Source picker: 顶部下拉选择 A/B（不可重复），第三列为 A−B。")
plt.show()
