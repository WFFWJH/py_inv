"""导出论文用高质量 PNG（不改动 okada_vs_comsol.py）。

用法：
  1. 在下方填好 SOURCE_A / SOURCE_B 与 MPH_FOR_EXPORT
  2. 运行本脚本，弹出图窗
  3. 用工具栏缩放选好区域（三列同步）
  4. 按 s，或点工具栏 Save HQ PNG 导出当前视野

色标：每行 A / B / Diff 三列共用一套 clim，一根 colorbar 放在最右侧。
轴名称：仅第一列显示 x/y；坐标为 m（与观测网格一致）。
"""
import os
import sys

import matplotlib.pyplot as plt
import mph
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calc_green import _xy2xy
from calc_okada import calc_okada
from load_fault_one_plane import load_fault_one_plane
from paragram import (
    n_layer, len_top, layers, width, ref, fault_file,
    l_ratio, w_ratio, dip, x1d_m, y1d_m, xe, yn,
)

# ---------------------------------------------------------------------------
# 论文图配置
# ---------------------------------------------------------------------------
SOURCE_A = "Okada"
SOURCE_B = "top0_most_refine.mph"

MPH_FOR_EXPORT = [
    ("top0_most_refine.mph", "11"),
]

I, J = 0, 0
FAULT_TYPE = 1
NU, U_SLIP = 0.25, 1.0

EXPORT_DIR = "figures"
EXPORT_DPI = 300
FIGSIZE = (13, 12)
CMAP = "jet"
DIGITS = 4


def _safe_stem(name):
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)


def set_comsol_slip_bc(model, fault_type: int):
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


def run_comsol(client, mph_path, L, W, i, j, fault_type, coords):
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
    client.remove(model_java)
    return data[0, 0], data[1, 0], data[2, 0]


def _format_cbar(cbar, vmin, vmax):
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"min: {vmin:.{DIGITS}f}", f"max: {vmax:.{DIGITS}f}"])


def plot_ab_diff(extent, results, name_a, name_b):
    """布局：A | B | Diff | 共用 colorbar（最右）。"""
    if name_a not in results or name_b not in results:
        raise KeyError(f"缺少数据源: A={name_a!r}, B={name_b!r}; 已有 {list(results)}")

    a_uvw, b_uvw = results[name_a], results[name_b]
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        3, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.05],
        wspace=0.15, hspace=0.25,
    )

    ref_ax = None
    for row, comp in enumerate(("E", "N", "U")):
        va, vb = a_uvw[row], b_uvw[row]
        vd = va - vb
        # 三列共用 clim（覆盖 A/B/Diff）
        vmin = float(min(np.min(va), np.min(vb), np.min(vd)))
        vmax = float(max(np.max(va), np.max(vb), np.max(vd)))

        if ref_ax is None:
            ax_a = fig.add_subplot(gs[row, 0])
            ref_ax = ax_a
            ax_b = fig.add_subplot(gs[row, 1], sharex=ref_ax, sharey=ref_ax)
            ax_d = fig.add_subplot(gs[row, 2], sharex=ref_ax, sharey=ref_ax)
        else:
            ax_a = fig.add_subplot(gs[row, 0], sharex=ref_ax, sharey=ref_ax)
            ax_b = fig.add_subplot(gs[row, 1], sharex=ref_ax, sharey=ref_ax)
            ax_d = fig.add_subplot(gs[row, 2], sharex=ref_ax, sharey=ref_ax)
        cax = fig.add_subplot(gs[row, 3])

        im_ref = None
        for col, (ax, data) in enumerate(((ax_a, va), (ax_b, vb), (ax_d, vd))):
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=CMAP,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                resample=True,
                rasterized=True,
            )
            if col == 0:
                ax.set_xlabel("x (m)")
                ax.set_ylabel("y (m)")
                im_ref = im
            else:
                ax.set_xlabel("")
                ax.set_ylabel("")
                # 非第一列不显示轴名称；刻度仍保留便于读数
                ax.xaxis.label.set_visible(False)
                ax.yaxis.label.set_visible(False)

        cbar = fig.colorbar(im_ref, cax=cax)
        cbar.set_label(f"u{comp.lower()} (m)", labelpad=1)
        _format_cbar(cbar, vmin, vmax)

    return fig


def save_hq_png(fig, stem):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"{_safe_stem(stem)}.png")
    fig.savefig(
        path, dpi=EXPORT_DPI, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    print(f"Saved ({EXPORT_DPI} dpi): {os.path.abspath(path)}")
    return path


def bind_export(fig, stem):
    """缩放后按 s 或点按钮导出当前视野。"""
    from matplotlib.backends.qt_compat import QtWidgets

    def do_save(_event=None):
        save_hq_png(fig, stem)

    fig.canvas.mpl_connect(
        "key_press_event",
        lambda e: do_save() if e.key == "s" else None,
    )

    toolbar = fig.canvas.manager.toolbar
    toolbar.addSeparator()
    btn = QtWidgets.QPushButton("Save HQ PNG")
    btn.setToolTip(f"导出当前缩放区域为 {EXPORT_DPI} dpi PNG（快捷键 s）")
    btn.clicked.connect(do_save)
    toolbar.addWidget(btn)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
slip_model = load_fault_one_plane(
    fault_file, dip=dip, **ref, l_ratio=l_ratio, w_ratio=w_ratio,
    width=width, len_top=len_top, layers=layers, coord_mode="local_xy",
)
idx = I * n_layer + J
assert slip_model[idx, 1] == idx + 1 and slip_model[idx, 2] == I + 1
slip_model[idx, 11] = 1

xp, yp, zp, L, W, strike_deg, dip_deg = slip_model[idx, 3:10]
d = max(-zp, 1e-10)
delta = np.radians(dip_deg)
strike = np.radians(strike_deg)

grid_shape = (y1d_m.size, x1d_m.size)
extent_m = [float(x1d_m[0]), float(x1d_m[-1]), float(y1d_m[0]), float(y1d_m[-1])]
coords = np.array([xe, yn, np.zeros(xe.size)])

theta = np.pi / 2 - strike
dx, dy = _xy2xy(np.array(L * 0.5), np.array(0.0), -theta)
xpt = xe - (xp + float(dx))
ypt = yn - (yp + float(dy))
tp = np.zeros(xe.size)

ue, un, uz = calc_okada(
    1.0, U_SLIP, xpt, ypt, NU, delta, d, L, W, FAULT_TYPE, strike, tp, backend="auto",
)
to2d = lambda a: np.asarray(a, dtype=np.float64).reshape(grid_shape)
results = {"Okada": tuple(to2d(a) for a in (ue, un, uz))}

needed = {n for n in (SOURCE_A, SOURCE_B) if n != "Okada"}
client = mph.start()
for mph_i0, mph_igt0 in MPH_FOR_EXPORT:
    mph_path = mph_i0 if I == 0 else mph_igt0
    name = os.path.basename(mph_path)
    if name not in needed:
        continue
    print(f"=== {name}: patch size=({L}, {W})  pos=({L * J}, {W * I}) ===")
    cu, cv, cw = run_comsol(client, mph_path, L, W, I, J, FAULT_TYPE, coords)
    results[name] = tuple(to2d(a) for a in (cu, cv, cw))

missing = needed - set(results)
if missing:
    raise FileNotFoundError(
        f"未算出: {missing}。请把对应 mph 加到 MPH_FOR_EXPORT，或检查 SOURCE_A/B 拼写。"
    )

fig = plot_ab_diff(extent_m, results, SOURCE_A, SOURCE_B)
bind_export(fig, f"{SOURCE_A}_vs_{SOURCE_B}")
print("用工具栏缩放选区（三列同步）。确认后按 s 或点 Save HQ PNG 导出。")
plt.show()
