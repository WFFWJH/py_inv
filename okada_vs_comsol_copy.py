"""Compare Okada vs COMSOL forward displacement for one fault patch."""
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAULT_FILE = os.path.join(os.path.dirname(__file__), "checkerboard.txt")
WIDTH, LENGTH, LEN_TOP, LAYERS = 20e3, 100e3, 4e3, 5
N_LAYER = int(LENGTH / LEN_TOP)

I, J = 1, 1                          # patch layer / along-strike index
FAULT_TYPE = 2                       # 1: strike-slip, 2: dip-slip
NU, U_SLIP = 0.25, 1.0


REF = dict(ref_lon=95, lonc=95.33, latc=19.61)
AXIS_RANGE = [0, 20, 0, 100, -20, 0]

PLOT = dict(
    s=8, cmap="jet", figsize=(13, 12), digits=4,
    cbar_shrink=0.75, cbar_fraction=0.035, cbar_pad=0.02, cbar_labelpad=1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def plot_field(ax, x, y, value, title, cbar_label, symmetric=False):
    if symmetric:
        vmax = np.max(np.abs(value))
        vmin = -vmax
    else:
        vmin, vmax = np.min(value), np.max(value)

    sc = ax.scatter(
        x / 1000, y / 1000, c=value,
        s=PLOT["s"], cmap=PLOT["cmap"], vmin=vmin, vmax=vmax,
    )
    ax.set(title=title, xlabel="x (km)", ylabel="y (km)")
    ax.set_aspect("equal")

    cbar = plt.colorbar(
        sc, ax=ax,
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
    """Set disp1/disp2 boundary conditions for unit slip."""
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

# patch geometry (cols: xp,yp,zp, L,W, strike, dip)
xp, yp, zp, L, W, strike_deg, dip_deg = slip_model[idx, 3:10]
d = max(-zp, 1e-10)
delta = np.radians(dip_deg)
strike = np.radians(strike_deg)

# observation grid (m)
NX, NY = 100, 140                    # observation grid
xe = np.linspace(-50, 50, NX) * 1000.0
yn = np.linspace(-20, 120, NY) * 1000.0
XE, YN = np.meshgrid(xe, yn)
xe, yn = XE.ravel(), YN.ravel()
nobs = xe.size

# patch center offset along strike (Okada reference point)
theta = np.pi / 2 - strike
dx, dy = _xy2xy(np.array(L * 0.5), np.array(0.0), -theta)
xpt = xe - (xp + float(dx))
ypt = yn - (yp + float(dy))
tp = np.zeros(nobs)

ue, un, uz = calc_okada(
    1.0, U_SLIP, xpt, ypt, NU, delta, d, L, W, FAULT_TYPE, strike, tp, backend="auto",
)

# ---------------------------------------------------------------------------
# COMSOL forward
# ---------------------------------------------------------------------------
client = mph.start()
model_java = client.load("top0.mph" if I == 0 else "top1.mph")
model = model_java.java

geom_r1 = model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")
geom_r1.set("size", [str(L), str(W)])
geom_r1.set("pos", [str(L * J), str(W * I)])
print(f"patch r1 size=({L}, {W})  pos=({L * J}, {W * I})")

set_comsol_slip_bc(model, FAULT_TYPE)
model.sol("sol1").runAll()

model.result().numerical().create("uvw", "Interp")
interp = model.result().numerical("uvw")
interp.setInterpolationCoordinates(np.array([xe, yn, np.zeros(nobs)]))
interp.set("expr", ["u", "v", "w"])
comsol = np.asarray(interp.getData())  # (3, 1, nobs)
cu, cv, cw = comsol[0, 0], comsol[1, 0], comsol[2, 0]

# ---------------------------------------------------------------------------
# Compare plots: Okada | COMSOL | Error
# ---------------------------------------------------------------------------
panels = [
    ("E", ue, cu),
    ("N", un, cv),
    ("U", uz, cw),
]

fig, axes = plt.subplots(3, 3, figsize=PLOT["figsize"])
for row, (name, okada, cos) in enumerate(panels):
    label = f"u{name.lower()} (m)"
    plot_field(axes[row, 0], xe, yn, okada, f"Okada: {name}", label)
    plot_field(axes[row, 1], xe, yn, cos, f"COMSOL: {name}", label)
    plot_field(axes[row, 2], xe, yn, cos - okada, f"Error: {name}", "error (m)", symmetric=True)

plt.tight_layout()
plt.show()
