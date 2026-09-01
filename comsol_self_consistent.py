# ============================================================
# COMSOL self-consistent checkerboard test
# ============================================================
import numpy as np
from load_fault_one_plane import load_fault_one_plane
from build_smooth_function import build_smooth_function
import os
from scipy.optimize import lsq_linear
from bounds_new import bounds_new
from show_slip_model import show_slip_model


fault_file = os.path.join(os.path.dirname(__file__), "checkerboard.txt")

width = 20e3
length = 100e3
# len_top = 4e3
# layers = 5
len_top = 10e3
layers = 2
n_layer = int(length/len_top)
slip_model = load_fault_one_plane(fault_file,dip=[80],
lonc=95.33,
latc=19.61,
ref_lon=95,
l_ratio=1,
w_ratio=1,
width=width,
len_top=len_top,
layers=layers,
coord_mode="local_xy"
);

slip_matrix = np.zeros((layers, n_layer))
one_layer_slip = np.zeros(n_layer)
next_layer_slip = np.zeros(n_layer)
for i in range(n_layer):
    if i%2 == 0:
        one_layer_slip[i] = 1
        next_layer_slip[i] = 0
    else:
        one_layer_slip[i] = 0
        next_layer_slip[i] = 1
for i in range(layers):
    if i%2 == 0:
        slip_matrix[i, :] = one_layer_slip
    else:
        slip_matrix[i, :] = next_layer_slip
for i in range(layers):
    for j in range(n_layer):
        slip_model[i*n_layer+j, 11] = slip_matrix[i, j]


true_slip = slip_model.copy()
slip_geo = slip_model.copy()
slip_geo[:, 11:13] = 0.0
slip_geo[:, 1] = np.arange(1, slip_geo.shape[0] + 1)

Npatch = slip_geo.shape[0]
lam = 0.001  # 平滑系数; 棋盘格可略小以免抹平 0/1 图案
Con = (1, 0, 0)  # 走滑 >= 0 (棋盘格真值为 0 或 1)

H, h1, _ = build_smooth_function(
    slip_geo, None, None, None, "none", dip_smooth=True,
)
nflt = int(slip_geo[:, 0].max())
fault_id = slip_geo[:, 0].astype(int)
tSm = np.zeros(nflt + 1, dtype=int)
for i in range(1, nflt + 1):
    tSm[i] = int(np.sum(fault_id == i))
lb, ub = bounds_new(nflt, 2, tSm, 2, 0, Con)


G_comsol = np.load("Green_comsol.npy")

u_true = np.zeros(2 * Npatch, dtype=np.float64)
u_true[:Npatch] = true_slip[:, 11]   # 只有 strike-slip
# u_true[Npatch:] 保持 0

# 用 COMSOL Green function 自己生成 synthetic data
d_comsol = G_comsol @ u_true

Greens = np.vstack([
    G_comsol,
    H * (lam / max(h1, 1))
])

bdata_sm = np.concatenate([
    d_comsol,
    np.zeros(H.shape[0], dtype=np.float64)
])

res = lsq_linear(
    np.ascontiguousarray(Greens, dtype=np.float64),
    np.ascontiguousarray(bdata_sm, dtype=np.float64),
    bounds=(lb, ub),
    method="trf",
    tol=1e-12,
    max_iter=200,
    verbose=0,
)

u = res.x

print("===== COMSOL SELF TEST =====")
print("success =", res.success)
print("strike true min/max =",
      u_true[:Npatch].min(),
      u_true[:Npatch].max())
print("strike inv  min/max =",
      u[:Npatch].min(),
      u[:Npatch].max())
print("dip true min/max =",
      u_true[Npatch:].min(),
      u_true[Npatch:].max())
print("dip inv  min/max =",
      u[Npatch:].min(),
      u[Npatch:].max())

misfit = G_comsol @ u - d_comsol
print("RMS =", np.sqrt(np.mean(misfit**2)))

show_slip_model(
    slip_model,
    ref_lon=95, lonc=95.33, latc=19.61,
    axis_range=[0, 20, 0, 100, -20, 0],
    apply_axis_range=True,
    out_path="fault_checkerboard_inv_okada.png",
    title="checkerboard inverted slip",
    block=False,
)


slip_model[:, 11] = u[:Npatch]
slip_model[:, 12] = u[Npatch:2 * Npatch]

show_slip_model(
    slip_model,
    ref_lon=95, lonc=95.33, latc=19.61,
    axis_range=[0, 20, 0, 100, -20, 0],
    apply_axis_range=True,
    out_path="fault_checkerboard_inv_comsol.png",
    title="checkerboard inverted slip",
    block=False,
)