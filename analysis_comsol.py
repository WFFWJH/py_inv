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
G_comsol = np.load("Green_comsol.npy")
G = np.load("Green_okada.npy")
Npatch = int(G.shape[1]/2)


print("\n==============================")
print("OKADA vs COMSOL")
print("==============================")

print(
    "total relative error =",
    np.linalg.norm(G_comsol-G)/np.linalg.norm(G)
)

print("\nPatch errors:")

for k in range(Npatch):

    es = np.linalg.norm(
        G_comsol[:,k]-G[:,k]
    ) / np.linalg.norm(G[:,k])

    ed = np.linalg.norm(
        G_comsol[:,k+Npatch]-G[:,k+Npatch]
    ) / np.linalg.norm(G[:,k+Npatch])

    cs = np.dot(
        G[:,k], G_comsol[:,k]
    ) / (
        np.linalg.norm(G[:,k]) *
        np.linalg.norm(G_comsol[:,k])
    )

    cd = np.dot(
        G[:,k+Npatch], G_comsol[:,k+Npatch]
    ) / (
        np.linalg.norm(G[:,k+Npatch]) *
        np.linalg.norm(G_comsol[:,k+Npatch])
    )

    print(
        f"patch {k:02d}: "
        f"strike err={es:.3e}, cos={cs:.8f}; "
        f"dip err={ed:.3e}, cos={cd:.8f}"
    )



# ----------------------------------
# residual caused by Okada/COMSOL mismatch
# ----------------------------------

u_true = np.zeros(2*Npatch)
u_true[:Npatch] = true_slip[:,11]

# d
from checkboard import calc_insar_patches_contrib_okada
ny = 700
x = np.unique(np.concatenate([
    np.linspace(-50, -20, 50),
    np.linspace(-20, 20, 200),
    np.linspace(20, 50, 50),
]))
y = np.linspace(-20, 120, ny)
X, Y = np.meshgrid(x, y)

xe = X.ravel() * 1000.0  # km -> m
yn = Y.ravel() * 1000.0
# 五列: x, y, ue, un, uz (位移初值为 0, 由 Okada 前向填充)
nobs = xe.size
data_insar = np.column_stack([
    xe,
    yn,
    np.zeros(nobs),  # ue
    np.zeros(nobs),  # un
    np.zeros(nobs),  # uz
])

data_insar = calc_insar_patches_contrib_okada(
    data_insar, slip_model, nu=0.25, backend="auto",
)
d = np.concatenate([data_insar[:, 2], data_insar[:, 3], data_insar[:, 4]])

assert np.allclose(d, G @ u_true)


r = d - G_comsol @ u_true

proj_s = G_comsol[:,:Npatch].T @ r
proj_d = G_comsol[:,Npatch:].T @ r

print("\nMismatch residual norm =", np.linalg.norm(r))
print(
    "dip/strike projection ratio =",
    np.linalg.norm(proj_d) /
    max(np.linalg.norm(proj_s), 1e-30)
)