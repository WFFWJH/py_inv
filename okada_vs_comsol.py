import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_fault_one_plane import load_fault_one_plane
from calc_okada import calc_okada


fault_file = os.path.join(os.path.dirname(__file__), "checkerboard.txt")

width = 20e3
length = 100e3
len_top = 4e3
n_layer = int(length/len_top)
layers = 5
wid_top = float(width/layers)
n_patch = n_layer*layers
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

import matplotlib
import numpy as np
print("backend =", matplotlib.get_backend())
print("SHOW_SLIP =", __import__("os").environ.get("SHOW_SLIP"))
from show_slip_model import show_slip_model


import mph


i=1
j=1
fault_type = 2 # 1:strike slip, 2:dip slip

index_patch = i*n_layer+j
assert slip_model[index_patch, 1]  == index_patch + 1, f"index_patch = {index_patch}, slip_model[index_patch, 1] = {slip_model[index_patch, 1]}"
assert slip_model[index_patch, 2] == i + 1, f"index_patch = {index_patch}, slip_model[index_patch, 2] = {slip_model[index_patch, 2]}"
print(f"xp = {slip_model[index_patch, 3]}, yp = {slip_model[index_patch, 4]}, zp = {slip_model[index_patch, 5]}")

# slip_model[index_all_patches, 0:10] = [
#                     current_fault_id,
#                     indx_patch,
#                     j + 1,
#                     xp,
#                     yp,
#                     zp,
#                     lp_this_layer[i, j],
#                     wp_layer[j],
#                     strikes[i],
#                     dip_i,
#                 ]
slip_model[index_patch,11]  = 1
show_slip_model(
    slip_model,
    ref_lon=95, lonc=95.33, latc=19.61,
    axis_range=[0, 20, 0, 100, -20, 0],
    apply_axis_range=True,
    out_path="fault_one_plane.png",
    block=False,  # 立刻返回; 脚本结束前会自动等你关掉图窗
)

from checkboard import _xy2xy
nx, ny = 100, 140  # 点数可改
x = np.linspace(-50, 50, nx)
y = np.linspace(-20, 120, ny)
X, Y = np.meshgrid(x, y)

xe = X.ravel() * 1000.0  # km -> m
yn = Y.ravel() * 1000.0
nobs = xe.size
data_insar = np.column_stack([
    xe,
    yn,
    np.zeros(nobs),  # ue
    np.zeros(nobs),  # un
    np.zeros(nobs),  # uz
])

d2r = np.pi / 180.0
HF = 1.0
nu = 0.25
delta = slip_model[index_patch,9]*d2r
d = -slip_model[index_patch,5]
if d == 0:
    d = 1e-10
L = slip_model[index_patch,6]
W = slip_model[index_patch,7]
strike_k = slip_model[index_patch,8]*d2r
tp = np.zeros(nobs, dtype=np.float64)

theta = np.pi/2-strike_k
dxf = L * 0.5
dx, dy = _xy2xy(np.array(dxf), np.array(0.0), -theta)
xxo = float(slip_model[index_patch,3] + dx)
yyo = float(slip_model[index_patch,4] + dy)
xpt = data_insar[:, 0] - xxo
ypt = data_insar[:, 1] - yyo
u1=1
ue1, un1, uz1 = calc_okada(HF, u1, xpt, ypt, nu, delta, d, L, W, fault_type, strike_k, tp, backend="auto")


data1 = np.array([xe,yn,np.zeros(nobs)])


client = mph.start()
if i==0:
    model_java = client.load("top0.mph")
else:
    model_java = client.load("top1.mph")

model = model_java.java

model.result().numerical().create("uvw","Interp")
model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("size", [str(L), str(W)]);
print(f"r1'size = {L}, {W}")
model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("pos", [str(L*j),str(W*i)])
print(f"r1'pos = {L*(j)}, {W*i}")
if fault_type == 1:
    model.component("comp1").physics("solid").feature("disp1").set("Direction", [["free"], ["prescribed"], ["free"]]);
    model.component("comp1").physics("solid").feature("disp1").set("U0", [[0],[0.5],[0]])
    model.component("comp1").physics("solid").feature("disp2").set("Direction", [["free"], ["prescribed"], ["free"]]);
    model.component("comp1").physics("solid").feature("disp2").set("U0", [[0],[0.5],[0]])
else:
    model.component("comp1").physics("solid").feature("disp1").set("Direction", [["prescribed"], ["free"], ["free"]]);
    model.component("comp1").physics("solid").feature("disp1").set("U0", [[-0.5],[0],[0]])
    model.component("comp1").physics("solid").feature("disp2").set("Direction", [["prescribed"], ["free"], ["free"]]);
    model.component("comp1").physics("solid").feature("disp2").set("U0", [[0.5],[0],[0]])
model.sol("sol1").runAll();
interp = model.result().numerical("uvw")
interp.setInterpolationCoordinates(data1)
interp.set("expr",["u","v","w"])
result = model.result().numerical("uvw").getData()
result_np = np.asarray(result)
result_u = result_np[0,0,:]
result_v = result_np[1,0,:]
result_w = result_np[2,0,:]


import numpy as np
import matplotlib.pyplot as plt

# =========================
# 公共绘图参数
# =========================
PLOT = {
    "s": 8,
    "cmap": "jet",
    "figsize": (13, 12),
    "digits": 4,       # colorbar 极值保留小数位数
        # colorbar
    "cbar_shrink": 0.75,
    "cbar_fraction": 0.035,
    "cbar_pad": 0.02,
    "cbar_labelpad": 1,
}


def plot_field(
    ax,
    x,
    y,
    value,
    title,
    cbar_label,
    vmin=None,
    vmax=None,
    symmetric=False,
):
    """绘制一个散点场，并在 colorbar 上显示极值"""

    if symmetric:
        vmax = np.max(np.abs(value))
        vmin = -vmax
    else:
        if vmin is None:
            vmin = np.min(value)
        if vmax is None:
            vmax = np.max(value)

    sc = ax.scatter(
        x / 1000,
        y / 1000,
        c=value,
        s=PLOT["s"],
        cmap=PLOT["cmap"],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")

    cbar = plt.colorbar(
    sc,
    ax=ax,
    shrink=PLOT["cbar_shrink"],
    fraction=PLOT["cbar_fraction"],
    pad=PLOT["cbar_pad"],
    )
    cbar.set_label(cbar_label, labelpad=PLOT["cbar_labelpad"])

    # colorbar 显示极值
    if symmetric:
        ticks = [vmin, 0, vmax]
        labels = [
            f"min: {vmin:.{PLOT['digits']}f}",
            "0",
            f"max: {vmax:.{PLOT['digits']}f}",
        ]
    else:
        ticks = [vmin, vmax]
        labels = [
            f"min: {vmin:.{PLOT['digits']}f}",
            f"max: {vmax:.{PLOT['digits']}f}",
        ]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

    return sc


# =========================
# 数据
# =========================
x = data_insar[:, 0]
y = data_insar[:, 1]

data = [
    ("E", ue1, result_u),
    ("N", un1, result_v),
    ("U", uz1, result_w),
]


# =========================
# 绘图
# =========================
fig, axes = plt.subplots(
    3, 3,
    figsize=PLOT["figsize"],
)

for i, (direction, okada, comsol) in enumerate(data):

    error = comsol - okada


    plot_field(
        axes[i, 0],
        x, y, okada,
        f"Okada forward: {direction}",
        f"u{direction.lower()} (m)",
    )

    plot_field(
        axes[i, 1],
        x, y, comsol,
        f"COMSOL forward: {direction}",
        f"u{direction.lower()} (m)",
    )

    # -------------------------------------------------
    # Error 使用对称 colorbar
    # -------------------------------------------------
    plot_field(
        axes[i, 2],
        x, y, error,
        f"Error: {direction}",
        "error (m)",
    )


plt.tight_layout()
plt.show()
exit()

# for i in range(0,1):
#     for j in range(0,1):
#         model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("pos", [str(len_top*(j+1)),str(wid_top*i)])
#         model.component("comp1").physics("solid").feature("disp1").set("Direction", [["free"], ["prescribed"], ["free"]]);
#         model.component("comp1").physics("solid").feature("disp1").set("U0", [[0],[0.5],[0]])
#         model.component("comp1").physics("solid").feature("disp2").set("Direction", [["free"], ["prescribed"], ["free"]]);
#         model.component("comp1").physics("solid").feature("disp2").set("U0", [[0],[0.5],[0]])
#         model.sol("sol1").runAll();

#         interp = model.result().numerical("uvw")
#         interp.setInterpolationCoordinates(data1)
#         interp.set("expr",["u","v","w"])
#         result = model.result().numerical("uvw").getData()
#         result_np = np.asarray(result)
#         result_u = result_np[0,0,:]
#         result_v = result_np[1,0,:]
#         result_w = result_np[2,0,:]
#         Green[:,i*n_layer+j] = np.concatenate([result_u,result_v,result_w])
#         print(i,j,i*n_layer+j)

#         model.component("comp1").physics("solid").feature("disp1").set("Direction", [["prescribed"], ["free"], ["free"]]);
#         model.component("comp1").physics("solid").feature("disp1").set("U0", [[-0.5],[0],[0]])
#         model.component("comp1").physics("solid").feature("disp2").set("Direction", [["prescribed"], ["free"], ["free"]]);
#         model.component("comp1").physics("solid").feature("disp2").set("U0", [[0.5],[0],[0]])
#         model.sol("sol1").runAll();
#         # model.result().numerical().create("uvw","Interp")
#         interp.setInterpolationCoordinates(data1)
#         interp.set("expr",["u","v","w"])
#         result = model.result().numerical("uvw").getData()
#         model_java.save("dip")
#         result_np = np.asarray(result)
#         result_u = result_np[0,0,:]
#         result_v = result_np[1,0,:]
#         result_w = result_np[2,0,:]
#         Green[:,n_patch+i*n_layer+j] = np.concatenate([result_u,result_v,result_w])
#         print(i,j,n_patch+i*n_layer+j)