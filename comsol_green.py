"""Build COMSOL Green matrix；BC 设置与 comsol_vs_comsol 对齐，可切换。"""
import mph
import numpy as np
import time

from paragram import (
    xe, yn, n_patch, n_layer, len_top, wid_top, layers, top0_model, top1_model, dip,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BC_MODE = "uniform"   # "uniform" | "taper1d" | "taper2d"
EDGE_TOL = 10         # 米；仅 taper 时有效
EXCL_MODE = "all"    # "none" → .set() 清空；"all" → .all() 排除全部边
OUT_PATH = "Green_comsol.npy"

nobs = xe.size
coords = np.array([xe, yn, np.zeros(nobs)])
Green = np.empty((nobs * 3, n_patch * 2))


# ---------------------------------------------------------------------------
# BC helpers（与 comsol_vs_comsol 一致）
# ---------------------------------------------------------------------------
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
    return (
        f"if(min(y-{a}[m],{b}[m]-y)<{edge_tol}[m],"
        f"0.5*(1-cos(pi*min(y-{a}[m],{b}[m]-y)/{edge_tol}[m])),1)"
    )


def _taper_z(c, d, edge_tol, dip_deg):
    return (
        f"if(min(z/cos({dip_deg}*pi/180)-{c}[m],{d}[m]-z/cos({dip_deg}*pi/180))<{edge_tol}[m],"
        f"0.5*(1-cos(pi*min(z/cos({dip_deg}*pi/180)-{c}[m],{d}[m]-z/cos({dip_deg}*pi/180))/{edge_tol}[m])),1)"
    )


def _u0_from_taper(fault_type, taper):
    if fault_type == 1:
        return ([["0"], [f"0.5*({taper})"], ["0"]], [["0"], [f"0.5*({taper})"], ["0"]])
    return ([[f"-0.5*({taper})"], ["0"], ["0"]], [[f"0.5*({taper})"], ["0"], ["0"]])


def set_slip_bc_taper(model, fault_type, i, j, L, W, edge_tol, dip_deg=None):
    dip_deg = float(dip[0] if dip_deg is None else dip_deg)
    a, b, c, d = L * j, L * (j + 1), -W * (i + 1), -W * i
    taper = f"({_taper_y(a, b, edge_tol)})*({_taper_z(c, d, edge_tol, dip_deg)})"
    _apply_u0(model, fault_type, _u0_from_taper(fault_type, taper))


def set_slip_bc_taper1d(model, fault_type, i, j, L, W, edge_tol, dip_deg=None):
    dip_deg = float(dip[0] if dip_deg is None else dip_deg)
    a, b, c, d = L * j, L * (j + 1), -W * (i + 1), -W * i
    taper = (
        _taper_y(a, b, edge_tol) if fault_type == 1
        else _taper_z(c, d, edge_tol, dip_deg)
    )
    _apply_u0(model, fault_type, _u0_from_taper(fault_type, taper))


_BC_SETTERS = {
    "uniform": set_slip_bc,
    "taper1d": set_slip_bc_taper1d,
    "taper2d": set_slip_bc_taper,
}


def apply_bc(model, fault_type, i, j):
    setter = _BC_SETTERS.get(BC_MODE)
    if setter is None:
        raise ValueError(f"BC_MODE must be one of {list(_BC_SETTERS)}, got {BC_MODE!r}")
    setter(model, fault_type, i, j, len_top, wid_top, EDGE_TOL)
    set_excluded_edges(model, EXCL_MODE)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
def interp_uvw(model, coords):
    interp = model.result().numerical("uvw")
    interp.setInterpolationCoordinates(coords)
    interp.set("expr", ["u", "v", "w"])
    data = np.asarray(interp.getData())
    return np.concatenate([data[0, 0], data[1, 0], data[2, 0]])


def solve_patch(model, i, j, fault_type, coords):
    r1 = model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")
    r1.set("size", [str(len_top), str(wid_top)])
    r1.set("pos", [str(len_top * j), str(wid_top * i)])
    apply_bc(model, fault_type, i, j)
    model.sol("sol1").runAll()
    return interp_uvw(model, coords)


def fill_layer_range(model, i_range, coords, Green):
    for i in i_range:
        for j in range(n_layer):
            col = i * n_layer + j
            Green[:, col] = solve_patch(model, i, j, 1, coords)
            print(i, j, col)
            Green[:, n_patch + col] = solve_patch(model, i, j, 2, coords)
            print(i, j, n_patch + col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
start = time.perf_counter()
print(f"BC_MODE={BC_MODE}  EDGE_TOL={EDGE_TOL}  EXCL_MODE={EXCL_MODE}")

client = mph.start()
model_java = client.load(top0_model)
model = model_java.java
model.result().numerical().create("uvw", "Interp")
fill_layer_range(model, range(0, 1), coords, Green)
top_time = time.perf_counter()

client.remove(model_java)
del model
client.clear()

model_java = client.load(top1_model)
model = model_java.java
model.result().numerical().remove("uvw")
model.result().numerical().create("uvw", "Interp")
fill_layer_range(model, range(1, layers), coords, Green)

client.clear()
over = time.perf_counter()
print("top time: ", top_time - start)
print("bottom time: ", over - top_time)
print("total time: ", over - start)
np.save(OUT_PATH, Green)
print(f"saved {OUT_PATH}")
