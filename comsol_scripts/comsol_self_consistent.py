"""Run the checkerboard COMSOL self-consistency test.

只使用走滑分量（``u_true[:Npatch]``），由 ``Green_comsol.npy`` 生成合成数据，
再以 Tikhonov 平滑 + 有界最小二乘反演，检验 COMSOL 模型自身的可还原性。

用法：``python comsol_scripts/comsol_self_consistent.py``（需先有 Green_comsol.npy）
"""


def main() -> None:
    global Con, G_comsol, Greens, H, Npatch, _, _ROOT
    global axis_range, bdata_sm, bounds_new, build_smooth_function, d_comsol, dip, fault_file
    global fault_id, h1, i, j, l_ratio, lam, layers
    global lb, len_top, load_fault_one_plane, lsq_linear, misfit, n_layer, next_layer_slip
    global nflt, np, one_layer_slip, os, ref, res, show_slip_model
    global slip_geo, slip_matrix, slip_model, sys, tSm, true_slip, u
    global u_true, ub, w_ratio, width
    import os
    import sys

    import numpy as np
    from scipy.optimize import lsq_linear

    # 项目根目录加入 sys.path，保证 geodetic / workflows 可导入
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from geodetic.constraints import bounds_new
    from geodetic.fault_geometry import load_fault_one_plane
    from geodetic.show_slip_model import show_slip_model
    from geodetic.smoothing import build_smooth_function
    from workflows.paragram_checkboard import (
        axis_range, dip, fault_file, l_ratio, layers, len_top, n_layer, ref, w_ratio, width,
    )

    slip_model = load_fault_one_plane(fault_file,dip=dip,
    **ref,
    l_ratio=l_ratio,
    w_ratio=w_ratio,
    width=width,
    len_top=len_top,
    layers=layers,
    coord_mode="local_xy",
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


    G_comsol = np.load(os.path.join(_ROOT, "Green_comsol.npy"))

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
        **ref,
        axis_range=axis_range,
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


if __name__ == "__main__":
    main()
