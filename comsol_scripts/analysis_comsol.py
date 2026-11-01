"""Compare Okada and COMSOL Green functions.

读取项目根目录下的 ``Green_okada.npy`` 与 ``Green_comsol.npy``：

  * 给出整体相对误差，以及每个片元走滑 / 倾滑列的相对误差与方向余弦；
  * 用 Okada 前向算出的合成观测，考察 Okada/COMSOL 失配残差在走滑、倾滑
    子空间上的投影比。

前置：先由 ``workflows.checkerboard`` 生成 ``Green_okada.npy``，
由 ``comsol_scripts/comsol_green.py`` 生成 ``Green_comsol.npy``。

用法：``python comsol_scripts/analysis_comsol.py``
"""


def main() -> None:
    global G, G_comsol, Npatch, _ROOT, calc_insar_patches_contrib_okada, cd, cs
    global d, data_insar, dip, ed, es, fault_file, i
    global j, k, l_ratio, layers, len_top, load_fault_one_plane, n_layer
    global next_layer_slip, nobs, np, one_layer_slip, os, proj_d, proj_s
    global r, ref, slip_matrix, slip_model, sys, true_slip, u_true
    global w_ratio, width, xe, yn
    import os
    import sys

    import numpy as np

    # 项目根目录加入 sys.path，保证 geodetic / workflows 可导入
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    from geodetic.fault_geometry import load_fault_one_plane
    from workflows.checkerboard import calc_insar_patches_contrib_okada
    from workflows.paragram_checkboard import (
        dip, fault_file, l_ratio, layers, len_top, n_layer, ref, w_ratio, width, xe, yn,
    )

    slip_model = load_fault_one_plane(fault_file,dip=dip,
    **ref,
    l_ratio=l_ratio,
    w_ratio=w_ratio,
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
    G_comsol = np.load(os.path.join(_ROOT, "Green_comsol.npy"))
    G = np.load(os.path.join(_ROOT, "Green_okada.npy"))
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


if __name__ == "__main__":
    main()
