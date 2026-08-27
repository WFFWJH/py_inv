import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_fault_one_plane import load_fault_one_plane
from calc_okada import calc_okada
from calc_green import _xy2xy
import numpy as np


# build Green function

def _build_green_patch_loop(data_slip_model: np.ndarray,
                            data_insar: np.ndarray,
                            nu: float = 0.25,
                            backend: str = "auto") -> np.ndarray:
    """Build Green function for checkboard model."""
    d2r = np.pi / 180.0
    xp = data_slip_model[:, 3]
    yp = data_slip_model[:, 4]
    zp = data_slip_model[:, 5]
    lp = data_slip_model[:, 6]
    wp = data_slip_model[:, 7]
    strkp = data_slip_model[:, 8]
    dip0 = data_slip_model[:, 9]

    Npatch = zp.size
    Npara = 2 * Npatch

    xe = np.asarray(data_insar[:, 0], dtype=np.float64)
    yn = np.asarray(data_insar[:, 1], dtype=np.float64)

    Nobs = xe.size
    G = np.zeros((Nobs*3, Npara), dtype=np.float64)

    HF = 1.0
    tp = np.zeros(Nobs, dtype=np.float64)

    for k in range(Npatch):
        strike_k = strkp[k] * d2r
        theta_k = (90.0 - strkp[k]) * d2r
        dxf = lp[k] * 0.5
        dx, dy = _xy2xy(np.asarray(dxf), np.asarray(0.0), -theta_k)
        xxo = xp[k] + float(dx)
        yyo = yp[k] + float(dy)
        zzo = zp[k]

        xpt = xe - xxo
        ypt = yn - yyo
        delta = dip0[k] * d2r
        d = -zzo
        L = lp[k]
        W = wp[k]

        ue1, un1, uz1 = calc_okada(HF, 1.0, xpt, ypt, nu, delta, d, L, W, 1, strike_k, tp, backend=backend)
        ue2, un2, uz2 = calc_okada(HF, 1.0, xpt, ypt, nu, delta, d, L, W, 2, strike_k, tp, backend=backend)

        # 0:Nobs is ue1, Nobs:2*Nobs is un1, 2*Nobs:3*Nobs is uz1
        # Nobs:2*Nobs is ue2, 2*Nobs:3*Nobs is un2, 3*Nobs:4*Nobs is uz2
        G[0:Nobs, k] = ue1
        G[Nobs:2*Nobs, k] = un1
        G[2*Nobs:3*Nobs, k] = uz1
        G[0:Nobs, k + Npatch] = ue2
        G[Nobs:2*Nobs, k + Npatch] = un2
        G[2*Nobs:3*Nobs, k + Npatch] = uz2

    return G
    
def calc_insar_patches_contrib_okada(
    data_insar: np.ndarray,
    slip_model_in: np.ndarray,
    nu: float,
    backend: str,
) -> np.ndarray:
    """在 data_insar 上累加 Okada 位移.

    data_insar 列: [x, y, ue, un, uz] (x,y 单位 m; ue/un/uz 初值通常为 0).
    """
    d2r = np.pi / 180.0
    HF = 1.0
    Npatch = slip_model_in.shape[0]
    Nobs = data_insar.shape[0]

    tp = np.zeros(Nobs, dtype=np.float64)
    xp = slip_model_in[:, 3]
    yp = slip_model_in[:, 4]
    zp = slip_model_in[:, 5]
    lp = slip_model_in[:, 6]
    wp = slip_model_in[:, 7]
    strkp = slip_model_in[:, 8]
    dip0 = slip_model_in[:, 9]
    s1 = slip_model_in[:, 11]
    s2 = slip_model_in[:, 12]
    for k in range(Npatch):
        theta = (90.0 - strkp[k]) * d2r
        dxf = lp[k] * 0.5
        dx, dy = _xy2xy(np.array(dxf), np.array(0.0), -theta)
        xxo = float(xp[k] + dx)
        yyo = float(yp[k] + dy)
        zzo = float(zp[k])
        u1, u2 = float(s1[k]), float(s2[k])
        xpt = data_insar[:, 0] - xxo
        ypt = data_insar[:, 1] - yyo
        delta = float(dip0[k] * d2r)
        d = float(-zzo)
        L, W = float(lp[k]), float(wp[k])
        strike_k = float(strkp[k] * d2r)
        ue1, un1, uz1 = calc_okada(HF, u1, xpt, ypt, nu, delta, d, L, W, 1, strike_k, tp, backend=backend)
        ue2, un2, uz2 = calc_okada(HF, u2, xpt, ypt, nu, delta, d, L, W, 2, strike_k, tp, backend=backend)
        data_insar[:, 2] += np.asarray(ue1, dtype=np.float64) + np.asarray(ue2, dtype=np.float64)
        data_insar[:, 3] += np.asarray(un1, dtype=np.float64) + np.asarray(un2, dtype=np.float64)
        data_insar[:, 4] += np.asarray(uz1, dtype=np.float64) + np.asarray(uz2, dtype=np.float64)

    return data_insar


if __name__ == "__main__":
    fault_file = os.path.join(os.path.dirname(__file__), "checkerboard.txt")

    width = 20e3
    length = 100e3
    len_top = 4e3
    n_layer = int(length/len_top)
    layers = 5

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

    import matplotlib
    print("backend =", matplotlib.get_backend())
    print("SHOW_SLIP =", __import__("os").environ.get("SHOW_SLIP"))
    from show_slip_model import show_slip_model

    show_slip_model(
        slip_model,
        ref_lon=95, lonc=95.33, latc=19.61,
        axis_range=[0, 20, 0, 100, -20, 0],
        apply_axis_range=True,
        out_path="fault_one_plane.png",
        block=False,  # 立刻返回; 脚本结束前会自动等你关掉图窗
    )
    # 后面可以继续写代码; 图窗会一直开着, 直到进程退出前由 show_slip_model 挂起等待
    print("继续运行 ... (关掉图窗后进程才会结束)", flush=True)

    nx, ny = 100, 140  # 点数可改
    x = np.linspace(-50, 50, nx)
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

    import matplotlib.pyplot as plt
    plt.figure()
    sc = plt.scatter(
        data_insar[:, 0] / 1000.0,
        data_insar[:, 1] / 1000.0,
        c=data_insar[:, 4],  # 着色用 uz; 可改成 ue/un
        s=8,
        cmap="jet",
    )
    plt.colorbar(sc, label="uz (m)")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.axis("equal")
    plt.title("checkerboard forward: uz")
    plt.show()

    # build Green function
    G = _build_green_patch_loop(slip_model, data_insar, nu=0.25, backend="auto")
    # d = [ue; un; uz], 与 G 行顺序一致 (长度 3*Nobs)
    d = np.concatenate([data_insar[:, 2], data_insar[:, 3], data_insar[:, 4]])
    print("G, d:", G.shape, d.shape)

    # ------------------------------------------------------------------
    # 求解 (参考 make_fault_from_insar1: Tikhonov 平滑 + 有界 lsq_linear)
    #   min || [G; (lam/h1)*H] u - [d; 0] ||^2   s.t.  lb <= u <= ub
    # ------------------------------------------------------------------
    from scipy.optimize import lsq_linear
    from build_smooth_function import build_smooth_function
    from bounds_new import bounds_new

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
    print(f"smooth H: {H.shape}, h1={h1}, lam={lam}")

    # np.save("Green_okada.npy", G)

# Okada inversion
    Greens = np.vstack([G, H * (lam / max(h1, 1))])
    bdata_sm = np.concatenate([d, np.zeros(H.shape[0], dtype=np.float64)])

    nflt = int(slip_geo[:, 0].max())
    fault_id = slip_geo[:, 0].astype(int)
    tSm = np.zeros(nflt + 1, dtype=int)
    for i in range(1, nflt + 1):
        tSm[i] = int(np.sum(fault_id == i))
    lb, ub = bounds_new(nflt, 2, tSm, 1, 0, Con)

    res = lsq_linear(
        np.ascontiguousarray(Greens, dtype=np.float64),
        np.ascontiguousarray(bdata_sm, dtype=np.float64).ravel(),
        bounds=(lb, ub),
        method="trf",
        tol=1e-12,
        max_iter=200,
        verbose=0,
    )
    u = res.x
    print(f"solver: success={res.success}  nit={res.nit}  {res.message}")

    slip_inv = slip_geo.copy()
    slip_inv[:, 11] = u[:Npatch]
    slip_inv[:, 12] = u[Npatch:2 * Npatch]

    misfit = G @ u - d
    rms = float(np.sqrt(np.mean(misfit ** 2)))
    print(f"data RMS misfit = {rms:.6e} m")
    print(f"true strike slip  min/max = {true_slip[:, 11].min():.3f} / {true_slip[:, 11].max():.3f}")
    print(f"inv  strike slip  min/max = {slip_inv[:, 11].min():.3f} / {slip_inv[:, 11].max():.3f}")

    show_slip_model(
        slip_inv,
        ref_lon=95, lonc=95.33, latc=19.61,
        axis_range=[0, 20, 0, 100, -20, 0],
        apply_axis_range=True,
        out_path="fault_checkerboard_inv_okada.png",
        title="checkerboard inverted slip",
        block=False,
    )


# Comsol inversion

    G_comsol = np.load("Green.npy")
    Greens = np.vstack([G_comsol, H * (lam / max(h1, 1))])
    res = lsq_linear(
        np.ascontiguousarray(Greens, dtype=np.float64),
        np.ascontiguousarray(bdata_sm, dtype=np.float64).ravel(),
        bounds=(lb, ub),
        method="trf",
        tol=1e-12,
        max_iter=200,
        verbose=0,
    )
    u = res.x
    print(f"solver: success={res.success}  nit={res.nit}  {res.message}")

    slip_inv = slip_geo.copy()
    slip_inv[:, 11] = u[:Npatch]
    slip_inv[:, 12] = u[Npatch:2 * Npatch]

    misfit = G_comsol @ u - d
    rms = float(np.sqrt(np.mean(misfit ** 2)))
    print(f"data RMS misfit = {rms:.6e} m")
    print(f"true strike slip  min/max = {true_slip[:, 11].min():.3f} / {true_slip[:, 11].max():.3f}")
    print(f"inv  strike slip  min/max = {slip_inv[:, 11].min():.3f} / {slip_inv[:, 11].max():.3f}")

    show_slip_model(
        slip_inv,
        ref_lon=95, lonc=95.33, latc=19.61,
        axis_range=[0, 20, 0, 100, -20, 0],
        apply_axis_range=True,
        out_path="fault_checkerboard_inv_comsol.png",
        title="checkerboard inverted slip",
        block=False,
    )
    print("继续运行 ... (关掉图窗后进程才会结束)", flush=True)
