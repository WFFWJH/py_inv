"""余弦窗（raised-cosine taper）演示：COMSOL 断层边界平滑过渡窗的形状。

绘制二维余弦 taper 曲面与沿 x 方向的一维剖面，用于对照
``comsol_vs_comsol.py`` / ``comsol_green.py`` 中 taper1d / taper2d 边界条件的
解析形式::

    if(min(x-a, b-x) < w, 0.5*(1-cos(pi*min(x-a, b-x)/w)), 1)

纯 numpy + matplotlib，不依赖项目内其他模块。
用法：``python comsol_scripts/comsol_slip_smooth.py``
"""
import numpy as np


def cosine_taper(x, y, a, b, c, d, w=5):
    dx = np.minimum(x - a, b - x)
    dy = np.minimum(y - c, d - y)

    tx = np.where(dx < w, 0.5 * (1 - np.cos(np.pi * dx / w)), 1)
    ty = np.where(dy < w, 0.5 * (1 - np.cos(np.pi * dy / w)), 1)

    return tx * ty


def main() -> None:
    import matplotlib.pyplot as plt

    a, b, c, d, w = 0, 50, 0, 50, 5
    x = np.linspace(a, b, 400)
    y = np.linspace(c, d, 400)
    X, Y = np.meshgrid(x, y)
    Z = cosine_taper(X, Y, a, b, c, d, w)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("2D Cosine Taper")
    ax.set_zlim(0, 1.05)
    fig.colorbar(surface, ax=ax, shrink=0.7, label="z")
    plt.tight_layout()
    plt.show()

    xx = np.linspace(a, b, 1000)
    yy = (c + d) / 2
    zx = cosine_taper(xx, yy, a, b, c, d, w)
    plt.figure(figsize=(8, 4))
    plt.plot(xx, zx)
    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.title(f"Cosine Taper in x direction, y={yy}")
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()