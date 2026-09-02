# if(min(x-a,b-x)<5,0.5*(1-cos(pi*min(x-a,b-x)/5)),1)*if(min(y-c,d-y)<5,0.5*(1-cos(pi*min(y-c,d-y)/5)),1)
# 
import numpy as np
import matplotlib.pyplot as plt


def cosine_taper(x, y, a, b, c, d, w=5):
    dx = np.minimum(x - a, b - x)
    dy = np.minimum(y - c, d - y)

    tx = np.where(dx < w, 0.5 * (1 - np.cos(np.pi * dx / w)), 1)
    ty = np.where(dy < w, 0.5 * (1 - np.cos(np.pi * dy / w)), 1)

    return tx * ty


# 参数
a, b = 0, 50
c, d = 0, 50
w = 5

# 二维网格
x = np.linspace(a, b, 400)
y = np.linspace(c, d, 400)
X, Y = np.meshgrid(x, y)

# 计算二维 taper
Z = cosine_taper(X, Y, a, b, c, d, w)


# ===== 三维 =====
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("2D Cosine Taper")
ax.set_zlim(0, 1.05)

fig.colorbar(surf, ax=ax, shrink=0.7, label="z")
plt.tight_layout()
plt.show()


# ===== x 方向一维窗 =====
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