"""棋盘格测试的共享参数与观测网格 (供 :mod:`workflows.checkerboard` 使用).

``fault_file`` 指向项目根目录的 ``checkerboard.txt``; ``xe`` / ``yn`` 为近断层加密的
1D 观测网格 (单位 m), 用于 Okada 正演合成棋盘格观测。
"""
import os

import numpy as np

# 观测网格: x 在近断层加密 (-20~20 km 共 200 点), 两侧约 1 km 间距; y 覆盖 -20~120 km
ny = 700
x1d_m = np.unique(np.concatenate([
    np.linspace(-50, -20, 50),
    np.linspace(-20, 20, 200),
    np.linspace(20, 50, 50),
]))
y1d_m = np.linspace(-20, 120, ny)
X, Y = np.meshgrid(x1d_m, y1d_m)
xe = X.ravel() * 1000.0  # km -> m
yn = Y.ravel() * 1000.0

# 断层参数 (项目根目录下的 checkerboard.txt 提供迹线)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fault_file = os.path.join(_PROJECT_ROOT, "checkerboard.txt")

ref = dict(ref_lon=95, lonc=95.33, latc=19.61)
axis_range = [0, 20, 0, 100, -20, 0]

dip = [80]
width = 20e3
length = 100e3
l_ratio = 1
w_ratio = 1
len_top = 10e3
layers = 2
n_layer = int(length / len_top)
wid_top = float(width / layers)
n_patch = n_layer * layers


# 棋盘格真值: 沿走向相邻 patch 交替 0/1, 逐层错开
slip_matrix = np.zeros((layers, n_layer))
one_layer_slip = np.zeros(n_layer)
next_layer_slip = np.zeros(n_layer)
for i in range(n_layer):
    if i % 2 == 0:
        one_layer_slip[i] = 1
        next_layer_slip[i] = 0
    else:
        one_layer_slip[i] = 0
        next_layer_slip[i] = 1
for i in range(layers):
    if i % 2 == 0:
        slip_matrix[i, :] = one_layer_slip
    else:
        slip_matrix[i, :] = next_layer_slip

# COMSOL 参考模型文件名 (可选)
top0_model = "top0_most_refine_extend.mph"
top1_model = "top1_most_refine_extend.mph"
