import numpy as np
import os
# build data: x 近断层加密（-20~20 km 共 200 点），两侧约 1 km 间距
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

# fault parameters

fault_file = os.path.join(os.path.dirname(__file__), "checkerboard.txt")

ref = dict(ref_lon=95, lonc=95.33, latc=19.61)
axis_range = [0, 20, 0, 100, -20, 0]

dip=[80]
width = 20e3
length = 100e3
l_ratio = 1
w_ratio = 1
len_top = 10e3
layers = 2
n_layer = int(length/len_top)
wid_top = float(width/layers)
n_patch = n_layer * layers


# checkboard parameters
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

# model
top0_model = "top0_most_refine_extend.mph"
top1_model = "top1_most_refine_extend.mph"