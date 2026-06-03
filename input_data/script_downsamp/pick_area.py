#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轻量化 GRD 多边形选点工具

功能
------
- 打开 GMT grd 文件
- 正确经纬度比例显示
- 左键添加点
- 右键撤销
- 滚轮缩放
- 中键拖动平移
- Enter 保存 polygon.txt

依赖
------
pip install xarray matplotlib numpy

使用
------
python pick_polygon.py input.grd
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP',
    'Noto Sans CJK SC',
    'SimHei',
    'Microsoft YaHei'
]

matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
class Picker:

    def __init__(self, grd):

        self.grd = Path(grd)

        self.points = []

        self.dragging = False
        self.last_mouse = None

        self.load_grd()

    # -------------------------------------------------
    # 读取 grd
    # -------------------------------------------------
    def load_grd(self):

        ds = xr.open_dataset(self.grd)

        zname = list(ds.data_vars.keys())[0]

        da = ds[zname]

        yname, xname = da.dims

        self.z = da.values

        self.x = ds[xname].values
        self.y = ds[yname].values

    # -------------------------------------------------
    # 绘图
    # -------------------------------------------------
    def show(self):

        self.fig, self.ax = plt.subplots(
            figsize=(10, 8)
        )

        self.im = self.ax.imshow(
            self.z,
            origin="lower",
            extent=[
                self.x.min(),
                self.x.max(),
                self.y.min(),
                self.y.max()
            ],
            cmap="jet",
            aspect="auto"
        )

        # 经纬度比例修正
        lat0 = np.mean(self.y)

        self.ax.set_aspect(
            1 / np.cos(np.deg2rad(lat0))
        )

        plt.colorbar(self.im, ax=self.ax)

        self.ax.set_title(
            "左键添加 | 右键撤销 | 滚轮缩放 | 中键拖动 | Enter保存"
        )

        self.scatter, = self.ax.plot(
            [],
            [],
            "ro",
            ms=5
        )

        self.line, = self.ax.plot(
            [],
            [],
            "r-",
            lw=1.5
        )

        # 鼠标
        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_press
        )

        self.fig.canvas.mpl_connect(
            "button_release_event",
            self.on_release
        )

        self.fig.canvas.mpl_connect(
            "motion_notify_event",
            self.on_motion
        )

        self.fig.canvas.mpl_connect(
            "scroll_event",
            self.on_scroll
        )

        # 键盘
        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        plt.show()

    # -------------------------------------------------
    # 更新显示
    # -------------------------------------------------
    def update_plot(self):

        if len(self.points) == 0:

            self.scatter.set_data([], [])
            self.line.set_data([], [])

        else:

            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]

            self.scatter.set_data(xs, ys)
            self.line.set_data(xs, ys)

        self.fig.canvas.draw_idle()

    # -------------------------------------------------
    # 鼠标按下
    # -------------------------------------------------
    def on_press(self, event):

        if event.inaxes != self.ax:
            return

        # 中键开始拖动
        if event.button == 2:

            self.dragging = True

            self.last_mouse = (
                event.x,
                event.y
            )

            return

        # 左键添加点
        if event.button == 1:

            if event.xdata is None:
                return

            self.points.append(
                (event.xdata, event.ydata)
            )

            print(
                f"[ADD] "
                f"{event.xdata:.6f} "
                f"{event.ydata:.6f}"
            )

            self.update_plot()

        # 右键撤销
        elif event.button == 3:

            if len(self.points) > 0:

                p = self.points.pop()

                print(
                    f"[UNDO] "
                    f"{p[0]:.6f} "
                    f"{p[1]:.6f}"
                )

                self.update_plot()

    # -------------------------------------------------
    # 鼠标释放
    # -------------------------------------------------
    def on_release(self, event):

        self.dragging = False

    # -------------------------------------------------
    # 鼠标移动（平移）
    # -------------------------------------------------
    def on_motion(self, event):

        if not self.dragging:
            return

        if self.last_mouse is None:
            return

        dx = event.x - self.last_mouse[0]
        dy = event.y - self.last_mouse[1]

        self.last_mouse = (
            event.x,
            event.y
        )

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        sx = (xlim[1] - xlim[0]) / self.ax.bbox.width
        sy = (ylim[1] - ylim[0]) / self.ax.bbox.height

        self.ax.set_xlim(
            xlim[0] - dx * sx,
            xlim[1] - dx * sx
        )

        self.ax.set_ylim(
            ylim[0] - dy * sy,
            ylim[1] - dy * sy
        )

        self.fig.canvas.draw_idle()

    # -------------------------------------------------
    # 滚轮缩放
    # -------------------------------------------------
    def on_scroll(self, event):

        if event.inaxes != self.ax:
            return

        scale = 1.2

        if event.button == "up":
            factor = 1 / scale
        else:
            factor = scale

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x = event.xdata
        y = event.ydata

        if x is None:
            return

        w = (xlim[1] - xlim[0]) * factor
        h = (ylim[1] - ylim[0]) * factor

        rx = (xlim[1] - x) / (xlim[1] - xlim[0])
        ry = (ylim[1] - y) / (ylim[1] - ylim[0])

        self.ax.set_xlim([
            x - w * (1 - rx),
            x + w * rx
        ])

        self.ax.set_ylim([
            y - h * (1 - ry),
            y + h * ry
        ])

        self.fig.canvas.draw_idle()

    # -------------------------------------------------
    # 键盘
    # -------------------------------------------------
    def on_key(self, event):

        if event.key == "enter":

            self.finish()

    # -------------------------------------------------
    # 保存
    # -------------------------------------------------
    def finish(self):

        if len(self.points) < 3:

            print("至少需要3个点")

            return

        pts = np.array(self.points)

        # 自动闭合
        if not np.allclose(
            pts[0],
            pts[-1]
        ):

            pts = np.vstack([
                pts,
                pts[0]
            ])

        outfile = Path("polygon.txt")

        np.savetxt(
            outfile,
            pts,
            fmt="%.8f"
        )

        print(f"\n保存成功: {outfile}")

        plt.close(self.fig)


# =====================================================
# main
# =====================================================
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python pick_polygon.py input.grd"
        )

        sys.exit(1)

    Picker(sys.argv[1]).show()
