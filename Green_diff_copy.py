"""
Green 函数对比可视化工具

对比 Okada 解析解与 COMSOL 数值解在每个 fault patch 上的前向位移场，
并以交互方式浏览不同 patch 编号 (i) 的结果。

布局说明 (3×3 子图):
    列 0: Okada 前向解
    列 1: COMSOL 前向解
    列 2: 误差 (COMSOL - Okada)
    行 0: 东向分量 ue
    行 1: 北向分量 un
    行 2: 垂向分量 uz

性能优化:
    - 使用 pcolormesh 代替 scatter（规则网格上快一个数量级以上）
    - 切换 patch 时仅更新数据数组，不重建图形对象
    - colorbar 用 FixedLocator/FixedFormatter 固定极值刻度
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib.widgets import Button, TextBox

# =============================================================================
# 断层几何参数（与 Green 函数计算时保持一致，此处仅作记录）
# =============================================================================
width = 20e3       # 断层宽度 (m)
length = 100e3     # 断层长度 (m)
len_top = 4e3      # 顶层 patch 长度 (m)
n_layer = int(length / len_top)
layers = 5

# =============================================================================
# 观测网格：在规则经纬网格上采样，用于将 Green 矩阵还原为 2D 场
# =============================================================================
nx, ny = 100, 140  # x、y 方向格点数，可酌情修改
x = np.linspace(-50, 50, nx)      # x 坐标 (km)
y = np.linspace(-20, 120, ny)     # y 坐标 (km)
X, Y = np.meshgrid(x, y)          # 2D 网格，shape = (ny, nx)

nobs = nx * ny  # 总观测点数，Green 矩阵每个分量占 nobs 行

# =============================================================================
# 加载 Green 函数矩阵
# Green 矩阵 shape = (3 * nobs, n_patch)
#   行 0         ~ nobs-1       : ue 分量
#   行 nobs      ~ 2*nobs-1     : un 分量
#   行 2*nobs    ~ 3*nobs-1     : uz 分量
#   列 j                          : 第 j 个 fault patch 的响应
# =============================================================================
Green_okada = np.load("Green_okada.npy")
Green_comsol = np.load("Green_comsol.npy")

# =============================================================================
# 公共绘图参数
# =============================================================================
PLOT = {
    "cmap": "jet",           # 色图
    "figsize": (13, 12),     # 图窗尺寸 (inch)
    "digits": 4,             # colorbar 极值显示的小数位数
    "cbar_shrink": 0.75,     # colorbar 高度缩放
    "cbar_fraction": 0.035,  # colorbar 宽度占子图比例
    "cbar_pad": 0.02,        # colorbar 与子图间距
    "cbar_labelpad": 1,      # colorbar 标签与刻度间距
}

# 行标签：三个位移分量
DIRECTIONS = ("ue", "un", "uz")

# 列定义：标题、colorbar 标签、是否使用对称色标（误差列以 0 为中心）
COLUMNS = (
    {"title": "Okada forward", "cbar_label": "displacement (m)", "symmetric": False},
    {"title": "COMSOL forward", "cbar_label": "displacement (m)", "symmetric": False},
    {"title": "Error", "cbar_label": "error (m)", "symmetric": True},
)

# =============================================================================
# 全局状态变量
# =============================================================================
i = 0                              # 当前显示的 patch 编号
cbar_mode = "individual"             # colorbar 模式: "individual" | "row"
pcms = []                          # pcolormesh 对象，二维列表 [row][col]
colorbars = []                     # colorbar 对象列表，布局随 cbar_mode 变化
suptitle_artist = None             # 顶部总标题（仅显示一次 patch 编号）

# =============================================================================
# 创建图窗与 3×3 子图
# bottom 留出空间给底部交互控件；top 留出空间给 suptitle
# =============================================================================
fig = plt.figure(figsize=PLOT["figsize"])
axes = fig.subplots(3, 3)
fig.subplots_adjust(bottom=0.12, top=0.93, wspace=0.25, hspace=0.30)

# =============================================================================
# 底部交互控件（坐标为 figure 归一化坐标 [left, bottom, width, height]）
# =============================================================================
ax_text = fig.add_axes([0.22, 0.025, 0.10, 0.04])
text_box = TextBox(ax_text, "i: ", initial=str(i))

ax_confirm = fig.add_axes([0.34, 0.025, 0.08, 0.04])
btn_confirm = Button(ax_confirm, "Confirm")

ax_prev = fig.add_axes([0.44, 0.025, 0.08, 0.04])
btn_prev = Button(ax_prev, "Previous")

ax_next = fig.add_axes([0.54, 0.025, 0.08, 0.04])
btn_next = Button(ax_next, "Next")

# 切换 colorbar 显示模式
ax_mode = fig.add_axes([0.66, 0.025, 0.14, 0.04])
btn_mode = Button(ax_mode, "CB: individual")


def _clim(value, symmetric=False):
    """
    计算单幅场的色标范围 (vmin, vmax)。

    参数
    ----
    value : ndarray, shape (ny, nx)
        待绘制的 2D 场
    symmetric : bool
        True 时以 0 为中心对称，适用于误差场；
        False 时取实际 min/max，适用于位移场。

    返回
    ----
    (vmin, vmax) : tuple of float
    """
    if symmetric:
        vmax = float(np.max(np.abs(value)))
        return -vmax, vmax
    return float(np.min(value)), float(np.max(value))


def _combined_clim(values, symmetric=False):
    """
    计算多幅场联合色标范围，用于共享 colorbar 的场景。

    参数
    ----
    values : sequence of ndarray
        多个待合并的 2D 场
    symmetric : bool
        同 _clim

    返回
    ----
    (vmin, vmax) : tuple of float
    """
    if symmetric:
        vmax = max(float(np.max(np.abs(v))) for v in values)
        return -vmax, vmax
    vmin = min(float(np.min(v)) for v in values)
    vmax = max(float(np.max(v)) for v in values)
    return vmin, vmax


def _update_cbar_ticks(cbar, pcm, vmin, vmax, symmetric=False):
    """
    在 colorbar 上显示极值刻度标签。

    说明
    ----
    pcolormesh 原地更新数据后，matplotlib 默认刻度定位器会覆盖自定义刻度。
    因此使用 FixedLocator + FixedFormatter 强制固定刻度位置和标签文本。

    对称模式显示三档: min / 0 / max
    非对称模式显示两档: min / max
    """
    # 先同步 colorbar 与 mappable 的色标范围
    cbar.update_normal(pcm)

    digits = PLOT["digits"]
    if symmetric:
        if vmax == 0.0:
            # 全场为零时的退化情况
            ticks = [0.0]
            labels = ["0"]
        else:
            ticks = [vmin, 0.0, vmax]
            labels = [
                f"min: {vmin:.{digits}f}",
                "0",
                f"max: {vmax:.{digits}f}",
            ]
    elif vmin == vmax:
        # 常数场
        ticks = [vmin]
        labels = [f"{vmin:.{digits}f}"]
    else:
        ticks = [vmin, vmax]
        labels = [
            f"min: {vmin:.{digits}f}",
            f"max: {vmax:.{digits}f}",
        ]

    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(labels))


def _clear_colorbars():
    """移除所有 colorbar 对象（切换 cbar_mode 时需要重建）。"""
    for cbar in colorbars:
        cbar.remove()
    colorbars.clear()


def _create_colorbars():
    """
    根据当前 cbar_mode 创建 colorbar。

    individual 模式（9 个 colorbar）:
        每个子图各自独立 colorbar，色标范围互不影响。

    row 模式（6 个 colorbar，每行 2 个）:
        - 前两列 (Okada + COMSOL) 共用 displacement colorbar
        - 第三列 (Error) 单独使用 error colorbar（对称色标）
        这样可避免误差量级远小于位移时被大色标"压扁"。

    colorbars 索引约定（row 模式）:
        colorbars[row * 2]     -> 第 row 行 displacement colorbar
        colorbars[row * 2 + 1] -> 第 row 行 error colorbar
    """
    _clear_colorbars()

    if cbar_mode == "individual":
        for row in range(3):
            for col in range(3):
                pcm = pcms[row][col]
                cbar = plt.colorbar(
                    pcm,
                    ax=axes[row, col],
                    shrink=PLOT["cbar_shrink"],
                    fraction=PLOT["cbar_fraction"],
                    pad=PLOT["cbar_pad"],
                )
                cbar.set_label(
                    COLUMNS[col]["cbar_label"],
                    labelpad=PLOT["cbar_labelpad"],
                )
                colorbars.append(cbar)
    else:
        for row in range(3):
            # Okada + COMSOL 共用：colorbar 挂在该行前两列右侧
            cbar_disp = fig.colorbar(
                pcms[row][0],
                ax=axes[row, :2],
                shrink=PLOT["cbar_shrink"],
                fraction=PLOT["cbar_fraction"],
                pad=PLOT["cbar_pad"],
            )
            cbar_disp.set_label(
                f"{DIRECTIONS[row]} displacement (m)",
                labelpad=PLOT["cbar_labelpad"],
            )
            colorbars.append(cbar_disp)

            # Error 独立 colorbar
            cbar_err = plt.colorbar(
                pcms[row][2],
                ax=axes[row, 2],
                shrink=PLOT["cbar_shrink"],
                fraction=PLOT["cbar_fraction"],
                pad=PLOT["cbar_pad"],
            )
            cbar_err.set_label(
                "error (m)",
                labelpad=PLOT["cbar_labelpad"],
            )
            colorbars.append(cbar_err)


def _init_axes():
    """
    初始化 9 个子图的 pcolormesh 对象（仅执行一次）。

    标签简化策略（避免 9 幅图重复标注）:
        - patch 编号: 顶部 suptitle 统一显示
        - 列名 (Okada/COMSOL/Error): 仅第一行子图显示 title
        - 分量名 (ue/un/uz): 仅每行最左子图的 ylabel 显示
        - x 轴标签: 仅最底行显示
    """
    global pcms

    pcms.clear()

    for row, direction in enumerate(DIRECTIONS):
        row_pcms = []
        for col, col_def in enumerate(COLUMNS):
            ax = axes[row, col]
            ax.clear()
            z0 = np.zeros((ny, nx))
            # pcolormesh 适合规则网格；shading="auto" 自动处理格点边界
            pcm = ax.pcolormesh(
                X,
                Y,
                z0,
                cmap=PLOT["cmap"],
                shading="auto",
            )
            ax.set_aspect("equal")

            # 仅最底行显示 x 轴标签
            if row == 2:
                ax.set_xlabel("x (km)")
            else:
                ax.set_xlabel("")

            # 仅最左列显示分量名 + y 轴标签
            if col == 0:
                ax.set_ylabel(f"{direction}\ny (km)")
            else:
                ax.set_ylabel("")

            # 仅第一行显示列标题
            if row == 0:
                ax.set_title(col_def["title"])
            else:
                ax.set_title("")

            row_pcms.append(pcm)
        pcms.append(row_pcms)

    _create_colorbars()


def _update_suptitle():
    """
    更新顶部总标题，显示当前 patch 编号。

    注意: 不可每次 remove() 后重建，否则多次 redraw 会触发
    matplotlib 内部列表不一致的 ValueError。应首次创建后仅 set_text()。
    """
    global suptitle_artist
    if suptitle_artist is None:
        suptitle_artist = fig.suptitle(f"patch = {i}", y=0.97)
    else:
        suptitle_artist.set_text(f"patch = {i}")


def redraw():
    """
    根据当前 patch 编号 i 和 cbar_mode 刷新全部子图。

    流程
    ----
    1. 从 Green 矩阵取出第 i 列，reshape 为 (ny, nx) 的 2D 场
    2. 更新 suptitle
    3. 按模式设置各子图数据与色标，并刷新 colorbar 极值刻度
    4. draw_idle() 异步重绘（比 draw() 更轻量）
    """
    # 从 Green 矩阵切片：每个分量占 nobs 行，第 i 列为当前 patch
    ue1 = Green_okada[0:nobs, i].reshape(ny, nx)
    un1 = Green_okada[nobs:2 * nobs, i].reshape(ny, nx)
    uz1 = Green_okada[2 * nobs:3 * nobs, i].reshape(ny, nx)

    result_u = Green_comsol[0:nobs, i].reshape(ny, nx)
    result_v = Green_comsol[nobs:2 * nobs, i].reshape(ny, nx)
    result_w = Green_comsol[2 * nobs:3 * nobs, i].reshape(ny, nx)

    # 每行: (Okada 场, COMSOL 场)，对应 ue / un / uz
    fields = [
        (ue1, result_u),
        (un1, result_v),
        (uz1, result_w),
    ]

    _update_suptitle()

    if cbar_mode == "individual":
        # 每幅子图独立色标
        for row, (okada, comsol) in enumerate(fields):
            values = (okada, comsol, comsol - okada)
            for col, value in enumerate(values):
                pcm = pcms[row][col]
                symmetric = COLUMNS[col]["symmetric"]
                vmin, vmax = _clim(value, symmetric=symmetric)
                pcm.set_array(value.ravel())
                pcm.set_clim(vmin, vmax)
                cbar_idx = row * 3 + col
                _update_cbar_ticks(
                    colorbars[cbar_idx], pcm, vmin, vmax, symmetric=symmetric
                )
    else:
        # row 模式：每行 displacement 共用色标，error 独立对称色标
        for row, (okada, comsol) in enumerate(fields):
            error = comsol - okada
            vmin_d, vmax_d = _combined_clim((okada, comsol), symmetric=False)
            vmin_e, vmax_e = _clim(error, symmetric=True)

            pcms[row][0].set_array(okada.ravel())
            pcms[row][0].set_clim(vmin_d, vmax_d)
            pcms[row][1].set_array(comsol.ravel())
            pcms[row][1].set_clim(vmin_d, vmax_d)
            pcms[row][2].set_array(error.ravel())
            pcms[row][2].set_clim(vmin_e, vmax_e)

            _update_cbar_ticks(
                colorbars[row * 2], pcms[row][0], vmin_d, vmax_d, symmetric=False
            )
            _update_cbar_ticks(
                colorbars[row * 2 + 1], pcms[row][2], vmin_e, vmax_e, symmetric=True
            )

    fig.canvas.draw_idle()


def toggle_cbar_mode(event):
    """在 individual / row 两种 colorbar 模式间切换，并重建 colorbar。"""
    global cbar_mode

    if cbar_mode == "individual":
        cbar_mode = "row"
        btn_mode.label.set_text("CB: row")
    else:
        cbar_mode = "individual"
        btn_mode.label.set_text("CB: individual")

    _create_colorbars()
    redraw()


def confirm(event):
    """读取文本框中的 patch 编号，校验后跳转并刷新。"""
    global i

    try:
        new_i = int(text_box.text)
    except ValueError:
        print("请输入整数，例如：37")
        return

    max_i = Green_okada.shape[1] - 1
    if new_i < 0 or new_i > max_i:
        print(f"i 必须在 0 ~ {max_i} 之间")
        return

    i = new_i
    redraw()
    text_box.set_val(str(i))


def next_patch(event):
    """切换到下一个 patch（不超过最大值）。"""
    global i

    if i < Green_okada.shape[1] - 1:
        i += 1

    redraw()
    text_box.set_val(str(i))


def previous_patch(event):
    """切换到上一个 patch（不低于 0）。"""
    global i

    if i > 0:
        i -= 1

    redraw()
    text_box.set_val(str(i))


# =============================================================================
# 绑定按钮回调并启动
# =============================================================================
btn_confirm.on_clicked(confirm)
btn_prev.on_clicked(previous_patch)
btn_next.on_clicked(next_patch)
btn_mode.on_clicked(toggle_cbar_mode)

_init_axes()   # 创建 pcolormesh 与 colorbar（仅一次）
redraw()       # 绘制初始 patch
plt.show()
