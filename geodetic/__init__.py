"""geodetic — InSAR 同震滑动反演核心库 (Okada 模型)。

模块地图
--------
基础
    :mod:`geodetic.config`        配置/数据列表解析 (configfile.txt, configpara.txt)
    :mod:`geodetic.coords`        横轴墨卡托投影与 2-D 旋转
    :mod:`geodetic.solver`        有界线性最小二乘 (MATLAB ``lsqlin`` 等价)
    :mod:`geodetic.constraints`   边界 (lb/ub) 与零滑动约束矩阵

正演
    :mod:`geodetic.okada`           Okada 1985 弹性半空间位移核 (numpy/numba)
    :mod:`geodetic.green`           单道 Green 矩阵 (LOS / AZO)
    :mod:`geodetic.green_function`  读采样数据 + ramp/模型分发, 组装 Green
    :mod:`geodetic.slip2insar`      滑移模型 -> 合成 LOS
    :mod:`geodetic.slip2azo`        滑移模型 -> 合成方位向偏移
    :mod:`geodetic.fault_geometry`  断层几何 -> 片元 (patch) 表

数据
    :mod:`geodetic.insar_data`    四叉树降采样与 .grd 读写
    :mod:`geodetic.resample`      按模型重采样
    :mod:`geodetic.multilook`     多视平均

反演与绘图
    :mod:`geodetic.invert`                     反演主核 (make_fault_from_insar1)
    :mod:`geodetic.smoothing`                  Tikhonov 一阶差分平滑
    :mod:`geodetic.show_slip_model`            3-D 滑移模型可视化
    :mod:`geodetic.show_slip_model_2d`         2-D 展开剖面 (沿走向-沿倾向)
    :mod:`geodetic.plot_insar_model_resampled` 逐道 观测/模型/残差 三联图

编排层见 :mod:`workflows`; 命令行入口见项目根目录 ``run_inversion.py``。
"""
from __future__ import annotations

from geodetic.config import (  # noqa: F401
    WorkflowConfig,
    count_fault_segments,
    load_workflow_config,
    read_data_list,
    resolve_geo_origin,
    resolve_under_root,
)
from geodetic.constraints import bounds_new, zero_slip_boundary  # noqa: F401
from geodetic.coords import ll2xy, xy2ll, xy2xy  # noqa: F401
from geodetic.fault_geometry import load_fault_one_plane  # noqa: F401
from geodetic.green import calc_green_AZO_okada, calc_green_insar_okada  # noqa: F401
from geodetic.green_function import build_green_function  # noqa: F401
from geodetic.insar_data import make_insar_data  # noqa: F401
from geodetic.invert import build_greens_mat_dict, make_fault_from_insar1  # noqa: F401
from geodetic.okada import NUMBA_AVAILABLE, calc_okada  # noqa: F401
from geodetic.plot_insar_model_resampled import (  # noqa: F401
    plot_insar_model_resampled,
)
from geodetic.resample import resamp_insar_data  # noqa: F401
from geodetic.show_slip_model import (  # noqa: F401
    load_slip_model_from_file,
    show_slip_model,
)
from geodetic.show_slip_model_2d import (  # noqa: F401
    show_slip_model_2d,
    unfold_patches,
)
from geodetic.smoothing import build_smooth_function  # noqa: F401
from geodetic.solver import SolveResult, solve_bounded_lsq  # noqa: F401

__version__ = "0.1.0"

__all__ = (
    # 配置 / 坐标 / 求解器
    "WorkflowConfig",
    "load_workflow_config",
    "read_data_list",
    "resolve_geo_origin",
    "count_fault_segments",
    "resolve_under_root",
    "ll2xy",
    "xy2ll",
    "xy2xy",
    "solve_bounded_lsq",
    "SolveResult",
    "bounds_new",
    "zero_slip_boundary",
    # 正演
    "NUMBA_AVAILABLE",
    "calc_okada",
    "calc_green_insar_okada",
    "calc_green_AZO_okada",
    "build_green_function",
    "build_smooth_function",
    "load_fault_one_plane",
    # 数据
    "make_insar_data",
    "resamp_insar_data",
    # 反演 / 绘图
    "make_fault_from_insar1",
    "build_greens_mat_dict",
    "show_slip_model",
    "show_slip_model_2d",
    "unfold_patches",
    "load_slip_model_from_file",
    "plot_insar_model_resampled",
)
