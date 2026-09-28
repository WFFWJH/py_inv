"""统一对外接口（将 `allfunction` 目录加入 PYTHONPATH 或在该目录下运行脚本）.

典型用法::

    import geodetic_api as g
    # 几何
    sm = g.load_fault_one_plane(path, dip=[...], lonc=..., latc=..., ref_lon=...)
    # 反演（单步）或 InversionExample 全流程
    # g.load_workflow_config(); g.run_workflow_steps(...); 或 InversionParams + run_okada_inversion
    ...

依赖仅 ``numpy`` + ``scipy``；``.grd`` 与 ``xarray``、对比脚本与 ``h5py`` 见 requirements.txt 注释行。
"""
from __future__ import annotations

from calc_okada import NUMBA_AVAILABLE, calc_okada
from load_fault_one_plane import load_fault_one_plane
from make_fault_from_insar1 import make_fault_from_insar1
from inversion_example_workflow import (
    InversionParams,
    InversionWorkflowState,
    WorkflowConfig,
    load_workflow_config,
    run_okada_inversion,
    run_workflow_steps,
)
from show_slip_model import load_slip_model_from_file, show_slip_model
from resamp_insar_data import resamp_insar_data

# 子模块按需引用名（避免 import *）
__all__ = (
    "NUMBA_AVAILABLE",
    "calc_okada",
    "load_fault_one_plane",
    "make_fault_from_insar1",
    "WorkflowConfig",
    "load_workflow_config",
    "InversionParams",
    "InversionWorkflowState",
    "run_okada_inversion",
    "run_workflow_steps",
    "load_slip_model_from_file",
    "show_slip_model",
    "resamp_insar_data",
)
