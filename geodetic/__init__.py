"""geodetic — 核心计算库。

顶层 API 采用按需导出，确保仅导入包不会加载绘图、数据或可选加速依赖。
重型操作仍可通过 ``from geodetic.<module> import ...`` 显式导入。
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "WorkflowConfig": ("geodetic.config", "WorkflowConfig"),
    "count_fault_segments": ("geodetic.config", "count_fault_segments"),
    "load_workflow_config": ("geodetic.config", "load_workflow_config"),
    "read_data_list": ("geodetic.config", "read_data_list"),
    "resolve_geo_origin": ("geodetic.config", "resolve_geo_origin"),
    "resolve_under_root": ("geodetic.config", "resolve_under_root"),
    "bounds_new": ("geodetic.constraints", "bounds_new"),
    "zero_slip_boundary": ("geodetic.constraints", "zero_slip_boundary"),
    "ll2xy": ("geodetic.coords", "ll2xy"),
    "xy2ll": ("geodetic.coords", "xy2ll"),
    "xy2xy": ("geodetic.coords", "xy2xy"),
    "load_fault_one_plane": ("geodetic.fault_geometry", "load_fault_one_plane"),
    "calc_green_AZO_okada": ("geodetic.green", "calc_green_AZO_okada"),
    "calc_green_insar_okada": ("geodetic.green", "calc_green_insar_okada"),
    "build_green_function": ("geodetic.green_function", "build_green_function"),
    "make_insar_data": ("geodetic.insar_data", "make_insar_data"),
    "build_greens_mat_dict": ("geodetic.invert", "build_greens_mat_dict"),
    "make_fault_from_insar1": ("geodetic.invert", "make_fault_from_insar1"),
    "NUMBA_AVAILABLE": ("geodetic.okada", "NUMBA_AVAILABLE"),
    "calc_okada": ("geodetic.okada", "calc_okada"),
    "plot_insar_model_resampled": (
        "geodetic.plot_insar_model_resampled",
        "plot_insar_model_resampled",
    ),
    "resamp_insar_data": ("geodetic.resample", "resamp_insar_data"),
    "load_slip_model_from_file": (
        "geodetic.show_slip_model",
        "load_slip_model_from_file",
    ),
    "show_slip_model": ("geodetic.show_slip_model", "show_slip_model"),
    "show_slip_model_2d": ("geodetic.show_slip_model_2d", "show_slip_model_2d"),
    "unfold_patches": ("geodetic.show_slip_model_2d", "unfold_patches"),
    "build_smooth_function": (
        "geodetic.smoothing",
        "build_smooth_function",
    ),
    "SolveResult": ("geodetic.solver", "SolveResult"),
    "solve_bounded_lsq": ("geodetic.solver", "solve_bounded_lsq"),
}

__version__ = "0.1.0"
__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
