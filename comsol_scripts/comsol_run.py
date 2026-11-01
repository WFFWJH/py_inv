"""Build the Myanmar COMSOL Green matrix from the inversion workflow state.

从 ``workflows.inversion`` 的配置/状态出发，读出各道降采样点，逐片元驱动 COMSOL
求解单位走滑 / 倾滑位移场，组装并保存：

  * ``Green_comsol.mat``    — 原始 uvw 三分量（``G_raw_uvw`` + ``sample_points``）
  * ``Green_comsolGD.mat``  — 投影到 InSAR LOS / AZO 的 Green 矩阵与 ``d_raw``

默认模型路径见 ``run_comsol`` 参数；产物写入当前工作目录（建议在项目根运行）。

用法：``python comsol_scripts/comsol_run.py``
"""
from __future__ import annotations

import os
import sys
import time
from typing import Sequence, Tuple

import numpy as np
from scipy.io import loadmat, savemat

# 项目根目录加入 sys.path，保证 geodetic / workflows 可导入
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from geodetic.fault_geometry import FaultLayout
from workflows.inversion import (
    InversionWorkflowState,
    step_load_config,
    step_load_fault,
)


def _load_sample_points(
    state: InversionWorkflowState,
) -> Tuple[np.ndarray, int]:
    if state.config is None:
        raise RuntimeError("请先运行 step_load_config 填充 state.config")
    if len(state.tracks) != len(state.dtypes):
        raise ValueError("state.tracks 与 state.dtypes 长度不一致")

    insar_data = []
    azi_data = []
    iint = int(state.config.iter_step)
    for track, data_type in zip(state.tracks, state.dtypes):
        path = os.path.join(track, "los_samp%d.mat" % iint)
        if not os.path.isfile(path):
            raise FileNotFoundError("找不到采样数据: %s" % path)
        mat = loadmat(path)
        if "sampled_insar_data" not in mat:
            raise KeyError("%s 中无变量 sampled_insar_data" % path)
        sampled = np.asarray(mat["sampled_insar_data"], dtype=np.float64)
        if sampled.ndim != 2 or sampled.shape[1] < 6:
            raise ValueError(
                "%s 中 sampled_insar_data 需要为 N×6 或更宽的二维数组，实际为 %r"
                % (path, sampled.shape)
            )

        kind = data_type.lower()
        if kind == "insar":
            insar_data.append(sampled[:, :6])
        elif kind in ("azi", "azo"):
            azi_data.append(sampled[:, :6])
        else:
            raise ValueError(
                "COMSOL Myanmar Green 矩阵不支持数据类型 %r (轨道 %s)"
                % (data_type, track)
            )

    insar_count = sum(data.shape[0] for data in insar_data)
    groups = insar_data + azi_data
    if not groups:
        raise ValueError("state 中没有可用采样点")
    return np.concatenate(groups, axis=0), insar_count


def _fault_geometry(
    state: InversionWorkflowState,
) -> FaultLayout:
    if state.config is None or not state.fault_abs:
        raise RuntimeError("请先运行 step_load_config 填充 state.config")
    if not os.path.isfile(state.fault_abs):
        raise FileNotFoundError("断层文件不存在: %s" % state.fault_abs)
    if state.fault_layout is None:
        step_load_fault(state, quiet=True)
    if state.fault_layout is None:
        raise RuntimeError("fault layout 未能从主反演几何构建")
    return state.fault_layout


def _interp_uvw(model, coordinates: np.ndarray) -> np.ndarray:
    interp = model.result().numerical("uvw")
    interp.setInterpolationCoordinates(coordinates)
    interp.set("expr", ["u", "v", "w"])
    values = np.asarray(interp.getData())
    return np.column_stack((values[0, 0], values[1, 0], values[2, 0]))


def _set_fault_parameters(model, x0: float, y0: float, x1: float, y1: float) -> None:
    params = model.param()
    params.set("x0", "%s[m]" % x0)
    params.set("y0", "%s[m]" % y0)
    params.set("x1", "%s[m]" % x1)
    params.set("y1", "%s[m]" % y1)


def _solve_layers(
    model,
    layer_range: Sequence[int],
    sample_points: np.ndarray,
    geometry: FaultLayout,
    green: np.ndarray,
    total_patches: int,
) -> None:
    xpt, ypt, _, _ = geometry.fault_segments_local.T
    patch_counts = geometry.patch_counts
    patch_lengths = geometry.patch_lengths
    n_faults = patch_counts.shape[0]
    coordinates = np.vstack(
        (sample_points[:, 0], sample_points[:, 1], np.zeros(sample_points.shape[0]))
    )
    solid = model.component("comp1").physics("solid")
    feature1 = solid.feature("disp1")
    feature2 = solid.feature("disp2")
    feature1.set("U0", [[-0.5], [0.5], [0]])
    feature2.set("U0", [[0.5], [0.5], [0]])
    rectangle = model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1")

    for fault_index in range(n_faults):
        _set_fault_parameters(
            model,
            xpt[fault_index],
            ypt[fault_index],
            xpt[fault_index + n_faults],
            ypt[fault_index + n_faults],
        )
        for layer in layer_range:
            patch_length = patch_lengths[fault_index, layer]
            patch_width = geometry.layer_widths[layer]
            layer_down = geometry.layer_offsets[layer]
            rectangle.set("size", [str(patch_length), str(patch_width)])
            for patch_index in range(patch_counts[fault_index, layer]):
                rectangle.set(
                    "pos",
                    [str(patch_index * patch_length), str(layer_down)],
                )
                column = (
                    int(geometry.green_column_starts[fault_index, layer])
                    + patch_index
                )

                for feature in (feature1, feature2):
                    feature.set(
                        "Direction", [["free"], ["prescribed"], ["free"]]
                    )
                model.sol("sol1").runAll()
                green[:, column, :] = _interp_uvw(model, coordinates)

                for feature in (feature1, feature2):
                    feature.set(
                        "Direction", [["prescribed"], ["free"], ["free"]]
                    )
                model.sol("sol1").runAll()
                green[:, total_patches + column, :] = _interp_uvw(
                    model, coordinates
                )
                print(
                    "fault=%d layer=%d patch=%d columns=(%d,%d)"
                    % (
                        fault_index + 1,
                        layer + 1,
                        patch_index + 1,
                        column,
                        total_patches + column,
                    ),
                    flush=True,
                )


def _run_model(
    client,
    model_path: str,
    layer_range: Sequence[int],
    sample_points: np.ndarray,
    geometry,
    green: np.ndarray,
    total_patches: int,
) -> None:
    model_java = client.load(model_path)
    try:
        model = model_java.java
        model.result().numerical().create("uvw", "Interp")
        _solve_layers(
            model,
            layer_range,
            sample_points,
            geometry,
            green,
            total_patches,
        )
    finally:
        client.remove(model_java)
        client.clear()


def run_comsol(
    state: InversionWorkflowState,
    *,
    # top_model_path: str = "mphs/v3_belt_1/touch_low_belt.mph",
    # deep_model_path: str = "mphs/v3_belt_1/touch_no_top_low_belt.mph",
    top_model_path: str = "mphs/v1_3035/touch.mph",
    deep_model_path: str = "mphs/v1_3035/touch_no_top.mph",
    uvw_output_path: str = "Green_comsol.mat",
    projected_output_path: str = "Green_comsolGD.mat",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run shallow/deep COMSOL models and save the MATLAB-compatible outputs."""
    if state.config is None:
        raise RuntimeError("请先运行 step_load_config 填充 state")
    sample_points, insar_count = _load_sample_points(state)
    geometry = _fault_geometry(state)
    patch_counts = geometry[2]
    total_patches = int(np.sum(patch_counts))
    if total_patches == 0:
        raise ValueError("断层几何没有生成任何片元")

    green = np.zeros((sample_points.shape[0], 2 * total_patches, 3))
    n_layers = state.config.layers
    start = time.perf_counter()
    import mph

    client = mph.start()
    try:
        _run_model(
            client,
            top_model_path,
            (0,),
            sample_points,
            geometry,
            green,
            total_patches,
        )
        top_finished = time.perf_counter()
        _run_model(
            client,
            deep_model_path,
            range(1, n_layers),
            sample_points,
            geometry,
            green,
            total_patches,
        )
    finally:
        client.clear()
    finished = time.perf_counter()

    savemat(
        uvw_output_path,
        {"G_raw_uvw": green, "sample_points": sample_points},
    )

    insar_uvw = green[:insar_count]
    azi_uvw = green[insar_count:]
    los_vectors = sample_points[:insar_count, 3:6]
    green_insar = np.sum(insar_uvw * los_vectors[:, None, :], axis=2)

    azi_vectors = sample_points[insar_count:, 3:5]
    azi_theta = -np.arctan2(azi_vectors[:, 1], azi_vectors[:, 0]) - np.pi
    green_azi = (
        azi_uvw[:, :, 0] * np.sin(azi_theta[:, None])
        + azi_uvw[:, :, 1] * np.cos(azi_theta[:, None])
    )
    d_raw = sample_points[:, 2] / 100.0
    savemat(
        projected_output_path,
        {
            "G_raw_azi": green_azi,
            "G_raw_insar": green_insar,
            "d_raw": d_raw,
        },
    )

    print("top model time: %.1f s" % (top_finished - start), flush=True)
    print("deep model time: %.1f s" % (finished - top_finished), flush=True)
    print("total time: %.1f s" % (finished - start), flush=True)
    print("saved %s and %s" % (uvw_output_path, projected_output_path), flush=True)
    return green, green_insar, green_azi


def main() -> None:
    workflow_state = InversionWorkflowState()
    step_load_config(workflow_state)
    run_comsol(workflow_state)


if __name__ == "__main__":
    main()
