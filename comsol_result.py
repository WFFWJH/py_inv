"""用 COMSOL GrF 替换 Step 5 快照中的 Green 矩阵并完成反演、出图。

先运行 comsol_run.py 生成 Green_comsol.mat，再运行本脚本。默认读取
py_inversion_iint0.mat 中的几何、平滑矩阵和边界约束，反演结果保存到
comsol_inversion_out/。
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.io import loadmat, savemat

from build_green_function import _build_ramp
from inversion_example_workflow import step_load_config, InversionWorkflowState
from make_fault_from_insar1 import (
    _add_col_of,
    _expand_last_columns,
    _track_orbit_number,
    build_greens_mat_dict,
)
from make_rms import invert_at_lambda, load_snapshot, plot_track_misfits


_HERE = os.path.dirname(os.path.abspath(__file__))


def _replace_with_comsol_matrices(
    snap: Dict[str, Any],
    state: InversionWorkflowState,
    track_files: Sequence[Dict[str, Any]],
    comsol_mat: str,
) -> None:
    """将 COMSOL 投影矩阵按配置轨道顺序组装成 Step 5 的 GrF/G_last。"""
    cfg = state.config
    if cfg is None:
        raise RuntimeError("请先运行 step_load_config")
    if len(track_files) != len(state.dtypes):
        raise ValueError("轨道文件数量与 state.dtypes 不一致")

    data_by_track: List[np.ndarray] = []
    dem_by_track: List[np.ndarray] = []
    for i, (track, expected_path, expected_type) in enumerate(
        zip(track_files, state.tracks, state.dtypes)
    ):
        expected_name = os.path.basename(os.path.normpath(expected_path))
        actual_name = track.get(
            "name", os.path.basename(os.path.dirname(track["path"]))
        )
        actual_type = track.get("dtype", expected_type)
        if actual_name != expected_name or actual_type != expected_type:
            raise ValueError(
                "轨道索引 %d 与配置顺序不符: 期望 %s (%s)，实际 %s (%s)"
                % (
                    i,
                    expected_name,
                    expected_type,
                    actual_name,
                    actual_type,
                )
            )
        if not os.path.isfile(track["path"]):
            raise FileNotFoundError(
                "配置轨道 %s 的采样文件不存在: %s" % (expected_name, track["path"])
            )
        values = loadmat(track["path"])
        if "sampled_insar_data" not in values:
            raise KeyError("%s 中无 sampled_insar_data" % track["path"])
        sampled = np.asarray(values["sampled_insar_data"], dtype=np.float64)
        if sampled.ndim != 2 or sampled.shape[1] < 6:
            raise ValueError(
                "%s 中 sampled_insar_data 需要至少 6 列，实际为 %r"
                % (track["path"], sampled.shape)
            )
        data_by_track.append(sampled)
        dem_by_track.append(
            np.asarray(
                values.get("dem_out", np.zeros(sampled.shape[0])),
                dtype=np.float64,
            ).reshape(-1)
        )
    original_bdata = np.asarray(snap["Bdata"], dtype=np.float64).reshape(-1)

    comsol = loadmat(comsol_mat)
    required = ("G_raw_insar", "G_raw_azi", "d_raw")
    missing = [name for name in required if name not in comsol]
    if missing:
        raise KeyError(
            "%s 缺少 %s；请使用 comsol_run.py 生成的 Green_comsolGD.mat"
            % (comsol_mat, ", ".join(missing))
        )
    green_insar = np.asarray(comsol["G_raw_insar"], dtype=np.float64)
    green_azi = np.asarray(comsol["G_raw_azi"], dtype=np.float64)
    d_raw = np.asarray(comsol["d_raw"], dtype=np.float64).reshape(-1)

    n_patches = int(snap["slip_model"].shape[0])
    if green_insar.ndim != 2 or green_azi.ndim != 2:
        raise ValueError("G_raw_insar 和 G_raw_azi 必须是二维矩阵")
    if green_insar.shape[1] != 2 * n_patches or green_azi.shape[1] != 2 * n_patches:
        raise ValueError(
            "COMSOL 投影矩阵列数应为 %d (快照片元数 %d)，实际为 insar=%d, azo=%d；"
            "请用相同断层几何重新生成 COMSOL 结果"
            % (2 * n_patches, n_patches, green_insar.shape[1], green_azi.shape[1])
        )

    # make_fault_from_insar1 读取轨道的顺序就是 data_list 顺序；
    # COMSOL 输出按数据类型分块，所以这里只用于切片映射，最终仍按原配置顺序拼接。
    types = [str(kind).lower() for kind in state.dtypes]
    unsupported = [kind for kind in types if kind not in ("insar", "azo", "azi")]
    if unsupported:
        raise ValueError("不支持的轨道类型: %s" % ", ".join(sorted(set(unsupported))))

    insar_indices = [i for i, kind in enumerate(types) if kind == "insar"]
    azi_indices = [i for i, kind in enumerate(types) if kind in ("azo", "azi")]
    insar_count = sum(data_by_track[i].shape[0] for i in insar_indices)
    azi_count = sum(data_by_track[i].shape[0] for i in azi_indices)
    if green_insar.shape[0] != insar_count or green_azi.shape[0] != azi_count:
        raise ValueError(
            "COMSOL 行数与配置轨道采样数不符: insar %d/%d, azo %d/%d"
            % (green_insar.shape[0], insar_count, green_azi.shape[0], azi_count)
        )
    if d_raw.size != insar_count + azi_count:
        raise ValueError(
            "d_raw 有 %d 行，但配置轨道采样点总数为 %d"
            % (d_raw.size, insar_count + azi_count)
        )

    # comsol_run 的 d_raw 单位是采样数据的 1/100，顺序是 InSAR 块后接 AZO 块。
    # 先把它恢复到输入数据单位，再按 make_fault_from_insar1 的权重检查 Bdata。
    grouped_indices = insar_indices + azi_indices
    grouped_data = np.concatenate(
        [data_by_track[i][:, 2] for i in grouped_indices]
    )
    grouped_bdata = np.concatenate(
        [
            data_by_track[i][:, 2] * (0.8 if types[i] == "insar" else 1.0)
            for i in grouped_indices
        ]
    )
    grouped_scale = np.concatenate(
        [
            np.full(
                data_by_track[i].shape[0],
                0.8 if types[i] == "insar" else 1.0,
            )
            for i in grouped_indices
        ]
    )
    if not np.allclose(
        d_raw * 100.0, grouped_data, rtol=0.0, atol=1e-6, equal_nan=True
    ):
        max_error = float(np.nanmax(np.abs(d_raw * 100.0 - grouped_data)))
        raise ValueError(
            "COMSOL d_raw 与配置轨道采样数据不一致 (乘 100 后最大差 %.6g); "
            "采样点或 InSAR/AZO 分组顺序不同，请重新运行 comsol_run.py"
            % max_error
        )
    if not np.allclose(
        d_raw * 100.0 * grouped_scale,
        grouped_bdata,
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError("内部错误：COMSOL d_raw 转换后与分组 Bdata 不一致")
    print(
        "d_raw 校验通过: %d 点；按配置轨道顺序恢复 Bdata 前，"
        "最大观测差 %.3g"
        % (d_raw.size, float(np.max(np.abs(d_raw * 100.0 - grouped_data)))),
        flush=True,
    )

    # 与 make_fault_from_insar1 一致：逐轨道顺序建矩阵，AZO 权重 1、
    # InSAR 权重 0.8；G 和 bd 均除以该轨道点数后再纵向堆叠。
    raw_by_track: List[np.ndarray] = []
    weighted_by_track: List[np.ndarray] = []
    raw_data_by_track: List[np.ndarray] = []
    weighted_data_by_track: List[np.ndarray] = []
    class_labels = np.zeros(len(track_files), dtype=np.int64)
    insar_offset = 0
    azi_offset = 0
    for i, (track, kind, sampled) in enumerate(
        zip(track_files, types, data_by_track)
    ):
        n_rows = sampled.shape[0]
        if n_rows == 0:
            raise ValueError("轨道采样点为空: %s" % track["path"])
        scale = 0.8 if kind == "insar" else 1.0
        if kind == "insar":
            projected = green_insar[insar_offset:insar_offset + n_rows]
            insar_offset += n_rows
        else:
            projected = green_azi[azi_offset:azi_offset + n_rows]
            azi_offset += n_rows

        ramp = _build_ramp(sampled, dem_by_track[i], cfg.ramp_choice)
        raw = np.column_stack((projected, ramp)) * scale
        raw_data = sampled[:, 2] * scale
        raw_by_track.append(raw)
        weighted_by_track.append(raw / n_rows)
        raw_data_by_track.append(raw_data)
        weighted_data_by_track.append(raw_data / n_rows)
        class_labels[i] = _track_orbit_number(track["path"]) or 0

    labels = np.asarray(
        class_labels,
        dtype=np.int64,
    )
    class_map = np.zeros(len(track_files), dtype=np.int64)
    for class_id, orbit in enumerate(np.unique(labels), start=1):
        class_map[labels == orbit] = class_id

    add_col = _add_col_of(cfg.ramp_choice)
    if add_col and class_map.size and class_map.max() > 1:
        raw_by_track = _expand_last_columns(raw_by_track, add_col, class_map)
        weighted_by_track = _expand_last_columns(
            weighted_by_track, add_col, class_map
        )

    snap["GrF"] = np.vstack(raw_by_track)
    snap["G_last"] = np.vstack(weighted_by_track)
    expected_bdata = np.concatenate(raw_data_by_track)
    if original_bdata.shape != expected_bdata.shape or not np.allclose(
        original_bdata, expected_bdata, rtol=0.0, atol=1e-10, equal_nan=True
    ):
        raise ValueError(
            "按配置文件轨道顺序生成的 Bdata 与 Step 5 快照不一致；"
            "快照可能使用了不同采样版本或轨道顺序"
        )
    print(
        "Bdata 校验通过: 输入快照与配置顺序逐轨道堆叠一致 (%d 点)"
        % expected_bdata.size,
        flush=True,
    )
    snap["Bdata"] = expected_bdata
    snap["bd_last"] = np.concatenate(weighted_data_by_track)
    snap["class_map"] = class_map

    # Keep the smoothing and boundary right-hand sides from the Step 5 snapshot.
    n_obs = snap["bd_last"].size
    if snap["bdata_sm"].size < n_obs:
        raise ValueError("快照 bdata_sm 行数小于观测点数")
    snap["bdata_sm"] = np.concatenate(
        (snap["bd_last"], snap["bdata_sm"].reshape(-1)[n_obs:])
    )

    n_cols = snap["G_last"].shape[1]
    for name in ("H", "Wb", "Wl", "Wr"):
        if snap[name].shape[1] != n_cols:
            raise ValueError(
                "%s 有 %d 列，但 COMSOL G_last 有 %d 列；"
                "请使用对应几何和 ramp_choice 生成的 step5 快照"
                % (name, snap[name].shape[1], n_cols)
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="用 COMSOL Green 矩阵替换 Step 5 矩阵，反演并绘制滑移/残差图。"
    )
    parser.add_argument(
        "--snapshot", default="py_inversion_iint0.mat",
        help="Step 5 反演快照路径",
    )
    parser.add_argument(
        "--comsol-mat", default="Green_comsolGD.mat",
        help="comsol_run.py 生成的投影 Green 矩阵和 d_raw",
    )
    parser.add_argument(
        "--config-dir", default=_HERE,
        help="configfile.txt 与 configpara.txt 所在目录",
    )
    parser.add_argument(
        "--out-dir", default="comsol_inversion_out",
        help="反演快照和图件输出目录",
    )
    parser.add_argument(
        "--smoothness", type=float, default=1.0,
        help="step5 平滑权重 lambda",
    )
    parser.add_argument(
        "--slip-max", type=float, default=10.0,
        help="滑移分量上下界绝对值 (m)",
    )
    parser.add_argument(
        "--no-track-plots", action="store_true",
        help="只绘制 3D 滑移图，不生成逐轨道残差图",
    )
    args = parser.parse_args(argv)

    for name in ("snapshot", "comsol_mat"):
        value = getattr(args, name)
        if not os.path.isabs(value):
            setattr(args, name, os.path.join(_HERE, value))
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(_HERE, args.out_dir)

    state = InversionWorkflowState()
    step_load_config(state, args.config_dir)
    cfg = state.config
    if cfg is None:
        raise RuntimeError("配置加载失败")

    snap = load_snapshot(args.snapshot)
    print("加载 step5 快照: %s" % args.snapshot, flush=True)
    track_files = [
        {
            "name": os.path.basename(os.path.normpath(track)),
            "path": os.path.join(track, "los_samp%d.mat" % int(cfg.iter_step)),
            "dtype": dtype,
        }
        for track, dtype in zip(state.tracks, state.dtypes)
    ]
    _replace_with_comsol_matrices(snap, state, track_files, args.comsol_mat)

    result = invert_at_lambda(
        snap,
        args.smoothness,
        Con=cfg.con,
        slip_max=args.slip_max,
        max_nfev=cfg.max_nfev,
        verbose=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_mat = os.path.join(args.out_dir, "comsol_inversion.mat")
    savemat(
        out_mat,
        build_greens_mat_dict(
            G_last=snap["G_last"],
            Bdata=snap["Bdata"],
            bd_last=snap["bd_last"],
            slip_model=result["slip_model"],
            bdata_sm=snap["bdata_sm"],
            GrF=snap["GrF"],
            H=snap["H"],
            h1=snap["h1"],
            Wb=snap["Wb"],
            Wl=snap["Wl"],
            Wr=snap["Wr"],
            ramp_choice=snap["ramp_choice"],
            u=result["u"],
            RMS_misfit=result["RMS_misfit"],
            model_roughness=result["roughness"],
            class_map=snap["class_map"],
        ),
        do_compression=True,
        oned_as="column",
    )
    print("反演快照 -> %s" % out_mat, flush=True)

    import matplotlib
    matplotlib.use("Agg", force=True)
    from show_slip_model import show_slip_model

    slip_png = os.path.join(args.out_dir, "comsol_slip_model.png")
    show_slip_model(
        result["slip_model"],
        ref_lon=cfg.ref_lon,
        lonc=cfg.lonc,
        latc=cfg.latc,
        fault=state.fault_abs,
        out_path=slip_png,
        show=False,
        title="COMSOL inversion (lambda=%g, RMS=%.4f)"
        % (result["lam"], result["rms"]),
    )
    print("滑移模型图 -> %s" % slip_png, flush=True)

    if not args.no_track_plots:
        from make_rms import _check_track_rows

        row_error = _check_track_rows(
            track_files, snap["GrF"].shape[0], int(cfg.iter_step)
        )
        if row_error:
            raise ValueError("无法绘制逐轨道残差图: %s" % row_error)
        plot_track_misfits(
            result,
            snap,
            track_files,
            out_dir=args.out_dir,
            iint=int(cfg.iter_step),
            misfit_range=1.0,
            defo_max=3.0,
            ref_lon=cfg.ref_lon,
            lonc=cfg.lonc,
            latc=cfg.latc,
            fault_abs=state.fault_abs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
