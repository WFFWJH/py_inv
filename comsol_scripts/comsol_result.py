"""用 COMSOL GrF 替换 Step 5 快照中的 Green 矩阵并完成反演、出图。

先运行 ``comsol_scripts/comsol_run.py`` 生成 ``Green_comsolGD.mat``，再运行本脚本。
默认读取项目根目录的 ``py_inversion_iint0.mat``（几何、平滑矩阵与边界约束），
反演结果写入 ``comsol_inversion_out/``。

图件（全部为**真实三维坐标**下的断层图，无二维展开的比例失真）：
  * ``slip3d_<模型>.png``   每个模型的滑移分布；多模型共用同一色标，可横向比较；
  * ``slip3d_diff_<模型>.png``  主结果 − 对比模型 的逐片元差值（对称色标）；
  * ``slip_comparison_stats.csv``  各模型 RMS/Mw/M0 与差值指标；
  * ``lam_*/track_*_misfit.png``  逐轨道 观测/模型/残差（可用 --no-track-plots 关掉）。
二维按分量的展开剖面不在本脚本生成，需要时调用
``geodetic/show_slip_model_2d.py``（横轴可选里程或纬度）。

用法：``python comsol_scripts/comsol_result.py [--snapshot ...] [--comsol-mat ...]``
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat, savemat

# 项目根目录加入 sys.path，保证 geodetic / workflows 可导入
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from geodetic.constraints import add_col_of
from geodetic.green_function import _build_ramp
from geodetic.invert import (
    _expand_last_columns,
    _track_orbit_number,
    build_greens_mat_dict,
)
from workflows.inversion import InversionWorkflowState, step_load_config
from workflows.make_rms import invert_at_lambda, load_snapshot, plot_track_misfits


# 配置 / 快照 / 产物默认都相对项目根目录
_HERE = _ROOT


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
    # 先把它恢复到输入数据单位，再与分组后的采样逐点核对（atol 1e-6）。
    grouped_indices = insar_indices + azi_indices
    grouped_data = np.concatenate(
        [data_by_track[i][:, 2] for i in grouped_indices]
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

    add_col = add_col_of(cfg.ramp_choice)
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


# ---------------------------------------------------------------------------
# 滑移模型对比
# ---------------------------------------------------------------------------
# 设计要点（全部为三维图，不画二维展开剖面）：
#   1. 每个模型用 ``SlipCase`` 描述，数量不限；对比对象可以是 Okada，
#      也可以是另一次 COMSOL 反演结果。
#   2. 两幅对比断层图 + 一幅差值图，都由 :func:`geodetic.show_slip_model.show_slip_model`
#      在**真实三维坐标**（Easting/Northing/Depth, km）下绘制：断层面的走向长度与
#      深度比例天然正确，不存在二维展开图里「深度被拉伸 / 长度被压缩」的问题。
#   3. 对比的两幅图共用同一 ``vmax``，颜色含义一致，可横向比较。
#   4. 差值图按逐片元 Δ|slip| 着色，配 RdBu_r 对称色标（正 = 主结果更大）。
#   5. 二维按分量的展开图不再在这里出——``geodetic/show_slip_model_2d.py`` 已经实现，
#      需要时可单独调用（横轴可选里程或纬度）。
#   6. 不修改传入的 slip_model（原版本的 ``a[:, 11:] -= ...`` 会污染结果）。


@dataclass
class SlipCase:
    """一个参与对比的滑移模型。"""

    label: str
    slip_model: np.ndarray          # (N, 13)，列 11/12 为走滑 / 倾滑
    rms: Optional[float] = None
    mw: Optional[float] = None
    source: str = ""

    @property
    def strike_slip(self) -> np.ndarray:
        return np.asarray(self.slip_model[:, 11], dtype=np.float64)

    @property
    def dip_slip(self) -> np.ndarray:
        return np.asarray(self.slip_model[:, 12], dtype=np.float64)

    @property
    def magnitude(self) -> np.ndarray:
        return np.hypot(self.strike_slip, self.dip_slip)


# 剪切模量 (Pa)，与 geodetic.invert / workflows.make_rms 一致
_MU = 30e9


def _slug(text: str) -> str:
    """把标签转成安全的文件名片段。"""
    kept = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(text))
    return kept.strip("_") or "case"


def _case_metrics(case: SlipCase) -> Dict[str, Any]:
    """单个模型的标量指标：RMS、Mw、M0、最大 / 平均滑移。"""
    mag = case.magnitude
    area = case.slip_model[:, 6] * case.slip_model[:, 7]
    m0 = float(np.sum(_MU * mag * area))
    if case.mw is not None:
        mw = float(case.mw)
    else:
        mw = 2.0 / 3.0 * (np.log10(m0) - 9.1) if m0 > 0 else float("nan")
    return {
        "label": case.label,
        "rms": "" if case.rms is None else float(case.rms),
        "Mw": mw,
        "M0_Nm": m0,
        "max_slip": float(np.max(mag)) if mag.size else float("nan"),
        "mean_slip": float(np.mean(mag)) if mag.size else float("nan"),
        "source": case.source,
    }


def _diff_metrics(case_a: SlipCase, case_b: SlipCase) -> Dict[str, float]:
    """``A − B`` 的差异指标（按滑移量比较）。"""
    a, b = case_a.magnitude, case_b.magnitude
    if a.shape != b.shape:
        raise ValueError("两个模型的片元数不同，无法比较")
    d = a - b
    denom = float(np.linalg.norm(b))
    corr = float("nan")
    if a.size > 1 and np.std(a) > 0 and np.std(b) > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "rms_diff": float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan"),
        "max_abs_diff": float(np.max(np.abs(d))) if d.size else float("nan"),
        "rel_l2": float(np.linalg.norm(d) / denom) if denom > 0 else float("nan"),
        "corr": corr,
    }


def _assert_same_geometry(ref: np.ndarray, other: np.ndarray, label: str) -> None:
    """对比前确认两者几何一致（列 0–9: 段号/片元号/层号/坐标/尺寸/走向/倾角）。"""
    if ref.shape != other.shape:
        raise ValueError(
            "几何不一致：片元数 %d vs %d（%s）。请用相同断层几何重新生成对比模型"
            % (ref.shape[0], other.shape[0], label)
        )
    same = np.all(np.isclose(ref[:, :10], other[:, :10], rtol=0.0, atol=1e-6), axis=1)
    if not np.all(same):
        bad = int(np.argmax(~same))
        raise ValueError(
            "几何不一致（%s）：第 %d 个片元（fault=%d, patch=%d）的几何列不同"
            % (label, bad, int(ref[bad, 0]), int(ref[bad, 1]))
        )


def _load_case_from_mat(label: str, path: str) -> SlipCase:
    """从反演快照 ``.mat`` 读取一个对比模型（只取 ``slip_model`` 与 ``return_var``）。"""
    if not os.path.isfile(path):
        raise FileNotFoundError("对比模型不存在: %s" % path)
    d = loadmat(path)
    if "slip_model" not in d:
        raise KeyError("%s 中没有 slip_model；请传入反演快照 .mat" % path)
    sm = np.asarray(d["slip_model"], dtype=np.float64)
    if sm.ndim != 2 or sm.shape[1] < 13:
        raise ValueError("%s 的 slip_model 期望 (N, 13)，实际 %r" % (path, sm.shape))

    rms = mw = None
    if "return_var" in d:
        ret = np.ravel(np.asarray(d["return_var"], dtype=np.float64))
        if ret.size >= 4:
            rms, mw = float(ret[2]), float(ret[3])
    return SlipCase(
        label=label, slip_model=sm, rms=rms, mw=mw,
        source=os.path.relpath(path, _ROOT),
    )


def _parse_compare(spec: str) -> Tuple[str, str]:
    """解析 ``LABEL=PATH`` 形式的 ``--compare`` 参数。"""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "--compare 需为 LABEL=PATH，例如：Okada=py_inversion_iint0.mat"
        )
    label, path = spec.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--compare 的标签与路径都不能为空")
    return label, path


def _write_stats_csv(path: str, cases: Sequence[SlipCase], base: SlipCase) -> str:
    """把各模型指标与相对基准的差异写成 CSV。"""
    fields = [
        "kind", "label", "rms", "Mw", "M0_Nm", "max_slip", "mean_slip",
        "rms_diff", "max_abs_diff", "rel_l2", "corr", "source",
    ]
    rows: List[Dict[str, Any]] = []
    for c in cases:
        row = {k: "" for k in fields}
        row.update(kind="model", **_case_metrics(c))
        rows.append(row)
    for c in cases:
        if c is base:
            continue
        row = {k: "" for k in fields}
        row.update(kind="diff", label="%s − %s" % (base.label, c.label))
        row.update(_diff_metrics(base, c))
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    return path


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
    parser.add_argument(
        "--compare", action="append", default=[], metavar="LABEL=PATH",
        help="额外对比的滑移模型 .mat（需含 slip_model）；可重复，"
             "例如 Okada=py_inversion_iint0.mat 或 COMSOL_v2=other/comsol_inversion.mat",
    )
    parser.add_argument(
        "--no-auto-compare", action="store_true",
        help="不与 Step 5 快照（Okada 反演结果）自动对比",
    )
    parser.add_argument(
        "--no-3d", action="store_true",
        help="跳过全部 3D 图（模型图与差值图）",
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
    # return_var 布局同 geodetic.invert.make_fault_from_insar1:
    # [redu_perc, exitflag, rms, Mw, M0/1e20]. 写进去, 日后本结果作为
    # --compare 对象时 rms/Mw 列才不为空 (_load_case_from_mat 只认这个键).
    # exitflag 此处无来源 (solver_msg 是字符串), 以 0 占位.
    result_return_var = np.array(
        [
            result["redu_perc"],
            0.0,
            result["rms"],
            result["Mw"],
            result["M0"] / 1e20,
        ],
        dtype=np.float64,
    )
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
            return_var=result_return_var,
            RMS_misfit=result["RMS_misfit"],
            model_roughness=result["roughness"],
            class_map=snap["class_map"],
        ),
        do_compression=True,
        oned_as="column",
    )
    print("反演快照 -> %s" % out_mat, flush=True)

    # ------------------------------------------------------------------
    # 绘图：主结果 + 对比模型
    # ------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg", force=True)
    from geodetic.show_slip_model import show_slip_model

    main_case = SlipCase(
        label="COMSOL",
        slip_model=result["slip_model"],
        rms=result["rms"],
        mw=result["Mw"],
        source=os.path.relpath(out_mat, _ROOT),
    )

    # 对比对象：默认自动取 Step 5 快照（即 Okada 反演结果），再叠加 --compare
    compare_cases: List[SlipCase] = []
    if not args.no_auto_compare:
        # return_var 已由 load_snapshot 一起读出, 不再重复 loadmat 87MB 快照
        okada_rms = okada_mw = None
        ret = snap.get("return_var")
        if ret is not None and np.asarray(ret).size >= 4:
            okada_rms, okada_mw = float(ret[2]), float(ret[3])
        compare_cases.append(
            SlipCase(
                label="Okada (Step 5)",
                slip_model=np.asarray(snap["slip_model"], dtype=np.float64),
                rms=okada_rms,
                mw=okada_mw,
                source=os.path.relpath(args.snapshot, _ROOT),
            )
        )
    for label, path in (_parse_compare(spec) for spec in args.compare):
        if not os.path.isabs(path):
            path = os.path.join(_HERE, path)
        compare_cases.append(_load_case_from_mat(label, path))

    # 逐片元对比要求几何完全一致
    for case in compare_cases:
        _assert_same_geometry(main_case.slip_model, case.slip_model, case.label)

    all_cases = [main_case] + compare_cases
    # 所有 3D 图共用同一色标上限，颜色才可横向比较
    vmax3d = max(float(np.max(c.magnitude)) for c in all_cases)

    if not args.no_3d:
        for case in all_cases:
            png = os.path.join(args.out_dir, "slip3d_%s.png" % _slug(case.label))
            show_slip_model(
                case.slip_model,
                ref_lon=cfg.ref_lon,
                lonc=cfg.lonc,
                latc=cfg.latc,
                fault=state.fault_abs,
                out_path=png,
                show=False,
                vmax=vmax3d,
                title="%s  (λ=%g, RMS=%s, Mw=%s)"
                % (
                    case.label,
                    result["lam"],
                    "-" if case.rms is None else "%.4f" % case.rms,
                    "-" if case.mw is None else "%.3f" % case.mw,
                ),
            )
            print("3D 滑移图 -> %s" % png, flush=True)

        # 三维差值图：每个对比对象一张（逐片元 Δ|slip|，正负对称色标）。
        # 不再画 A|B|A−B 的二维展开图——那类图 geodetic/show_slip_model_2d.py
        # 已经实现 (且横轴可选里程或纬度)，在这里重复一遍还会带来
        # 「深度被拉伸 / 长度被压缩」的二维比例失真问题。
        for case in compare_cases:
            diff_values = main_case.magnitude - case.magnitude
            diff = _diff_metrics(main_case, case)
            png = os.path.join(
                args.out_dir, "slip3d_diff_%s.png" % _slug(case.label)
            )
            show_slip_model(
                main_case.slip_model,      # 几何与对比模型逐片元一致 (上面已校验)
                ref_lon=cfg.ref_lon,
                lonc=cfg.lonc,
                latc=cfg.latc,
                fault=state.fault_abs,
                out_path=png,
                show=False,
                color_values=diff_values,
                cmap_name="RdBu_r",
                symmetric=True,
                color_label="Δ|slip| (m)",
                title="%s − %s  (λ=%g, RMS diff=%.4f m)"
                % (main_case.label, case.label, result["lam"], diff["rms_diff"]),
            )
            print(
                "3D 差值图 -> %s  (RMS差=%.4f m, max|Δ|=%.4f m, 相对L2=%.2f%%, corr=%.4f)"
                % (
                    png,
                    diff["rms_diff"],
                    diff["max_abs_diff"],
                    100.0 * diff["rel_l2"],
                    diff["corr"],
                ),
                flush=True,
            )

    stats_csv = os.path.join(args.out_dir, "slip_comparison_stats.csv")
    _write_stats_csv(stats_csv, all_cases, main_case)
    print("对比指标 -> %s" % stats_csv, flush=True)

    if not args.no_track_plots:
        from workflows.make_rms import _check_track_rows

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
