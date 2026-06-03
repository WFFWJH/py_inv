r"""与 ``InversionExample.m`` 主流程对齐(默认至原脚本第 162 行).

对应 MATLAB 步骤:
  Step 0: ``configfile.txt`` + ``configpara.txt`` + ``data_list``
  Step 3: ``make_insar_data`` (quadtree)
  Step 4: ``load_fault_one_plane`` (断层几何)
  Step 5: ``make_fault_from_insar1`` (iter_step) → ``savemat`` → ``show_slip_model``
  Step 6: ``resamp_insar_data`` (iter_step2)
  Step 7: ``make_fault_from_insar1`` (iter_step2) → ``savemat`` → ``show_slip_model``

``__main__`` 用 ``START_STEP`` / ``END_STEP`` (3, 5, 6, 7 可选); Step 0 与 Step 4 任意运行前强制执行.

直接运行: 在文件末尾 ``if __name__ == "__main__"`` 中改变量后
``python inversion_example_workflow.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import loadmat, savemat

from load_fault_one_plane import load_fault_one_plane
from make_fault_from_insar1 import build_greens_mat_dict, make_fault_from_insar1
from make_insar_data import make_insar_data
from resamp_insar_data import resamp_insar_data
from show_slip_model import show_slip_model

# 与 InversionExample.m 第 74-75 行一致(未从 config 读取)
_DIP_ANGLE_DEFAULT: List[float] = [82.0, 82.0, 82.0, 82.0]
_RAMP_CHOICE = "qu_ramp_7"

# load_fault_one_plane 默认 — 须与重构前 inversion_workflow.InversionParams 一致
# (MATLAB InversionExample.m 为 w_ratio=1.2, layers=5; 改此处会改变反演结果)
_FAULT_W_RATIO = 1.1
_FAULT_L_RATIO = 1.2
_FAULT_WIDTH = 20e3
_FAULT_LEN_TOP = 2e3
_FAULT_LAYERS = 5
_OKADA_BACKEND = "numpy"
_MAX_NFEV = 100

# 与 InversionExample.m 第 105-106 行默认采样框一致
_DEFAULT_REGION: Tuple[float, float, float, float] = (95.0, 97.52, 15.17, 24.05)


def _in_step_range(step: int, lo: int, hi: int) -> bool:
    return lo <= step <= hi


def _run_required_steps(
    state: InversionWorkflowState,
    config_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    dip_per_segment: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> None:
    """Step 0 + Step 4: 无论从哪一步开始, 均须先读配置并重建断层几何."""
    if verbose:
        print("=== Step 0 (required) ===", flush=True)
    step_load_config(state, config_dir, dip_per_segment=dip_per_segment)
    if verbose:
        print("=== Step 4 (required) ===", flush=True)
    step_load_fault(state)


@dataclass
class InversionParams:
    """单步反演参数容器 (``geodetic_api`` / 脚本一次性调用)."""

    track_paths: Sequence[str]
    paths_type: Sequence[str]
    fault_file: str
    seg_connect_file: str
    output_mat: str
    iter_step: int = 0
    lonc: float = 95.33
    latc: float = 19.61
    ref_lon: float = 95.0
    ramp_choice: str = _RAMP_CHOICE
    con: Tuple[int, int, int] = (0, 0, 0)
    dip_per_segment: Optional[List[float]] = None
    w_ratio: float = _FAULT_W_RATIO
    l_ratio: float = _FAULT_L_RATIO
    width: float = _FAULT_WIDTH
    len_top: float = _FAULT_LEN_TOP
    layers: int = _FAULT_LAYERS
    max_nfev: int = _MAX_NFEV
    model_type: str = "okada"
    okada_backend: str = _OKADA_BACKEND
    save_snapshot: bool = True


def _save_inversion_mat(
    out_mat: str,
    slip: np.ndarray,
    rms_m: float,
    rough: float,
    ret: np.ndarray,
    extras: Dict[str, Any],
    *,
    ramp_choice: str = _RAMP_CHOICE,
) -> None:
    savemat(
        out_mat,
        build_greens_mat_dict(
            G_last=extras["G_last"],
            Bdata=extras["Bdata"],
            bd_last=extras["bd_last"],
            slip_model=slip,
            bdata_sm=extras["bdata_sm"],
            GrF=extras["GrF"],
            H=extras["H"],
            h1=extras["h1"],
            Wb=extras["Wb"],
            Wl=extras["Wl"],
            Wr=extras["Wr"],
            ramp_choice=ramp_choice,
            u=extras["u"],
            return_var=ret,
            RMS_misfit=rms_m,
            model_roughness=rough,
            class_map=extras["class_map"],
        ),
        do_compression=True,
        oned_as="column",
    )
    print("Saved -> %s" % out_mat, flush=True)


def run_okada_inversion(params: InversionParams) -> Tuple[
    np.ndarray, float, float, np.ndarray, Dict[str, Any],
]:
    """构几何 → 反演 → 可选存盘 (一次性调用, 供 ``geodetic_api`` 等)."""
    if params.dip_per_segment is None:
        raise ValueError("dip_per_segment: 请传入每段走滑断层倾角列表，如 [80, 80, 90, 110]")

    slip_vs = load_fault_one_plane(
        params.fault_file,
        dip=list(params.dip_per_segment),
        lonc=params.lonc, latc=params.latc, ref_lon=params.ref_lon,
        l_ratio=params.l_ratio, w_ratio=params.w_ratio,
        width=params.width, len_top=params.len_top, layers=params.layers,
    )
    print("slip_model_vs shape = %s" % (slip_vs.shape,), flush=True)

    slip, rms_m, rough, ret, extras = make_fault_from_insar1(
        slip_vs, None, params.iter_step, list(params.track_paths),
        paths_type=params.paths_type,
        ramp_choice=params.ramp_choice,
        segment_smooth_file=params.seg_connect_file,
        intersect_smooth_file=None,
        fault_file=params.fault_file,
        ref_lon=params.ref_lon, lonc=params.lonc, latc=params.latc,
        Con=params.con,
        model_type=params.model_type,
        backend=params.okada_backend,
        max_nfev=params.max_nfev,
        verbose=True,
    )

    if params.save_snapshot and params.output_mat:
        _save_inversion_mat(
            params.output_mat, slip, rms_m, rough, ret, extras,
            ramp_choice=params.ramp_choice,
        )
    return slip, rms_m, rough, ret, extras


def _lines_skip_hash(path: str) -> List[str]:
    """行内出现 ``#`` 则整行丢弃(与 MATLAB 对含 ``#`` 行跳过类似)."""
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "#" in line:
                continue
            s = line.strip()
            if s:
                out.append(s)
    return out


def read_configfile(configfile_path: str) -> SimpleNamespace:
    """解析 ``configfile.txt``: 6 行路径(无含 ``#`` 行)."""
    lines = _lines_skip_hash(configfile_path)
    keys = (
        "grdin",
        "grdout",
        "data_list",
        "fault_file",
        "segment_smooth_file",
        "segment_file",
    )
    if len(lines) < 6:
        raise ValueError(
            "configfile 需要至少 6 条无注释行(grdin, grdout, data_list, fault, 2x segment), 得到 %d: %s"
            % (len(lines), configfile_path)
        )
    d = {k: lines[i] for i, k in enumerate(keys)}
    return SimpleNamespace(**d)


@dataclass
class ConfigPara:
    """与 ``configpara.txt`` 中数值顺序一致."""

    lonf: float
    latf: float
    ref_lon: float
    threshold: float
    lonc: float
    latc: float
    iter_step: int
    iter_step2: int
    con: Tuple[int, int, int]


def read_configpara(configpara_path: str) -> ConfigPara:
    """前 8 个为原 MATLAB ``para(1)..(8)``; 第 9–11 个为 Con(若缺省为 -1,0,0)."""
    nums: List[float] = []
    with open(configpara_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "#" in line:
                continue
            s = line.strip()
            if not s:
                continue
            try:
                nums.append(float(s))
            except ValueError:
                pass
    if len(nums) < 8:
        raise ValueError("configpara 中至少需要 8 个数值, 得到 %d: %s" % (len(nums), configpara_path))
    it0, it1 = int(nums[6]), int(nums[7])
    con: Tuple[int, int, int]
    if len(nums) >= 11:
        con = (int(nums[8]), int(nums[9]), int(nums[10]))
    else:
        con = (0, 0, 0)
    return ConfigPara(
        lonf=nums[0],
        latf=nums[1],
        ref_lon=nums[2],
        threshold=nums[3],
        lonc=nums[4],
        latc=nums[5],
        iter_step=it0,
        iter_step2=it1,
        con=con,
    )


def read_data_list(
    data_list_path: str,
    *,
    default_region: Tuple[float, float, float, float] = _DEFAULT_REGION,
) -> Tuple[List[str], List[int], np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    每行 ``track_path npt [xmin xmax ymin ymax] type``.

    与 ``InversionExample.m`` 一致: 列数不足时用默认 region; ``insar``/``azi`` 设 Nmin/Nmax.
    """
    base = os.path.dirname(os.path.abspath(data_list_path))
    tracks: List[str] = []
    npt: List[int] = []
    regions: List[Tuple[float, float, float, float]] = []
    dtypes: List[str] = []
    with open(data_list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "#" in line:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tr_rel = parts[0]
            n = int(float(parts[1]))
            if len(parts) >= 7:
                reg = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                dtp = parts[6].lower()
            elif len(parts) == 3:
                reg = default_region
                dtp = parts[2].lower()
            else:
                reg = default_region
                dtp = "insar"
            tr_abs = os.path.normpath(os.path.join(base, tr_rel))
            tracks.append(tr_abs)
            npt.append(n)
            regions.append(reg)
            dtypes.append(dtp)
    if not tracks:
        raise ValueError("data_list 中无有效道: %s" % data_list_path)
    region = np.asarray(regions, dtype=np.float64)
    nmin: List[int] = []
    nmax: List[int] = []
    for t in dtypes:
        if t == "insar":
            nmin.append(8)
            nmax.append(500)
        elif t == "azi":
            nmin.append(4)
            nmax.append(50)
        else:
            nmin.append(8)
            nmax.append(500)
    return (
        tracks,
        npt,
        region,
        dtypes,
        np.asarray(nmin, dtype=np.int64),
        np.asarray(nmax, dtype=np.int64),
    )


def count_fault_segments(fault_path: str) -> int:
    n = 0
    with open(fault_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n


def _resolve_under_root(root: str, p: str) -> str:
    p = p.strip()
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(root, p))


def load_slip_from_mat(mat_path: str) -> np.ndarray:
    """从 ``py_inversion_iint*.mat`` 等快照读 ``slip_model``."""
    if not os.path.isfile(mat_path):
        raise FileNotFoundError("未找到 slip 快照: %s" % mat_path)
    data = loadmat(mat_path)
    if "slip_model" not in data:
        raise KeyError("%s 中无变量 slip_model" % mat_path)
    return np.asarray(data["slip_model"], dtype=np.float64)


@dataclass
class InversionWorkflowState:
    """全流程中间变量容器; 各 step_* 函数读写此对象."""

    root: str = ""
    cfg: Optional[SimpleNamespace] = None
    para: Optional[ConfigPara] = None
    tracks: List[str] = field(default_factory=list)
    npt: List[int] = field(default_factory=list)
    region: Optional[np.ndarray] = None
    dtypes: List[str] = field(default_factory=list)
    nmin: Optional[np.ndarray] = None
    nmax: Optional[np.ndarray] = None
    fault_abs: str = ""
    seg_file_abs: str = ""
    dangles: List[float] = field(default_factory=list)
    nseg: int = 0
    ntrack: int = 0
    out_mat: str = ""
    out_png: str = ""
    slip_vs: Optional[np.ndarray] = None
    slip: Optional[np.ndarray] = None
    rms: Optional[float] = None
    rough: Optional[float] = None
    ret: Optional[np.ndarray] = None
    extras: Optional[Dict[str, Any]] = None
    out_mat_step2: str = ""
    out_png_step2: str = ""
    slip2: Optional[np.ndarray] = None
    rms2: Optional[float] = None
    rough2: Optional[float] = None
    ret2: Optional[np.ndarray] = None
    extras2: Optional[Dict[str, Any]] = None


def _require_config(state: InversionWorkflowState) -> None:
    if state.cfg is None or state.para is None or not state.tracks:
        raise RuntimeError("缺少配置: 请先运行 step_load_config 或从 load_config 步骤开始")


def _los_samp_path(track: str, iint: int) -> str:
    return os.path.join(track, "los_samp%d.mat" % int(iint))


def _check_los_samp_files(state: InversionWorkflowState, iint: int) -> None:
    _require_config(state)
    missing = [p for tr in state.tracks if not os.path.isfile(p := _los_samp_path(tr, iint))]
    if missing:
        raise FileNotFoundError(
            "步骤需要各道 los_samp%d.mat, 缺失例如: %s" % (iint, missing[0])
        )


def _ensure_slip(state: InversionWorkflowState, checkpoint_mat: Optional[str] = None) -> None:
    if state.slip is not None:
        return
    path = checkpoint_mat or state.out_mat
    if path and os.path.isfile(path):
        state.slip = load_slip_from_mat(path)
        print("已从快照加载 slip: %s" % path, flush=True)
        return
    raise RuntimeError(
        "步骤 resamp 需要 state.slip 或可用的 CHECKPOINT_SLIP_MAT / out_mat 快照"
    )


def _resolve_output_paths(state: InversionWorkflowState) -> None:
    if not state.out_mat:
        state.out_mat = os.path.join(state.root, "py_inversion_iint0.mat")
    if not state.out_png:
        state.out_png = os.path.splitext(state.out_mat)[0] + "_show.png"
    if state.para and not state.out_mat_step2:
        it2 = int(state.para.iter_step2)
        state.out_mat_step2 = os.path.join(state.root, "py_inversion_iint%d.mat" % it2)
    if not state.out_png_step2 and state.out_mat_step2:
        state.out_png_step2 = os.path.splitext(state.out_mat_step2)[0] + "_show_step7.png"


def _show_slip_figure(
    slip: np.ndarray,
    state: InversionWorkflowState,
    out_png: str,
    title: str,
    *,
    show_figure: bool,
) -> None:
    do_show = show_figure and os.environ.get("SHOW_SLIP", "1").strip() not in (
        "0", "false", "False", "no",
    )
    if not do_show:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")
    elif show_figure:
        print(
            "show_slip_model(关图继续: SHOW_SLIP_BLOCK=0; 不弹窗: SHOW_SLIP=0) ...",
            flush=True,
        )
    show_slip_model(
        slip,
        ref_lon=state.para.ref_lon,
        lonc=state.para.lonc,
        latc=state.para.latc,
        fault=state.fault_abs,
        out_path=out_png,
        show=do_show,
        title=title,
        block=False,
    )


def _ensure_slip_vs(state: InversionWorkflowState) -> None:
    if state.slip_vs is None:
        raise RuntimeError("缺少 slip_vs; 请先运行 step_load_fault (或 _run_required_steps)")


def step_load_fault(state: InversionWorkflowState) -> InversionWorkflowState:
    """Step 4: ``load_fault_one_plane`` 构建断层几何."""
    _require_config(state)
    if not state.dangles:
        raise ValueError("dip 角列表为空; 请先 step_load_config")
    state.slip_vs = load_fault_one_plane(
        state.fault_abs,
        dip=state.dangles,
        lonc=state.para.lonc,
        latc=state.para.latc,
        ref_lon=state.para.ref_lon,
        l_ratio=_FAULT_L_RATIO,
        w_ratio=_FAULT_W_RATIO,
        width=_FAULT_WIDTH,
        len_top=_FAULT_LEN_TOP,
        layers=_FAULT_LAYERS,
    )
    print("slip_model_vs shape = %s" % (state.slip_vs.shape,), flush=True)
    return state


def _run_make_fault(
    state: InversionWorkflowState,
    iter_step: int,
) -> Tuple[np.ndarray, float, float, np.ndarray, Dict[str, Any]]:
    _ensure_slip_vs(state)
    _check_los_samp_files(state, iter_step)
    return make_fault_from_insar1(
        state.slip_vs, None, int(iter_step), state.tracks,
        paths_type=state.dtypes,
        ramp_choice=_RAMP_CHOICE,
        segment_smooth_file=state.seg_file_abs,
        intersect_smooth_file=None,
        fault_file=state.fault_abs,
        ref_lon=state.para.ref_lon,
        lonc=state.para.lonc,
        latc=state.para.latc,
        Con=state.para.con,
        model_type="okada",
        backend=_OKADA_BACKEND,
        max_nfev=_MAX_NFEV,
        verbose=True,
    )


def step_load_config(
    state: InversionWorkflowState,
    config_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    dip_per_segment: Optional[Sequence[float]] = None,
) -> InversionWorkflowState:
    """Step 0: 读 config / data_list, 填充 state."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(config_dir) if config_dir is not None else here
    state.root = root

    cfile = os.path.join(root, "configfile.txt")
    pfile = os.path.join(root, "configpara.txt")
    if not os.path.isfile(cfile) or not os.path.isfile(pfile):
        raise FileNotFoundError("需要 %s 与 %s" % (cfile, pfile))

    state.cfg = read_configfile(cfile)
    state.para = read_configpara(pfile)

    data_list_abs = _resolve_under_root(root, state.cfg.data_list)
    state.fault_abs = _resolve_under_root(root, state.cfg.fault_file)
    state.seg_file_abs = _resolve_under_root(root, state.cfg.segment_file)

    tracks, npt, region, dtypes, nmin, nmax = read_data_list(data_list_abs)
    state.tracks = tracks
    state.npt = npt
    state.region = region
    state.dtypes = dtypes
    state.nmin = nmin
    state.nmax = nmax
    state.ntrack = len(tracks)
    state.nseg = count_fault_segments(state.fault_abs)

    dangles = list(dip_per_segment) if dip_per_segment is not None else _DIP_ANGLE_DEFAULT
    if state.nseg != len(dangles):
        raise ValueError(
            "fault 段数 %d 与 dip 角个数 %d 不一致; 请改 ``_DIP_ANGLE_DEFAULT`` 或传 dip_per_segment=..."
            % (state.nseg, len(dangles))
        )
    state.dangles = [float(x) for x in dangles]

    _resolve_output_paths(state)

    print("There are %d segments of fault." % state.nseg, flush=True)
    print("There are %d tracks of data." % state.ntrack, flush=True)
    for i, tr in enumerate(state.tracks):
        print(
            "  track[%d] %s  npt=%s  %s  Nmin=%s Nmax=%s"
            % (i, tr, state.npt[i], state.dtypes[i], state.nmin[i], state.nmax[i]),
            flush=True,
        )
    return state


def step_downsample(
    state: InversionWorkflowState,
    *,
    skip: bool = False,
    save_plot: bool = True,
) -> InversionWorkflowState:
    """Step 3: ``make_insar_data`` (quadtree)."""
    _require_config(state)
    if skip:
        iint = int(state.para.iter_step)
        _check_los_samp_files(state, iint)
        print("已跳过 downsample (假定各道已有 los_samp%d.mat)." % iint, flush=True)
        return state
    make_insar_data(
        state.tracks,
        state.npt,
        state.region,
        state.nmin,
        state.nmax,
        method="quadtree",
        lonc=state.para.lonc,
        latc=state.para.latc,
        ref_lon=state.para.ref_lon,
        fault_file=state.fault_abs,
        save_mat=True,
        save_plot=save_plot,
    )
    return state


def step_invert1(state: InversionWorkflowState) -> InversionWorkflowState:
    """Step 5a: ``make_fault_from_insar1`` (iter_step)."""
    _require_config(state)
    _resolve_output_paths(state)
    slip, rms, rough, ret, extras = _run_make_fault(state, int(state.para.iter_step))
    state.slip, state.rms, state.rough = slip, rms, rough
    state.ret, state.extras = ret, extras
    return state


def step_save1(state: InversionWorkflowState) -> InversionWorkflowState:
    """Step 5b: 保存第一次反演 ``.mat``."""
    _require_config(state)
    _resolve_output_paths(state)
    if state.slip is None or state.extras is None:
        raise RuntimeError("step_save1 需要先运行 step_invert1")
    _save_inversion_mat(
        state.out_mat, state.slip, state.rms, state.rough, state.ret, state.extras,
    )
    return state


def step_show1(
    state: InversionWorkflowState,
    *,
    show_figure: bool = True,
) -> InversionWorkflowState:
    """Step 5: 第一次滑动模型出图."""
    _require_config(state)
    if state.slip is None:
        _ensure_slip(state)
    _resolve_output_paths(state)
    if show_figure:
        _show_slip_figure(
            state.slip,
            state,
            state.out_png,
            "InversionExample step5",
            show_figure=True,
        )
    else:
        print("已跳过 show1.", flush=True)
    return state


def step_resamp(
    state: InversionWorkflowState,
    *,
    dec: int = 1,
    patch_workers: Optional[int] = None,
    checkpoint_slip_mat: Optional[str] = None,
) -> InversionWorkflowState:
    """Step 6: ``resamp_insar_data``."""
    _require_config(state)
    _ensure_slip(state, checkpoint_slip_mat)
    it2 = int(state.para.iter_step2)
    print("Step6: resamp_insar_data 开始 ...", flush=True)
    resamp_insar_data(
        state.slip,
        state.tracks,
        state.npt,
        list(np.asarray(state.nmin).ravel()),
        list(np.asarray(state.nmax).ravel()),
        state.dtypes,
        it2,
        lonc=state.para.lonc,
        latc=state.para.latc,
        ref_lon=state.para.ref_lon,
        fault_file=state.fault_abs,
        dec=dec,
        patch_workers=patch_workers,
    )
    print("resamp_insar_data: 已写出各道 los_samp%d.mat" % it2, flush=True)
    return state


def step_invert2(state: InversionWorkflowState) -> InversionWorkflowState:
    """Step 7a: ``make_fault_from_insar1`` (iter_step2)."""
    _require_config(state)
    _resolve_output_paths(state)
    it2 = int(state.para.iter_step2)
    slip2, rms2, rough2, ret2, extras2 = _run_make_fault(state, it2)
    state.slip2 = slip2
    state.rms2, state.rough2 = rms2, rough2
    state.ret2, state.extras2 = ret2, extras2
    return state


def step_save2(state: InversionWorkflowState) -> InversionWorkflowState:
    """Step 7b: 保存第二次反演 ``.mat``."""
    _require_config(state)
    _resolve_output_paths(state)
    if state.slip2 is None or state.extras2 is None:
        raise RuntimeError("step_save2 需要先运行 step_invert2")
    _save_inversion_mat(
        state.out_mat_step2, state.slip2, state.rms2, state.rough2,
        state.ret2, state.extras2,
    )
    return state


def step_show2(
    state: InversionWorkflowState,
    *,
    show_figure: bool = True,
) -> InversionWorkflowState:
    """Step 7: 第二次滑动模型出图."""
    _require_config(state)
    if state.slip2 is None:
        raise RuntimeError("step_show2 需要 state.slip2; 请先运行 step_invert2")
    _resolve_output_paths(state)
    if show_figure:
        it2 = int(state.para.iter_step2)
        _show_slip_figure(
            state.slip2,
            state,
            state.out_png_step2,
            "InversionExample step7 (iter_step2=%d)" % it2,
            show_figure=True,
        )
    else:
        print("已跳过 show2.", flush=True)
    return state


def run_workflow_steps(
    state: InversionWorkflowState,
    *,
    start_step: int = 0,
    end_step: int = 7,
    config_dir: Optional[Union[str, os.PathLike]] = None,
    dip_per_segment: Optional[Sequence[float]] = None,
    skip_downsample: bool = False,
    make_insar_save_plot: bool = True,
    show_figure: bool = True,
    dec: int = 1,
    patch_workers: Optional[int] = None,
    checkpoint_slip_mat: Optional[str] = None,
    out_mat: Optional[str] = None,
    out_png: Optional[str] = None,
    out_mat_step2: Optional[str] = None,
    out_png_step2: Optional[str] = None,
) -> InversionWorkflowState:
    """按 ``start_step``..``end_step`` 执行; Step 0/4 始终先跑."""
    if out_mat:
        state.out_mat = out_mat
    if out_png:
        state.out_png = out_png
    if out_mat_step2:
        state.out_mat_step2 = out_mat_step2
    if out_png_step2:
        state.out_png_step2 = out_png_step2

    _run_required_steps(state, config_dir, dip_per_segment=dip_per_segment)

    if _in_step_range(3, start_step, end_step):
        step_downsample(state, skip=skip_downsample, save_plot=make_insar_save_plot)

    if _in_step_range(5, start_step, end_step):
        step_invert1(state)
        step_save1(state)
        step_show1(state, show_figure=show_figure)

    if _in_step_range(6, start_step, end_step):
        step_resamp(
            state,
            dec=dec,
            patch_workers=patch_workers,
            checkpoint_slip_mat=checkpoint_slip_mat,
        )

    if _in_step_range(7, start_step, end_step):
        step_invert2(state)
        step_save2(state)
        step_show2(state, show_figure=show_figure)

    return state


if __name__ == "__main__":
    CONFIG_DIR: Optional[Union[str, os.PathLike]] = None
    START_STEP = 5   # 3 | 5 | 6 | 7  (Step 0/4 始终自动执行)
    END_STEP = 5
    SKIP_DOWNSAMPLE = True
    OUT_MAT: Optional[str] = None
    OUT_PNG: Optional[str] = None
    OUT_MAT_STEP2: Optional[str] = None
    OUT_PNG_STEP2: Optional[str] = None
    MAKE_INSAR_SAVE_PLOT = True
    SHOW_FIGURE = True
    RESAMP_DEC = 1
    RESAMP_PATCH_WORKERS: Optional[int] = 4
    CHECKPOINT_SLIP_MAT: Optional[str] = None

    state = InversionWorkflowState()
    if OUT_MAT:
        state.out_mat = OUT_MAT
    if OUT_PNG:
        state.out_png = OUT_PNG
    if OUT_MAT_STEP2:
        state.out_mat_step2 = OUT_MAT_STEP2
    if OUT_PNG_STEP2:
        state.out_png_step2 = OUT_PNG_STEP2

    _run_required_steps(state, CONFIG_DIR)

    if _in_step_range(3, START_STEP, END_STEP):
        print("=== Step 3 ===", flush=True)
        step_downsample(state, skip=SKIP_DOWNSAMPLE, save_plot=MAKE_INSAR_SAVE_PLOT)
    if _in_step_range(5, START_STEP, END_STEP):
        print("=== Step 5 ===", flush=True)
        step_invert1(state)
        step_save1(state)
        step_show1(state, show_figure=SHOW_FIGURE)
    if _in_step_range(6, START_STEP, END_STEP):
        print("=== Step 6 ===", flush=True)
        step_resamp(
            state, dec=RESAMP_DEC, patch_workers=RESAMP_PATCH_WORKERS,
            checkpoint_slip_mat=CHECKPOINT_SLIP_MAT,
        )
    if _in_step_range(7, START_STEP, END_STEP):
        print("=== Step 7 ===", flush=True)
        step_invert2(state)
        step_save2(state)
        step_show2(state, show_figure=SHOW_FIGURE)

    print("workflow 完成 (step %d .. %d)." % (START_STEP, END_STEP), flush=True)
    if state.slip is not None:
        print("  slip1  -> %s  rms=%s" % (state.out_mat, state.rms), flush=True)
    if state.slip2 is not None:
        print("  slip2  -> %s  rms=%s" % (state.out_mat_step2, state.rms2), flush=True)
