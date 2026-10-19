r"""Flattened InversionExample workflow.

Reading path::

    run_inversion.py / run()
        -> load_config_and_fault
        -> invert_round / run_downsample / run_resamp
        -> make_fault_from_insar1 / show_slip_model

MATLAB step numbers (3/5/6/7) map to phase names; see ``PHASE_ORDER``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import loadmat, savemat

from geodetic.config import (
    WorkflowConfig,
    count_fault_segments,
    load_workflow_config,
    read_data_list,
    resolve_under_root,
)
from geodetic.fault_geometry import load_fault_one_plane
from geodetic.insar_data import make_insar_data
from geodetic.invert import build_greens_mat_dict, make_fault_from_insar1
from geodetic.resample import resamp_insar_data
from geodetic.show_slip_model import show_slip_model

# Phase names（对应 MATLAB Step 3 / 5 / 6 / 7）
PHASE_ORDER = (
    "downsample",  # Step 3
    "invert1",     # Step 5: invert + save + show
    "resamp",      # Step 6
    "invert2",     # Step 7: invert + save + show
)


@dataclass
class InversionParams:
    """One-shot inversion API (does not read config files)."""

    track_paths: Sequence[str]
    paths_type: Sequence[str]
    fault_file: str
    seg_connect_file: str
    output_mat: str
    iter_step: int = WorkflowConfig.iter_step
    lonc: float = WorkflowConfig.lonc
    latc: float = WorkflowConfig.latc
    ref_lon: float = WorkflowConfig.ref_lon
    ramp_choice: str = WorkflowConfig.ramp_choice
    con: Tuple[int, int, int] = WorkflowConfig.con
    dip_per_segment: Optional[List[float]] = None
    w_ratio: float = WorkflowConfig.w_ratio
    l_ratio: float = WorkflowConfig.l_ratio
    width: float = WorkflowConfig.width
    len_top: float = WorkflowConfig.len_top
    layers: int = WorkflowConfig.layers
    max_nfev: int = WorkflowConfig.max_nfev
    model_type: str = WorkflowConfig.model_type
    okada_backend: str = WorkflowConfig.okada_backend
    save_snapshot: bool = True


@dataclass
class InversionWorkflowState:
    """Runtime state for the inversion workflow."""

    # --- config / IO ---
    root: str = ""
    config: Optional[WorkflowConfig] = None
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

    # --- round 1 (iter_step) ---
    out_mat: str = ""
    out_png: str = ""
    slip_vs: Optional[np.ndarray] = None
    slip: Optional[np.ndarray] = None
    rms: Optional[float] = None
    rough: Optional[float] = None
    ret: Optional[np.ndarray] = None
    extras: Optional[Dict[str, Any]] = None

    # --- round 2 (iter_step2) ---
    out_mat_step2: str = ""
    out_png_step2: str = ""
    slip2: Optional[np.ndarray] = None
    rms2: Optional[float] = None
    rough2: Optional[float] = None
    ret2: Optional[np.ndarray] = None
    extras2: Optional[Dict[str, Any]] = None


def load_slip_from_mat(mat_path: str) -> np.ndarray:
    if not os.path.isfile(mat_path):
        raise FileNotFoundError("未找到 slip 快照: %s" % mat_path)
    data = loadmat(mat_path)
    if "slip_model" not in data:
        raise KeyError("%s 中无变量 slip_model" % mat_path)
    return np.asarray(data["slip_model"], dtype=np.float64)


def save_inversion_mat(
    out_mat: str,
    slip: np.ndarray,
    rms_m: float,
    rough: float,
    ret: np.ndarray,
    extras: Dict[str, Any],
    *,
    ramp_choice: str = WorkflowConfig.ramp_choice,
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
    if params.dip_per_segment is None:
        raise ValueError("dip_per_segment: 请传入每段倾角列表，如 [80, 80, 90, 110]")

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
        save_inversion_mat(
            params.output_mat, slip, rms_m, rough, ret, extras,
            ramp_choice=params.ramp_choice,
        )
    return slip, rms_m, rough, ret, extras


def _require_config(state: InversionWorkflowState) -> WorkflowConfig:
    if state.config is None or not state.tracks:
        raise RuntimeError("缺少配置: 请先 load_config_and_fault / step_load_config")
    return state.config


def _resolve_output_paths(state: InversionWorkflowState) -> None:
    cfg = state.config
    if not state.out_mat:
        state.out_mat = os.path.join(state.root, "py_inversion_iint0.mat")
    if not state.out_png:
        state.out_png = os.path.splitext(state.out_mat)[0] + "_show.png"
    if cfg is not None and not state.out_mat_step2:
        state.out_mat_step2 = os.path.join(
            state.root, "py_inversion_iint%d.mat" % int(cfg.iter_step2)
        )
    if not state.out_png_step2 and state.out_mat_step2:
        state.out_png_step2 = os.path.splitext(state.out_mat_step2)[0] + "_show_step7.png"


def _check_los_samp(state: InversionWorkflowState, iint: int) -> None:
    missing = [
        p for tr in state.tracks
        if not os.path.isfile(p := os.path.join(tr, "los_samp%d.mat" % int(iint)))
    ]
    if missing:
        raise FileNotFoundError(
            "需要各道 los_samp%d.mat, 缺失例如: %s" % (iint, missing[0])
        )


def _show_slip(
    slip: np.ndarray,
    state: InversionWorkflowState,
    out_png: str,
    title: str,
    show_figure: bool,
) -> None:
    cfg = _require_config(state)
    do_show = show_figure and os.environ.get("SHOW_SLIP", "1").strip() not in (
        "0", "false", "False", "no",
    )
    if not do_show:
        import matplotlib
        try:
            matplotlib.use("Agg", force=True)
        except TypeError:
            matplotlib.use("Agg")
    show_slip_model(
        slip,
        ref_lon=cfg.ref_lon, lonc=cfg.lonc, latc=cfg.latc,
        fault=state.fault_abs, out_path=out_png,
        show=do_show, title=title, block=False,
    )


# ---------------------------------------------------------------------------
# Config + fault (always required)
# ---------------------------------------------------------------------------

def step_load_config(
    state: InversionWorkflowState,
    config_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    dip_per_segment: Optional[Sequence[float]] = None,
    quiet: bool = False,
) -> InversionWorkflowState:
    """Load configfile/configpara/data_list into state (MATLAB Step 0)."""
    root, cfg = load_workflow_config(config_dir)
    state.root = root
    state.config = cfg
    state.fault_abs = resolve_under_root(root, cfg.fault_file)
    state.seg_file_abs = resolve_under_root(root, cfg.segment_file)

    tracks, npt, region, dtypes, nmin, nmax = read_data_list(
        resolve_under_root(root, cfg.data_list),
        default_region=cfg.default_region,
    )
    state.tracks, state.npt, state.region = tracks, npt, region
    state.dtypes, state.nmin, state.nmax = dtypes, nmin, nmax
    state.ntrack = len(tracks)
    state.nseg = count_fault_segments(state.fault_abs)

    dangles = list(dip_per_segment) if dip_per_segment is not None else list(cfg.dip)
    if state.nseg != len(dangles):
        raise ValueError(
            "fault 段数 %d 与 dip 角个数 %d 不一致; 请改 ##dip 或传 dip_per_segment"
            % (state.nseg, len(dangles))
        )
    state.dangles = [float(x) for x in dangles]
    _resolve_output_paths(state)

    if not quiet:
        print("There are %d segments of fault." % state.nseg, flush=True)
        print("There are %d tracks of data." % state.ntrack, flush=True)
        for i, tr in enumerate(state.tracks):
            print(
                "  track[%d] %s  npt=%s  %s  Nmin=%s Nmax=%s"
                % (i, tr, state.npt[i], state.dtypes[i], state.nmin[i], state.nmax[i]),
                flush=True,
            )
    return state


def step_load_fault(
    state: InversionWorkflowState,
    *,
    quiet: bool = False,
) -> InversionWorkflowState:
    """Build fault geometry (MATLAB Step 4)."""
    cfg = _require_config(state)
    if not state.dangles:
        raise ValueError("dip 角列表为空; 请先 step_load_config")
    state.slip_vs = load_fault_one_plane(
        state.fault_abs, dip=state.dangles,
        lonc=cfg.lonc, latc=cfg.latc, ref_lon=cfg.ref_lon,
        l_ratio=cfg.l_ratio, w_ratio=cfg.w_ratio,
        width=cfg.width, len_top=cfg.len_top, layers=cfg.layers,
    )
    if not quiet:
        print("slip_model_vs shape = %s" % (state.slip_vs.shape,), flush=True)
    return state


def load_config_and_fault(
    state: InversionWorkflowState,
    config_dir: Optional[Union[str, os.PathLike]] = None,
    *,
    dip_per_segment: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> InversionWorkflowState:
    """Step 0 + Step 4 combined."""
    if verbose:
        print("=== load config + fault ===", flush=True)
    step_load_config(state, config_dir, dip_per_segment=dip_per_segment)
    step_load_fault(state)
    return state


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def run_downsample(
    state: InversionWorkflowState,
    *,
    skip: bool = False,
    save_plot: bool = True,
) -> None:
    """MATLAB Step 3."""
    cfg = _require_config(state)
    if skip:
        _check_los_samp(state, int(cfg.iter_step))
        print("已跳过 downsample (假定已有 los_samp%d.mat)." % int(cfg.iter_step), flush=True)
        return
    make_insar_data(
        state.tracks, state.npt, state.region, state.nmin, state.nmax,
        method="quadtree",
        lonc=cfg.lonc, latc=cfg.latc, ref_lon=cfg.ref_lon,
        fault_file=state.fault_abs, save_mat=True, save_plot=save_plot,
    )


def invert_round(
    state: InversionWorkflowState,
    which: int = 1,
    *,
    show_figure: bool = True,
) -> None:
    """One invert + save + show round. ``which=1`` → Step 5; ``which=2`` → Step 7."""
    cfg = _require_config(state)
    if state.slip_vs is None:
        raise RuntimeError("缺少 slip_vs; 请先 load_config_and_fault")
    _resolve_output_paths(state)

    if which == 1:
        iint = int(cfg.iter_step)
        out_mat, out_png, title = state.out_mat, state.out_png, "InversionExample invert1"
    elif which == 2:
        iint = int(cfg.iter_step2)
        out_mat, out_png = state.out_mat_step2, state.out_png_step2
        title = "InversionExample invert2 (iter_step2=%d)" % iint
    else:
        raise ValueError("which must be 1 or 2")

    _check_los_samp(state, iint)
    slip, rms, rough, ret, extras = make_fault_from_insar1(
        state.slip_vs, None, iint, state.tracks,
        paths_type=state.dtypes,
        ramp_choice=cfg.ramp_choice,
        segment_smooth_file=state.seg_file_abs,
        intersect_smooth_file=None,
        fault_file=state.fault_abs,
        ref_lon=cfg.ref_lon, lonc=cfg.lonc, latc=cfg.latc,
        Con=cfg.con, model_type=cfg.model_type,
        backend=cfg.okada_backend, max_nfev=cfg.max_nfev, verbose=True,
    )
    save_inversion_mat(out_mat, slip, rms, rough, ret, extras, ramp_choice=cfg.ramp_choice)

    if which == 1:
        state.slip, state.rms, state.rough = slip, rms, rough
        state.ret, state.extras = ret, extras
    else:
        state.slip2, state.rms2, state.rough2 = slip, rms, rough
        state.ret2, state.extras2 = ret, extras

    if show_figure:
        _show_slip(slip, state, out_png, title, show_figure=True)


def run_resamp(
    state: InversionWorkflowState,
    *,
    dec: int = 1,
    patch_workers: Optional[int] = None,
    checkpoint_slip_mat: Optional[str] = None,
) -> None:
    """MATLAB Step 6."""
    cfg = _require_config(state)
    if state.slip is None:
        path = checkpoint_slip_mat or state.out_mat
        if path and os.path.isfile(path):
            state.slip = load_slip_from_mat(path)
            print("已从快照加载 slip: %s" % path, flush=True)
        else:
            raise RuntimeError("resamp 需要 state.slip 或可用快照")
    it2 = int(cfg.iter_step2)
    print("resamp_insar_data 开始 ...", flush=True)
    resamp_insar_data(
        state.slip, state.tracks, state.npt,
        list(np.asarray(state.nmin).ravel()),
        list(np.asarray(state.nmax).ravel()),
        state.dtypes, it2,
        lonc=cfg.lonc, latc=cfg.latc, ref_lon=cfg.ref_lon,
        fault_file=state.fault_abs, dec=dec, patch_workers=patch_workers,
    )
    print("resamp_insar_data: 已写出各道 los_samp%d.mat" % it2, flush=True)


def _phases_in_range(start: str, end: str) -> List[str]:
    if start not in PHASE_ORDER or end not in PHASE_ORDER:
        raise ValueError("phase 须为 %s 之一" % (PHASE_ORDER,))
    i0, i1 = PHASE_ORDER.index(start), PHASE_ORDER.index(end)
    if i0 > i1:
        raise ValueError("start phase 不能晚于 end phase")
    return list(PHASE_ORDER[i0 : i1 + 1])


def run(
    state: Optional[InversionWorkflowState] = None,
    *,
    start: str = "invert1",
    end: str = "invert1",
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
    """Run phases from ``start`` to ``end``; always loads config + fault first."""
    if state is None:
        state = InversionWorkflowState()
    if out_mat:
        state.out_mat = out_mat
    if out_png:
        state.out_png = out_png
    if out_mat_step2:
        state.out_mat_step2 = out_mat_step2
    if out_png_step2:
        state.out_png_step2 = out_png_step2

    load_config_and_fault(state, config_dir, dip_per_segment=dip_per_segment)

    for name in _phases_in_range(start, end):
        print("=== %s ===" % name, flush=True)
        if name == "downsample":
            run_downsample(state, skip=skip_downsample, save_plot=make_insar_save_plot)
        elif name == "invert1":
            invert_round(state, which=1, show_figure=show_figure)
        elif name == "resamp":
            run_resamp(
                state, dec=dec, patch_workers=patch_workers,
                checkpoint_slip_mat=checkpoint_slip_mat,
            )
        elif name == "invert2":
            invert_round(state, which=2, show_figure=show_figure)
    return state
