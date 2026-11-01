"""Explicit, cached configuration and sampled data for COMSOL comparisons."""
from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace
from typing import NamedTuple, Optional, Union

import numpy as np
from scipy.io import loadmat

from workflows.inversion import (
    InversionWorkflowState,
    step_load_config,
    step_load_fault,
)


class PatchMapping(NamedTuple):
    row_index: int
    model_role: str
    green_column: int
    position_x: float
    position_y: float
    length: float
    width: float


@lru_cache(maxsize=4)
def _load_context(root: str) -> SimpleNamespace:
    state = InversionWorkflowState()
    step_load_config(state, root, quiet=True)
    step_load_fault(state, quiet=True)
    cfg = state.config
    if cfg is None or state.slip_vs is None or state.fault_layout is None:
        raise RuntimeError("Unable to initialize the COMSOL workflow context")

    fault_file = state.fault_abs
    layout = state.fault_layout
    slip_model = state.slip_vs
    ref = dict(ref_lon=cfg.ref_lon, lonc=cfg.lonc, latc=cfg.latc)
    patch_mappings = {}
    for (segment, layer), rows in layout.patch_groups.items():
        start = int(layout.green_column_starts[segment - 1, layer - 1])
        for patch_id, row_index in enumerate(rows, start=1):
            patch = slip_model[row_index]
            patch_mappings[(segment, layer, patch_id)] = PatchMapping(
                row_index=int(row_index),
                model_role="top" if layer == 1 else "deep",
                green_column=start + patch_id - 1,
                position_x=(patch_id - 1) * float(patch[6]),
                position_y=float(layout.layer_offsets[layer - 1]),
                length=float(patch[6]),
                width=float(patch[7]),
            )

    iint = int(cfg.iter_step)
    track_samples = {}
    sampled_parts = []
    track_labels = []
    point_indices = []
    for track, data_type in zip(state.tracks, state.dtypes):
        if data_type not in ("insar", "azo", "azi"):
            raise ValueError(f"Unsupported sample type {data_type!r} for track {track}")
        sample_file = os.path.join(track, f"los_samp{iint}.mat")
        if not os.path.isfile(sample_file):
            raise FileNotFoundError(f"Sample file not found: {sample_file}")
        mat = loadmat(sample_file)
        if "sampled_insar_data" not in mat:
            raise KeyError(f"{sample_file} does not contain 'sampled_insar_data'")
        sample = np.asarray(mat["sampled_insar_data"], dtype=np.float64)
        if sample.ndim != 2 or sample.shape[1] < 6:
            raise ValueError(
                f"{sample_file}: expected sampled_insar_data shape (N, >=6), "
                f"got {sample.shape}"
            )
        if not np.all(np.isfinite(sample[:, :6])):
            raise ValueError(f"{sample_file} contains non-finite sampled data")

        name = os.path.basename(os.path.normpath(track))
        track_samples[name] = sample[:, :6].copy()
        sampled_parts.append(sample[:, :6])
        track_labels.append(np.full(sample.shape[0], name, dtype=object))
        point_indices.append(np.arange(sample.shape[0], dtype=np.int64))

    if not sampled_parts:
        raise ValueError("No track samples were loaded from the configured data list")
    sampled_data = np.concatenate(sampled_parts, axis=0)
    sampled_tracks = np.concatenate(track_labels)
    sampled_point_indices = np.concatenate(point_indices)
    xe, yn = sampled_data[:, 0], sampled_data[:, 1]

    from geodetic.coords import ll2xy, xy2ll

    x_origin, y_origin = ll2xy(cfg.lonc, cfg.latc, cfg.ref_lon)
    sampled_longitude, sampled_latitude = xy2ll(
        x_origin + xe, y_origin + yn, cfg.ref_lon
    )

    return SimpleNamespace(
        _root=state.root,
        cfg=cfg,
        ref_lon=cfg.ref_lon,
        lonc=cfg.lonc,
        latc=cfg.latc,
        ref=ref,
        axis_range=None,
        dip=state.dangles,
        width=cfg.width,
        len_top=cfg.len_top,
        l_ratio=cfg.l_ratio,
        w_ratio=cfg.w_ratio,
        layers=cfg.layers,
        fault_file=fault_file,
        tracks=state.tracks,
        npt=state.npt,
        region=state.region,
        dtypes=state.dtypes,
        nmin=state.nmin,
        nmax=state.nmax,
        iint=iint,
        nu=0.25,
        fault_type=1,
        unit_slip=1.0,
        fault_segments=state.fault_segments,
        slip_model=slip_model,
        patch_groups=layout.patch_groups,
        fault_segments_local=layout.fault_segments_local,
        layer_widths=layout.layer_widths,
        layer_offsets=layout.layer_offsets,
        patch_counts=layout.patch_counts,
        patch_lengths=layout.patch_lengths,
        patch_mappings=patch_mappings,
        track_samples=track_samples,
        sampled_data=sampled_data,
        sampled_tracks=sampled_tracks,
        sampled_point_indices=sampled_point_indices,
        xe=xe,
        yn=yn,
        coords=np.vstack((xe, yn, np.zeros(xe.size, dtype=np.float64))),
        sampled_longitude=sampled_longitude,
        sampled_latitude=sampled_latitude,
        n_patch=int(slip_model.shape[0]),
        top0_model="touch.mph",
        top1_model="touch_no_top.mph",
    )


def load_paragram_context(
    config_dir: Optional[Union[str, os.PathLike]] = None,
) -> SimpleNamespace:
    """Load main-workflow configuration and COMSOL samples only when requested."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.abspath(config_dir) if config_dir is not None else project_root
    return _load_context(root)
