"""Port of build_green_function.m (only 'insar' and 'AZO' paths; GPS skipped).

Returns G_raw, G, bdata_raw, bdata where:
  G_raw = [okada_green | ramp]            shape (n_obs, 2*Np + add_col)
  G     = diag(w) * G_raw                 same shape
  bdata_raw = sampled LOS / AZO value     shape (n_obs,)
  bdata = diag(w) * bdata_raw

Default weighting is uniform. Optional RMS weighting uses
``w_i = (1/rms_i) / sum(1/rms_i)`` with a 1 mm floor for zero/small RMS.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.io import loadmat

from geodetic.green import calc_green_AZO_okada, calc_green_insar_okada


def _build_ramp(sampled: np.ndarray, dem_out: np.ndarray, ramp_choice: str) -> np.ndarray:
    """Return ramp columns (m, k) or (m, 0) when no ramp."""
    ramp = ramp_choice.lower() if ramp_choice else "none"
    xsar = sampled[:, 0] / 1000.0
    ysar = sampled[:, 1] / 1000.0
    dem = np.asarray(dem_out, dtype=np.float64).reshape(-1) / 1000.0
    ones = np.ones_like(xsar)
    if ramp == "bi_ramp":
        return np.column_stack([xsar, ysar, dem, ones])
    if ramp == "qu_ramp_7":
        return np.column_stack([xsar * xsar, ysar * ysar, xsar * ysar,
                                xsar, ysar, dem, ones])
    if ramp == "qu_ramp_5":
        return np.column_stack([xsar * ysar, xsar, ysar, dem, ones])
    return np.zeros((xsar.size, 0), dtype=np.float64)


def build_green_function(slip_model: np.ndarray,
                         sampled_data_file: str,
                         option: str,
                         ramp_choice: str,
                         model_type: str = "okada",
                         nu: float = 0.25,
                         backend: str = "auto",
                         *,
                         use_rms_weight: bool = False,
                         use_topography: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(sampled_data_file)
    sampled = np.asarray(data["sampled_insar_data"], dtype=np.float64)
    if sampled.ndim != 2 or sampled.shape[1] < 6:
        raise ValueError(
            "%s: sampled_insar_data must have shape (N, >=6), got %r"
            % (sampled_data_file, sampled.shape)
        )
    if "dem_out" not in data:
        if use_topography:
            raise KeyError(
                "%s does not contain 'dem_out'; topography was requested."
                % sampled_data_file
            )
        warnings.warn(
            "%s does not contain 'dem_out'; using zeros for the ramp DEM column."
            % sampled_data_file,
            UserWarning,
            stacklevel=2,
        )
        dem_out = np.zeros(sampled.shape[0], dtype=np.float64)
    else:
        dem_out = np.asarray(data["dem_out"], dtype=np.float64).reshape(-1)
        if dem_out.size != sampled.shape[0] or not np.all(np.isfinite(dem_out)):
            raise ValueError(
                "%s: dem_out must contain one finite value per observation"
                % sampled_data_file
            )

    surface_elevation = dem_out if use_topography else None

    if model_type.lower() != "okada":
        raise NotImplementedError("Only 'okada' model_type is supported in this Python port.")

    opt = option.lower()
    if opt == "insar":
        G_green = calc_green_insar_okada(
            slip_model, sampled, nu=nu, backend=backend,
            surface_elevation=surface_elevation,
        )
    elif opt == "azo":
        G_green = calc_green_AZO_okada(
            slip_model, sampled, nu=nu, backend=backend,
            surface_elevation=surface_elevation,
        )
    else:
        raise ValueError(f"Unsupported option: {option!r}; only 'insar' and 'AZO' are implemented.")

    bdata_raw = np.asarray(sampled[:, 2], dtype=np.float64)

    ramp = _build_ramp(sampled, dem_out, ramp_choice)
    G_raw = np.concatenate([G_green, ramp], axis=1).astype(np.float64, copy=False)

    h1 = sampled.shape[0]
    if use_rms_weight:
        if "rms_out" not in data:
            raise KeyError(
                "%s does not contain 'rms_out'; RMS weighting was requested."
                % sampled_data_file
            )
        rms_insar = np.asarray(data["rms_out"], dtype=np.float64).reshape(-1)
        if (
            rms_insar.size != h1
            or not np.all(np.isfinite(rms_insar))
            or np.any(rms_insar < 0.0)
        ):
            raise ValueError(
                "%s: rms_out must contain one finite non-negative value per observation"
                % sampled_data_file
            )
        rms_insar = np.maximum(rms_insar, 1e-3)
    else:
        rms_insar = np.ones(h1, dtype=np.float64)
    w_data = 1.0 / rms_insar
    w_vec = w_data / w_data.sum()

    G = G_raw * w_vec[:, None]
    bdata = bdata_raw * w_vec

    return G_raw, G, bdata_raw, bdata
