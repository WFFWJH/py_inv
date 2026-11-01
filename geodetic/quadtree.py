"""Shared Python implementation of the InSAR quadtree partition."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


_BLOCK_COVERAGE_MIN = 0.4
_SINGLE_PIXEL_SPLIT_RMS = 1.0e30


def decompose_quadtree(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    threshold: float,
    nres_min: int,
    nres_max: int,
    *,
    method: str = "mean",
    min_stop_mode: str = "or",
    representative: str = "mean",
) -> List[Tuple[float, float, float, int, float, float, float, float, float]]:
    """Partition a gridded field and return one nine-column row per accepted block.

    The stopping mode and representative are explicit so legacy downsampling and
    the inversion workflow can share the partitioner without changing defaults.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"z shape {z.shape} does not match (ny={y.size}, nx={x.size})"
        )
    if method not in ("mean", "trend"):
        raise ValueError(f"method must be 'mean' or 'trend', got {method!r}")
    if min_stop_mode not in ("and", "or"):
        raise ValueError(
            f"min_stop_mode must be 'and' or 'or', got {min_stop_mode!r}"
        )
    if representative not in ("mean", "median_pixel"):
        raise ValueError(
            "representative must be 'mean' or 'median_pixel', "
            f"got {representative!r}"
        )
    if nres_min < 1 or nres_max < 1:
        raise ValueError("nres_min and nres_max must be positive")

    stack = [(np.arange(x.size), np.arange(y.size))]
    output = []

    while stack:
        x_idx, y_idx = stack.pop()
        z_sub = z[np.ix_(y_idx, x_idx)]
        valid = ~np.isnan(z_sub)
        n_good = int(valid.sum())
        if n_good == 0:
            continue

        lx, ly = len(x_idx), len(y_idx)
        good_ratio = n_good / (lx * ly)
        z_data = z_sub[valid]
        rows, cols = np.nonzero(valid)
        x_data = x[x_idx[cols]]
        y_data = y[y_idx[rows]]
        x_mean = float(x_data.mean())
        y_mean = float(y_data.mean())
        z_mean = float(z_data.mean())

        rms_true = float(np.sqrt(np.mean((z_data - z_mean) ** 2)))
        if n_good == 1:
            rms_for_split = _SINGLE_PIXEL_SPLIT_RMS
        elif method == "trend":
            design = np.column_stack((x_data, y_data, np.ones(n_good)))
            coeffs, *_ = np.linalg.lstsq(design, z_data, rcond=None)
            residual = z_data - design @ coeffs
            rms_for_split = float(np.sqrt(np.mean(residual**2)))
        else:
            rms_for_split = rms_true

        can_split_x, can_split_y = lx >= 2, ly >= 2
        min_reached = (
            lx <= nres_min and ly <= nres_min
            if min_stop_mode == "and"
            else lx <= nres_min or ly <= nres_min
        )
        need_split = False
        if min_reached or not (can_split_x or can_split_y):
            if good_ratio <= _BLOCK_COVERAGE_MIN:
                continue
        elif lx > 2 and lx < nres_max and ly > 2 and ly < nres_max:
            if rms_for_split > threshold:
                need_split = True
            elif good_ratio <= _BLOCK_COVERAGE_MIN:
                continue
        else:
            need_split = True

        if need_split and (can_split_x or can_split_y):
            x_mid, y_mid = lx // 2, ly // 2
            if can_split_x and can_split_y:
                for ys_part in (y_idx[:y_mid], y_idx[y_mid:]):
                    for xs_part in (x_idx[:x_mid], x_idx[x_mid:]):
                        if xs_part.size and ys_part.size:
                            stack.append((xs_part, ys_part))
            elif can_split_x:
                stack.extend(
                    (xs_part, y_idx)
                    for xs_part in (x_idx[:x_mid], x_idx[x_mid:])
                    if xs_part.size
                )
            else:
                stack.extend(
                    (x_idx, ys_part)
                    for ys_part in (y_idx[:y_mid], y_idx[y_mid:])
                    if ys_part.size
                )
            continue

        if representative == "median_pixel" and z_data.size > 1:
            z_median = float(np.median(z_data))
            distances = np.abs(z_data - z_median)
            tolerance = 1e-12 * max(abs(z_median), 1.0)
            candidates = np.flatnonzero(distances <= distances.min() + tolerance)
            candidate_x = x[x_idx[cols[candidates]]]
            candidate_y = y[y_idx[rows[candidates]]]
            nearest = int(
                np.argmin(
                    (candidate_x - x_mean) ** 2 + (candidate_y - y_mean) ** 2
                )
            )
            selected = int(candidates[nearest])
            x_out = float(x[x_idx[cols[selected]]])
            y_out = float(y[y_idx[rows[selected]]])
            z_out = float(z_data[selected])
        else:
            x_out, y_out, z_out = x_mean, y_mean, z_mean

        output.append(
            (
                x_out,
                y_out,
                z_out,
                n_good,
                rms_true,
                float(x[x_idx[0]]),
                float(x[x_idx[-1]]),
                float(y[y_idx[0]]),
                float(y[y_idx[-1]]),
            )
        )

    return output
