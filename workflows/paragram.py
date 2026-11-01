"""Lazy compatibility facade for the main-workflow COMSOL context.

Importing this module is safe; configuration and MAT data are read only when a
legacy COMSOL script accesses one of the exported attributes.
"""
from __future__ import annotations

def __getattr__(name: str):
    if name == "PatchMapping":
        from workflows.comsol_context import PatchMapping

        globals()[name] = PatchMapping
        return PatchMapping
    if name not in _CONTEXT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from workflows.comsol_context import load_paragram_context

    try:
        value = getattr(load_paragram_context(), name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    globals()[name] = value
    return value


def load_paragram_context(config_dir=None):
    from workflows.comsol_context import load_paragram_context as load_context

    return load_context(config_dir)


_CONTEXT_NAMES = {
    "_root", "cfg", "ref_lon", "lonc", "latc", "ref", "axis_range", "dip",
    "width", "len_top", "l_ratio", "w_ratio", "layers", "fault_file",
    "tracks", "npt", "region", "dtypes", "nmin", "nmax", "iint", "nu",
    "fault_type", "unit_slip", "fault_segments", "slip_model", "patch_groups",
    "fault_segments_local", "layer_widths", "layer_offsets", "patch_counts",
    "patch_lengths", "patch_mappings", "track_samples", "sampled_data",
    "sampled_tracks", "sampled_point_indices", "xe", "yn", "coords",
    "sampled_longitude", "sampled_latitude", "n_patch", "top0_model",
    "top1_model",
}


def __dir__():
    return sorted(set(globals()) | _CONTEXT_NAMES | {"PatchMapping"})
