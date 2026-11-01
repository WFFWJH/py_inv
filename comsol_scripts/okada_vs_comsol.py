"""Compare single-patch Okada and COMSOL displacements at four AZO tracks' samples.

List the available fault patches:
    python -m comsol_scripts.okada_vs_comsol --list-patches

Compare one patch (segment/layer/patch IDs are one-based):
    python -m comsol_scripts.okada_vs_comsol --segment 2 --layer 3 --patch 5

Patch IDs start at the second endpoint in the fault file (COMSOL's x1/y1 origin)
and proceed toward its first endpoint. ``patch_index.csv`` records the
matching Okada row, COMSOL model role, rectangle position, and rectangle size.
Layer 1 uses the top model; all deeper layers use the no-top model.

``--all-patches`` runs a separate COMSOL solve for every fault patch and may take
many hours. Point-wise results and figures are saved under
``comsol_scripts/okada_vs_comsol_out`` by default.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from geodetic.config import WorkflowConfig
from geodetic.coords import ll2xy, xy2ll, xy2xy
from geodetic.okada import calc_okada
from workflows import paragram


PatchKey = Tuple[int, int]


def _patch_row_index(
    groups: Dict[PatchKey, np.ndarray],
    segment: int,
    layer: int,
    patch: int,
) -> int:
    key = (segment, layer)
    if key not in groups:
        raise ValueError(f"No patches exist for segment={segment}, layer={layer}")
    rows = groups[key]
    if patch < 1 or patch > rows.size:
        raise ValueError(
            f"Patch {patch} is out of range for segment={segment}, layer={layer}; "
            f"valid patch IDs are 1..{rows.size}"
        )
    return paragram.patch_mappings[(segment, layer, patch)].row_index


def _write_patch_index(
    output_file: Path,
    slip_model: np.ndarray,
    groups: Dict[PatchKey, np.ndarray],
    cfg: WorkflowConfig,
) -> None:
    x_origin, y_origin = ll2xy(cfg.lonc, cfg.latc, cfg.ref_lon)
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "segment_id",
                "layer_id",
                "patch_id_from_comsol_origin",
                "comsol_model_role",
                "comsol_green_column_0based",
                "comsol_position_x_m",
                "comsol_position_y_m",
                "comsol_length_m",
                "comsol_width_m",
                "slip_model_row_0based",
                "fault_geometry_patch_id_from_segment_start",
                "center_x_m",
                "center_y_m",
                "center_longitude",
                "center_latitude",
                "center_depth_m",
                "length_m",
                "width_m",
            ]
        )
        for (segment, layer), rows in sorted(groups.items()):
            for patch_id in range(1, rows.size + 1):
                mapping = paragram.patch_mappings[(segment, layer, patch_id)]
                row_index = mapping.row_index
                patch = slip_model[row_index]
                theta = np.pi / 2.0 - np.radians(patch[8])
                along_x, along_y = xy2xy(
                    np.asarray(patch[6] * 0.5), np.asarray(0.0), -theta
                )
                down_x, down_y = xy2xy(
                    np.asarray(0.0),
                    np.asarray(-patch[7] * 0.5 * np.cos(np.radians(patch[9]))),
                    -theta,
                )
                center_x = patch[3] + float(along_x) + float(down_x)
                center_y = patch[4] + float(along_y) + float(down_y)
                center_depth = -patch[5] + patch[7] * 0.5 * np.sin(
                    np.radians(patch[9])
                )
                lon, lat = xy2ll(
                    x_origin + center_x, y_origin + center_y, cfg.ref_lon
                )
                writer.writerow(
                    [
                        segment,
                        layer,
                        patch_id,
                        mapping.model_role,
                        mapping.green_column,
                        f"{mapping.position_x:.12g}",
                        f"{mapping.position_y:.12g}",
                        f"{mapping.length:.12g}",
                        f"{mapping.width:.12g}",
                        row_index,
                        int(round(patch[1])),
                        f"{center_x:.12g}",
                        f"{center_y:.12g}",
                        f"{lon:.12g}",
                        f"{lat:.12g}",
                        f"{center_depth:.12g}",
                        f"{patch[6]:.12g}",
                        f"{patch[7]:.12g}",
                    ]
                )


def _set_comsol_slip_bc(model, fault_type: int) -> None:
    solid = model.component("comp1").physics("solid")
    if fault_type == 1:
        direction = [["free"], ["prescribed"], ["free"]]
        displacements = ([[0], [0.5], [0]], [[0], [0.5], [0]])
    else:
        direction = [["prescribed"], ["free"], ["free"]]
        displacements = ([[-0.5], [0], [0]], [[0.5], [0], [0]])
    for feature_name, displacement in zip(("disp1", "disp2"), displacements):
        feature = solid.feature(feature_name)
        feature.set("Direction", direction)
        feature.set("U0", displacement)


def _run_comsol(
    client,
    model_path: Path,
    segment_endpoints: np.ndarray,
    patch_size: Tuple[float, float],
    patch_position: Tuple[float, float],
    fault_type: int,
    coords: np.ndarray,
) -> np.ndarray:
    """Return one patch's COMSOL E/N/U displacements at all sample coordinates."""
    model_java = client.load(str(model_path))
    try:
        model = model_java.java
        params = model.param()
        params.set("x0", f"{segment_endpoints[0]}[m]")
        params.set("y0", f"{segment_endpoints[1]}[m]")
        params.set("x1", f"{segment_endpoints[2]}[m]")
        params.set("y1", f"{segment_endpoints[3]}[m]")

        rectangle = (
            model.component("comp1")
            .geom("geom1")
            .feature("wp2")
            .geom()
            .feature("r1")
        )
        rectangle.set("size", [str(patch_size[0]), str(patch_size[1])])
        rectangle.set("pos", [str(patch_position[0]), str(patch_position[1])])
        _set_comsol_slip_bc(model, fault_type)
        model.sol("sol1").runAll()

        model.result().numerical().create("uvw", "Interp")
        interp = model.result().numerical("uvw")
        interp.setInterpolationCoordinates(coords)
        interp.set("expr", ["u", "v", "w"])
        values = np.asarray(interp.getData(), dtype=np.float64)
        return np.column_stack((values[0, 0], values[1, 0], values[2, 0]))
    finally:
        client.remove(model_java)


def _run_okada(
    patch: np.ndarray,
    coords: np.ndarray,
    fault_type: int,
    nu: float,
) -> np.ndarray:
    xp, yp, zp, length, width, strike_deg, dip_deg = patch[3:10]
    strike = np.radians(strike_deg)
    delta = np.radians(dip_deg)
    theta = np.pi / 2.0 - strike
    dx, dy = xy2xy(np.asarray(length * 0.5), np.asarray(0.0), -theta)
    east, north, up = calc_okada(
        1.0,
        paragram.unit_slip,
        coords[0] - (xp + float(dx)),
        coords[1] - (yp + float(dy)),
        nu,
        delta,
        max(-float(zp), 1e-10),
        length,
        width,
        fault_type,
        strike,
        np.zeros(coords.shape[1], dtype=np.float64),
        backend="auto",
    )
    return np.column_stack((east, north, up))


def _project_azimuth(uvw: np.ndarray, look: np.ndarray) -> np.ndarray:
    azimuth = -np.arctan2(look[:, 1], look[:, 0]) - np.pi
    return uvw[:, 0] * np.sin(azimuth) + uvw[:, 1] * np.cos(azimuth)


def _write_comparison_csv(
    output_file: Path,
    patch_id: Tuple[int, int, int],
    row_index: int,
    samples: np.ndarray,
    tracks: np.ndarray,
    point_indices: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    okada: np.ndarray,
    comsol: np.ndarray,
) -> None:
    okada_az = _project_azimuth(okada, samples[:, 3:6])
    comsol_az = _project_azimuth(comsol, samples[:, 3:6])
    columns = [
        "track",
        "sample_index_0based",
        "segment_id",
        "layer_id",
        "patch_id_from_comsol_origin",
        "slip_model_row_0based",
        "comsol_green_column_0based",
        "longitude",
        "latitude",
        "local_x_m",
        "local_y_m",
        "observed_azimuth_cm",
        "observed_azimuth_m",
        "okada_e_m",
        "okada_n_m",
        "okada_u_m",
        "comsol_e_m",
        "comsol_n_m",
        "comsol_u_m",
        "difference_e_m",
        "difference_n_m",
        "difference_u_m",
        "okada_azimuth_m",
        "comsol_azimuth_m",
        "difference_azimuth_m",
    ]
    segment, layer, patch = patch_id
    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(samples.shape[0]):
            writer.writerow(
                [
                    tracks[index],
                    int(point_indices[index]),
                    segment,
                    layer,
                    patch,
                    row_index,
                    paragram.patch_mappings[
                        (segment, layer, patch)
                    ].green_column,
                    f"{longitude[index]:.12g}",
                    f"{latitude[index]:.12g}",
                    f"{samples[index, 0]:.12g}",
                    f"{samples[index, 1]:.12g}",
                    f"{samples[index, 2]:.12g}",
                    f"{samples[index, 2] / 100.0:.12g}",
                    *[f"{value:.12g}" for value in okada[index]],
                    *[f"{value:.12g}" for value in comsol[index]],
                    *[f"{value:.12g}" for value in comsol[index] - okada[index]],
                    f"{okada_az[index]:.12g}",
                    f"{comsol_az[index]:.12g}",
                    f"{comsol_az[index] - okada_az[index]:.12g}",
                ]
            )


def _plot_comparison(
    output_file: Path,
    patch_label: str,
    samples: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    okada: np.ndarray,
    comsol: np.ndarray,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    okada_az = _project_azimuth(okada, samples[:, 3:6])
    comsol_az = _project_azimuth(comsol, samples[:, 3:6])
    fields = [
        ("East (m)", okada[:, 0], comsol[:, 0]),
        ("North (m)", okada[:, 1], comsol[:, 1]),
        ("Up (m)", okada[:, 2], comsol[:, 2]),
        ("Azimuth projection (m)", okada_az, comsol_az),
    ]
    fig, axes = plt.subplots(
        len(fields), 3, figsize=(15, 16), constrained_layout=True
    )
    fig.suptitle(f"Okada vs COMSOL at four tracks' sample points\n{patch_label}")
    for row, (label, oka, cos) in enumerate(fields):
        maximum = float(max(np.max(np.abs(oka)), np.max(np.abs(cos))))
        maximum = maximum if maximum > 0 else 1.0
        difference = cos - oka
        error_max = float(np.max(np.abs(difference)))
        error_max = error_max if error_max > 0 else 1.0
        for column, (values, title, limits) in enumerate(
            (
                (oka, "Okada", (-maximum, maximum)),
                (cos, "COMSOL", (-maximum, maximum)),
                (difference, "COMSOL - Okada", (-error_max, error_max)),
            )
        ):
            scatter = axes[row, column].scatter(
                longitude,
                latitude,
                c=values,
                cmap="jet",
                vmin=limits[0],
                vmax=limits[1],
                s=7,
            )
            axes[row, column].set_title(f"{label}: {title}")
            axes[row, column].set_xlabel("Longitude (deg)")
            axes[row, column].set_ylabel("Latitude (deg)")
            axes[row, column].set_aspect("equal", adjustable="datalim")
            axes[row, column].grid(True, alpha=0.25)
            fig.colorbar(scatter, ax=axes[row, column], shrink=0.8, label=label)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def _make_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", type=int, default=1, help="One-based fault segment ID.")
    parser.add_argument("--layer", type=int, default=1, help="One-based down-dip layer ID.")
    parser.add_argument(
        "--patch",
        type=int,
        default=1,
        help=(
            "One-based patch ID from fault_trace.txt's second endpoint "
            "(COMSOL x1/y1 origin), toward the first endpoint."
        ),
    )
    parser.add_argument(
        "--fault-type", type=int, choices=(1, 2), default=1
    )
    parser.add_argument("--nu", type=float, default=0.25)
    parser.add_argument("--all-patches", action="store_true")
    parser.add_argument("--list-patches", action="store_true")
    parser.add_argument("--show", action="store_true", help="Display the saved figure.")
    parser.add_argument(
        "--top-model",
        type=Path,
        default=root / "touch.mph",
        help="COMSOL template for layer 1.",
    )
    parser.add_argument(
        "--deep-model",
        type=Path,
        default=root / "touch_no_top.mph",
        help="COMSOL template for layers 2 and deeper.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "comsol_scripts" / "okada_vs_comsol_out",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    root_path = _ROOT
    args = _make_parser(root_path).parse_args(argv)
    context = paragram.load_paragram_context()
    cfg = context.cfg
    slip_model = context.slip_model
    groups = context.patch_groups

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_file = args.output_dir / "patch_index.csv"
    _write_patch_index(index_file, slip_model, groups, cfg)
    if args.list_patches:
        for (segment, layer), rows in sorted(groups.items()):
            print(f"segment={segment} layer={layer} patches={rows.size}")
        print(f"Patch location table: {index_file}")
        return 0

    if args.all_patches:
        patch_ids = [
            (segment, layer, patch)
            for (segment, layer), rows in sorted(groups.items())
            for patch in range(1, rows.size + 1)
        ]
    else:
        patch_ids = [(args.segment, args.layer, args.patch)]

    resolved_patch_ids: List[Tuple[int, int, int, int]] = []
    for segment, layer, patch_id in patch_ids:
        row_index = _patch_row_index(groups, segment, layer, patch_id)
        mapping = paragram.patch_mappings[(segment, layer, patch_id)]
        model_path = (
            args.top_model if mapping.model_role == "top" else args.deep_model
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"COMSOL model not found: {model_path}")
        resolved_patch_ids.append((segment, layer, patch_id, row_index))

    samples = paragram.sampled_data
    coords = paragram.coords
    tracks = paragram.sampled_tracks
    point_indices = paragram.sampled_point_indices
    longitude = paragram.sampled_longitude
    latitude = paragram.sampled_latitude

    import matplotlib.pyplot as plt
    import mph

    client = mph.start()
    try:
        for segment, layer, patch_id, row_index in resolved_patch_ids:
            patch = slip_model[row_index]
            mapping = paragram.patch_mappings[(segment, layer, patch_id)]
            model_path = (
                args.top_model if mapping.model_role == "top" else args.deep_model
            )

            segment_endpoints = paragram.fault_segments_local[segment - 1]
            okada = _run_okada(patch, coords, args.fault_type, args.nu)

            label = (
                f"segment_{segment:02d}_layer_{layer:02d}_patch_{patch_id:03d}"
                f"_type_{args.fault_type}"
            )
            print(
                f"{label}: slip_model_row={row_index}, "
                f"COMSOL model={mapping.model_role} ({model_path.name}), "
                f"position=({mapping.position_x:.1f}, "
                f"{mapping.position_y:.1f}) m, "
                f"size=({mapping.length:.1f}, {mapping.width:.1f}) m, "
                f"samples={samples.shape[0]}",
                flush=True,
            )
            comsol = _run_comsol(
                client,
                model_path,
                segment_endpoints,
                (mapping.length, mapping.width),
                (mapping.position_x, mapping.position_y),
                args.fault_type,
                coords,
            )

            csv_path = args.output_dir / f"{label}.csv"
            figure_path = args.output_dir / f"{label}.png"
            _write_comparison_csv(
                csv_path,
                (segment, layer, patch_id),
                row_index,
                samples,
                tracks,
                point_indices,
                longitude,
                latitude,
                okada,
                comsol,
            )
            _plot_comparison(
                figure_path,
                label,
                samples,
                longitude,
                latitude,
                okada,
                comsol,
                args.show,
            )
            print(f"Saved {csv_path} and {figure_path}", flush=True)

    finally:
        client.clear()

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
