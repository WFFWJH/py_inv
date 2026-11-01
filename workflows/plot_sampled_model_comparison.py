"""Export and plot sampled InSAR observations against the saved model values.

Run from the repository root with ``python -m workflows.plot_sampled_model_comparison``.
The default inputs are ``input_data/data_v5_opt1/<track>/los_samp0.mat`` and
``los_model.mat``. Outputs are written to ``input_data/data_v5_opt1/comparison_output``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from geodetic.coords import ll2xy, xy2ll


def load_comparison_data(
    sampled_file: Path,
    model_file: Path,
    *,
    ref_lon: float,
    lonc: float,
    latc: float,
) -> np.ndarray:
    """Return rows of longitude, latitude, observed LOS, and modeled LOS."""
    sampled_values = loadmat(sampled_file).get("sampled_insar_data")
    model_values = loadmat(model_file).get("sampled_model")
    if sampled_values is None:
        raise KeyError(f"sampled_insar_data not found in {sampled_file}")
    if model_values is None:
        raise KeyError(f"sampled_model not found in {model_file}")

    sampled = np.asarray(sampled_values, dtype=np.float64)
    model = np.asarray(model_values, dtype=np.float64)
    if sampled.ndim != 2 or sampled.shape[1] < 3:
        raise ValueError(f"{sampled_file}: expected sampled_insar_data with shape (N, >=3), got {sampled.shape}")
    if model.ndim != 2 or model.shape[1] < 3:
        raise ValueError(f"{model_file}: expected sampled_model with shape (N, >=3), got {model.shape}")
    if sampled.shape[0] != model.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {sampled_file} has {sampled.shape[0]} rows, "
            f"{model_file} has {model.shape[0]}"
        )
    if not np.allclose(sampled[:, :2], model[:, :2], rtol=0.0, atol=1e-8):
        raise ValueError(
            f"Coordinates/order differ between {sampled_file} and {model_file}; "
            "cannot safely pair LOS values by row"
        )

    x_origin, y_origin = ll2xy(lonc, latc, ref_lon)
    longitude, latitude = xy2ll(
        x_origin + sampled[:, 0],
        y_origin + sampled[:, 1],
        ref_lon,
    )
    return np.column_stack(
        [longitude, latitude, sampled[:, 2], model[:, 2]]
    )


def plot_track_comparison(
    data: np.ndarray,
    track: str,
    csv_path: Path,
    figure_path: Path,
) -> None:
    """Save a four-column data table and a spatial observation/model/residual plot."""
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="longitude,latitude,los_observed,los_model",
        comments="",
        fmt="%.17g",
    )

    longitude, latitude = data[:, 0], data[:, 1]
    observed, modeled = data[:, 2], data[:, 3]
    fields = (observed, modeled, observed - modeled)
    titles = ("Observed LOS", "Modeled LOS", "Residual (observed - model)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, values, title in zip(axes, fields, titles):
        points = ax.scatter(
            longitude,
            latitude,
            c=values,
            cmap="jet",
            s=5,
        )
        fig.colorbar(points, ax=ax, shrink=0.8, label="LOS")
        ax.set_title(f"{track}: {title}")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)

    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "input_data" / "data_v5_opt1",
        help="Folder containing one subfolder per track.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "input_data" / "data_v5_opt1" / "comparison_output",
        help="Folder for per-track CSV files and comparison plots.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--ref-lon", type=float, default=95.0)
    parser.add_argument("--lonc", type=float, default=95.33)
    parser.add_argument("--latc", type=float, default=19.61)
    args = parser.parse_args(argv)

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Track data directory not found: {args.data_root}")
    output_dir = args.output_dir.resolve()
    track_dirs = sorted(
        path
        for path in args.data_root.iterdir()
        if path.is_dir() and path.resolve() != output_dir
    )
    if not track_dirs:
        raise FileNotFoundError(f"No track folders found in {args.data_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for track_dir in track_dirs:
        sampled_file = track_dir / f"los_samp{args.sample_index}.mat"
        model_file = track_dir / "los_model.mat"
        for required_file in (sampled_file, model_file):
            if not required_file.is_file():
                raise FileNotFoundError(required_file)

        data = load_comparison_data(
            sampled_file,
            model_file,
            ref_lon=args.ref_lon,
            lonc=args.lonc,
            latc=args.latc,
        )
        csv_path = args.output_dir / f"{track_dir.name}_los_samp{args.sample_index}.csv"
        figure_path = args.output_dir / f"{track_dir.name}_los_samp{args.sample_index}_comparison.png"
        plot_track_comparison(data, track_dir.name, csv_path, figure_path)
        print(f"{track_dir.name}: {data.shape[0]} points -> {csv_path}, {figure_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
