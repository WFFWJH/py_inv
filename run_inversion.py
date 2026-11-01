r"""Main entry for the geodetic inversion workflow.

Run the workflow from a directory containing the configuration files::

    py-inv --start invert1 --end invert1

Phases: ``downsample`` | ``invert1`` | ``resamp`` | ``invert2``
(MATLAB Steps 3 / 5 / 6 / 7). Config + fault geometry always load first.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from geodetic.config import load_workflow_config
from workflows.inversion import InversionWorkflowState, run


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", help="Directory containing workflow config files.")
    parser.add_argument(
        "--start",
        default=None,
        choices=("downsample", "invert1", "resamp", "invert2"),
        help="First workflow phase.",
    )
    parser.add_argument(
        "--end",
        default=None,
        choices=("downsample", "invert1", "resamp", "invert2"),
        help="Last workflow phase.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write inversion snapshots and figures to this directory.",
    )
    parser.add_argument(
        "--plot-gmt",
        action="store_true",
        help="Export and plot each inversion result with GMT.",
    )
    parser.add_argument(
        "--rms-weight",
        action="store_true",
        help="Weight observations by their sampled-block RMS uncertainty.",
    )
    parser.add_argument(
        "--surface-elevation",
        action="store_true",
        help="Use sampled DEM elevations in the Okada Green function.",
    )
    args = parser.parse_args(argv)

    CONFIG_DIR: Optional[Union[str, os.PathLike]] = args.config_dir
    START = args.start or "invert1"
    END = args.end or START
    phase_order = ("downsample", "invert1", "resamp", "invert2")
    if phase_order.index(START) > phase_order.index(END):
        parser.error("--start must not come after --end")
    SKIP_DOWNSAMPLE = False
    OUT_MAT: Optional[str] = None
    OUT_PNG: Optional[str] = None
    OUT_MAT_STEP2: Optional[str] = None
    OUT_PNG_STEP2: Optional[str] = None
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _, cfg = load_workflow_config(CONFIG_DIR)
        first_stem = args.output_dir / f"py_inversion_iint{cfg.iter_step}"
        second_stem = args.output_dir / f"py_inversion_iint{cfg.iter_step2}"
        OUT_MAT = str(first_stem.with_suffix(".mat"))
        OUT_PNG = str(first_stem.with_name(first_stem.name + "_show.png"))
        OUT_MAT_STEP2 = str(second_stem.with_suffix(".mat"))
        OUT_PNG_STEP2 = str(
            second_stem.with_name(second_stem.name + "_show_step7.png")
        )
    MAKE_INSAR_SAVE_PLOT = True
    SHOW_FIGURE = not args.no_show
    PLOT_GMT = args.plot_gmt
    USE_RMS_WEIGHT = args.rms_weight
    USE_TOPOGRAPHY = args.surface_elevation
    RESAMP_DEC = 1
    RESAMP_PATCH_WORKERS: Optional[int] = 4
    CHECKPOINT_SLIP_MAT: Optional[str] = None

    state = run(
        InversionWorkflowState(),
        start=START,
        end=END,
        config_dir=CONFIG_DIR,
        skip_downsample=SKIP_DOWNSAMPLE,
        make_insar_save_plot=MAKE_INSAR_SAVE_PLOT,
        show_figure=SHOW_FIGURE,
        plot_gmt=PLOT_GMT,
        use_rms_weight=USE_RMS_WEIGHT,
        use_topography=USE_TOPOGRAPHY,
        dec=RESAMP_DEC,
        patch_workers=RESAMP_PATCH_WORKERS,
        checkpoint_slip_mat=CHECKPOINT_SLIP_MAT,
        out_mat=OUT_MAT,
        out_png=OUT_PNG,
        out_mat_step2=OUT_MAT_STEP2,
        out_png_step2=OUT_PNG_STEP2,
    )
    print("workflow 完成 (%s .. %s)." % (START, END), flush=True)
    if state.slip is not None:
        print("  slip1  -> %s  rms=%s" % (state.out_mat, state.rms), flush=True)
    if state.slip2 is not None:
        print("  slip2  -> %s  rms=%s" % (state.out_mat_step2, state.rms2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
