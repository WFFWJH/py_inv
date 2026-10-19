r"""Main entry for the geodetic inversion workflow.

Edit the variables below, then::

    python run_inversion.py

Phases: ``downsample`` | ``invert1`` | ``resamp`` | ``invert2``
(MATLAB Steps 3 / 5 / 6 / 7). Config + fault geometry always load first.
"""
from __future__ import annotations

from typing import Optional, Union
import os

from workflows.inversion import InversionWorkflowState, run


def main() -> None:
    CONFIG_DIR: Optional[Union[str, os.PathLike]] = None
    # Prefer phase names; integers 3/5/6/7 still accepted
    START = "invert1"
    END = "invert1"
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

    state = run(
        InversionWorkflowState(),
        start=START,
        end=END,
        config_dir=CONFIG_DIR,
        skip_downsample=SKIP_DOWNSAMPLE,
        make_insar_save_plot=MAKE_INSAR_SAVE_PLOT,
        show_figure=SHOW_FIGURE,
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


if __name__ == "__main__":
    main()
