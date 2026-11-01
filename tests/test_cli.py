import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_inversion
from workflows.inversion import InversionWorkflowState


class InversionCliTests(unittest.TestCase):
    def test_default_phase_matches_the_existing_single_inversion_run(self):
        state = InversionWorkflowState()
        with patch.object(run_inversion, "run", return_value=state) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_inversion.main([]), 0)

        self.assertEqual(run.call_args.kwargs["start"], "invert1")
        self.assertEqual(run.call_args.kwargs["end"], "invert1")
        self.assertFalse(run.call_args.kwargs["plot_gmt"])
        self.assertFalse(run.call_args.kwargs["use_rms_weight"])
        self.assertFalse(run.call_args.kwargs["use_topography"])
        self.assertTrue(run.call_args.kwargs["show_figure"])

    def test_output_paths_follow_configured_iteration_numbers(self):
        config = SimpleNamespace(iter_step=5, iter_step2=9)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "results"
            with (
                patch.object(
                    run_inversion,
                    "load_workflow_config",
                    return_value=(temporary, config),
                ),
                patch.object(
                    run_inversion, "run", return_value=InversionWorkflowState()
                ) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                run_inversion.main(
                    [
                        "--config-dir",
                        temporary,
                        "--start",
                        "invert1",
                        "--end",
                        "invert1",
                        "--output-dir",
                        str(output_dir),
                        "--no-show",
                        "--rms-weight",
                        "--surface-elevation",
                    ]
                )
            kwargs = run.call_args.kwargs
            self.assertEqual(
                kwargs["out_mat"], str(output_dir / "py_inversion_iint5.mat")
            )
            self.assertEqual(
                kwargs["out_mat_step2"],
                str(output_dir / "py_inversion_iint9.mat"),
            )
            self.assertTrue(kwargs["use_rms_weight"])
            self.assertTrue(kwargs["use_topography"])
            self.assertTrue(output_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
