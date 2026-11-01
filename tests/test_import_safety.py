import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportSafetyTests(unittest.TestCase):
    def run_in_clean_interpreter(self, source):
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_geodetic_package_import_is_lazy(self):
        self.run_in_clean_interpreter(
            "import sys, geodetic; "
            "assert 'numpy' not in sys.modules; "
            "assert 'geodetic.okada' not in sys.modules"
        )

    def test_comsol_context_is_not_loaded_by_import(self):
        self.run_in_clean_interpreter(
            "import sys, workflows.paragram as p; "
            "assert 'cfg' not in vars(p); "
            "assert 'workflows.comsol_context' not in sys.modules"
        )

    def test_checkerboard_config_import_is_lazy(self):
        self.run_in_clean_interpreter(
            "import sys, workflows.paragram_checkboard as p; "
            "assert 'numpy' not in sys.modules; "
            "assert p.n_patch == 20; "
            "assert p.xe.shape == p.yn.shape == (208600,)"
        )

    def test_comsol_command_imports_do_not_start_comsol(self):
        self.run_in_clean_interpreter(
            "import importlib, sys; "
            "names = ('analysis_comsol', 'comsol_green', "
            "'comsol_gui_vs_java', 'comsol_self_consistent', "
            "'comsol_slip_smooth', 'comsol_vs_comsol', "
            "'export_okada_vs_comsol_png', 'Green_diff_copy', "
            "'okada_vs_comsol_copy', 'comsol_run', 'okada_vs_comsol'); "
            "modules = [importlib.import_module('comsol_scripts.' + name) "
            "for name in names]; "
            "assert all(callable(module.main) for module in modules); "
            "assert 'mph' not in sys.modules"
        )


if __name__ == "__main__":
    unittest.main()
