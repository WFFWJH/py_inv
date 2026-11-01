import unittest

import numpy as np

from geodetic.fault_geometry import build_fault_layout, load_fault_one_plane


class FaultLayoutTests(unittest.TestCase):
    def test_layout_uses_inversion_patch_table_as_source_of_truth(self):
        fault_segments = np.array(
            [
                [95.0, 19.0, 95.1, 19.0],
                [95.1, 19.0, 95.2, 19.0],
            ],
            dtype=np.float64,
        )
        slip_model = load_fault_one_plane(
            fault_segments,
            dip=[80.0, 85.0],
            lonc=95.1,
            latc=19.0,
            ref_lon=95.0,
            width=20_000.0,
            layers=2,
            len_top=5_000.0,
            l_ratio=1.2,
            w_ratio=1.1,
        )

        layout = build_fault_layout(
            fault_segments,
            slip_model,
            lonc=95.1,
            latc=19.0,
            ref_lon=95.0,
            layers=2,
        )
        self.assertEqual(layout.patch_counts.shape, (2, 2))
        self.assertEqual(int(layout.patch_counts.sum()), slip_model.shape[0])
        self.assertAlmostEqual(float(layout.layer_widths.sum()), 20_000.0)
        self.assertEqual(layout.green_column_starts[0, 0], 0)
        self.assertEqual(
            layout.green_column_starts[1, 0],
            int(layout.patch_counts[0].sum()),
        )
        for key, rows in layout.patch_groups.items():
            self.assertTrue(np.all(slip_model[rows, 0] == key[0]))
            self.assertTrue(np.all(slip_model[rows, 2] == key[1]))

    def test_rejects_geometry_missing_a_layer(self):
        fault_segments = np.array([[95.0, 19.0, 95.1, 19.0]])
        slip_model = load_fault_one_plane(
            fault_segments,
            dip=[80.0],
            lonc=95.0,
            latc=19.0,
            ref_lon=95.0,
            width=10_000.0,
            layers=1,
            len_top=5_000.0,
        )
        with self.assertRaisesRegex(ValueError, "no patches"):
            build_fault_layout(
                fault_segments,
                slip_model,
                lonc=95.0,
                latc=19.0,
                ref_lon=95.0,
                layers=2,
            )


if __name__ == "__main__":
    unittest.main()
