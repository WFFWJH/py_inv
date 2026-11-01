import unittest
from unittest.mock import patch

import numpy as np

from geodetic.green_function import build_green_function
from geodetic.green import calc_green_insar_okada


class GreenFunctionOptionsTests(unittest.TestCase):
    def setUp(self):
        self.sampled = np.array(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                [1000.0, 0.0, 2.0, 1.0, 0.0, 1.0],
            ]
        )
        self.data = {
            "sampled_insar_data": self.sampled,
            "rms_out": np.array([[0.2], [0.4]]),
            "dem_out": np.array([[100.0], [250.0]]),
        }
        self.green = np.array([[3.0], [6.0]])

    def build(self, **options):
        with (
            patch("geodetic.green_function.loadmat", return_value=self.data),
            patch(
                "geodetic.green_function.calc_green_insar_okada",
                return_value=self.green,
            ) as calc_green,
        ):
            result = build_green_function(
                np.zeros((1, 13)), "track.mat", "insar", "none", **options
            )
        return result, calc_green

    def test_default_preserves_uniform_weights_and_zero_surface_elevation(self):
        (raw, weighted, data_raw, data_weighted), calc_green = self.build()
        np.testing.assert_array_equal(raw[:, 0], self.green[:, 0])
        np.testing.assert_allclose(weighted[:, 0], [1.5, 3.0])
        np.testing.assert_array_equal(data_raw, [1.0, 2.0])
        np.testing.assert_allclose(data_weighted, [0.5, 1.0])
        self.assertIsNone(calc_green.call_args.kwargs["surface_elevation"])

    def test_rms_weights_and_surface_elevation_are_forwarded(self):
        (raw, weighted, _, data_weighted), calc_green = self.build(
            use_rms_weight=True, use_topography=True
        )
        np.testing.assert_array_equal(raw[:, 0], self.green[:, 0])
        np.testing.assert_allclose(weighted[:, 0], [2.0, 2.0])
        np.testing.assert_allclose(data_weighted, [2.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_array_equal(
            calc_green.call_args.kwargs["surface_elevation"], [100.0, 250.0]
        )

    def test_zero_rms_uses_finite_weight_floor(self):
        self.data["rms_out"] = np.array([0.0, 1.0])
        (_, weighted, _, _), _ = self.build(use_rms_weight=True)
        self.assertTrue(np.all(np.isfinite(weighted)))
        np.testing.assert_allclose(weighted[:, 0], [3000.0 / 1001.0, 6.0 / 1001.0])

    def test_requested_rms_and_topography_data_are_required(self):
        self.data.pop("rms_out")
        with (
            patch("geodetic.green_function.loadmat", return_value=self.data),
            patch(
                "geodetic.green_function.calc_green_insar_okada",
                return_value=self.green,
            ),
        ):
            with self.assertRaisesRegex(KeyError, "rms_out"):
                build_green_function(
                    np.zeros((1, 13)), "track.mat", "insar", "none",
                    use_rms_weight=True,
                )
            self.data.pop("dem_out")
            with self.assertRaisesRegex(KeyError, "dem_out"):
                build_green_function(
                    np.zeros((1, 13)), "track.mat", "insar", "none",
                    use_topography=True,
                )

    def test_surface_elevation_reaches_okada_tp(self):
        slip_model = np.zeros((1, 13), dtype=np.float64)
        slip_model[0, 6:10] = [1000.0, 500.0, 90.0, 80.0]
        zero = np.zeros(2, dtype=np.float64)
        with patch(
            "geodetic.green.calc_okada", side_effect=[(zero, zero, zero)] * 4
        ) as calc_okada:
            calc_green_insar_okada(
                slip_model,
                self.sampled,
                surface_elevation=np.array([100.0, 250.0]),
            )
            calc_green_insar_okada(slip_model, self.sampled)

        for call in calc_okada.call_args_list[:2]:
            np.testing.assert_array_equal(call.args[11], [100.0, 250.0])
        for call in calc_okada.call_args_list[2:]:
            np.testing.assert_array_equal(call.args[11], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
