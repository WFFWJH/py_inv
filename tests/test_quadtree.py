import unittest
from unittest.mock import patch

import numpy as np

from geodetic.quadtree import decompose_quadtree
from input_data.script_downsamp.sample import (
    _alloc_block_buf,
    _quad_decomp_mean2,
    iter_quad_downsample,
)


class SharedQuadtreeTests(unittest.TestCase):
    def test_legacy_python_api_uses_selected_minimum_stop_mode(self):
        x = np.arange(8, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        z = np.zeros((4, 8), dtype=np.float64)

        outputs = {}
        for mode, min_stop_and in (("or", False), ("and", True)):
            expected = decompose_quadtree(
                x, y, z, 1.0, 2, 8, min_stop_mode=mode
            )
            buffer = _alloc_block_buf(z)
            count = _quad_decomp_mean2(
                x,
                y,
                z,
                1.0,
                2,
                8,
                buffer,
                0,
                min_stop_and=min_stop_and,
            )
            np.testing.assert_allclose(buffer[:count], expected)
            outputs[mode] = count
        self.assertEqual(outputs, {"or": 4, "and": 16})

    def test_median_pixel_representative_is_an_input_pixel(self):
        x = np.arange(2, dtype=np.float64)
        y = np.arange(2, dtype=np.float64)
        z = np.array([[0.0, 10.0], [10.0, 0.0]])
        result = decompose_quadtree(
            x,
            y,
            z,
            1.0,
            2,
            4,
            min_stop_mode="and",
            representative="median_pixel",
        )
        self.assertEqual(len(result), 1)
        self.assertIn(result[0][2], (0.0, 10.0))
        self.assertEqual(result[0][3], 4)

    def test_numba_selector_falls_back_to_python_when_numba_is_unavailable(self):
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        z = np.arange(16, dtype=np.float64).reshape(4, 4)
        expected = iter_quad_downsample(
            x, y, z, 4, 1, 4, 0, initial_threshold=2.0, verbose=False
        )
        with patch("input_data.script_downsamp.sample.NUMBA_AVAILABLE", False):
            actual = iter_quad_downsample(
                x, y, z, 4, 1, 4, 1, initial_threshold=2.0, verbose=False
            )

        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(expected_array, actual_array)


if __name__ == "__main__":
    unittest.main()
