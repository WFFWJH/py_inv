import tempfile
import unittest
from pathlib import Path

from geodetic.config import load_workflow_config, read_data_list


class WorkflowConfigTests(unittest.TestCase):
    def test_loads_known_keys_and_rejects_misspelled_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configfile.txt").write_text(
                "##data_list\ntracks.list\n"
                "##fault_file\nfault.txt\n"
                "##segment_file\nsegments.txt\n",
                encoding="utf-8",
            )
            (root / "configpara.txt").write_text(
                "##dip\n80 85\n##layers\n2\n", encoding="utf-8"
            )
            _, config = load_workflow_config(root)
            self.assertEqual(config.dip, [80.0, 85.0])
            self.assertEqual(config.layers, 2)

            (root / "configpara.txt").write_text(
                "##widht\n1000\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "未知配置键"):
                load_workflow_config(root)

    def test_rejects_incomplete_constraint_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configfile.txt").write_text(
                "##data_list\ntracks.list\n"
                "##fault_file\nfault.txt\n"
                "##segment_file\nsegments.txt\n",
                encoding="utf-8",
            )
            (root / "configpara.txt").write_text("##con\n1 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "##con"):
                load_workflow_config(root)

    def test_data_list_accepts_supported_formats_and_aliases_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_list = root / "tracks.list"
            data_list.write_text(
                "track_a 10\n"
                "track_b 20 rng\n"
                "track_c 30 1 2 3 4 azo\n",
                encoding="utf-8",
            )
            tracks, counts, regions, kinds, nmin, nmax = read_data_list(
                str(data_list), default_region=(0, 5, 1, 6)
            )
            self.assertEqual(counts, [10, 20, 30])
            self.assertEqual(kinds, ["insar", "insar", "azo"])
            self.assertEqual(regions[0].tolist(), [0, 5, 1, 6])
            self.assertEqual(nmin.tolist(), [8, 8, 2])
            self.assertEqual(nmax.tolist(), [500, 500, 50])
            self.assertEqual(len(tracks), 3)

    def test_data_list_rejects_unknown_type_and_nonintegral_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_list = Path(temporary) / "tracks.list"
            data_list.write_text("track_a 10 typo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不支持的数据类型"):
                read_data_list(str(data_list))
            data_list.write_text("track_a 10.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "正整数"):
                read_data_list(str(data_list))


if __name__ == "__main__":
    unittest.main()
