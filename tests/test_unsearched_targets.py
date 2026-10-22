import tempfile
import unittest
from pathlib import Path

import pandas as pd
import unsearched_targets as targets


class TargetSelectionTests(unittest.TestCase):
    def test_tic_normalization(self):
        self.assertEqual(targets.tic_id("TIC 00123456789"), "123456789")
        self.assertEqual(targets.tic_id("123456789.0"), "123456789")
        self.assertIsNone(targets.tic_id("not a target"))

    def test_union_mes_bounds_and_exclusions(self):
        tces = pd.DataFrame({"tic_id": ["100000001", "100000002", "100000002"],
                             "mes": [7.5, 20, 6]})
        self.assertEqual(targets.select_candidates(
            tces, {"TIC 100000001", "100000004"}, {"100000004"}),
            ["100000001", "100000002"])

    def test_limit_and_deterministic_order(self):
        tces = pd.DataFrame({"tic_id": ["100000003"], "mes": [8]})
        self.assertEqual(targets.select_candidates(
            tces, {"100000001", "100000002"}, set(), limit=2),
            ["100000001", "100000002"])

    def test_text_list_and_append_only_emitted_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            tried = Path(directory) / "tried.txt"
            tried.write_text("100000001\nTIC 100000002\n", encoding="utf-8")
            self.assertEqual(targets.load_ids(tried), {"100000001", "100000002"})
            ledger = Path(directory) / "emitted.csv"
            targets.append_emitted(ledger, ["100000003", "100000003"])
            targets.append_emitted(ledger, ["100000003", "100000004"])
            self.assertEqual(targets.load_ids(ledger), {"100000003", "100000004"})

    def test_reads_target_list_without_import(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "search.py"
            script.write_text("TARGET_LIST = [100000001, 100000002]\n", encoding="utf-8")
            self.assertEqual(targets.load_target_list(script), {"100000001", "100000002"})


if __name__ == "__main__":
    unittest.main()
