import unittest
import pandas as pd

from unsearched_targets import select_candidates


class TargetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tce = pd.DataFrame({
            "tic_id": [101, 101, 202, 303, 404, 505],
            "mes": [9.0, 12.0, 8.1, 21.0, 7.0, 10.0],
        })

    def test_excludes_catalog_and_persistent_seen_ids_and_deduplicates(self):
        result = select_candidates(self.tce, {202}, {303}, limit=0)
        self.assertEqual(result.tic_id.tolist(), ["101", "505"])
        self.assertEqual(result.iloc[0].max_mes, 12.0)
        self.assertEqual(result.iloc[0].tce_count, 2)

    def test_mes_bounds_and_limit_are_applied(self):
        result = select_candidates(self.tce, set(), set(), limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].tic_id, "101")

    def test_tic_cleaning_and_empty_result(self):
        result = select_candidates(pd.DataFrame({"tic_id": ["TIC 12", None], "mes": [8.0, 9.0]}), {"12"}, set())
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()


