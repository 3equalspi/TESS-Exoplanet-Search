import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

from unsearched_targets import (
    select_candidates, select_qlp_candidates, parse_sectors,
    load_qlp_target_catalog, coordinate_matched_tics,
    append_emitted_tics, read_seen_tics, tce_catalog_urls,
)
from vetting import primary_transit_passes


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

    def test_qlp_pool_deduplicates_and_omits_cataloged_or_seen_targets(self):
        pool = pd.DataFrame({
            "tic_id": ["TIC 101", "202", "303"],
            "sector_count": [3, 2, 1],
            "sectors": ["1;2;3", "2;3", "3"],
        })
        result = select_qlp_candidates(pool, {"202"}, {"303"}, limit=5)
        self.assertEqual(result.tic_id.tolist(), ["101"])
        ranked = select_qlp_candidates(pool, set(), set())
        self.assertEqual(ranked.tic_id.tolist(), ["303", "202", "101"])

    def test_same_coverage_tier_is_stably_shuffled_not_numeric_prefix_sorted(self):
        pool = pd.DataFrame({"tic_id": ["1000000001", "1000000002", "1000000003", "1000000004"],
                             "sector_count": [1, 1, 1, 1]})
        first = select_qlp_candidates(pool, set(), set()).tic_id.tolist()
        second = select_qlp_candidates(pool, set(), set()).tic_id.tolist()
        self.assertEqual(first, second)
        self.assertNotEqual(first, sorted(first))

    def test_live_tce_index_finds_late_sector_and_multisector_products(self):
        html = ('<a href="/missions/tess/catalogs/tce/a-s0001-s0001_dvr-tcestats.csv">S1</a>'
                '<a href="b-s0097-s0097_dvr-tcestats.csv">S97</a>'
                '<a href="c-s0001-s0096_dvr-tcestats.csv">multi</a>')
        urls = tce_catalog_urls(html)
        self.assertEqual(len(urls), 3)
        self.assertTrue(any("s0097-s0097" in url for url in urls))
        self.assertTrue(any("s0001-s0096" in url for url in urls))

    def test_sector_ranges(self):
        self.assertEqual(parse_sectors("2,4-5,8"), [2, 4, 5, 8])

    def test_qlp_lists_merge_sector_coverage(self):
        tmp = Path(".test_qlp_cache")
        tmp.mkdir(exist_ok=True)
        try:
            # Match the actual MAST QLP sector-list format: headerless TIC, RA, Dec.
            (tmp / "qlp_targets_s0001.csv").write_text("101,10,20\n", encoding="utf-8")
            (tmp / "qlp_targets_s0002.csv").write_text("101,10,20\n202,30,-5\n", encoding="utf-8")
            with patch("unsearched_targets.CACHE_DIR", str(tmp)):
                merged = load_qlp_target_catalog(session=None, sectors=[1, 2])
            row = merged.set_index("tic_id").loc["101"]
            self.assertEqual(row.sector_count, 2)
            self.assertEqual(row.sectors, "1;2")
        finally:
            for item in tmp.iterdir():
                item.unlink()
            tmp.rmdir()

    def test_coordinate_crossmatch_catches_catalog_host(self):
        targets = pd.DataFrame({"tic_id": ["123", "456"], "ra": [10.0, 20.0], "dec": [5.0, 5.0]})
        planets = pd.DataFrame({"ra": [10.0001], "dec": [5.0]})
        self.assertEqual(coordinate_matched_tics(targets, [planets]), {"123"})

    def test_emitted_tics_are_persistently_skipped(self):
        root = Path(".test_ledgers")
        root.mkdir(exist_ok=True)
        emitted = root / "emitted.csv"
        processed = root / "processed.csv"
        candidates = root / "candidates.csv"
        try:
            append_emitted_tics(emitted, ["123", "456"])
            seen = read_seen_tics(processed, candidates, emitted)
            self.assertEqual(seen, {"123", "456"})
        finally:
            for item in root.iterdir():
                item.unlink()
            root.rmdir()

    def test_primary_gate_rejects_flat_folds_and_underresolved_events(self):
        base = dict(fitted_depth=0.001, out_of_transit_scatter=0.0001,
                    n_primary_bins=4, cadence_days=0.005, duration_days=0.02)
        self.assertFalse(primary_transit_passes(measured_depth=0.0, **base))
        self.assertFalse(primary_transit_passes(measured_depth=0.001, **{**base, "cadence_days": 0.02}))
        self.assertTrue(primary_transit_passes(measured_depth=0.001, **base))


if __name__ == "__main__":
    unittest.main()

