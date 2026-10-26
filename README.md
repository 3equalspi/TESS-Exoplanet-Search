# TESS-Exoplanet-Search

A Python/lightkurve workflow for searching TESS SPOC light curves for transit-like signals. The search stores completed TIC IDs in `tess_lc_cache/processed_targets.csv` and detected signals in `tess_lc_cache/candidate_signals.csv`. The post-search review measures transit depth by sector and event, checks odd/even depth and possible secondary eclipses, and generates a review plot. These checks prioritize human review; they do not confirm planets.

## Find and search fresh targets

Install the packages used by the scripts (`lightkurve`, `astropy`, `transitleastsquares`, `pandas`, `matplotlib`, `requests`, and `astroquery`), then run:

```bash
python unsearched_targets.py --limit 100
python tess_search.py --targets-file catalog_cache/undersearched_candidates.csv
python post_search_vetting.py
```

The selector draws from the archived SPOC TCE catalog, removes TICs in the TOI catalog and confirmed transiting planet hosts in the NASA Exoplanet Archive, and removes TICs already present in the processed or candidate-signal ledgers. It applies a conservative MES range of 7.5–20 when the TCE catalog provides MES, collapses multiple TCE rows to one TIC, and sorts stronger candidates first. The TCE catalog currently covers sectors 1–96, so this is an underreviewed-target list from that catalog, not a census of every TESS target or a guarantee that signals were never examined elsewhere.

`--processed-file` and `--candidate-file` can point to alternate ledgers. `--output` changes the selector CSV path. `--limit` limits how many candidates are emitted. The searcher accepts either that CSV (`tic_id` column) or a plain list of TIC IDs. If `--targets-file` is omitted, the existing built-in `TARGET_LIST` is used. The searcher still skips any TIC already in its processed ledger.

## Review and benchmarks

```bash
python post_search_vetting.py --tic 123456789
python -m unittest test_target_selection.py
python benchmark_transit_search.py --injections 10 --controls 10
```

The review plot now includes a local BLS period scan around the recorded period, alongside folded light curves, sector depths, and per-event depths. The synthetic benchmark reports recovery for injected box-shaped transits and peak power on noise-only controls. It is a reproducible algorithm/code check, not an estimate of real-world planet yield or a calibrated false-positive rate. Candidates remain targets worth vetting, not planet claims.

