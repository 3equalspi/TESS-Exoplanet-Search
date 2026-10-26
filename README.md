# TESS-Exoplanet-Search

A Python/lightkurve workflow for searching TESS SPOC light curves for transit-like signals. The search stores completed TIC IDs in `tess_lc_cache/processed_targets.csv` and detected signals in `tess_lc_cache/candidate_signals.csv`. The post-search review measures transit depth by sector and event, checks odd/even depth and possible secondary eclipses, and generates a review plot. These checks prioritize human review; they do not confirm planets.

## Find catalog-screened targets for independent transit vetting

Install the packages used by the scripts (`lightkurve`, `astropy`, `transitleastsquares`, `pandas`, `matplotlib`, `requests`, and `astroquery`), then run:

```bash
python unsearched_targets.py --limit 500
python tess_search.py --targets-file catalog_cache/undersearched_candidates.csv
python post_search_vetting.py
```

The selector now defaults to MAST's sector-by-sector QLP full-frame target lists, which cover far more stars than the archived SPOC TCE catalog. It removes TICs found in the TOI and confirmed transiting-planet catalogs, and downloads all SPOC TCE-statistics files linked by MAST's live bulk-download index (including later sectors and multi-sector searches). It also coordinate-matches against ATLAS-VAR DR1's published variable-star catalog and TOI/confirmed-host positions within 10 arcseconds to catch missing or inconsistent TIC cross-identifications. The ATLAS-VAR catalog is cached locally after the first run (~74 MB). If a required catalog cannot be retrieved or parsed, selection stops rather than silently using a weaker screen. It also skips TICs in the processed, candidate-signal, and persistent emitted-target ledgers. Its default sectors are 97–98; sectors 99–104 are skipped because MAST currently flags timing corrections there. Choose other sectors with `--sectors 1-96` or `--sectors 40-60,70-75`. Each run emits up to 500 new TICs, ranked from fewer observed sectors toward more observed sectors, using sector count as a rough prioritization proxy. This favors targets with less TESS coverage while still letting you require multiple sectors.

```bash
python unsearched_targets.py --sectors 1-98 --min-sectors 2 --limit 1000
```

A catalog-screen pass is **not a new discovery** and does not prove that a star has never been studied. Because this selector starts from QLP-observed target lists, every selected target has already received QLP light-curve processing and large-scale transit-search coverage; this pool is not a way to find stars untouched by mass searches. Papers, private analyses, and signal lists absent from the queried records may also exist. Catalog screening only means no match was found in the queried TOI, confirmed-planet, SPOC TCE, and ATLAS-VAR records at the time of download. A transit-shape screen pass means only that the light-curve signal passed internal automated checks; novelty and planet status remain unverified and require literature/archive review, independent data, and expert follow-up. The selector's source is [MAST's QLP archive](https://archive.stsci.edu/hlsp/qlp), which publishes per-sector TIC target lists and light curves. Use `--source tce` to retain the legacy SPOC-TCE selection; that pool is smaller and consists of targets already associated with a SPOC transit-like event.

`--processed-file`, `--candidate-file`, and `--emitted-file` can point to alternate ledgers. `--output` changes the selected-target CSV. The selector writes emitted TICs to `tess_lc_cache/emitted_targets.csv` by default, so later runs do not repeat earlier batches. The candidate-signal CSV explicitly records `internal_transit_shape_passed`, `novelty_status=NOT_CHECKED`, and `planet_status=NOT_ESTABLISHED`; older logs are upgraded on the next write. The searcher uses SPOC light curves when present and falls back to QLP FFI light curves when the target is not in SPOC's short-cadence target list. It still skips completed TICs in `tess_lc_cache/processed_targets.csv`. Candidate plots are saved under `tess_candidate_plots/` and closed after saving, so long runs do not open a window or wait for input. Set the `TESS_PLOT_DIR` environment variable to choose another output directory. Within the same sector-coverage tier, TICs are ordered by a stable hash so a limited batch is not just the numerically smallest IDs.

The initial search may still report no signal or noisy plots for most targets; that is normal for a transit search. Every plot and console pass is labeled as an internal transit-shape screen only: it does not establish a novel signal or a planet. A TESS light curve alone does not establish a planet. TLS analysis no longer applies an extra 60,000-point binning step after the initial cap. A signal is not automatically marked as passed unless the folded primary dip is positive, significant, agrees with the TLS depth, and the cadence is at least twice as fine as the fitted transit duration. Short or under-sampled signals are failed/inconclusive for automated triage and need a higher-cadence light curve or manual review.

## Review and benchmarks

```bash
python post_search_vetting.py --tic 123456789
python -m unittest test_target_selection.py
python benchmark_transit_search.py --injections 10 --controls 10
```

The review plot now includes a local BLS period scan around the recorded period, alongside folded light curves, sector depths, and per-event depths. The synthetic benchmark reports recovery for injected box-shaped transits and peak power on noise-only controls. It is a reproducible algorithm/code check, not an estimate of real-world planet yield or a calibrated false-positive rate. Candidates remain targets worth vetting, not planet claims.
