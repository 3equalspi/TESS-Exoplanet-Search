# TESS-Exoplanet-Search
TESS exoplanet hunt using Python/lightkurve. Pipeline downloads SPOC light curves, runs TLS/BLS, vets candidates by depth, transit coverage, sector consistency. For promising signals, performs per-transit coherence test to reject stellar variability/eclipsing binaries.

Main code for extracting data and initial analysis is withon tess_search.py

## Post-search candidate review

After tess_search.py finishes, it records TLS detections, including signals that failed its initial gates, in tess_lc_cache/candidate_signals.csv. The existing processed_targets.csv remains the resume list; it records attempted TICs, not candidate detections. This manifest is created for searches run with the updated code; earlier runs are not reconstructed automatically. To rescan an earlier TIC, remove only that TIC's line from processed_targets.csv and rerun tess_search.py.

Run the follow-up review with:

    python post_search_vetting.py

The reviewer downloads or reuses available SPOC sectors, estimates depths by sector and observed event, checks odd/even event depths and a possible secondary eclipse, then writes diagnostic plots and tess_vetting_results/review_summary.csv. To review one target, run python post_search_vetting.py --tic 41232189. Use --input, --output-dir, or --cache-dir to change paths.

Flags are prompts for manual inspection. No flags do not validate a planet, and a flag does not prove a false positive. Review the plots and follow up with appropriate astrophysical checks.
