# TESS-Exoplanet-Search

TESS exoplanet hunt using Python/lightkurve. The pipeline downloads SPOC light curves, searches with TLS/BLS, and applies transit and signal vetting.

## Finding targets not tried before

Run \`python unsearched_targets.py\` after installing \`pandas\`, \`requests\`, and \`astroquery\`. It combines TICs with TESS-SPOC time-series observations returned by MAST (including sectors newer than the static bulk TCE catalog) and TICs in the existing SPOC TCE catalog whose MES is 7.5–20, preserving the prior source and filter.

It excludes TOIs, confirmed TESS-discovered planets, the \`TARGET_LIST\` in \`tess_search.py\`, \`tess_lc_cache/processed_targets.csv\`, the prior \`catalog_cache/undersearched_candidates.csv\`, and persistent \`catalog_cache/emitted_candidates.csv\`. The output is \`catalog_cache/undersearched_candidates.csv\`. Emitted TICs are recorded so reruns move on to unseen IDs. The first MAST query may be large and needs network access; downloaded catalogs are cached in \`catalog_cache/\` (override with \`TESS_CATALOG_CACHE\`).

Import manually tried TICs with a CSV containing a \`tic_id\` column or a newline-delimited file:

\`\`\`sh
python unsearched_targets.py --tried-file tried_tics.csv --limit 500
\`\`\`

Repeat \`--tried-file\` for multiple lists. \`--processed-file\`, \`--emitted-file\`, and \`--output\` accept alternate paths. Use \`--no-mast\` to use the legacy TCE-only source. Candidates are targets for vetting, not planet detections; check current catalogs and review the light curves before drawing conclusions.

Run the offline selection tests with \`python -m unittest discover -s tests\`.

Main search code: \`tess_search.py\`.
