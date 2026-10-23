# TESS-Exoplanet-Search
TESS exoplanet hunt using Python/lightkurve. Pipeline downloads SPOC light curves, runs TLS/BLS, vets candidates by depth, transit coverage, sector consistency. For promising signals, performs per-transit coherence test to reject stellar variability/eclipsing binaries.

Main code for extracting data and initial analysis is withon tess_search.py
