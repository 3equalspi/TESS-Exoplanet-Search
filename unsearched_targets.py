#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

CACHE_DIR = "catalog_cache"
TCE_URL = ("https://archive.stsci.edu/missions/tess/catalogs/tce/"
           "tess2018206190142-s0001-s0096_dvr-tcestats.csv")
TOI_URL = "https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv"
TOI_URL_FALLBACK = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
MES_MIN, MES_MAX = 7.5, 20.0


def _session_with_retries():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504])))
    return s


def download_once(url, fname, session=None, timeout=180):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        print(f"  -> Using cached {path}")
        return path
    session = session or _session_with_retries()
    print(f"  -> Downloading {url}")
    t0 = time.time()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"  -> Saved {path} ({len(response.content)/1e6:.1f} MB, {time.time()-t0:.1f}s)")
    return path


def find_tic_column(df):
    candidates = [c for c in df.columns if "tic" in c.lower().replace("_", "").replace(" ", "")]
    if not candidates:
        raise ValueError(f"No TIC-ID-like column found: {list(df.columns)}")
    return candidates[0]


def clean_tics(series):
    return series.astype(str).str.extract(r"(\d+)", expand=False).dropna().astype(str)


def load_tce_catalog(session):
    path = download_once(TCE_URL, "tce_all_sectors.csv", session)
    df = pd.read_csv(path, comment="#")
    tic_col = find_tic_column(df)
    mes_col = next((c for c in df.columns if "mes" in c.lower() or "mult_ev" in c.lower()), None)
    df = df.rename(columns={tic_col: "tic_id"})
    if mes_col:
        df = df.rename(columns={mes_col: "mes"})
    return df


def load_toi_catalog(session):
    try:
        path = download_once(TOI_URL, "toi.csv", session)
        df = pd.read_csv(path, comment="#")
    except Exception as exc:
        print(f"  -> Primary TOI URL failed ({exc}), trying ExoFOP mirror...")
        path = download_once(TOI_URL_FALLBACK, "toi.csv", session)
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    return df.rename(columns={find_tic_column(df): "tic_id"})


def load_confirmed_tess_planets():
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars", select="tic_id,pl_name",
        where="tran_flag=1 and tic_id is not null")
    df = table.to_pandas()
    return df.rename(columns={find_tic_column(df): "tic_id"})


def read_seen_tics(processed_file, candidate_file):
    seen = set()
    if processed_file.exists():
        with processed_file.open(encoding="utf-8-sig") as f:
            for line in f:
                seen.update(clean_tics(pd.Series([line])).tolist())
    if candidate_file.exists():
        with candidate_file.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                seen.update(clean_tics(pd.Series([row.get("tic_id", "")])).tolist())
    return seen


def select_candidates(tce, excluded_tics, seen_tics, limit=0):
    excluded_tics = {str(tic) for tic in excluded_tics}
    seen_tics = {str(tic) for tic in seen_tics}
    tce = tce.copy()
    tce["tic_id"] = clean_tics(tce["tic_id"])
    tce = tce.dropna(subset=["tic_id"])
    tce = tce[~tce["tic_id"].isin(excluded_tics | seen_tics)]
    if "mes" in tce:
        tce["mes"] = pd.to_numeric(tce["mes"], errors="coerce")
        tce = tce[tce["mes"].between(MES_MIN, MES_MAX)]
        result = tce.groupby("tic_id", as_index=False).agg(
            max_mes=("mes", "max"), tce_count=("mes", "size"))
        result["source_catalog"] = "SPOC TCE catalog"
        result = result.sort_values(["max_mes", "tic_id"], ascending=[False, True])
    else:
        result = pd.DataFrame({"tic_id": sorted(tce["tic_id"].unique())})
        result["tce_count"], result["source_catalog"] = 1, "SPOC TCE catalog"
    return result.head(limit) if limit > 0 else result


def main():
    parser = argparse.ArgumentParser(description="Select undersearched TESS targets for transit vetting.")
    parser.add_argument("--processed-file", type=Path, default=Path("tess_lc_cache/processed_targets.csv"))
    parser.add_argument("--candidate-file", type=Path, default=Path("tess_lc_cache/candidate_signals.csv"))
    parser.add_argument("--output", type=Path, default=Path(CACHE_DIR) / "undersearched_candidates.csv")
    parser.add_argument("--limit", type=int, default=0, help="Maximum targets to emit; 0 means all")
    args = parser.parse_args()
    session = _session_with_retries()
    print("Loading SPOC TCE catalog...")
    tce = load_tce_catalog(session)
    print("Loading TOI catalog...")
    toi = load_toi_catalog(session)
    print("Loading confirmed transiting planet hosts from NASA Exoplanet Archive...")
    confirmed = load_confirmed_tess_planets()
    excluded = set(clean_tics(toi["tic_id"])) | set(clean_tics(confirmed["tic_id"]))
    seen = read_seen_tics(args.processed_file, args.candidate_file)
    candidates = select_candidates(tce, excluded, seen, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    print(f"Excluded {len(excluded)} known TOI/confirmed-host TICs and {len(seen)} processed or emitted TICs.")
    print(f"Wrote {len(candidates)} candidates to {args.output}")
    print(f"Run: python tess_search.py --targets-file {args.output}")
    print("First 20 TICs:", candidates["tic_id"].astype(int).tolist()[:20])


if __name__ == "__main__":
    main()

