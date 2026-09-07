#!/usr/bin/env python3
import os
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

CACHE_DIR = "catalog_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
#https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html -
TCE_URL = ("https://archive.stsci.edu/missions/tess/catalogs/tce/"
           "tess2018206190142-s0001-s0096_dvr-tcestats.csv")
TOI_URL = "https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv"
TOI_URL_FALLBACK = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

MES_MIN = 7.5
MES_MAX = 20.0


def _session_with_retries():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


def download_once(url, fname, session=None, timeout=180):
    path = os.path.join(CACHE_DIR, fname)
    if os.path.exists(path):
        print(f"  -> Using cached {path}")
        return path
    session = session or _session_with_retries()
    print(f"  -> Downloading {url}")
    t0 = time.time()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"  -> Saved {path} ({len(r.content)/1e6:.1f} MB, {time.time()-t0:.1f}s)")
    return path


def find_tic_column(df):
    candidates = [c for c in df.columns if "tic" in c.lower().replace("_", "").replace(" ", "")]
    if not candidates:
        raise ValueError(f"No TIC-ID-like column found. Columns were: {list(df.columns)}")
    return candidates[0]


def load_tce_catalog(session):
    path = download_once(TCE_URL, "tce_all_sectors.csv", session)
    df = pd.read_csv(path, comment="#")
    print(f"  -> TCE catalog columns: {list(df.columns)}")
    tic_col = find_tic_column(df)
    mes_col = next((c for c in df.columns if "mes" in c.lower() or "mult_ev" in c.lower()), None)
    print(f"  -> Using TIC column '{tic_col}'" + (f", MES-like column '{mes_col}'" if mes_col else ", no MES-like column found"))
    df = df.rename(columns={tic_col: "tic_id"})
    if mes_col:
        df = df.rename(columns={mes_col: "mes"})
    return df


def load_toi_catalog(session):
    try:
        path = download_once(TOI_URL, "toi.csv", session)
        df = pd.read_csv(path, comment="#")
    except Exception as e:
        print(f"  -> Primary TOI URL failed ({e}), trying ExoFOP mirror...")
        path = download_once(TOI_URL_FALLBACK, "toi.csv", session)
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
    print(f"  -> TOI catalog columns: {list(df.columns)}")
    tic_col = find_tic_column(df)
    df = df.rename(columns={tic_col: "tic_id"})
    return df


def load_confirmed_tess_planets():
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select="tic_id,pl_name",
        where="disc_facility like '%TESS%'",
    )
    df = table.to_pandas()
    print(f"  -> Confirmed TESS planets columns: {list(df.columns)}")
    tic_col = find_tic_column(df)
    df = df.rename(columns={tic_col: "tic_id"})
    return df


def _clean_tic_series(s):
    return (
        s.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .dropna()
        .astype(str)
    )


def main():
    session = _session_with_retries()

    print("Loading TCE catalog (every signal SPOC's pipeline has already found)...")
    tce = load_tce_catalog(session)
    tce["tic_id"] = _clean_tic_series(tce["tic_id"])

    print("\nLoading TOI catalog (every signal a human/QLP already reviewed)...")
    toi = load_toi_catalog(session)
    toi["tic_id"] = _clean_tic_series(toi["tic_id"])

    print("\nLoading confirmed TESS planets from the NASA Exoplanet Archive...")
    confirmed = load_confirmed_tess_planets()
    confirmed["tic_id"] = _clean_tic_series(confirmed["tic_id"])

    already_seen = set(toi["tic_id"]) | set(confirmed["tic_id"])
    print(f"\n{len(already_seen)} unique TIC IDs already have a TOI or confirmed planet.")

    unreviewed = tce[~tce["tic_id"].isin(already_seen)].copy()
    print(f"{tce['tic_id'].nunique()} unique TIC IDs have at least one TCE; "
          f"{unreviewed['tic_id'].nunique()} of those were never promoted to a TOI.")

    if "mes" in unreviewed.columns:
        before = unreviewed["tic_id"].nunique()
        unreviewed = unreviewed[(unreviewed["mes"] >= MES_MIN) & (unreviewed["mes"] <= MES_MAX)]
        print(f"Restricting to MES in [{MES_MIN}, {MES_MAX}]: "
              f"{before} -> {unreviewed['tic_id'].nunique()} unique TIC IDs.")

    candidate_tic_ids = sorted(unreviewed["tic_id"].astype(int).unique().tolist())
    out_path = os.path.join(CACHE_DIR, "undersearched_candidates.csv")
    pd.DataFrame({"tic_id": candidate_tic_ids}).to_csv(out_path, index=False)
    print(f"\nWrote {len(candidate_tic_ids)} candidate TIC IDs to {out_path}")
    print("Drop these into TARGET_LIST in tess_search_fixed.py (or read the CSV there).")
    print("\nFirst 20 candidates:", candidate_tic_ids[:20])


if __name__ == "__main__":
    main()
