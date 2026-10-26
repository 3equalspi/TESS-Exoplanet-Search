#!/usr/bin/env python3
import argparse
import csv
import hashlib
import re
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

CACHE_DIR = "catalog_cache"
TCE_URL = ("https://archive.stsci.edu/missions/tess/catalogs/tce/"
           "tess2018206190142-s0001-s0096_dvr-tcestats.csv")
TCE_INDEX_URL = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html"
TOI_URL = "https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv"
TOI_URL_FALLBACK = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
MES_MIN, MES_MAX = 7.5, 20.0
QLP_TARGET_LIST_URL = "https://archive.stsci.edu/hlsps/qlp/target_lists/s{sector:04d}.csv"
ATLAS_VAR_URL = ("https://archive.stsci.edu/hlsps/atlas-var/"
                 "hlsp_atlas-var_atlas_ccd_all_cyan-orange_dr1_object.fits")


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


def tce_catalog_urls(index_html):
    """Extract unique TCE statistics CSV URLs from the current MAST index."""
    hrefs = re.findall(r'href=["\']([^"\']*_dvr-tcestats\.csv)["\']', index_html, re.I)
    return sorted({urljoin(TCE_INDEX_URL, href) for href in hrefs})


def load_tce_catalog(session):
    """Load every SPOC TCE stats CSV linked by MAST's live bulk-download index."""
    response = session.get(TCE_INDEX_URL, timeout=60)
    response.raise_for_status()
    urls = tce_catalog_urls(response.text)
    if not urls:
        raise RuntimeError("MAST TCE index returned no TCE statistics CSV links; refusing an incomplete screen")
    frames = []
    for url in urls:
        path = download_once(url, url.rsplit("/", 1)[-1], session)
        df = pd.read_csv(path, comment="#")
        tic_col = find_tic_column(df)
        df = df.rename(columns={tic_col: "tic_id"})
        mes_col = next((c for c in df.columns if "mes" in c.lower() or "mult_ev" in c.lower()), None)
        if mes_col and mes_col != "mes":
            df = df.rename(columns={mes_col: "mes"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def load_atlas_variable_catalog(session):
    """Return ATLAS-VAR DR1 positions (4.7M published variable candidates)."""
    path = download_once(ATLAS_VAR_URL, "atlas_var_dr1_object.fits", session, timeout=300)
    from astropy.table import Table
    frame = Table.read(path, hdu=1).to_pandas()
    ra_col = next((c for c in frame.columns if c.lower() in {"ra", "raj2000"}), None)
    dec_col = next((c for c in frame.columns if c.lower() in {"dec", "dej2000"}), None)
    if not ra_col or not dec_col:
        raise ValueError(f"ATLAS-VAR table has no RA/Dec columns: {list(frame.columns)}")
    return frame.rename(columns={ra_col: "ra", dec_col: "dec"})[["ra", "dec"]]


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
        table="pscomppars", select="tic_id,pl_name,ra,dec",
        where="tran_flag=1 and tic_id is not null")
    df = table.to_pandas()
    return df.rename(columns={find_tic_column(df): "tic_id"})



def parse_sectors(spec):
    """Parse comma-separated sector numbers/ranges, e.g. 97-98,105-107."""
    sectors = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            if lo < 1 or hi < lo:
                raise ValueError(f"Invalid sector range: {part}")
            sectors.update(range(lo, hi + 1))
        else:
            value = int(part)
            if value < 1:
                raise ValueError(f"Invalid sector number: {value}")
            sectors.add(value)
    if not sectors:
        raise ValueError("At least one sector is required")
    return sorted(sectors)


def load_qlp_target_catalog(session, sectors):
    """Load TICs with public QLP FFI light curves, grouped across sectors."""
    by_tic = {}
    for sector in sectors:
        name = f"qlp_targets_s{sector:04d}.csv"
        path = download_once(QLP_TARGET_LIST_URL.format(sector=sector), name, session)
        # QLP's published sector target lists are headerless TIC, RA, Dec CSVs;
        # also accept a headered variant in case the archive format changes.
        frame = pd.read_csv(path, comment="#", header=None)
        first_row = [str(value).strip().lower() for value in frame.iloc[0].tolist()] if not frame.empty else []
        has_header = any("tic" in value for value in first_row) and any("ra" in value for value in first_row)
        if has_header:
            frame.columns = frame.iloc[0].tolist()
            frame = frame.iloc[1:].reset_index(drop=True)
        else:
            if frame.shape[1] < 3:
                raise ValueError(f"Expected TIC, RA, Dec columns in {path}; found {frame.shape[1]} columns")
            frame = frame.rename(columns={frame.columns[0]: "tic_id", frame.columns[1]: "ra", frame.columns[2]: "dec"})
        tic_col = find_tic_column(frame)
        ra_col = next((c for c in frame.columns if c.lower().replace("_", "").strip() in {"ra", "radeg", "raj2000"}), None)
        dec_col = next((c for c in frame.columns if c.lower().replace("_", "").strip() in {"dec", "decdeg", "dej2000"}), None)
        frame["_clean_tic"] = clean_tics(frame[tic_col])
        for _, row in frame.dropna(subset=["_clean_tic"]).iterrows():
            tic = row["_clean_tic"]
            entry = by_tic.setdefault(tic, {"sectors": set(), "ra": np.nan, "dec": np.nan})
            entry["sectors"].add(sector)
            if ra_col and dec_col:
                entry["ra"] = pd.to_numeric(row[ra_col], errors="coerce")
                entry["dec"] = pd.to_numeric(row[dec_col], errors="coerce")
    rows = [{"tic_id": tic, "sector_count": len(info["sectors"]),
             "sectors": ";".join(str(sector) for sector in sorted(info["sectors"])),
             "ra": info["ra"], "dec": info["dec"],
             "source_catalog": "MAST QLP observed-target list"}
            for tic, info in by_tic.items()]
    return pd.DataFrame(rows, columns=["tic_id", "sector_count", "sectors", "ra", "dec", "source_catalog"])


def select_qlp_candidates(targets, excluded_tics, seen_tics, limit=0, min_sectors=1):
    excluded = {str(tic) for tic in excluded_tics} | {str(tic) for tic in seen_tics}
    if targets.empty:
        return targets.copy()
    result = targets.copy()
    result["tic_id"] = clean_tics(result["tic_id"])
    result = result.dropna(subset=["tic_id"])
    result = result[~result["tic_id"].isin(excluded)]
    if "sector_count" in result:
        result = result[result["sector_count"] >= min_sectors]
        # TIC IDs are labels, not a scientific priority. Hash within each
        # coverage tier to avoid every batch being the numerically smallest IDs.
        result["_stable_order"] = result["tic_id"].map(
            lambda tic: hashlib.sha256(str(tic).encode("ascii")).hexdigest()
        )
        result = result.sort_values(["sector_count", "_stable_order"])
        result = result.drop(columns=["_stable_order"])
    return result.head(limit) if limit > 0 else result


def coordinate_matched_tics(targets, catalogs, radius_arcsec=10.0):
    """Return TICs near catalog positions using chunked spherical matching."""
    if targets.empty or not {"ra", "dec"}.issubset(targets.columns):
        return set()
    catalog_parts = []
    for frame in catalogs:
        if frame is None or frame.empty:
            continue
        ra_col = next((c for c in frame.columns if c.lower().replace("_", "").strip() in {"ra", "radeg", "raj2000"}), None)
        dec_col = next((c for c in frame.columns if c.lower().replace("_", "").strip() in {"dec", "decdeg", "dej2000"}), None)
        if ra_col and dec_col:
            part = frame[[ra_col, dec_col]].copy()
            part.columns = ["ra", "dec"]
            catalog_parts.append(part)
    if not catalog_parts:
        return set()
    catalog = pd.concat(catalog_parts, ignore_index=True)
    catalog["ra"] = pd.to_numeric(catalog["ra"], errors="coerce")
    catalog["dec"] = pd.to_numeric(catalog["dec"], errors="coerce")
    catalog = catalog.dropna()
    catalog = catalog[catalog.ra.between(0.0, 360.0) & catalog.dec.between(-90.0, 90.0)]
    if catalog.empty:
        return set()

    target_ra = pd.to_numeric(targets.ra, errors="coerce").to_numpy(dtype=float)
    target_dec = pd.to_numeric(targets.dec, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(target_ra) & np.isfinite(target_dec) & (target_ra >= 0) & (target_ra <= 360) & (target_dec >= -90) & (target_dec <= 90)
    valid_idx = np.flatnonzero(valid)
    if not len(valid_idx):
        return set()
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        target_coords = SkyCoord(ra=target_ra[valid_idx] * u.deg, dec=target_dec[valid_idx] * u.deg)
        matched_targets = set()
        # Chunk the 4.7M-row ATLAS catalog to keep memory use bounded.
        for start in range(0, len(catalog), 250_000):
            part = catalog.iloc[start:start + 250_000]
            catalog_coords = SkyCoord(ra=part.ra.to_numpy(dtype=float) * u.deg,
                                      dec=part.dec.to_numpy(dtype=float) * u.deg)
            # SkyCoord.search_around_sky returns indices into its argument
            # first, then indices into self (the target coordinates here).
            _, target_idx, _, _ = target_coords.search_around_sky(
                catalog_coords, radius_arcsec * u.arcsec
            )
            if len(target_idx):
                matched_targets.update(targets.iloc[valid_idx[np.unique(target_idx)]]["tic_id"].astype(str))
                if len(matched_targets) == len(set(targets.iloc[valid_idx]["tic_id"].astype(str))):
                    break
        return matched_targets
    except ImportError:
        # Keep a dependency-light fallback for small catalogs and unit tests.
        if len(catalog) > 100_000:
            raise RuntimeError("Large-catalog coordinate screening requires SciPy; install the project dependencies")

    order = valid_idx[np.argsort(target_dec[valid_idx])]
    sorted_dec = target_dec[order]
    sorted_ra = target_ra[order]
    sorted_tics = targets.iloc[order]["tic_id"].astype(str).to_numpy()
    radius_deg = radius_arcsec / 3600.0
    radius_rad = np.deg2rad(radius_deg)
    matched = set()
    for ra0, dec0 in catalog[["ra", "dec"]].itertuples(index=False, name=None):
        lo = np.searchsorted(sorted_dec, dec0 - radius_deg, side="left")
        hi = np.searchsorted(sorted_dec, dec0 + radius_deg, side="right")
        if hi <= lo:
            continue
        ras, decs, tics = sorted_ra[lo:hi], sorted_dec[lo:hi], sorted_tics[lo:hi]
        cos_dec = max(abs(np.cos(np.deg2rad(dec0))), 1e-8)
        ra_window = min(180.0, radius_deg / cos_dec)
        delta_ra = np.abs((ras - ra0 + 180.0) % 360.0 - 180.0)
        near = delta_ra <= ra_window
        if not np.any(near):
            continue
        ras, decs, tics = ras[near], decs[near], tics[near]
        dra = np.deg2rad((ras - ra0 + 180.0) % 360.0 - 180.0)
        ddec = np.deg2rad(decs - dec0)
        hav = np.sin(ddec / 2.0) ** 2 + np.cos(np.deg2rad(dec0)) * np.cos(np.deg2rad(decs)) * np.sin(dra / 2.0) ** 2
        matched.update(tics[2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0))) <= radius_rad])
    return matched


def append_emitted_tics(path, tics):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if is_new:
            writer.writerow(["tic_id"])
        writer.writerows([[tic] for tic in tics])


def read_seen_tics(processed_file, candidate_file, emitted_file=None):
    seen = set()
    if processed_file.exists():
        with processed_file.open(encoding="utf-8-sig") as f:
            for line in f:
                seen.update(clean_tics(pd.Series([line])).tolist())
    if candidate_file.exists():
        with candidate_file.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                seen.update(clean_tics(pd.Series([row.get("tic_id", "")])).tolist())
    if emitted_file and emitted_file.exists():
        with emitted_file.open(newline="", encoding="utf-8-sig") as f:
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
    parser = argparse.ArgumentParser(description="Select QLP-observed TESS targets without matches in the queried signal catalogs.")
    parser.add_argument("--source", choices=["qlp", "tce"], default="qlp",
                        help="QLP observed-target pool (default) or legacy SPOC TCE pool")
    parser.add_argument("--sectors", default="97-98",
                        help="QLP sectors as numbers/ranges; default skips sectors 99-104 with timing corrections")
    parser.add_argument("--min-sectors", type=int, default=1,
                        help="Require this many QLP sectors for each TIC")
    parser.add_argument("--processed-file", type=Path, default=Path("tess_lc_cache/processed_targets.csv"))
    parser.add_argument("--candidate-file", type=Path, default=Path("tess_lc_cache/candidate_signals.csv"))
    parser.add_argument("--emitted-file", type=Path, default=Path("tess_lc_cache/emitted_targets.csv"),
                        help="Persistent ledger of TICs already emitted by this selector")
    parser.add_argument("--output", type=Path, default=Path(CACHE_DIR) / "undersearched_candidates.csv")
    parser.add_argument("--limit", type=int, default=500, help="Maximum new targets to emit per run")
    args = parser.parse_args()
    session = _session_with_retries()
    print("Loading TOI catalog...")
    toi = load_toi_catalog(session)
    print("Loading confirmed transiting planet hosts from NASA Exoplanet Archive...")
    confirmed = load_confirmed_tess_planets()
    excluded = set(clean_tics(toi["tic_id"])) | set(clean_tics(confirmed["tic_id"]))
    if args.source == "qlp":
        sectors = parse_sectors(args.sectors)
        print(f"Loading public QLP target lists for sectors {sectors}...")
        pool = load_qlp_target_catalog(session, sectors)
        print("Loading all public SPOC TCE catalogs linked by MAST...")
        tce = load_tce_catalog(session)
        excluded |= set(clean_tics(tce["tic_id"]))
        print("Loading ATLAS-VAR DR1 positions to remove previously cataloged variable stars...")
        atlas_var = load_atlas_variable_catalog(session)
        excluded |= coordinate_matched_tics(pool, [toi, confirmed, atlas_var], radius_arcsec=10.0)
    else:
        print("Loading SPOC TCE catalog...")
        pool = load_tce_catalog(session)
    seen = read_seen_tics(args.processed_file, args.candidate_file, args.emitted_file)
    if args.source == "qlp":
        candidates = select_qlp_candidates(pool, excluded, seen, args.limit, args.min_sectors)
    else:
        candidates = select_candidates(pool, excluded, seen, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    append_emitted_tics(args.emitted_file, candidates["tic_id"].tolist())
    print(f"Excluded {len(excluded)} TICs matching TOIs, confirmed planets, any indexed SPOC TCE, or ATLAS-VAR positions; skipped {len(seen)} processed/emitted TICs.")
    print(f"Wrote {len(candidates)} TICs not previously emitted by this selector to {args.output}")
    print(f"Persistent emission ledger: {args.emitted_file}")
    print(f"Run: python tess_search.py --targets-file {args.output}")
    print("The QLP source targets have already had QLP processing; catalog clearance is not a new-discovery claim.")
    print("First 20 TICs:", candidates["tic_id"].astype(int).tolist()[:20])


if __name__ == "__main__":
    main()
