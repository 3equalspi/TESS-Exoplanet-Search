#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, os, re, tempfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("TESS_CATALOG_CACHE", ROOT / "catalog_cache"))
TCE_URL = "https://archive.stsci.edu/missions/tess/catalogs/tce/tess2018206190142-s0001-s0096_dvr-tcestats.csv"
TOI_URL = "https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv"
TOI_URL_FALLBACK = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
MES_MIN, MES_MAX = 7.5, 20.0

def session_with_retries():
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    return session

def download_once(url, filename, session=None, timeout=180):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    if path.exists():
        return path
    response = (session or session_with_retries()).get(url, timeout=timeout)
    response.raise_for_status()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(path)
    return path

def tic_id(value):
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(?:TIC\s*)?(\d{5,})", str(value), flags=re.IGNORECASE)
    return str(int(match.group(1))) if match else None

def ids_from_values(values):
    return {identifier for value in values if (identifier := tic_id(value)) is not None}

def find_tic_column(frame):
    columns = [c for c in frame.columns if "tic" in c.lower().replace("_", "").replace(" ", "")]
    if not columns:
        raise ValueError(f"No TIC-ID-like column found. Columns: {list(frame.columns)}")
    return columns[0]

def load_ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, comment="#", dtype=str)
        return ids_from_values(frame[find_tic_column(frame)])
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, KeyError):
        return ids_from_values(path.read_text(encoding="utf-8").splitlines())

def load_target_list(path=ROOT / "tess_search.py"):
    path = Path(path)
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TARGET_LIST" for t in node.targets
        ):
            try:
                return ids_from_values(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                return set()
    return set()

def load_tce_catalog(session):
    frame = pd.read_csv(download_once(TCE_URL, "tce_all_sectors.csv", session),
                        comment="#", dtype=str)
    tic_column = find_tic_column(frame)
    mes_column = next((c for c in frame.columns if "mes" in c.lower() or "mult_ev" in c.lower()), None)
    frame = frame.rename(columns={tic_column: "tic_id"})
    frame["tic_id"] = frame["tic_id"].map(tic_id)
    if mes_column:
        frame["mes"] = pd.to_numeric(frame[mes_column], errors="coerce")
    return frame.dropna(subset=["tic_id"])

def load_toi_catalog(session):
    try:
        path = download_once(TOI_URL, "toi.csv", session)
        frame = pd.read_csv(path, comment="#", dtype=str)
    except Exception as error:
        print(f"TOI source failed ({error}); trying ExoFOP")
        frame = pd.read_csv(download_once(TOI_URL_FALLBACK, "toi_exofop.csv", session),
                            sep=None, engine="python", comment="#", dtype=str)
    return ids_from_values(frame[find_tic_column(frame)])

def load_confirmed_tess_planets():
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    table = NasaExoplanetArchive.query_criteria(
        table="pscomppars", select="tic_id,pl_name",
        where="disc_facility like '%TESS%'")
    frame = table.to_pandas()
    return ids_from_values(frame[find_tic_column(frame)])

def load_spoc_observed_ids():
    from astroquery.mast import Observations
    observations = Observations.query_criteria(
        obs_collection="TESS", dataproduct_type="timeseries", provenance_name="SPOC")
    if "target_name" not in observations.colnames:
        raise ValueError(f"MAST response lacks target_name: {observations.colnames}")
    return ids_from_values(observations["target_name"])

def select_candidates(tce_frame, observed_ids, excluded_ids,
                      mes_min=MES_MIN, mes_max=MES_MAX, limit=None):
    eligible = tce_frame.copy()
    if "mes" in eligible.columns:
        mes = pd.to_numeric(eligible["mes"], errors="coerce")
        eligible = eligible[mes.between(mes_min, mes_max, inclusive="both")]
    identifiers = ids_from_values(eligible["tic_id"]) | ids_from_values(observed_ids)
    result = sorted(identifiers - set(excluded_ids), key=int)
    return result[:limit] if limit is not None else result

def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                     dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(path)

def append_emitted(path, identifiers):
    path = Path(path)
    existing = load_ids(path)
    new = sorted(set(identifiers) - existing, key=int)
    if not new:
        return
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        if "tic_id" not in old.columns:
            old = pd.DataFrame({"tic_id": sorted(existing, key=int)})
    else:
        old = pd.DataFrame(columns=["tic_id", "emitted_utc"])
    rows = pd.DataFrame({"tic_id": new, "emitted_utc":
                         [datetime.now(timezone.utc).isoformat()] * len(new)})
    atomic_csv(pd.concat([old, rows], ignore_index=True), path)

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="maximum new TICs to emit")
    parser.add_argument("--tried-file", type=Path, action="append", default=[],
                        help="additional CSV or newline list of tried TICs (repeatable)")
    parser.add_argument("--processed-file", type=Path,
                        default=ROOT / "tess_lc_cache/processed_targets.csv")
    parser.add_argument("--emitted-file", type=Path, default=CACHE_DIR / "emitted_candidates.csv")
    parser.add_argument("--output", type=Path, default=CACHE_DIR / "undersearched_candidates.csv")
    parser.add_argument("--no-mast", action="store_true",
                        help="use only the legacy TCE catalog; skip MAST observed-target source")
    parser.add_argument("--mes-min", type=float, default=MES_MIN)
    parser.add_argument("--mes-max", type=float, default=MES_MAX)
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    session = session_with_retries()
    tces = load_tce_catalog(session)
    toi_ids = load_toi_catalog(session)
    confirmed = load_confirmed_tess_planets()
    observed = set() if args.no_mast else load_spoc_observed_ids()
    processed = load_ids(args.processed_file)
    emitted = load_ids(args.emitted_file)
    # Migrate the prior script's last overwritten CSV as a tried-ID list.
    old_output = load_ids(args.output)
    excluded = toi_ids | confirmed | processed | emitted | old_output | load_target_list()
    for path in args.tried_file:
        excluded |= load_ids(path)
    candidates = select_candidates(tces, observed, excluded,
                                   args.mes_min, args.mes_max, args.limit)
    atomic_csv(pd.DataFrame({"tic_id": candidates}), args.output)
    append_emitted(args.emitted_file, candidates)
    print(f"Wrote {len(candidates):,} new TIC IDs to {args.output}")
    print(f"Updated emitted-ID ledger: {args.emitted_file}")
    print("These are targets to vet, not planet detections.")
    print("First 20:", candidates[:20])

if __name__ == "__main__":
    main()
