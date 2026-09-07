#!/usr/bin/env python3
import os
import re
import signal
import warnings
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astroquery.mast import Catalogs

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CACHE_DIR = "tess_verify_cache"   # change to "tess_lc_cache" if you already have downloads
SAVE_DIR = "tess_verify_plots"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

QUALITY_BITMASK = "hard"
FLATTEN_WINDOW = 401

R_SUN_IN_R_EARTH = 109.2
R_SUN_IN_R_JUPITER = 9.73

CANDIDATES = [
    {"tic_id": 382145521, "label": "TLS pass 1/2 signal", "period": 3.75482, "duration_hours": 0.30},
    {"tic_id": 382145521, "label": "TLS pass 3 signal", "period": 3.00386, "duration_hours": 0.41},
    {"tic_id": 41232189, "label": "TLS signal", "period": 0.78257, "duration_hours": 0.25},
    {"tic_id": 41232189, "label": "disagreeing BLS period", "period": 0.52, "duration_hours": 0.25},
    {"tic_id": 685125879, "label": "TLS signal", "period": 3.46615, "duration_hours": 1.3},
]

def sanitize_filename(text):
    """Replace any character that is not alphanumeric, underscore, hyphen,
    or period with an underscore."""
    return re.sub(r'[^A-Za-z0-9_.\-]', '_', text)

class DownloadTimeoutError(Exception):
    pass

def _download_with_timeout(search_row, cache_dir, quality_bitmask, timeout=60):
    """Download a single sector with a timeout (seconds)."""
    def handler(signum, frame):
        raise DownloadTimeoutError("Download timed out")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        lc = search_row.download(download_dir=cache_dir, quality_bitmask=quality_bitmask)
        signal.alarm(0)
        return lc
    except DownloadTimeoutError:
        signal.alarm(0)
        print("  Download timed out – skipping this sector")
        return None
    except Exception as e:
        signal.alarm(0)
        print(f"  Download failed: {e}")
        return None

def get_full_cleaned_lc(tic_id, max_sectors=4, lc_coll=None):
    """Download (or reuse) up to max_sectors sectors, stitch, clean."""
    if lc_coll is not None:
        return lc_coll.stitch(), lc_coll

    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if len(search) == 0:
        return None, None

    search = search[:max_sectors]

    print(f"  Downloading up to {len(search)} sector(s) individually...")
    lc_list = []
    for idx, row in enumerate(search):
        print(f"  Sector {idx+1}/{len(search)}: downloading...", flush=True)
        lc = _download_with_timeout(row, CACHE_DIR, QUALITY_BITMASK)
        if lc is not None:
            lc_list.append(lc)
            print(f"  Sector {idx+1} downloaded successfully")
        else:
            print(f"  Sector {idx+1} skipped")

    if not lc_list:
        return None, None

    lc_coll = lk.LightCurveCollection(lc_list)
    lc = lc_coll.stitch()
    lc = lc.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
    lc = lc.flatten(window_length=FLATTEN_WINDOW)
    lc = lc.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
    lc = lc.normalize()
    return lc, lc_coll

def find_epoch_for_period(lc, period, duration_hours, margin=0.02):
    t = lc.time.value
    y = lc.flux.value
    dy = lc.flux_err.value

    period_grid = np.linspace(period * (1 - margin), period * (1 + margin), 201)
    durations = np.array([duration_hours / 24])
    model = BoxLeastSquares(t, y, dy)
    result = model.power(period_grid, durations, oversample=5)
    power = result.power
    if not np.isfinite(power).any():
        return None
    best_idx = np.nanargmax(power)
    return float(result.transit_time[best_idx]), float(result.period[best_idx])

def get_stellar_radius(tic_id):
    try:
        table = Catalogs.query_criteria(catalog="Tic", ID=int(tic_id))
    except Exception as e:
        print(f"  Stellar radius lookup failed: {e}")
        return None
    if len(table) == 0:
        print("  No TIC catalog entry found.")
        return None
    row = table[0]
    if "rad" in table.colnames and row["rad"] is not None and not np.ma.is_masked(row["rad"]):
        return float(row["rad"])
    print(f"  'rad' column not usable - available columns: {table.colnames}")
    return None

def check_physical_plausibility(tic_id, depth):
    """Convert depth to planet radius using actual stellar radius."""
    print(f"\n  --- Physical plausibility (TIC {tic_id}) ---")
    if depth is None or not np.isfinite(depth) or depth <= 0:
        print("  No usable depth measurement to check.")
        return
    r_star = get_stellar_radius(tic_id)
    rp_rs = np.sqrt(depth)
    print(f"  Measured depth: {depth*1e3:.3f} ppt -> Rp/Rs = sqrt(depth) = {rp_rs:.4f}")
    if r_star is None or not np.isfinite(r_star):
        print("  No stellar radius available (or radius is NaN) - can't convert to a physical size.")
        return
    rp_rsun = rp_rs * r_star
    rp_rearth = rp_rsun * R_SUN_IN_R_EARTH
    rp_rjup = rp_rsun * R_SUN_IN_R_JUPITER
    print(f"  Star radius (TIC catalog): {r_star:.3f} R_sun")
    print(f"  Implied planet radius: {rp_rearth:.2f} R_earth ({rp_rjup:.3f} R_jupiter)")
    if rp_rjup > 20:
        print("  -> Far too large to be a planet at this depth/radius combination - "
              "this points to an eclipsing binary or a grazing/diluted system, not a planet.")
    elif rp_rjup > 2:
        print("  -> Larger than any known planet - possible brown dwarf or EB, treat with real skepticism.")
    elif rp_rjup > 1:
        print("  -> Roughly Jupiter-sized or larger - plausible for a hot Jupiter, especially around a small star.")
    else:
        print("  -> Physically plausible planet size.")

def validate_one_period(tic_id, label, period, duration_hours, lc_coll=None):
    print(f"\n{'='*70}\nTIC {tic_id} - {label} (P={period:.5f} d)\n{'='*70}")

    full_lc, lc_coll = get_full_cleaned_lc(tic_id, max_sectors=4, lc_coll=lc_coll)
    if full_lc is None:
        print("Could not download/clean this target - skipping.")
        return lc_coll

    epoch_result = find_epoch_for_period(full_lc, period, duration_hours)
    if epoch_result is None:
        print("Could not determine an epoch for this period - skipping.")
        return lc_coll
    epoch, refined_period = epoch_result
    print(f"Determined epoch T0={epoch:.4f} (refined period={refined_period:.5f}d, "
          f"input was {period:.5f}d)")

    # Use the refined period for all subsequent folding/sector checks
    period = refined_period
    duration = duration_hours / 24

    print(f"\n{'Sector':<8}{'N pts':<10}{'In-transit':<12}{'Depth (ppt)':<14}{'SNR':<8}")
    depths = []
    for i, sector_lc in enumerate(lc_coll):
        try:
            sec = sector_lc.remove_nans()
            sec = sec.remove_outliers(sigma_lower=15, sigma_upper=5)
            sec = sec.flatten(window_length=FLATTEN_WINDOW)
            sec = sec.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
            sec = sec.normalize()
        except Exception as e:
            print(f"{i+1:<8}sector failed to clean: {e}")
            continue

        t = sec.time.value
        y = sec.flux.value
        if len(t) < 50:
            continue
        phase = ((t - epoch + period / 2) % period) - period / 2
        in_transit = np.abs(phase) < duration / 2
        n_in = int(in_transit.sum())
        if n_in < 5 or n_in > len(t) * 0.5:
            print(f"{i+1:<8}{len(t):<10}{n_in:<12}insufficient/ambiguous coverage")
            continue
        out_flux = y[~in_transit]
        in_flux = y[in_transit]
        d = np.median(out_flux) - np.median(in_flux)
        noise = np.std(out_flux) / np.sqrt(n_in)
        snr = d / noise if noise > 0 else np.nan
        depths.append(d)
        print(f"{i+1:<8}{len(t):<10}{n_in:<12}{d*1e3:<14.3f}{snr:<8.2f}")

    mean_depth = None
    if depths:
        depths_arr = np.array(depths)
        mean_depth, std_depth = np.mean(depths_arr), np.std(depths_arr)
        print(f"\n  -> {len(depths_arr)} sector(s) measured independently: "
              f"mean={mean_depth*1e3:.3f} ppt, std={std_depth*1e3:.3f} ppt, "
              f"range=[{np.min(depths_arr)*1e3:.3f}, {np.max(depths_arr)*1e3:.3f}] ppt")
        sign_flips = np.any(depths_arr < 0) and np.any(depths_arr > 0)
        if std_depth > abs(mean_depth) or sign_flips:
            print("  -> WARNING: sector-to-sector scatter is large relative to the mean depth"
                  + (" and the sign flips between sectors" if sign_flips else "")
                  + " - not consistent with a single stable periodic signal.")
        else:
            print("  -> Depth is reasonably consistent across independent sectors.")
    else:
        print("  -> No sector had usable in-transit coverage for this period.")

    check_physical_plausibility(tic_id, mean_depth)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={'height_ratios': [1, 2]})

    t_full = full_lc.time.value
    y_full = full_lc.flux.value
    ax1.plot(t_full, y_full, 'k.', ms=1, alpha=0.5)
    transit_times = np.arange(epoch, t_full[-1], period)
    transit_times = transit_times[(transit_times >= t_full[0]) & (transit_times <= t_full[-1])]
    for tt in transit_times:
        ax1.axvline(tt, color='r', alpha=0.2, lw=1)
    ax1.set_title(f"TIC {tic_id} - {label} (P={period:.5f}d, T0={epoch:.2f})")
    ax1.set_xlabel("Time (BTJD)")
    ax1.set_ylabel("Normalized flux")

    folded = full_lc.fold(period=period, epoch_time=epoch)
    folded.scatter(ax=ax2, c='k', alpha=0.4)
    folded.bin(50).scatter(ax=ax2, c='r', s=20)
    ax2.set_xlim(-2*duration/period, 2*duration/period)
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Normalized flux")
    ax2.set_title("Folded transit (binned in red)")

    plt.tight_layout()

    safe_label = sanitize_filename(label)
    fname = os.path.join(SAVE_DIR, f"TIC{tic_id}_{safe_label}_P{period:.5f}d_verify.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to {fname}")
    plt.close(fig)

    return lc_coll

if __name__ == "__main__":
    lc_coll_cache = {}
    for cand in CANDIDATES:
        tic_id = cand["tic_id"]
        cached_coll = lc_coll_cache.get(tic_id)
        result_coll = validate_one_period(
            tic_id, cand["label"], cand["period"], cand["duration_hours"], lc_coll=cached_coll
        )
        if result_coll is not None:
            lc_coll_cache[tic_id] = result_coll

    print("\n" + "=" * 70)
    print("ALL CANDIDATES CHECKED")
    print("=" * 70)
