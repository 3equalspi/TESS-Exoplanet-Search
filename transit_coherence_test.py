#!/usr/bin/env python3
import os
import re
import signal
import warnings
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
TARGET_TIC = 41232189
CACHE_DIR = "tess_coherence_cache"
SAVE_DIR = "tess_coherence_plots"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

QUALITY_BITMASK = "hard"
FLATTEN_WINDOW = 401
MAX_SECTORS = 6 
TRANSIT_WINDOW = 0.5
OUT_TRANSIT_PAD = 0.1 
OUT_TRANSIT_WIDTH = 0.2
PERIODS = {
    0.3821: 0.20,
    0.5095: 0.25,
    0.5200: 0.25,
    0.78257: 0.30,
    1.56514: 0.50,
}
class DownloadTimeoutError(Exception):
    pass

def _download_with_timeout(search_row, cache_dir, quality_bitmask, timeout=60):
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

def get_cleaned_sectors(tic_id, max_sectors=MAX_SECTORS):
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if len(search) == 0:
        print("No SPOC data found.")
        return None
    search = search[:max_sectors]
    print(f"Downloading up to {len(search)} sectors...")
    sector_list = []
    for idx, row in enumerate(search):
        print(f"  Sector {idx+1}/{len(search)}: downloading...", flush=True)
        lc = _download_with_timeout(row, CACHE_DIR, QUALITY_BITMASK)
        if lc is not None:
            sector_list.append(lc)
            print(f"  Sector {idx+1} downloaded")
    if not sector_list:
        return None
    cleaned_sectors = []
    for sec in sector_list:
        try:
            sec = sec.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
            sec = sec.flatten(window_length=FLATTEN_WINDOW)
            sec = sec.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
            sec = sec.normalize()
            cleaned_sectors.append(sec)
        except Exception as e:
            print(f"  Sector cleaning failed: {e}")
    return lk.LightCurveCollection(cleaned_sectors)

def find_epoch_for_period(lc, period, duration_hours, margin=0.02):
    t = lc.time.value
    y = lc.flux.value
    dy = lc.flux_err.value
    period_grid = np.linspace(period * (1 - margin), period * (1 + margin), 201)
    durations = np.array([duration_hours / 24])
    model = BoxLeastSquares(t, y, dy)
    result = model.power(period_grid, durations, oversample=5)
    if not np.isfinite(result.power).any():
        return None, None
    best_idx = np.nanargmax(result.power)
    return float(result.transit_time[best_idx]), float(result.period[best_idx])

def measure_individual_transits(sector_lc, period, epoch, duration_hours):
    t = sector_lc.time.value
    y = sector_lc.flux.value
    dy = sector_lc.flux_err.value
    duration = duration_hours / 24.0
    half_dur = duration / 2.0
    pad = OUT_TRANSIT_PAD
    width = OUT_TRANSIT_WIDTH
    t_start, t_end = t[0], t[-1]
    transit_times = np.arange(epoch, t_end + period, period)
    transit_times = transit_times[(transit_times >= t_start) & (transit_times <= t_end)]

    depths = []
    errors = []
    times = []
    for tt in transit_times:
        in_mask = (t >= tt - half_dur) & (t <= tt + half_dur)
        n_in = np.sum(in_mask)
        if n_in < 3:
            continue
        baseline_mask = (t >= tt - width - half_dur) & (t <= tt + width + half_dur)
        baseline_mask &= ~in_mask
        baseline_mask &= (np.abs(t - tt) > half_dur + pad)
        if np.sum(baseline_mask) < 5:
            continue

        out_flux = y[baseline_mask]
        in_flux = y[in_mask]
        baseline = np.median(out_flux)
        depth = baseline - np.median(in_flux)
        sigma_out = np.std(out_flux) / np.sqrt(n_in)
        depth_err = sigma_out
        sigma_in = np.std(in_flux) / np.sqrt(n_in)
        depth_err = np.sqrt(sigma_in**2 + sigma_out**2)

        depths.append(depth)
        errors.append(depth_err)
        times.append(tt)
    return np.array(depths), np.array(errors), np.array(times)

def analyze_period(tic_id, period, duration_hours, sector_collection):
    print(f"\n{'='*70}\nTesting period P = {period:.5f} d (duration guess {duration_hours:.2f} h)\n{'='*70}")

    full_lc = sector_collection.stitch()
    epoch, refined_period = find_epoch_for_period(full_lc, period, duration_hours)
    if epoch is None:
        print("Could not determine epoch – skipping.")
        return
    period = refined_period
    print(f"Adopted T0 = {epoch:.4f}, refined period = {period:.5f} d")

    all_depths = []
    all_errors = []
    all_times = []
    sector_info = []
    for i, sec in enumerate(sector_collection):
        d, e, t = measure_individual_transits(sec, period, epoch, duration_hours)
        if len(d) > 0:
            all_depths.extend(d)
            all_errors.extend(e)
            all_times.extend(t)
            sector_info.append((i+1, len(d)))
        else:
            sector_info.append((i+1, 0))

    if len(all_depths) == 0:
        print("No individual transits with enough data.")
        return

    all_depths = np.array(all_depths)
    all_errors = np.array(all_errors)
    all_times = np.array(all_times)

    mean_depth = np.mean(all_depths)
    std_depth = np.std(all_depths)
    chi2 = np.sum(((all_depths - mean_depth) / all_errors)**2)
    ndof = len(all_depths) - 1
    reduced_chi2 = chi2 / ndof if ndof > 0 else np.nan

    print(f"\nMeasured {len(all_depths)} individual transits across {len(sector_collection)} sectors.")
    print(f"Depth statistics: mean = {mean_depth*1e3:.3f} ppt, std = {std_depth*1e3:.3f} ppt")
    print(f"Reduced chi² (consistency with constant depth): {reduced_chi2:.2f} (ndof={ndof})")
    if reduced_chi2 > 2.0:
        print("  → Depths are NOT consistent with a constant value (likely variability/EB).")
    else:
        print("  → Depths are reasonably consistent (but still check plots).")
    print("\nSector  #transits")
    for sidx, ntr in sector_info:
        print(f"{sidx:<6} {ntr}")
    safe_label = re.sub(r'[^A-Za-z0-9_.\-]', '_', f"P{period:.5f}d")
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(3, 1, 1)
    folded = full_lc.fold(period=period, epoch_time=epoch)
    folded.scatter(ax=ax1, c='k', alpha=0.3, label='individual points')
    folded.bin(50).scatter(ax=ax1, c='r', s=30, label='binned')
    ax1.axvline(0, color='b', linestyle='--', alpha=0.5)
    ax1.set_xlim(-3*duration_hours/24/period, 3*duration_hours/24/period)  # zoom
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Normalized flux")
    ax1.set_title(f"Folded light curve (P={period:.5f} d, T0={epoch:.2f})")
    ax1.legend()
    ax2 = plt.subplot(3, 1, 2)
    ax2.errorbar(all_times, all_depths*1e3, yerr=all_errors*1e3,
                 fmt='o', capsize=3, alpha=0.7)
    ax2.axhline(mean_depth*1e3, color='r', linestyle='--', label=f'mean = {mean_depth*1e3:.2f} ppt')
    ax2.axhline(0, color='k', alpha=0.3)
    ax2.set_xlabel("Time (BTJD)")
    ax2.set_ylabel("Depth (ppt)")
    ax2.set_title("Individual transit depths vs time")
    ax2.legend()
    ax3 = plt.subplot(3, 1, 3)
    ax3.hist(all_depths*1e3, bins=20, alpha=0.6, edgecolor='black')
    ax3.axvline(mean_depth*1e3, color='r', linestyle='--')
    ax3.set_xlabel("Depth (ppt)")
    ax3.set_ylabel("Number")
    ax3.set_title("Depth distribution")

    plt.tight_layout()
    fname = os.path.join(SAVE_DIR, f"TIC{tic_id}_P{period:.5f}d_coherence.png")
    plt.savefig(fname, dpi=150)
    print(f"Plot saved to {fname}")
    plt.close(fig)
    return {
        "period": period,
        "n_transits": len(all_depths),
        "mean_depth_ppt": mean_depth*1e3,
        "std_depth_ppt": std_depth*1e3,
        "reduced_chi2": reduced_chi2,
    }

# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    print(f"Running per‑transit coherence test for TIC {TARGET_TIC} (η¹ Dor)")
    sectors = get_cleaned_sectors(TARGET_TIC)
    if sectors is None or len(sectors) == 0:
        print("No data available – exiting.")
        exit(1)

    results = []
    for period, dur_hours in PERIODS.items():
        res = analyze_period(TARGET_TIC, period, dur_hours, sectors)
        if res is not None:
            results.append(res)
    print("\n" + "="*80)
    print("SUMMARY OF PERIOD TESTS")
    print("="*80)
    print(f"{'Period (d)':<12}{'N transits':<12}{'Mean depth (ppt)':<18}{'Std (ppt)':<12}{'Red. chi²':<12}")
    for r in results:
        print(f"{r['period']:<12.5f}{r['n_transits']:<12}{r['mean_depth_ppt']:<18.3f}{r['std_depth_ppt']:<12.3f}{r['reduced_chi2']:<12.2f}")
    print("\nA real transiting planet should have a low std relative to mean and reduced chi² near 1.")
