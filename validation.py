#!/usr/bin/env python3
import os
import sys
import glob
import time
import signal
import warnings
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from transitleastsquares import transitleastsquares

class StepTimeout(Exception):
    pass

@contextmanager
def time_limit(seconds, label="step"):
    def _handler(signum, frame):
        raise StepTimeout(f"{label} exceeded its {seconds//60:.0f}-minute budget")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
SDE_THRESHOLD = 7.0
PERIOD_MIN = 0.5
PERIOD_MAX = 20.0
MIN_TRANSITS = 3
TRANSIT_DEPTH_MIN = 0.0001
FLATTEN_WINDOW = 401
MAX_POINTS = 200_000
MIN_OBSERVED_TRANSITS = 5
MIN_TRANSIT_COVERAGE_FRAC = 0.2
SHORT_LONG_SPLIT_DAYS = 3.0
SHORT_REGIME_DURATIONS = np.linspace(0.02, 0.3, 8)
LONG_REGIME_DURATIONS = np.linspace(0.1, 0.3, 5)
TLS_MAX_POINTS = 60_000
TLS_OVERSAMPLING_FACTOR = 1
QUALITY_BITMASK = "hard"
MAX_MEAN_FLUX_DEVIATION = 0.01
CATASTROPHIC_FLUX_LOW = 0.5
CATASTROPHIC_FLUX_HIGH = 1.5
USE_BLS_PRETRIAGE = True
BLS_PEAK_SNR_MIN = 6.0
BLS_N_PERIODS = 15_000
MASK_AND_RESEARCH_PASSES = 3

CACHE_DIR = "tess_lc_cache"
SAVE_DIR = "/home/life/hail_mary"
PLOT_ALL = True
BLS_TIMEOUT_S = 10 * 60
TLS_TIMEOUT_S = 40 * 60

warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
TARGET_LIST = [
    25117273,
    30537037,
    32089054,
    32152605,
    37302368,
    38846471,
    41232189,
    41483281,
    55557836,
    65541415,
    140757590,
    140938247,
    141412532,
    141770047,
    149120276,
    149154927,
    149542947,
    150066631,
    150100627,
    150165657,
    150441015,
    167344197,
    177082055
]
def _purge_cache_for_tic(tic_id):
    pattern = os.path.join(CACHE_DIR, "**", f"*{int(tic_id):016d}*")
    removed = 0
    for path in glob.glob(pattern, recursive=True):
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed

def _looks_transient(msg):
    m = msg.lower()
    return any(k in m for k in (
        "remotedisconnected", "connection aborted", "connection reset",
        "max retries exceeded", "timeout", "timed out", "ssleoferror",
        "unexpected_eof", "temporarily unavailable", "502", "503", "504",
    ))

DOWNLOAD_MAX_ATTEMPTS = 4
DOWNLOAD_RETRY_BACKOFF = [15, 45, 120]
DOWNLOAD_STEP_TIMEOUT_S = 900
CLEAN_STEP_TIMEOUT_S = 180

def get_stitched_lc(tic_id):
    last_err = None
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            suffix = f" (attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS})" if attempt > 1 else ""
            print(f"  -> Searching MAST...{suffix}", flush=True)
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
            if len(search) == 0:
                print(f"  -> No SPOC data for TIC {tic_id}")
                return None

            print(f"  -> Downloading {len(search)} sector(s)...", flush=True)
            t0 = time.time()
            with time_limit(DOWNLOAD_STEP_TIMEOUT_S, "download"):
                lc_coll = search.download_all(download_dir=CACHE_DIR, quality_bitmask=QUALITY_BITMASK)
            print(f"  -> Download done ({time.time()-t0:.1f}s)", flush=True)
            if lc_coll is None:
                return None
            t_step = time.time()
            with time_limit(CLEAN_STEP_TIMEOUT_S, "stitch"):
                lc = lc_coll.stitch()
            print(f"     stitched: {len(lc)} points ({time.time()-t_step:.1f}s)", flush=True)
            if len(lc.time) > MAX_POINTS:
                factor = -(-len(lc.time) // MAX_POINTS)
                t_step = time.time()
                with time_limit(CLEAN_STEP_TIMEOUT_S, "pre-bin"):
                    lc = bin_points(lc, factor)
                    lc = lc.remove_nans()
                print(f"     pre-binned by {factor}: {len(lc)} points ({time.time()-t_step:.1f}s)", flush=True)

            t_step = time.time()
            with time_limit(CLEAN_STEP_TIMEOUT_S, "clean pass 1"):
                lc = lc.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
            print(f"     cleaned (pass 1): {len(lc)} points ({time.time()-t_step:.1f}s)", flush=True)

            t_step = time.time()
            with time_limit(CLEAN_STEP_TIMEOUT_S, "flatten"):
                lc = lc.flatten(window_length=FLATTEN_WINDOW)
            print(f"     flattened: {len(lc)} points ({time.time()-t_step:.1f}s)", flush=True)
            t_step = time.time()
            with time_limit(CLEAN_STEP_TIMEOUT_S, "clean pass 2"):
                lc = lc.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
            print(f"     cleaned (pass 2): {len(lc)} points ({time.time()-t_step:.1f}s)", flush=True)

            t_step = time.time()
            with time_limit(CLEAN_STEP_TIMEOUT_S, "normalize"):
                lc = lc.normalize()
            print(f"     normalized ({time.time()-t_step:.1f}s)", flush=True)
            yv = lc.flux.value
            print(f"     flux diagnostics: mean={np.mean(yv):.4f} median={np.median(yv):.4f} "
                  f"min={np.min(yv):.4f} max={np.max(yv):.4f}", flush=True)
            print(f"       p0.1={np.percentile(yv,0.1):.4f} p1={np.percentile(yv,1):.4f} "
                  f"p99={np.percentile(yv,99):.4f} p99.9={np.percentile(yv,99.9):.4f}", flush=True)
            n_catastrophic = int(np.sum((yv < CATASTROPHIC_FLUX_LOW) | (yv > CATASTROPHIC_FLUX_HIGH)))
            if n_catastrophic > 0:
                print(f"     -> removing {n_catastrophic} catastrophic point(s) "
                      f"(flux < {CATASTROPHIC_FLUX_LOW} or > {CATASTROPHIC_FLUX_HIGH})", flush=True)
                lc = lc[(yv >= CATASTROPHIC_FLUX_LOW) & (yv <= CATASTROPHIC_FLUX_HIGH)]

            mean_flux = float(np.nanmean(lc.flux.value))
            if not np.isfinite(mean_flux) or abs(mean_flux - 1) > MAX_MEAN_FLUX_DEVIATION:
                print(f"  -> Post-cleaning mean flux is {mean_flux:.3f} (should be ~1). "
                      f"Data still looks contaminated (stray light / cosmic rays?); skipping.")
                return None

            return lc

        except StepTimeout as e:
            last_err = e
            print(f"  -> {e} for TIC {tic_id}.", flush=True)
            if attempt < DOWNLOAD_MAX_ATTEMPTS:
                wait = DOWNLOAD_RETRY_BACKOFF[min(attempt - 1, len(DOWNLOAD_RETRY_BACKOFF) - 1)]
                print(f"     Retrying in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            print(f"  -> Giving up on TIC {tic_id} after repeated timeouts.")
            return None

        except Exception as e:
            last_err = e
            msg = str(e)

            if "corrupt" in msg.lower() or "truncated" in msg.lower():
                n = _purge_cache_for_tic(tic_id)
                print(f"  -> Cached file for TIC {tic_id} looks corrupted (likely an "
                      f"interrupted download from a previous Ctrl-C). Removed {n} file(s) "
                      f"from the cache, retrying immediately...", flush=True)
                continue  # no backoff needed, this is a local fix, not a server issue

            if attempt < DOWNLOAD_MAX_ATTEMPTS and _looks_transient(msg):
                wait = DOWNLOAD_RETRY_BACKOFF[min(attempt - 1, len(DOWNLOAD_RETRY_BACKOFF) - 1)]
                print(f"  -> {msg}\n     Looks like a transient MAST connection issue - "
                      f"waiting {wait}s before retrying...", flush=True)
                time.sleep(wait)
                continue

            print(f"  -> Error downloading TIC {tic_id}: {e}")
            return None

    print(f"  -> Giving up on TIC {tic_id} after {DOWNLOAD_MAX_ATTEMPTS} attempts: {last_err}")
    return None

def bls_pretriage(lc, period_min=PERIOD_MIN, period_max=PERIOD_MAX, durations=None):
    durations = SHORT_REGIME_DURATIONS if durations is None else durations
    t = lc.time.value
    y = lc.flux.value
    dy = lc.flux_err.value
    period_max = min(period_max, (t[-1] - t[0]) / 2)
    if period_max <= period_min:
        return False, 0.0, np.nan

    period_grid = np.exp(np.linspace(np.log(period_min), np.log(period_max), BLS_N_PERIODS))

    model = BoxLeastSquares(t, y, dy)
    result = model.power(period_grid, durations, oversample=5)
    power = result.power
    if not np.isfinite(power).any():
        return True, np.nan, np.nan

    med = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - med)) * 1.4826 + 1e-12
    peak_snr = (np.nanmax(power) - med) / mad
    best_period = result.period[np.nanargmax(power)]
    return peak_snr >= BLS_PEAK_SNR_MIN, peak_snr, best_period

TLS_USE_DY = False

def run_tls(lc, period_min=PERIOD_MIN, period_max=PERIOD_MAX):
    if len(lc.time) > TLS_MAX_POINTS:
        factor = -(-len(lc.time) // TLS_MAX_POINTS)  # ceiling division
        print(f"  -> Light curve has {len(lc.time)} points; binning by {factor} "
              f"for TLS (cap={TLS_MAX_POINTS}) to keep runtime reasonable...", flush=True)
        lc = bin_points(lc, factor)
        lc = lc.remove_nans()

    t = np.array(lc.time.value, dtype=float)
    y = np.array(lc.flux.value, dtype=float)
    dy = np.array(lc.flux_err.value, dtype=float)

    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy)
    t, y, dy = t[good], y[good], dy[good]
    med = np.nanmedian(y)
    y = y / med
    dy = dy / med
    if TLS_USE_DY:
        tls = transitleastsquares(t, y, dy)
    else:
        tls = transitleastsquares(t, y)

    results = tls.power(
        period_min=period_min,
        period_max=min(period_max, (t[-1] - t[0]) / 2),
        n_transits_min=MIN_TRANSITS,
        transit_depth_min=TRANSIT_DEPTH_MIN,
        oversampling_factor=TLS_OVERSAMPLING_FACTOR,
    )
    return results

def transit_coverage(lc, tls_results):
    period = tls_results.period
    epoch = tls_results.T0
    duration = tls_results.duration
    if period <= 0 or np.isnan(period):
        return 0.0, 0

    t = lc.time.value
    t_start, t_end = t[0], t[-1]

    transit_times = np.arange(epoch, t_end, period)
    transit_times = transit_times[transit_times >= t_start]

    total = len(transit_times)
    observed = 0
    half_dur = duration * 0.5

    for tt in transit_times:
        in_window = (t >= tt - half_dur) & (t <= tt + half_dur)
        if np.any(in_window):
            observed += 1

    frac = observed / total if total > 0 else 0.0
    return frac, observed

def bin_points(lc, factor):
    t = lc.time.value
    y = lc.flux.value
    dy = lc.flux_err.value if lc.flux_err is not None else np.full(len(t), np.nan)
    n_bins = len(t) // factor
    if n_bins == 0:
        return lc
    usable = n_bins * factor
    t_b = t[:usable].reshape(n_bins, factor)
    y_b = y[:usable].reshape(n_bins, factor)
    dy_b = dy[:usable].reshape(n_bins, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        t_mean = np.nanmean(t_b, axis=1)
        y_mean = np.nanmean(y_b, axis=1)
        dy_mean = np.sqrt(np.nanmean(dy_b ** 2, axis=1)) / np.sqrt(factor)
    return lk.LightCurve(time=t_mean, flux=y_mean, flux_err=dy_mean)

MASK_MARGIN_FACTOR = 0.75

def mask_transits(lc, period, epoch, duration, margin_factor=MASK_MARGIN_FACTOR):
    t = lc.time.value
    if period <= 0 or np.isnan(period) or duration <= 0 or np.isnan(duration):
        return lc
    half_window = margin_factor * duration
    phase = ((t - epoch + period / 2) % period) - period / 2
    in_transit = np.abs(phase) < half_window
    print(f"     masking {in_transit.sum()} of {len(t)} points in-transit "
          f"(P={period:.4f}d, dur={duration*24:.2f}h) before re-searching residual", flush=True)
    return lc[~in_transit]

def _fold_and_bin(t, y, fold_period, epoch, n_bins):
    phase = (((t - epoch) / fold_period + 0.5) % 1.0) - 0.5
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)
    bin_medians = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = bin_idx == i
        if np.any(sel):
            bin_medians[i] = np.median(y[sel])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, bin_medians

def vet_candidate(lc, tls_results):
    period = tls_results.period
    epoch = tls_results.T0
    duration = tls_results.duration
    depth = 1.0 - tls_results.depth

    t = lc.time.value
    y = lc.flux.value
    MIN_BINS_ACROSS_TRANSIT = 6
    BIN_COUNT_CAP = 2000
    n_bins_2x = int(np.clip(MIN_BINS_ACROSS_TRANSIT * (2 * period) / max(duration, 1e-6), 30, BIN_COUNT_CAP))
    n_bins_1x = int(np.clip(MIN_BINS_ACROSS_TRANSIT * period / max(duration, 1e-6), 50, BIN_COUNT_CAP))

    phases, flux = _fold_and_bin(t, y, 2 * period, epoch, n_bins_2x)
    half_width_2x = 0.5 * duration / (2 * period)
    transit_mask = np.abs(phases) < half_width_2x
    transit_mask_05 = np.abs(phases - 0.5) < half_width_2x
    have_2x = np.isfinite(flux)
    resolvable_2x = (np.any(transit_mask & have_2x) and
                      np.any(~transit_mask & ~transit_mask_05 & have_2x))

    if resolvable_2x:
        median_out = np.nanmedian(flux[~transit_mask & ~transit_mask_05])
        depth_even = median_out - np.nanmedian(flux[transit_mask])
        depth_odd = (median_out - np.nanmedian(flux[transit_mask_05])
                     if np.any(transit_mask_05 & have_2x) else np.nan)
        odd_even_diff = 0.0 if np.isnan(depth_odd) else abs(depth_even - depth_odd) / max(abs(depth_even), 1e-6)
        odd_even_ok = odd_even_diff < 0.5
    else:
        odd_even_diff = np.nan
        odd_even_ok = False

    phases_s, flux_s = _fold_and_bin(t, y, period, epoch, n_bins_1x)
    half_width_1x = 0.5 * duration / period
    sec_mask = np.abs(phases_s - 0.5) < half_width_1x
    have_1x = np.isfinite(flux_s)
    resolvable_1x = np.any(sec_mask & have_1x) and np.any(~sec_mask & have_1x)

    if resolvable_1x:
        median_out_s = np.nanmedian(flux_s[~sec_mask])
        sec_depth = median_out_s - np.nanmedian(flux_s[sec_mask])
        secondary_ok = sec_depth < 0.3 * abs(depth)
    else:
        sec_depth = np.nan
        secondary_ok = False

    resolvable = resolvable_2x and resolvable_1x
    passed = resolvable and odd_even_ok and secondary_ok

    return {
        "period": period, "epoch": epoch, "depth": depth,
        "duration": duration, "odd_even_diff": odd_even_diff,
        "secondary_depth": sec_depth, "odd_even_ok": odd_even_ok,
        "secondary_ok": secondary_ok, "passed": passed, "resolvable": resolvable
    }

def plot_candidate(lc, tls_results, vet, tic_id, pass_suffix=""):
    period = vet["period"]
    epoch = vet["epoch"]
    duration = vet["duration"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})
    t = lc.time.value
    y = lc.flux.value
    axes[0].plot(t, y, 'k.', markersize=1, alpha=0.6)
    transit_times = np.arange(epoch, t[-1], period)
    transit_times = transit_times[(transit_times >= t[0]) & (transit_times <= t[-1])]
    for tt in transit_times:
        axes[0].axvline(tt, color='r', alpha=0.3, lw=1)
    axes[0].set_xlabel("Time (BTJD)")
    axes[0].set_ylabel("Normalised flux")
    title_suffix = f" [{pass_suffix}]" if pass_suffix else ""
    axes[0].set_title(f"TIC {tic_id}{title_suffix}: P={period:.3f} d  Depth={vet['depth']*1e3:.2f} ppt")
    fold = lc.fold(period=period, epoch_time=epoch, normalize_phase=True)
    fold.scatter(ax=axes[1], c='k', alpha=0.4)
    plot_n_bins = int(np.clip(6 * period / max(duration, 1e-6), 50, 2000))
    bin_phases, bin_flux = _fold_and_bin(t, y, period, epoch, plot_n_bins)
    axes[1].scatter(bin_phases, bin_flux, c='r', s=20)
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Normalised flux")
    axes[1].set_xlim(-2*duration/period, 2*duration/period)
    axes[1].set_title(f"Folded transit (vet passed = {vet['passed']})")
    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname_suffix = f"_{pass_suffix.replace(' ', '')}" if pass_suffix else ""
    fname = os.path.join(SAVE_DIR, f"TIC{tic_id}{fname_suffix}_candidate.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  -> Plot saved to {fname}")
    plt.show()



def validate_candidate_per_sector(tic_id, period, epoch, duration, extra_periods=None):
    print(f"\n{'='*70}\nPer-sector validation for TIC {tic_id}\n{'='*70}")
    try:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    except Exception as e:
        print(f"Search failed: {e}")
        return
    if len(search) == 0:
        print("No SPOC data found.")
        return
    try:
        lc_coll = search.download_all(download_dir=CACHE_DIR, quality_bitmask=QUALITY_BITMASK)
    except Exception as e:
        print(f"Download failed: {e}")
        return
    if lc_coll is None:
        print("Download returned nothing.")
        return

    periods_to_test = [("candidate", period, epoch, duration)]
    for label, p in (extra_periods or []):
        periods_to_test.append((label, p, epoch, duration))

    for label, p, ep, dur in periods_to_test:
        print(f"\n--- {label}: P={p:.5f} d, duration={dur*24:.2f} h ---")
        print(f"{'Sector':<8}{'N pts':<10}{'In-transit':<12}{'Depth (ppt)':<14}{'SNR':<8}")
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
            phase = ((t - ep + p / 2) % p) - p / 2
            in_transit = np.abs(phase) < dur / 2
            n_in = int(in_transit.sum())
            if n_in < 5 or n_in > len(t) * 0.5:
                print(f"{i+1:<8}{len(t):<10}{n_in:<12}insufficient/ambiguous coverage")
                continue
            out_flux = y[~in_transit]
            in_flux = y[in_transit]
            depth = np.median(out_flux) - np.median(in_flux)
            noise = np.std(out_flux) / np.sqrt(n_in)
            snr = depth / noise if noise > 0 else np.nan
            depths.append(depth)
            print(f"{i+1:<8}{len(t):<10}{n_in:<12}{depth*1e3:<14.3f}{snr:<8.2f}")

        if depths:
            depths = np.array(depths)
            mean_d, std_d = np.mean(depths), np.std(depths)
            print(f"  -> {len(depths)} sector(s) measured independently: "
                  f"mean={mean_d*1e3:.3f} ppt, std={std_d*1e3:.3f} ppt, "
                  f"range=[{np.min(depths)*1e3:.3f}, {np.max(depths)*1e3:.3f}] ppt")
            sign_flips = np.any(depths < 0) and np.any(depths > 0)
            if std_d > abs(mean_d) or sign_flips:
                print("  -> WARNING: sector-to-sector scatter is large relative to the mean "
                      "depth" + (" and the sign flips between sectors" if sign_flips else "") +
                      " - not consistent with a single stable periodic signal.")
        else:
            print("  -> No sector had usable in-transit coverage for this period.")

RESULTS_LOG = os.path.join(CACHE_DIR, "processed_targets.csv")

def load_processed():
    if not os.path.exists(RESULTS_LOG):
        return set()
    with open(RESULTS_LOG) as f:
        return {line.strip() for line in f if line.strip()}

def mark_processed(tic_id):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(RESULTS_LOG, "a") as f:
        f.write(f"{tic_id}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        if len(sys.argv) < 6:
            print("Usage: python3 tess_search.py --validate <tic_id> <period_days> "
                  "<epoch_btjd> <duration_hours> [extra_period_days ...]")
            print("Example (TIC 41232189, TLS period vs. the disagreeing BLS period):")
            print("  python3 tess_search.py --validate 41232189 0.78257 1325.7 0.25 0.52")
            sys.exit(1)
        v_tic = int(sys.argv[2])
        v_period = float(sys.argv[3])
        v_epoch = float(sys.argv[4])
        v_duration = float(sys.argv[5]) / 24.0
        v_extra = [(f"extra P={float(p):.5f}d", float(p)) for p in sys.argv[6:]]
        validate_candidate_per_sector(v_tic, v_period, v_epoch, v_duration, extra_periods=v_extra)
        sys.exit(0)

    processed = load_processed()
    remaining = [t for t in TARGET_LIST if str(t) not in processed]
    print(f"Searching {len(TARGET_LIST)} stars "
          f"({len(processed)} already done in a previous run, {len(remaining)} remaining).\n")

    for i, tic_id in enumerate(remaining):
        if i > 0:
            time.sleep(3)  # be a little polite to MAST between targets
        print(f"\n{'='*60}")
        print(f"Processing TIC {tic_id}...")

        try:
            lc = get_stitched_lc(tic_id)  # handles its own per-step timeouts/retries internally

            if lc is None or len(lc.time) < 100:
                print("  -> Skipped")
                continue

            if len(lc.time) > MAX_POINTS:
                # Defensive fallback - get_stitched_lc already caps to MAX_POINTS
                # before flatten() internally, so this shouldn't normally fire.
                factor = -(-len(lc.time) // MAX_POINTS)
                print(f"  -> Binning by {factor}")
                lc = bin_points(lc, factor)
                lc = lc.remove_nans()

            print(f"  -> Light curve: {len(lc.time)} points, span {lc.time[0].value:.2f} - {lc.time[-1].value:.2f}")
            baseline_days = lc.time.value[-1] - lc.time.value[0]
            median_cadence = float(np.median(np.diff(lc.time.value))) if len(lc.time) > 1 else 0.0
            duty_cycle = (len(lc.time) * median_cadence / baseline_days) if baseline_days > 0 else 0.0
            print(f"  -> Observational duty cycle: {duty_cycle*100:.1f}% of the {baseline_days:.0f}-day baseline has data")
            if duty_cycle < 0.25 * MIN_TRANSIT_COVERAGE_FRAC:
                print(f"  -> Too little of the baseline is actually observed for most periods to "
                      f"reach the {MIN_TRANSIT_COVERAGE_FRAC*100:.0f}% coverage requirement - "
                      f"skipping BLS/TLS for this star.")
                continue
            lc_search = lc
            for search_pass in range(1, MASK_AND_RESEARCH_PASSES + 1):
                if search_pass > 1:
                    print(f"  -- Pass {search_pass}/{MASK_AND_RESEARCH_PASSES}: re-searching residual "
                          f"({len(lc_search.time)} points left after masking) --")

                if USE_BLS_PRETRIAGE:
                    print("  -> Running BLS pre-triage...", flush=True)
                    t_bls = time.time()
                    try:
                        with time_limit(BLS_TIMEOUT_S, "BLS pre-triage"):
                            bls_passed, peak_snr, best_bls_period = bls_pretriage(lc_search)
                    except StepTimeout as e:
                        print(f"  -> {e} - stopping search passes for this star.")
                        break
                    print(f"  -> BLS pre-triage: peak SNR={peak_snr:.1f} at P~{best_bls_period:.2f}d "
                          f"({time.time()-t_bls:.1f}s)")
                    if not bls_passed:
                        if search_pass == 1:
                            print("  -> No plausible periodic dip in fast scan, skipping expensive TLS run.")
                        else:
                            print("  -> No further significant signal in the residual - stopping here.")
                        break

                print("  -> Running TLS search (this can take several minutes)...", flush=True)
                t_tls = time.time()
                try:
                    with time_limit(TLS_TIMEOUT_S, "TLS search"):
                        tls_results = run_tls(lc_search)
                except StepTimeout as e:
                    print(f"  -> {e} - stopping search passes for this star.")
                    break
                except Exception as e:
                    print(f"  -> TLS failed: {e}")
                    break
                print(f"  -> TLS done ({time.time()-t_tls:.1f}s)")

                if np.isnan(tls_results.SDE) or tls_results.period == 0:
                    print("  -> No transit found.")
                    break

                period = tls_results.period
                depth = 1.0 - tls_results.depth
                duration = tls_results.duration
                sde = tls_results.SDE

                coverage_frac, n_obs = transit_coverage(lc_search, tls_results)
                print(f"  -> Period: {period:.4f} d, SDE={sde:.2f}, Depth={depth*1e3:.2f} ppt, "
                      f"Duration={duration*24:.2f} h")
                print(f"  -> Transits with data: {n_obs} ({coverage_frac*100:.1f}%)")
                if depth > 0.1 or depth < 0.0:
                    print(f"  -> NOTE: depth {depth*1e3:.1f} ppt is not a physically plausible "
                          f"planet transit - likely an eclipsing binary or residual normalization issue.")

                passed_all_gates = True
                if coverage_frac < MIN_TRANSIT_COVERAGE_FRAC:
                    print("  -> Coverage fraction too low, skipping.")
                    passed_all_gates = False
                elif n_obs < MIN_OBSERVED_TRANSITS:
                    print(f"  -> Only {n_obs} transits in data, need >= {MIN_OBSERVED_TRANSITS}.")
                    passed_all_gates = False
                elif sde < SDE_THRESHOLD:
                    print("  -> SDE below threshold, skipping.")
                    passed_all_gates = False
                elif depth > 0.1 or depth < 0.0:
                    print("  -> Depth unrealistic, skipping.")
                    passed_all_gates = False
                elif duration > 0.3 * period:
                    print("  -> Duration too long, skipping.")
                    passed_all_gates = False

                if passed_all_gates:
                    try:
                        vet = vet_candidate(lc_search, tls_results)
                        if vet["resolvable"]:
                            print(f"  -> Odd/even diff: {vet['odd_even_diff']:.2f}, "
                                  f"Secondary depth: {vet['secondary_depth']*1e3:.2f} ppt")
                            print(f"  -> Vetting {'PASSED' if vet['passed'] else 'FAILED'}")
                        else:
                            print("  -> Vetting INCONCLUSIVE - transit too narrow relative to the "
                                  "phase-folded bins to resolve odd/even or secondary-eclipse depth.")
                        if PLOT_ALL:
                            plot_candidate(lc_search, tls_results, vet, tic_id,
                                            pass_suffix=f"pass{search_pass}" if search_pass > 1 else "")
                    except Exception as e:
                        print(f"  -> Vetting/plotting failed unexpectedly: {e}")
                        print("  -> Continuing without vetting for this pass.")
                if passed_all_gates and search_pass < MASK_AND_RESEARCH_PASSES:
                    lc_search = mask_transits(lc_search, period, tls_results.T0, duration)
                    if len(lc_search.time) < 100:
                        print("  -> Too few points left after masking - stopping here.")
                        break
                elif not passed_all_gates:
                    print("  -> Not masking - this result didn't pass the gates, so it "
                          "shouldn't get to delete data from the residual search.")
                    break
        finally:
            mark_processed(tic_id)

    print("\nDone.")
