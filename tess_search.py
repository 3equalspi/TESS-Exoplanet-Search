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


#Settings
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

# -----------------------------------------------------------------
# Your target list
# -----------------------------------------------------------------
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

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
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
                factor = -(-len(lc.time) // MAX_POINTS)  # ceiling division
                t_step = time.time()
                with time_limit(CLEAN_STEP_TIMEOUT_S, "pre-bin"):
                    lc = lc.bin(binsize=factor)
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
        lc = lc.bin(binsize=factor)
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

MASK_MARGIN_FACTOR = 0.75  # mask +/- this many transit durations around each predicted center

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

def _phase_bin_numpy(phase, flux, n_bins):
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)

    valid = np.isfinite(phase) & np.isfinite(flux)

    phase = phase[valid]
    flux = flux[valid]

    if len(phase) == 0:
        return np.array([]), np.array([])

    phase = ((phase + 0.5) % 1.0) - 0.5

    edges = np.linspace(-0.5, 0.5, int(n_bins) + 1)

    indices = np.digitize(phase, edges) - 1

    indices = np.clip(indices, 0, int(n_bins) - 1)

    binned_phase = []
    binned_flux = []

    for i in range(int(n_bins)):

        mask = indices == i

        if not np.any(mask):
            continue
        binned_phase.append(
            np.median(phase[mask])
        )

        binned_flux.append(
            np.median(flux[mask])
        )

    return (
        np.asarray(binned_phase),
        np.asarray(binned_flux),
    )


def _phase_distance(phase, center):
    return np.abs(
        ((phase - center + 0.5) % 1.0) - 0.5
    )


def vet_candidate(lc, tls_results):

    period = float(tls_results.period)
    epoch = float(tls_results.T0)
    duration = float(tls_results.duration)
    depth = 1.0 - float(tls_results.depth)


    if (
        not np.isfinite(period)
        or not np.isfinite(epoch)
        or not np.isfinite(duration)
        or period <= 0
        or duration <= 0
        or depth <= 0
    ):
        return {
            "period": period,
            "epoch": epoch,
            "depth": depth,
            "duration": duration,
            "odd_even_diff": np.nan,
            "secondary_depth": np.nan,
            "odd_even_ok": False,
            "secondary_ok": False,
            "passed": False,
            "resolvable": False,
        }

    MIN_BINS_ACROSS_TRANSIT = 6
    BIN_COUNT_CAP = 2000

    n_bins_2x = int(
        np.clip(
            MIN_BINS_ACROSS_TRANSIT * (2.0 * period) / duration,
            30,
            BIN_COUNT_CAP,
        )
    )

    n_bins_1x = int(
        np.clip(
            MIN_BINS_ACROSS_TRANSIT * period / duration,
            50,
            BIN_COUNT_CAP,
        )
    )

    print(
        f"     vetting phase bins: "
        f"2x={n_bins_2x}, 1x={n_bins_1x}",
        flush=True,
    )
    time = np.asarray(
        lc.time.value,
        dtype=float,
    )

    flux = np.asarray(
        lc.flux.value,
        dtype=float,
    )

    valid = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    time = time[valid]
    flux = flux[valid]

    if len(time) < 100:
        return {
            "period": period,
            "epoch": epoch,
            "depth": depth,
            "duration": duration,
            "odd_even_diff": np.nan,
            "secondary_depth": np.nan,
            "odd_even_ok": False,
            "secondary_ok": False,
            "passed": False,
            "resolvable": False,
        }

    phase_2x = (
        (time - epoch) / (2.0 * period)
    )

    phase_2x = (
        phase_2x + 0.5
    ) % 1.0 - 0.5

    phases_2x, flux_2x = _phase_bin_numpy(
        phase_2x,
        flux,
        n_bins_2x,
    )

    half_width_2x = (
        0.5 * duration / (2.0 * period)
    )

    primary_mask = (
        _phase_distance(
            phases_2x,
            0.0,
        )
        < half_width_2x
    )

    secondary_mask_2x = (
        _phase_distance(
            phases_2x,
            0.5,
        )
        < half_width_2x
    )

    outside_mask = (
        ~primary_mask
        & ~secondary_mask_2x
    )

    resolvable_2x = (
        np.any(primary_mask)
        and np.any(outside_mask)
    )

    if resolvable_2x:

        median_out = np.median(
            flux_2x[outside_mask]
        )

        depth_even = (
            median_out
            - np.median(flux_2x[primary_mask])
        )

        if np.any(secondary_mask_2x):

            depth_odd = (
                median_out
                - np.median(flux_2x[secondary_mask_2x])
            )

            odd_even_diff = (
                abs(depth_even - depth_odd)
                / max(abs(depth_even), 1e-6)
            )

        else:

            depth_odd = np.nan
            odd_even_diff = np.nan
        if np.isfinite(odd_even_diff):

            odd_even_ok = (
                odd_even_diff < 0.5
            )

        else:

            odd_even_ok = False

    else:

        depth_even = np.nan
        depth_odd = np.nan
        odd_even_diff = np.nan
        odd_even_ok = False
    phase_1x = (
        (time - epoch) / period
    )

    phase_1x = (
        phase_1x + 0.5
    ) % 1.0 - 0.5

    phases_1x, flux_1x = _phase_bin_numpy(
        phase_1x,
        flux,
        n_bins_1x,
    )
    half_width_1x = (
        0.5 * duration / period
    )

    secondary_mask = (
        _phase_distance(
            phases_1x,
            0.5,
        )
        < half_width_1x
    )

    outside_secondary = (
        ~secondary_mask
    )

    resolvable_1x = (
        np.any(secondary_mask)
        and np.any(outside_secondary)
    )

    if resolvable_1x:

        median_out_s = np.median(
            flux_1x[outside_secondary]
        )

        secondary_depth = (
            median_out_s
            - np.median(flux_1x[secondary_mask])
        )
        secondary_depth_abs = abs(
            secondary_depth
        )

        secondary_ok = (
            secondary_depth_abs
            < 0.3 * abs(depth)
        )

    else:

        secondary_depth = np.nan
        secondary_ok = False
    resolvable = (
        resolvable_2x
        and resolvable_1x
    )

    passed = (
        resolvable
        and odd_even_ok
        and secondary_ok
    )
    print(
        f"     vetting depth: "
        f"{depth * 1e3:.3f} ppt",
        flush=True,
    )

    if np.isfinite(odd_even_diff):

        print(
            f"     odd/even difference: "
            f"{odd_even_diff:.3f}",
            flush=True,
        )

    else:

        print(
            "     odd/even difference: unresolved",
            flush=True,
        )

    if np.isfinite(secondary_depth):

        print(
            f"     secondary depth: "
            f"{secondary_depth * 1e3:.3f} ppt",
            flush=True,
        )

    else:

        print(
            "     secondary depth: unresolved",
            flush=True,
        )

    print(
        f"     vetting: "
        f"{'PASSED' if passed else 'FAILED'}",
        flush=True,
    )

    return {
        "period": period,
        "epoch": epoch,
        "depth": depth,
        "duration": duration,
        "odd_even_diff": odd_even_diff,
        "secondary_depth": secondary_depth,
        "odd_even_ok": odd_even_ok,
        "secondary_ok": secondary_ok,
        "passed": passed,
        "resolvable": resolvable,
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
    phase = np.asarray(fold.phase.value, dtype=float)
    flux = np.asarray(fold.flux.value, dtype=float)
    binned_phase, binned_flux = _phase_bin_numpy(phase, flux, plot_n_bins)
    if len(binned_phase) > 0:
        axes[1].scatter(binned_phase, binned_flux, c='r', s=20, zorder=5,
                        label=f'{plot_n_bins} bins')
        axes[1].legend(loc='best')

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

# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
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
    processed = load_processed()
    remaining = [t for t in TARGET_LIST if str(t) not in processed]
    print(f"Searching {len(TARGET_LIST)} stars "
          f"({len(processed)} already done in a previous run, {len(remaining)} remaining).\n")

    for i, tic_id in enumerate(remaining):
        if i > 0:
            time.sleep(3)
        print(f"\n{'='*60}")
        print(f"Processing TIC {tic_id}...")

        try:
            lc = get_stitched_lc(tic_id)

            if lc is None or len(lc.time) < 100:
                print("  -> Skipped")
                continue

            if len(lc.time) > MAX_POINTS:
                factor = -(-len(lc.time) // MAX_POINTS)
                print(f"  -> Binning by {factor}")
                lc = lc.bin(binsize=factor)
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

                    if vet["passed"]:
                        # Only mask a signal that passed FULL vetting.
                        if search_pass < MASK_AND_RESEARCH_PASSES:
                            lc_search = mask_transits(lc_search, period, tls_results.T0, duration)
                            if len(lc_search.time) < 100:
                                print("  -> Too few points left after masking - stopping here.")
                                break
                        # If this is the last pass, nothing left to mask.
                    else:
                        print("  -> Vetting failed; not masking and stopping search passes.")
                        break
                else:
                    print("  -> Not masking - this result didn't pass the basic gates, so it "
                          "shouldn't get to delete data from the residual search.")
                    break
        finally:
            mark_processed(tic_id)

    print("\nDone.")
