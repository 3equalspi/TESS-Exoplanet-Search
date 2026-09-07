#!/usr/bin/env python3
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks
from transitleastsquares import transitleastsquares

TIC_ID = 469979014

P_BINARY = 1.16342
PLANET_PERIOD_MIN = 1.30
PLANET_PERIOD_MAX = 20.0
QUALITY_BITMASK = 24319
AUTHOR = "SPOC"
EXPTIME = 120
OUTPUT_DIR = f"TIC{TIC_ID}_binary_planet_diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_PHASE_BINS = 1500
TEMPLATE_SMOOTHING_PASSES = 2
N_PERIODS = 20000
BLS_SNR_THRESHOLD = 8.0
MIN_SECTORS_WITH_SIGNAL = 2
HARMONIC_TOLERANCE = 0.01
RUN_TLS = True
TLS_PERIOD_WINDOW = 0.05
TLS_OVERSAMPLING_FACTOR = 2
TLS_MAX_POINTS = 120000

def circular_phase(x):
    """
    Return phase in [-0.5, 0.5).
    """
    return ((x + 0.5) % 1.0) - 0.5


def robust_median(x):
    """
    Median ignoring NaNs.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return np.nanmedian(x)


def phase_bin_template(
    time,
    flux,
    period,
    epoch,
    n_bins=1500,
):

    phase = circular_phase(
        (time - epoch) / period
    )

    valid = (
        np.isfinite(phase)
        & np.isfinite(flux)
    )

    phase = phase[valid]
    flux = flux[valid]

    edges = np.linspace(
        -0.5,
        0.5,
        n_bins + 1,
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    template = np.full(
        n_bins,
        np.nan,
        dtype=float,
    )

    for i in range(n_bins):

        mask = (
            (phase >= edges[i])
            & (phase < edges[i + 1])
        )

        if np.any(mask):
            template[i] = np.nanmedian(
                flux[mask]
            )
    good = np.isfinite(template)

    if np.sum(good) < 10:
        raise RuntimeError(
            "Not enough populated phase bins "
            "to construct binary template."
        )

    interp = PchipInterpolator(
        centers[good],
        template[good],
        extrapolate=True,
    )

    template = interp(centers)

    return centers, template


def evaluate_template(
    time,
    phase_centers,
    template,
    period,
    epoch,
):
    """
    Evaluate a phase template at arbitrary times.
    """

    phase = circular_phase(
        (time - epoch) / period
    )

    # PCHIP interpolation requires monotonically increasing x.
    interp = PchipInterpolator(
        phase_centers,
        template,
        extrapolate=True,
    )

    return interp(phase)


def harmonic_distance(
    period,
    binary_period,
    max_harmonic=10,
):
    aliases = []

    for n in range(1, max_harmonic + 1):

        aliases.append(
            (
                binary_period / n,
                f"Pbin/{n}",
            )
        )

        aliases.append(
            (
                binary_period * n,
                f"{n}*Pbin",
            )
        )

    best = None

    for alias_period, alias_name in aliases:

        if alias_period <= 0:
            continue

        frac = abs(
            period - alias_period
        ) / alias_period

        if best is None or frac < best[0]:

            best = (
                frac,
                alias_period,
                alias_name,
            )

    return best


def print_header(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)

def download_data():

    print_header(
        f"TIC {TIC_ID} — SEARCHING FOR PLANET AROUND BINARY"
    )

    print(
        f"Assumed binary period: "
        f"{P_BINARY:.8f} d"
    )

    print("\nSearching MAST for SPOC 2-minute data...")

    search = lk.search_lightcurve(
        f"TIC {TIC_ID}",
        mission="TESS",
        author=AUTHOR,
        exptime=EXPTIME,
    )

    if len(search) == 0:
        raise RuntimeError(
            "No SPOC 2-minute data found."
        )

    print(
        f"\nFound {len(search)} SPOC products."
    )

    print(search)

    print("\nDownloading...")

    collection = search.download_all(
        quality_bitmask=QUALITY_BITMASK
    )

    if collection is None or len(collection) == 0:
        raise RuntimeError(
            "No SPOC light curves downloaded."
        )

    print(
        f"Downloaded {len(collection)} products."
    )

    return collection
def prepare_sectors(collection):

    print_header(
        "PREPARING INDIVIDUAL TESS SECTORS"
    )

    sectors = []

    for i, lc in enumerate(collection):

        try:

            lc = lc.remove_nans()

            if len(lc.time) < 100:

                print(
                    f"Sector/product {i}: too few points, skipping."
                )

                continue
            lc = lc.normalize()
            lc = lc.remove_outliers(
                sigma_lower=8,
                sigma_upper=8,
            )

            lc = lc.remove_nans()

            if len(lc.time) < 100:

                continue

            sectors.append(
                {
                    "index": i,
                    "lc": lc,
                }
            )

            print(
                f"Product {i:3d}: "
                f"{len(lc.time):7d} points, "
                f"{lc.time.value[0]:.3f} - "
                f"{lc.time.value[-1]:.3f}"
            )

        except Exception as e:

            print(
                f"Product {i}: failed: {e}"
            )

    if not sectors:

        raise RuntimeError(
            "No usable sectors."
        )

    return sectors


def concatenate_sectors(sectors):

    times = []
    fluxes = []
    sector_ids = []

    for s in sectors:

        lc = s["lc"]

        t = np.asarray(
            lc.time.value,
            dtype=float
        )

        y = np.asarray(
            lc.flux.value,
            dtype=float
        )

        valid = (
            np.isfinite(t)
            & np.isfinite(y)
        )

        times.append(t[valid])
        fluxes.append(y[valid])

        sector_ids.append(
            np.full(
                np.sum(valid),
                s["index"],
                dtype=int,
            )
        )

    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    sector = np.concatenate(sector_ids)

    order = np.argsort(time)

    return (
        time[order],
        flux[order],
        sector[order],
    )

def build_binary_template(
    time,
    flux,
):

    print_header(
        "BUILDING BINARY PHASE TEMPLATE"
    )
    epoch = time[0]

    phase, template = phase_bin_template(
        time,
        flux,
        P_BINARY,
        epoch,
        n_bins=N_PHASE_BINS,
    )
    baseline = np.nanpercentile(
        template,
        95,
    )

    template = template / baseline
    for _ in range(
        TEMPLATE_SMOOTHING_PASSES
    ):

        padded = np.r_[
            template[-5:],
            template,
            template[:5],
        ]

        kernel = np.ones(11) / 11.0

        smoothed = np.convolve(
            padded,
            kernel,
            mode="same",
        )

        template = smoothed[5:-5]

    binary_model = evaluate_template(
        time,
        phase,
        template,
        P_BINARY,
        epoch,
    )

    print(
        f"Binary template constructed using "
        f"{N_PHASE_BINS} phase bins."
    )

    print(
        f"Template range: "
        f"{np.min(template):.5f} - "
        f"{np.max(template):.5f}"
    )

    print(
        f"Binary peak-to-peak variation: "
        f"{(np.max(template) - np.min(template))*100:.2f}%"
    )

    return (
        epoch,
        phase,
        template,
        binary_model,
    )

def remove_binary(
    flux,
    binary_model,
):

    """
    Divide by the fitted binary template rather than subtracting.

    This is preferable for multiplicative flux effects and keeps
    a small planetary transit approximately in fractional-depth
    units.

        residual = observed / binary_model
    """

    valid = (
        np.isfinite(flux)
        & np.isfinite(binary_model)
        & (binary_model > 0)
    )

    residual = np.full_like(
        flux,
        np.nan,
    )

    residual[valid] = (
        flux[valid]
        / binary_model[valid]
    )

    med = np.nanmedian(residual)

    residual /= med

    return residual

def run_bls(
    time,
    residual,
):

    print_header(
        "BLS SEARCH OF BINARY-SUBTRACTED RESIDUALS"
    )

    valid = (
        np.isfinite(time)
        & np.isfinite(residual)
    )
    t = time[valid]
    y = residual[valid]
    y = y / np.median(y)
    duration_hours = np.array(
        [
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ]
    )

    durations = (
        duration_hours / 24.0
    )

    periods = np.linspace(
        PLANET_PERIOD_MIN,
        PLANET_PERIOD_MAX,
        N_PERIODS,
    )

    model = BoxLeastSquares(
        t,
        y - 1.0,
    )

    results = model.power(
        periods,
        durations,
    )
    power = results.power

    best = np.nanargmax(power)

    best_period = float(
        results.period[best]
    )

    best_duration = float(
        results.duration[best]
    )

    best_depth = float(
        results.depth[best]
    )

    best_power = float(
        results.power[best]
    )
    med_power = np.nanmedian(power)

    mad_power = np.nanmedian(
        np.abs(
            power - med_power
        )
    )

    mad_power *= 1.4826

    if mad_power > 0:

        bls_snr = (
            best_power - med_power
        ) / mad_power

    else:

        bls_snr = np.nan

    print(
        f"Best residual BLS period: "
        f"{best_period:.8f} d"
    )

    print(
        f"Depth: "
        f"{best_depth * 1000:.3f} ppt"
    )

    print(
        f"Duration: "
        f"{best_duration * 24:.2f} h"
    )

    print(
        f"Periodogram SNR-like statistic: "
        f"{bls_snr:.2f}"
    )
    alias = harmonic_distance(
        best_period,
        P_BINARY,
        max_harmonic=10,
    )

    print(
        f"Closest binary harmonic: "
        f"{alias[2]} = {alias[1]:.6f} d"
    )

    print(
        f"Fractional distance from harmonic: "
        f"{alias[0]:.4f}"
    )

    harmonic = (
        alias[0] < HARMONIC_TOLERANCE
    )

    if harmonic:

        print(
            "WARNING: Candidate period is "
            "very close to a binary harmonic."
        )

    return {
        "period": best_period,
        "duration": best_duration,
        "depth": best_depth,
        "power": best_power,
        "snr": bls_snr,
        "periods": periods,
        "powers": power,
        "harmonic": harmonic,
    }
def sector_consistency(
    sectors,
    period,
    epoch,
):

    print_header(
        "SECTOR-BY-SECTOR CANDIDATE TEST"
    )

    results = []
    for s in sectors:

        lc = s["lc"]

        t = np.asarray(
            lc.time.value,
            dtype=float
        )

        y = np.asarray(
            lc.flux.value,
            dtype=float
        )

        valid = (
            np.isfinite(t)
            & np.isfinite(y)
        )

        t = t[valid]
        y = y[valid]

        if len(t) < 200:

            continue

        phase = circular_phase(
            (t - epoch) / period
        )
        half_width = (
            2.0 / 24.0 / period
        )

        in_transit = (
            np.abs(phase)
            < half_width
        )

        out_transit = (
            np.abs(phase)
            > 3.0 * half_width
        )

        if (
            np.sum(in_transit) < 10
            or np.sum(out_transit) < 30
        ):

            continue

        baseline = np.median(
            y[out_transit]
        )

        depth = (
            baseline
            - np.median(
                y[in_transit]
            )
        )

        noise = 1.4826 * np.median(
            np.abs(
                y[out_transit]
                - baseline
            )
        )

        if noise <= 0:
            snr = np.nan
        else:
            snr = (
                depth
                / noise
                * np.sqrt(
                    np.sum(in_transit)
                )
            )

        results.append(
            {
                "sector": s["index"],
                "n": len(t),
                "depth": depth,
                "snr": snr,
            }
        )

        print(
            f"Sector {s['index']:3d}: "
            f"N={len(t):6d}, "
            f"depth={depth*1000:+.3f} ppt, "
            f"SNR={snr:.2f}"
        )

    n_good = sum(
        np.isfinite(r["snr"])
        and r["snr"] >= 2.0
        for r in results
    )

    print()
    print(
        f"Sectors with candidate-like "
        f"signal (SNR >= 2): {n_good}"
    )

    if n_good >= MIN_SECTORS_WITH_SIGNAL:

        print(
            "Sector consistency: POSSIBLE"
        )

    else:

        print(
            "Sector consistency: WEAK"
        )

    return results
def run_tls_confirmation(
    time,
    residual,
    bls_period,
):

    print_header(
        "TLS CONFIRMATION AROUND BLS PERIOD"
    )

    valid = (
        np.isfinite(time)
        & np.isfinite(residual)
    )

    t = time[valid]
    y = residual[valid]
    if len(t) > TLS_MAX_POINTS:

        indices = np.linspace(
            0,
            len(t) - 1,
            TLS_MAX_POINTS,
        ).astype(int)

        t = t[indices]
        y = y[indices]

    y = y / np.median(y)

    print(
        f"TLS input points: {len(t)}"
    )

    print(
        f"Searching period window: "
        f"{max(0.5, bls_period - TLS_PERIOD_WINDOW):.6f} - "
        f"{bls_period + TLS_PERIOD_WINDOW:.6f} d"
    )

    tls = transitleastsquares(
        t,
        y,
    )

    results = tls.power(
        period_min=max(
            0.5,
            bls_period - TLS_PERIOD_WINDOW,
        ),
        period_max=(
            bls_period
            + TLS_PERIOD_WINDOW
        ),
        oversampling_factor=TLS_OVERSAMPLING_FACTOR,
        n_transits_min=3,
    )

    if not np.isfinite(results.period):

        print(
            "TLS did not find a valid transit."
        )

        return None

    actual_depth = (
        1.0
        - float(results.depth)
    )

    print()
    print(
        f"TLS period: "
        f"{results.period:.8f} d"
    )

    print(
        f"TLS SDE: "
        f"{results.SDE:.3f}"
    )

    print(
        f"Physical depth: "
        f"{actual_depth * 1000:.3f} ppt"
    )

    print(
        f"Duration: "
        f"{results.duration * 24:.3f} h"
    )

    print(
        f"Rp/Rs: "
        f"{results.rp_rs:.6f}"
    )

    print(
        f"Distinct transits: "
        f"{getattr(results, 'distinct_transit_count', np.nan)}"
    )

    print(
        f"Empty transits: "
        f"{getattr(results, 'empty_transit_count', np.nan)}"
    )

    return results
def make_plots(
    time,
    flux,
    binary_model,
    residual,
    binary_phase,
    binary_template,
    epoch,
    bls,
):
    fig, ax = plt.subplots(
        figsize=(16, 6)
    )
    step = max(
        1,
        len(time) // 100000
    )

    ax.plot(
        time[::step],
        flux[::step],
        ".",
        markersize=1,
        alpha=0.3,
        label="SPOC flux",
    )

    ax.plot(
        time[::step],
        binary_model[::step],
        linewidth=1.5,
        label="Binary template",
    )

    ax.set_xlabel(
        "Time [BTJD]"
    )

    ax.set_ylabel(
        "Normalized flux"
    )

    ax.set_title(
        f"TIC {TIC_ID} — binary model"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "01_binary_model.png",
        ),
        dpi=200,
    )

    plt.close()
    phase = circular_phase(
        (time - epoch)
        / P_BINARY
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        phase[::step],
        flux[::step],
        ".",
        markersize=1,
        alpha=0.25,
    )

    ax.plot(
        binary_phase,
        binary_template,
        linewidth=2,
        label="Binary template",
    )

    ax.set_xlabel(
        "Binary phase"
    )

    ax.set_ylabel(
        "Normalized flux"
    )

    ax.set_title(
        f"TIC {TIC_ID} — binary fold, P={P_BINARY:.5f} d"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "02_binary_fold.png",
        ),
        dpi=200,
    )

    plt.close()

    fig, ax = plt.subplots(
        figsize=(16, 6)
    )

    ax.plot(
        time[::step],
        residual[::step],
        ".",
        markersize=1,
        alpha=0.35,
    )

    ax.axhline(
        1.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Time [BTJD]"
    )

    ax.set_ylabel(
        "Binary-divided flux"
    )

    ax.set_title(
        f"TIC {TIC_ID} — residual after binary removal"
    )

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "03_binary_removed_residual.png",
        ),
        dpi=200,
    )

    plt.close()
    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        bls["periods"],
        bls["powers"],
        linewidth=1,
    )

    ax.axvline(
        P_BINARY,
        linestyle="--",
        label="Binary period",
    )

    ax.axvline(
        bls["period"],
        linestyle="--",
        label=(
            f"BLS candidate "
            f"{bls['period']:.4f} d"
        ),
    )

    ax.set_xlabel(
        "Period [days]"
    )

    ax.set_ylabel(
        "BLS power"
    )

    ax.set_title(
        f"TIC {TIC_ID} — BLS of binary-subtracted residual"
    )

    ax.legend()

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "04_residual_bls.png",
        ),
        dpi=200,
    )

    plt.close()
    candidate_period = bls["period"]

    residual_phase = circular_phase(
        (time - epoch)
        / candidate_period
    )
    mask = (
        np.abs(residual_phase)
        < 0.10
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        residual_phase[mask],
        residual[mask],
        ".",
        markersize=2,
        alpha=0.3,
    )

    ax.axhline(
        1.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Candidate phase"
    )

    ax.set_ylabel(
        "Binary-divided flux"
    )

    ax.set_title(
        f"TIC {TIC_ID} — residual candidate "
        f"P={candidate_period:.6f} d"
    )

    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "05_candidate_fold.png",
        ),
        dpi=200,
    )

    plt.close()
def main():

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )
    collection = download_data()
    sectors = prepare_sectors(
        collection
    )
    time, flux, sector_ids = (
        concatenate_sectors(sectors)
    )

    print_header(
        "COMBINED LIGHT CURVE"
    )

    print(
        f"Points: {len(time)}"
    )

    print(
        f"Time span: "
        f"{time[-1] - time[0]:.3f} days"
    )

    print(
        f"Flux min: {np.min(flux):.6f}"
    )

    print(
        f"Flux max: {np.max(flux):.6f}"
    )

    print(
        f"Flux std: {np.std(flux):.6f}"
    )
    (
        epoch,
        binary_phase,
        binary_template,
        binary_model,
    ) = build_binary_template(
        time,
        flux,
    )
    residual = remove_binary(
        flux,
        binary_model,
    )

    print(
        f"Residual std: "
        f"{np.nanstd(residual):.8f}"
    )
    bls = run_bls(
        time,
        residual,
    )
    sector_results = sector_consistency(
        sectors,
        bls["period"],
        epoch,
    )
    tls_results = None

    if RUN_TLS:

        tls_results = run_tls_confirmation(
            time,
            residual,
            bls["period"],
        )
    make_plots(
        time,
        flux,
        binary_model,
        residual,
        binary_phase,
        binary_template,
        epoch,
        bls,
    )
    print_header(
        "FINAL DIAGNOSTIC SUMMARY"
    )

    print(
        f"Binary period assumed: "
        f"{P_BINARY:.6f} d"
    )

    print(
        f"Residual BLS period: "
        f"{bls['period']:.6f} d"
    )

    print(
        f"Residual BLS depth: "
        f"{bls['depth'] * 1000:.3f} ppt"
    )

    print(
        f"Residual BLS statistic: "
        f"{bls['snr']:.2f}"
    )

    print(
        f"Binary harmonic: "
        f"{bls['harmonic']}"
    )

    if tls_results is not None:

        tls_depth = (
            1.0
            - float(tls_results.depth)
        )

        print(
            f"TLS period: "
            f"{tls_results.period:.6f} d"
        )

        print(
            f"TLS SDE: "
            f"{tls_results.SDE:.2f}"
        )

        print(
            f"TLS physical depth: "
            f"{tls_depth * 1000:.3f} ppt"
        )

    n_good = sum(
        np.isfinite(r["snr"])
        and r["snr"] >= 2
        for r in sector_results
    )

    print(
        f"Sectors with candidate-like signal: "
        f"{n_good}"
    )
    if (
        bls["snr"] >= BLS_SNR_THRESHOLD
        and not bls["harmonic"]
        and n_good >= MIN_SECTORS_WITH_SIGNAL
        and tls_results is not None
        and np.isfinite(tls_results.SDE)
        and tls_results.SDE >= 7
    ):

        print()
        print(
            ">>> SURVIVING PLANET-TRANSIT CANDIDATE <<<"
        )

        print(
            "The binary-subtracted residual contains "
            "a repeatable transit-like signal."
        )

    else:

        print()
        print(
            ">>> NO CONFIRMED PLANETARY TRANSIT <<<"
        )

        print(
            "The current data do not satisfy all "
            "independent residual-search criteria."
        )

    print()
    print(
        f"Plots saved in: {OUTPUT_DIR}/"
    )


if __name__ == "__main__":

    main()
