import numpy as np
import matplotlib.pyplot as plt


def circular_phase(phase):
    return (phase + 0.5) % 1.0 - 0.5


def phase_distance(phase, center):
    return np.abs(((phase - center + 0.5) % 1.0) - 0.5)


def phase_bin_numpy(phase, flux, n_bins=400):


    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)

    good = np.isfinite(phase) & np.isfinite(flux)

    phase = phase[good]
    flux = flux[good]

    if len(phase) == 0:
        return np.array([]), np.array([]), np.array([])

    phase = circular_phase(phase)

    edges = np.linspace(-0.5, 0.5, n_bins + 1)

    bin_index = np.digitize(phase, edges) - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    bphase = []
    bflux = []
    bcount = []

    for i in range(n_bins):

        mask = bin_index == i

        if not np.any(mask):
            continue

        bphase.append(np.median(phase[mask]))
        bflux.append(np.median(flux[mask]))
        bcount.append(np.sum(mask))

    return (
        np.asarray(bphase),
        np.asarray(bflux),
        np.asarray(bcount),
    )


def measure_eclipse_depth(
    phase,
    flux,
    center,
    half_width,
    baseline_width=0.05,
):


    phase = circular_phase(phase)

    eclipse_mask = (
        phase_distance(phase, center)
        <= half_width
    )

    outer_mask = (
        (phase_distance(phase, center) > half_width * 2.0)
        &
        (phase_distance(phase, center) < half_width + baseline_width)
    )

    if np.sum(eclipse_mask) < 3:
        return np.nan, np.nan, np.nan, 0

    if np.sum(outer_mask) < 10:

        outer_mask = ~eclipse_mask

    baseline = np.median(flux[outer_mask])
    eclipse = np.median(flux[eclipse_mask])

    depth = baseline - eclipse

    return (
        depth,
        baseline,
        eclipse,
        int(np.sum(eclipse_mask)),
    )


def secondary_diagnostic(
    lc,
    tls_results,
    n_bins=500,
    plot_window=0.08,
):

    P_tls = float(tls_results.period)
    T0 = float(tls_results.T0)

    tls_bottom = float(tls_results.depth)

    primary_depth_tls = 1.0 - tls_bottom

    P_binary = 2.0 * P_tls

    duration = float(tls_results.duration)

    time = np.asarray(
        lc.time.value,
        dtype=float,
    )

    flux = np.asarray(
        lc.flux.value,
        dtype=float,
    )

    good = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    time = time[good]
    flux = flux[good]

    phase = (
        (time - T0) / P_binary
    )

    phase = circular_phase(phase)

    bphase, bflux, bcount = phase_bin_numpy(
        phase,
        flux,
        n_bins=n_bins,
    )

    half_width = (
        0.5 * duration / P_binary
    )

    half_width = max(
        half_width,
        0.003,
    )

    primary_depth, primary_base, primary_flux, n_primary = (
        measure_eclipse_depth(
            phase,
            flux,
            center=0.0,
            half_width=half_width,
        )
    )

    secondary_depth, secondary_base, secondary_flux, n_secondary = (
        measure_eclipse_depth(
            phase,
            flux,
            center=0.5,
            half_width=half_width,
        )
    )

    if (
        np.isfinite(primary_depth)
        and primary_depth > 0
        and np.isfinite(secondary_depth)
    ):

        secondary_ratio = (
            secondary_depth / primary_depth
        )

    else:

        secondary_ratio = np.nan

    print()
    print("=" * 70)
    print("SECONDARY ECLIPSE DIAGNOSTIC")
    print("=" * 70)

    print(f"TLS period             : {P_tls:.8f} d")
    print(f"Tested binary period   : {P_binary:.8f} d")
    print(f"TLS duration           : {duration * 24 * 60:.2f} min")
    print()

    print(
        f"Primary depth          : "
        f"{primary_depth * 1000:.3f} ppt"
    )

    print(
        f"Secondary depth        : "
        f"{secondary_depth * 1000:.3f} ppt"
    )

    print(
        f"Secondary / primary    : "
        f"{secondary_ratio:.3f}"
    )

    print(
        f"Primary points         : "
        f"{n_primary}"
    )

    print(
        f"Secondary points       : "
        f"{n_secondary}"
    )

    print()

    if np.isfinite(secondary_ratio):

        if secondary_ratio > 0.5:

            print(
                ">>> STRONG EB INDICATION:"
                " secondary eclipse is >50% of primary."
            )

        elif secondary_ratio > 0.3:

            print(
                ">>> POSSIBLE EB:"
                " significant secondary eclipse detected."
            )

        elif secondary_ratio > 0.1:

            print(
                ">>> WEAK SECONDARY:"
                " candidate requires further validation."
            )

        else:

            print(
                ">>> NO STRONG SECONDARY DETECTED."
            )

    print("=" * 70)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 13),
        gridspec_kw={
            "height_ratios": [1.0, 1.0, 1.0]
        },
    )

    axes[0].scatter(
        phase,
        flux,
        s=1,
        alpha=0.15,
        rasterized=True,
    )

    axes[0].plot(
        bphase,
        bflux,
        "o-",
        markersize=3,
        linewidth=1,
    )

    axes[0].axvline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    axes[0].axvline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_xlim(-0.5, 0.5)

    axes[0].set_ylabel("Normalised flux")

    axes[0].set_title(
        f"TIC {getattr(lc, 'label', '')} — "
        f"2P secondary diagnostic\n"
        f"TLS P={P_tls:.6f} d, "
        f"tested P={P_binary:.6f} d"
    )

    axes[0].grid(alpha=0.2)

    primary_zoom = (
        np.abs(bphase) <= plot_window
    )

    axes[1].scatter(
        phase[
            np.abs(phase) <= plot_window
        ],
        flux[
            np.abs(phase) <= plot_window
        ],
        s=2,
        alpha=0.20,
    )

    axes[1].plot(
        bphase[primary_zoom],
        bflux[primary_zoom],
        "o-",
        markersize=4,
        linewidth=1,
    )

    axes[1].axhline(
        primary_base,
        linestyle="--",
        linewidth=1,
        label=f"baseline = {primary_base:.6f}",
    )

    axes[1].axhline(
        primary_flux,
        linestyle="--",
        linewidth=1,
        label=f"eclipse = {primary_flux:.6f}",
    )

    axes[1].set_xlim(
        -plot_window,
        plot_window,
    )

    axes[1].set_ylabel("Normalised flux")

    axes[1].set_title(
        f"Primary eclipse — "
        f"depth = {primary_depth * 1000:.3f} ppt"
    )

    axes[1].legend()

    axes[1].grid(alpha=0.2)

    secondary_zoom = (
        np.abs(
            circular_phase(bphase - 0.5)
        )
        <= plot_window
    )

    secondary_phase = circular_phase(
        phase - 0.5
    )

    secondary_bphase = circular_phase(
        bphase - 0.5
    )

    axes[2].scatter(
        secondary_phase[
            np.abs(secondary_phase) <= plot_window
        ],
        flux[
            np.abs(secondary_phase) <= plot_window
        ],
        s=2,
        alpha=0.20,
    )

    axes[2].plot(
        secondary_bphase[secondary_zoom],
        bflux[secondary_zoom],
        "o-",
        markersize=4,
        linewidth=1,
    )

    axes[2].axhline(
        secondary_base,
        linestyle="--",
        linewidth=1,
        label=f"baseline = {secondary_base:.6f}",
    )

    axes[2].axhline(
        secondary_flux,
        linestyle="--",
        linewidth=1,
        label=f"eclipse = {secondary_flux:.6f}",
    )

    axes[2].set_xlim(
        -plot_window,
        plot_window,
    )

    axes[2].set_xlabel(
        "Phase relative to secondary eclipse"
    )

    axes[2].set_ylabel("Normalised flux")

    axes[2].set_title(
        f"Secondary eclipse — "
        f"depth = {secondary_depth * 1000:.3f} ppt\n"
        f"Secondary / primary = {secondary_ratio:.3f}"
    )

    axes[2].legend()

    axes[2].grid(alpha=0.2)

    plt.tight_layout()

    plt.show()

    return {
        "tls_period": P_tls,
        "binary_period": P_binary,
        "duration": duration,
        "primary_depth": primary_depth,
        "secondary_depth": secondary_depth,
        "secondary_ratio": secondary_ratio,
        "primary_points": n_primary,
        "secondary_points": n_secondary,
    }
