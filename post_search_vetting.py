#!/usr/bin/env python3
import argparse
import csv
import re
import warnings
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
DEFAULT_MANIFEST = Path("tess_lc_cache") / "candidate_signals.csv"


def num(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_candidates(path):
    unique = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                row["tic_id"] = int(row["tic_id"])
                row["search_pass"] = int(row.get("search_pass") or 1)
                for name in ("period_days", "epoch_btjd", "duration_days"):
                    row[name] = float(row[name])
                row["sde"] = num(row.get("sde"))
                if row["tic_id"] > 0 and row["period_days"] > 0 and row["duration_days"] > 0:
                    unique[(row["tic_id"], row["search_pass"], round(row["period_days"], 6))] = row
            except (KeyError, TypeError, ValueError):
                continue
    return list(unique.values())


def clean_sector(lc):
    lc = lc.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5)
    n = len(lc.time)
    window = min(401, n if n % 2 else n - 1)
    if window >= 5:
        try:
            lc = lc.flatten(window_length=window)
        except Exception:
            pass
    return lc.remove_nans().remove_outliers(sigma_lower=15, sigma_upper=5).normalize()


def event_depth(t, y, center, duration):
    distance = np.abs(t - center)
    inside = distance <= duration / 2
    baseline = (distance >= 1.5 * duration) & (distance <= 3 * duration)
    if inside.sum() < 3 or baseline.sum() < 5:
        return np.nan
    return float(np.nanmedian(y[baseline]) - np.nanmedian(y[inside]))


def review(row, cache, output):
    tic, period = row["tic_id"], row["period_days"]
    epoch, duration = row["epoch_btjd"], row["duration_days"]
    search = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC")
    if len(search) == 0:
        raise RuntimeError("No SPOC light curves found.")
    cache.mkdir(parents=True, exist_ok=True)
    coll = search.download_all(download_dir=str(cache), quality_bitmask="hard")
    if coll is None or len(coll) == 0:
        raise RuntimeError("No SPOC sectors downloaded.")
    sectors, sector_ids = [], []
    for i, raw in enumerate(coll):
        try:
            sec = clean_sector(raw)
            if len(sec.time) >= 20:
                sectors.append(sec)
                sector_ids.append(str(sec.meta.get("SECTOR", i + 1)))
        except Exception:
            continue
    if not sectors:
        raise RuntimeError("No usable sectors after cleaning.")

    t = np.concatenate([np.asarray(s.time.value, float) for s in sectors])
    y = np.concatenate([np.asarray(s.flux.value, float) for s in sectors])
    valid = np.isfinite(t) & np.isfinite(y)
    order = np.argsort(t[valid])
    t, y = t[valid][order], y[valid][order]

    sd = []
    for sec in sectors:
        st, sy = np.asarray(sec.time.value), np.asarray(sec.flux.value)
        ph = ((st - epoch + period / 2) % period) - period / 2
        inside = np.abs(ph) <= duration / 2
        outside = (np.abs(ph) >= 1.5 * duration) & (np.abs(ph) <= min(3 * duration, period / 2))
        sd.append(float(np.nanmedian(sy[outside]) - np.nanmedian(sy[inside]))
                  if inside.sum() >= 3 and outside.sum() >= 8 else np.nan)

    k0, k1 = int(np.ceil((t[0] - epoch) / period)), int(np.floor((t[-1] - epoch) / period))
    centers, depths, event_cycles = [], [], []
    for k in range(k0, k1 + 1):
        center = epoch + k * period
        d = event_depth(t, y, center, duration)
        if np.isfinite(d):
            centers.append(center)
            depths.append(d)
            event_cycles.append(k)
    depths = np.asarray(depths)
    flags = []
    finite = np.asarray(sd)[np.isfinite(sd)]
    if not len(finite):
        flags.append("no_sector_has_resolved_transit")
    scatter_sector = float(np.nanstd(finite)) if len(finite) > 1 else np.nan
    if len(finite) > 1:
        if (finite < 0).any() and (finite > 0).any():
            flags.append("sector_depth_sign_change")
        if scatter_sector > abs(float(np.nanmedian(finite))):
            flags.append("sector_depth_inconsistent")
    odd_even = np.nan
    if len(depths) >= 4:
        cycles = np.asarray(event_cycles)
        even = np.nanmedian(depths[cycles % 2 == 0])
        odd = np.nanmedian(depths[cycles % 2 == 1])
        odd_even = float(abs(even - odd) / max(abs(float(np.nanmedian(depths))), 1e-6))
        if odd_even > .5:
            flags.append("odd_even_depth_difference")
    elif len(depths):
        flags.append("few_individual_transits")

    phase = ((t - epoch + period / 2) % period) - period / 2
    secondary = np.abs(np.abs(phase) - period / 2) <= duration / 2
    primary = np.abs(phase) <= duration / 2
    baseline = ~(secondary | primary)
    secondary_depth = np.nan
    if secondary.sum() >= 3 and baseline.sum() >= 10:
        secondary_depth = float(np.nanmedian(y[baseline]) - np.nanmedian(y[secondary]))
        if len(depths) and abs(secondary_depth) >= .3 * abs(float(np.nanmedian(depths))):
            flags.append("possible_secondary_eclipse")
    median_depth = float(np.nanmedian(depths)) if len(depths) else np.nan
    event_scatter = float(1.4826 * np.nanmedian(np.abs(depths - median_depth))) if len(depths) else np.nan
    if len(depths) >= 4 and event_scatter > abs(median_depth):
        flags.append("transit_depth_inconsistent")
    if not flags:
        flags = ["no_automatic_flags"]

    # Local BLS scan checks whether nearby aliases rival the recorded period. It is diagnostic, not a new verdict.
    local_periods = np.linspace(max(0.5, period * 0.8), min(20.0, period * 1.2), 241)
    try:
        local_power = BoxLeastSquares(t, y).power(local_periods, duration=[min(duration, period * 0.2)])
        best_local = int(np.nanargmax(local_power.power))
        local_best_period = float(local_power.period[best_local])
        local_best_power = float(local_power.power[best_local])
    except Exception:
        local_power, local_best_period, local_best_power = None, np.nan, np.nan

    fig, ax = plt.subplots(5, 1, figsize=(13, 15), constrained_layout=True)
    ax[0].plot(t, y, "k.", ms=1, alpha=.4)
    for center in epoch + np.arange(k0, k1 + 1) * period:
        ax[0].axvline(center, color="tab:red", alpha=.2, lw=.6)
    ax[0].set(ylabel="Normalized flux", xlabel="Time (BTJD)")
    ax[0].set_title(f"TIC {tic} P={period:.6f} d SDE={row['sde']:.1f}: {'; '.join(flags)}")
    ax[1].plot(phase / period, y, "k.", ms=1, alpha=.25)
    ax[1].set_xlim(-2 * duration / period, 2 * duration / period)
    ax[1].set(ylabel="Normalized flux", xlabel="Orbital phase")
    ax[1].set_title("All sectors folded on search ephemeris")
    if local_power is not None:
        ax[2].plot(local_power.period, local_power.power, color="tab:blue")
        ax[2].axvline(period, color="tab:red", ls="--", label="Search period")
        ax[2].axvline(local_best_period, color="tab:green", ls=":", label="Local peak")
        ax[2].legend()
    ax[2].set(xlabel="Period (days)", ylabel="BLS power")
    ax[2].set_title(f"Local period scan; best={local_best_period:.6f} d")
    xx = np.arange(len(sd))
    ax[3].axhline(0, color=".5", lw=.8)
    ax[3].plot(xx, np.asarray(sd) * 1e3, "o-")
    ax[3].set_xticks(xx, sector_ids, rotation=45)
    ax[3].set(ylabel="Depth (ppt)", xlabel="TESS sector")
    ax[3].set_title("Independent sector depth estimates")
    ax[4].axhline(0, color=".5", lw=.8)
    if len(depths):
        ax[4].scatter(np.arange(len(depths)), depths * 1e3,
                      c=np.asarray(event_cycles) % 2, cmap="coolwarm", s=24)
    ax[4].set(ylabel="Depth (ppt)", xlabel="Observed event sequence")
    ax[4].set_title("Individual event depths (alternating colors)")
    output.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"TIC{tic}_pass{row['search_pass']}_P{period:.6f}")
    plot = output / f"{safe}_review.png"
    fig.savefig(plot, dpi=150)
    plt.close(fig)
    return {
        "tic_id": tic, "search_pass": row["search_pass"], "period_days": period,
        "epoch_btjd": epoch, "duration_hours": duration * 24, "sde": row["sde"],
        "sectors_downloaded": len(sectors), "transits_measured": len(depths),
        "median_transit_depth": median_depth, "transit_depth_scatter": event_scatter,
        "sector_depth_scatter": scatter_sector, "odd_even_relative_difference": odd_even,
        "local_bls_best_period_days": local_best_period, "local_bls_peak_power": local_best_power,
        "secondary_depth": secondary_depth, "review_flags": ";".join(flags),
        "plot_path": str(plot), "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Triage TLS detections; not a planet verdict.")
    parser.add_argument("--input", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("tess_vetting_results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("tess_lc_cache"))
    parser.add_argument("--tic", type=int, help="Review a single TIC.")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing {args.input}; run tess_search.py first.")
    rows = load_candidates(args.input)
    if args.tic:
        rows = [r for r in rows if r["tic_id"] == args.tic]
    if not rows:
        raise SystemExit("No usable TLS detections in the manifest.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = args.output_dir / "review_summary.csv"
    done = set()
    if summary.exists():
        with open(summary, newline="", encoding="utf-8") as f:
            done = {(r.get("tic_id"), r.get("search_pass"), f"{num(r.get('period_days')):.6f}")
                    for r in csv.DictReader(f)}
    results = []
    for row in rows:
        key = (str(row["tic_id"]), str(row["search_pass"]), f"{row['period_days']:.6f}")
        if key in done:
            continue
        print(f"Reviewing TIC {row['tic_id']} at P={row['period_days']:.6f} d...")
        try:
            result = review(row, args.cache_dir, args.output_dir)
        except Exception as exc:
            result = {name: "" for name in ("sectors_downloaded", "transits_measured",
                      "median_transit_depth", "transit_depth_scatter", "sector_depth_scatter",
                      "odd_even_relative_difference", "local_bls_best_period_days", "local_bls_peak_power",
                      "secondary_depth", "review_flags", "plot_path")}
            result.update({"tic_id": row["tic_id"], "search_pass": row["search_pass"],
                           "period_days": row["period_days"], "epoch_btjd": row["epoch_btjd"],
                           "duration_hours": row["duration_days"] * 24, "sde": row["sde"],
                           "error": str(exc)})
        results.append(result)
    if results:
        fields = ["tic_id", "search_pass", "period_days", "epoch_btjd", "duration_hours", "sde",
                  "sectors_downloaded", "transits_measured", "median_transit_depth",
                  "transit_depth_scatter", "sector_depth_scatter", "odd_even_relative_difference",
                  "local_bls_best_period_days", "local_bls_peak_power", "secondary_depth", "review_flags", "plot_path", "error"]
        header = not summary.exists() or summary.stat().st_size == 0
        with open(summary, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if header:
                writer.writeheader()
            writer.writerows(results)
    print(f"Saved {len(results)} review row(s) to {summary}")


if __name__ == "__main__":
    main()

