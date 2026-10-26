#!/usr/bin/env python3
"""Small deterministic BLS benchmark with injected transits and noise-only controls.

This measures recovery on synthetic light curves, not astrophysical false-positive rates.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.timeseries import BoxLeastSquares


def one_trial(rng, injected, duration=0.12):
    time = np.arange(0, 27.0, 30.0 / (24 * 60))
    time = time[(rng.random(len(time)) > 0.035)]
    period = float(rng.uniform(1.2, 12.0))
    epoch = float(rng.uniform(0.2, period))
    depth = float(rng.uniform(0.001, 0.01))
    noise = rng.normal(0, 0.0015, len(time))
    flux = 1.0 + noise
    if injected:
        phase = ((time - epoch + period / 2) % period) - period / 2
        flux[np.abs(phase) < duration / 2] -= depth
    periods = np.linspace(0.5, 20.0, 5000)
    result = BoxLeastSquares(time, flux).power(periods, duration=[duration])
    best = int(np.nanargmax(result.power))
    found = float(result.period[best])
    tol = max(0.02, 0.01 * period)
    recovered = min(abs(found - period), abs(found - period / 2), abs(found - 2 * period)) <= tol
    return {"label": "injected_transit" if injected else "noise_only_control",
            "true_period_days": period if injected else "", "best_period_days": found,
            "period_recovered": bool(recovered) if injected else "",
            "peak_power": float(result.power[best]), "depth": depth if injected else "",
            "n_points": len(time)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injections", type=int, default=10)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/synthetic_bls.csv"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = [one_trial(rng, True) for _ in range(args.injections)]
    rows.extend(one_trial(rng, False) for _ in range(args.controls))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    recovered = sum(row["period_recovered"] is True for row in rows)
    controls = [row["peak_power"] for row in rows if row["label"] == "noise_only_control"]
    positives = [row["peak_power"] for row in rows if row["label"] == "injected_transit"]
    print(f"Synthetic BLS benchmark: {recovered}/{args.injections} injected periods recovered")
    separation = sum(p > c for p in positives for c in controls) / (len(positives) * len(controls))
    print(f"Median peak power: injections={np.median(positives):.3e}, controls={np.median(controls):.3e}")
    print(f"Pairwise injected/control peak-power separation: {separation:.0%}")
    print(f"Per-trial results: {args.output}")
    print("These synthetic controls are a code check, not a calibrated astrophysical false-positive rate.")


if __name__ == "__main__":
    main()



