"""Small dependency-light conservative gates used by the transit vetter."""
import math

def primary_transit_passes(measured_depth, fitted_depth, out_of_transit_scatter,
                           n_primary_bins, cadence_days, duration_days,
                           min_snr=3.0):
    """Require a resolved, positive, significant depth agreeing with the TLS fit."""
    values = (measured_depth, fitted_depth, out_of_transit_scatter,
              cadence_days, duration_days)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if fitted_depth <= 0 or duration_days <= 0 or n_primary_bins < 1:
        return False
    if cadence_days > duration_days / 2.0 or measured_depth <= 0:
        return False
    depth_sigma = max(out_of_transit_scatter, 0.0) / math.sqrt(n_primary_bins)
    snr = measured_depth / max(depth_sigma, 1e-12)
    depth_ratio = measured_depth / fitted_depth
    return snr >= min_snr and 0.5 <= depth_ratio <= 1.5

