"""Absolute temperature extremes as an interval, with a simple lapse-rate elevation correction."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from .extremes import gumbel_fit

LAPSE_C_PER_KM = 6.5          # standard-atmosphere lapse rate
LARGE_ELEVATION_GAP_M = 300.0  # above this the correction itself is unreliable; flag it


def _elevation_shift(station_elev_m: float, site_elev_m: float) -> float:
    return -LAPSE_C_PER_KM * (site_elev_m - station_elev_m) / 1000.0


def temperature_extreme_interval(annual_extremes: Sequence[float], kind: str,
                                 station_elev_m: float, site_elev_m: float,
                                 return_period_years: float = 50.0,
                                 min_years: int = 15) -> Optional[dict]:
    """Interval [record extreme, Gumbel return level] on the site, or ``None`` if too few years.

    ``kind`` is ``'max'`` (annual maximum temperatures) or ``'min'`` (annual minima).
    The interval spans the observed record extreme and the fitted return level rather than
    choosing one of them, because records of different length are not comparable.
    """
    if kind not in ("max", "min"):
        raise ValueError("kind must be 'max' or 'min'")
    xs = [float(x) for x in annual_extremes if x is not None and math.isfinite(x)]
    if len(xs) < min_years:
        return None
    sign = 1.0 if kind == "max" else -1.0
    fit = gumbel_fit([sign * x for x in xs], min_years)
    if fit is None:
        return None
    rl = sign * fit.return_level(return_period_years)
    record = max(xs) if kind == "max" else min(xs)
    shift = _elevation_shift(station_elev_m, site_elev_m)
    lo, hi = sorted((record + shift, rl + shift))
    gap = abs(site_elev_m - station_elev_m)
    return {
        "interval_c": (round(lo, 2), round(hi, 2)),
        "n_years": len(xs),
        "elevation_gap_m": gap,
        "elevation_correction_unreliable": gap > LARGE_ELEVATION_GAP_M,
    }
