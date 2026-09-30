"""Extreme wind speed helpers: Gumbel fit and conversions to the design basis.

All inputs are supplied by the caller (e.g. annual maxima from a public re-analysis or a
storm track archive); this module reads no files and downloads nothing.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional, Sequence

EULER_GAMMA = 0.5772156649015329

# Commonly used factor from a 1-minute to a 10-minute mean wind (tropical-cyclone best-track
# winds are 1-minute means). Check the value against the guidance you follow.
ONE_MIN_TO_TEN_MIN = 0.88


@dataclass(frozen=True)
class GumbelFit:
    mu: float
    beta: float
    n_years: int

    def return_level(self, return_period_years: float) -> float:
        if not math.isfinite(return_period_years) or return_period_years <= 1:
            raise ValueError("return period must exceed 1 year")
        return self.mu - self.beta * math.log(-math.log(1.0 - 1.0 / return_period_years))


def gumbel_fit(annual_maxima: Sequence[float], min_years: int = 15) -> Optional[GumbelFit]:
    """Method-of-moments Gumbel fit. Returns ``None`` if fewer than ``min_years`` valid values.

    An extreme-value fit on a short sample is more dangerous than no answer, so the
    caller must handle ``None`` explicitly.
    """
    xs = [float(x) for x in annual_maxima if x is not None and math.isfinite(x)]
    if len(xs) < max(min_years, 2):
        return None
    s = statistics.stdev(xs)
    if s == 0:
        return None
    beta = s * math.sqrt(6.0) / math.pi
    mu = statistics.fmean(xs) - EULER_GAMMA * beta
    return GumbelFit(mu, beta, len(xs))


def extrapolate_to_hub(v_ref_height: float, ref_height_m: float, hub_height_m: float,
                       exponent: float) -> float:
    """Power-law extrapolation with the fixed extreme-wind exponent.

    Use the exponent your class table prescribes for extreme conditions
    (``ClassTable.ewm_shear_exponent``), not the site's measured normal-conditions shear.
    """
    if not all(math.isfinite(x) for x in (v_ref_height, ref_height_m, hub_height_m, exponent)) \
            or ref_height_m <= 0 or hub_height_m <= 0:
        raise ValueError("extrapolate_to_hub needs finite positive heights and finite inputs")
    return v_ref_height * (hub_height_m / ref_height_m) ** exponent


def v50_hub_from_annual_maxima(annual_max_1min_10m: Sequence[float], hub_height_m: float,
                               exponent: float, min_years: int = 15,
                               ref_height_m: float = 10.0) -> Optional[float]:
    """50-year, 10-minute mean wind at hub height from annual maxima of 1-minute winds at 10 m."""
    fit = gumbel_fit([v * ONE_MIN_TO_TEN_MIN for v in annual_max_1min_10m], min_years)
    if fit is None:
        return None
    return extrapolate_to_hub(fit.return_level(50.0), ref_height_m, hub_height_m, exponent)
