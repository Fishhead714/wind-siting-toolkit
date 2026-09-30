"""Interval turbulence estimate and comparison with a turbine's reference intensity.

Chain: z0 interval -> neutral mean TI -> x scatter factor (mean -> 90th percentile) -> TI interval.
The interval is carried all the way to the verdict; taking its midpoint would pretend the
stacked assumptions (z0 class, neutral stability, scatter factor) are better known than they are.
"""
from __future__ import annotations

import math
from typing import Tuple

from .iec import ClassTable
from .roughness import z0_interval_ti

# 1 + 1.28 * CoV(sigma1) for CoV 0.15 .. 0.30 (normal-quantile approximation).
# The CoV range is an ASSUMPTION for onshore sites; replace with a measured value if you have a mast.
DEFAULT_COV_SIGMA1 = (0.15, 0.30)


def scatter_factor_range(cov: Tuple[float, float] = DEFAULT_COV_SIGMA1) -> Tuple[float, float]:
    return 1.0 + 1.28 * cov[0], 1.0 + 1.28 * cov[1]


def ti_p90_interval(z0_low: float, z0_high: float, height_m: float,
                    cov: Tuple[float, float] = DEFAULT_COV_SIGMA1) -> Tuple[float, float]:
    lo, hi = z0_interval_ti(z0_low, z0_high, height_m)
    f_lo, f_hi = scatter_factor_range(cov)
    return round(lo * f_lo, 4), round(hi * f_hi, 4)


def verdict_against_iref(ti_interval: Tuple[float, float], turbine_iref: float,
                         table: ClassTable) -> str:
    """pass / marginal / fail at the 15 m/s reference speed (90th percentile basis).

    This is a coarse classification helper. ``SuitabilityEngine`` is stricter: it compares against the
    tightest bin of the limit curve up to cut-out. Expect the engine to fail cases this helper passes.
    """
    limit = turbine_iref * table.ntm_p90_factor()
    if not all(math.isfinite(x) for x in (limit, *ti_interval)):
        raise ValueError("non-finite turbulence input")
    lo, hi = ti_interval
    if hi <= limit:
        return "pass"
    if lo > limit:
        return "fail"
    return "marginal"
