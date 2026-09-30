"""Core formulae. All uncertainties are 1-sigma, expressed as a FRACTION of the P50 energy
(0.05 means 5 %). Everything is a plain textbook formula; nothing here is site data.

Method
------
1. Loss chain: net = gross * prod(1 - loss_i)  (losses are multiplicative, not additive).
2. Long-term components are combined as a root-sum-square (independent) or with an
   explicit correlation matrix: sigma^2 = s^T R s.
3. Year-to-year variability (``interannual``) does not average out over a short horizon:
   sigma_N^2 = sigma_lt^2 + sigma_iav^2 / N  for an N-year window (N=1 gives the
   single-year value, N=10 the ten-year value).
4. Exceedance: with a normal distribution, the value exceeded with probability p is
   P_p = P50 * (1 - z_p * sigma) with z_p = Phi^-1(p) (p = 0.90 -> z = 1.2816).
   P50 is the mean/median of the assumed normal distribution.

Assumption: the energy distribution is normal and sigma is small enough that
P50*(1 - z*sigma) stays positive. For large sigma the normal tail is unphysical; the
functions raise ValueError when the result would be non-positive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

DEFAULT_LEVELS = (50, 75, 90, 99)


@dataclass(frozen=True)
class Component:
    """One uncertainty source. ``sigma`` is a 1-sigma fraction of P50.

    kind: ``"long_term"`` (systematic, does not shrink with more years) or
    ``"interannual"`` (year-to-year variability, shrinks as 1/sqrt(N) over N years).
    """
    name: str
    sigma: float
    kind: str = "long_term"

    def __post_init__(self):
        if self.kind not in ("long_term", "interannual"):
            raise ValueError(f"kind must be 'long_term' or 'interannual', got {self.kind!r}")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError(f"sigma must be finite and >= 0, got {self.sigma!r} for {self.name}")


#: ILLUSTRATIVE values only, chosen to show the mechanics. They are not measured, not
#: recommended and not taken from any assessment. Replace them with your own.
EXAMPLE_COMPONENTS = (
    Component("measurement", 0.03),
    Component("long_term_reference", 0.03),
    Component("wind_flow_model", 0.04),
    Component("loss_estimate", 0.02),
    Component("interannual_variability", 0.06, kind="interannual"),
)


def apply_loss_chain(gross: float, losses: Mapping[str, float] | Iterable[float]) -> Dict[str, float]:
    """Multiplicative loss chain. ``losses`` are fractions in [0, 1).

    Returns ``{"gross", "net", "net_factor", "total_loss"}``.
    """
    vals = list(losses.values()) if isinstance(losses, Mapping) else list(losses)
    factor = 1.0
    for v in vals:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"loss must be in [0, 1), got {v}")
        factor *= 1.0 - v
    return {"gross": gross, "net": gross * factor, "net_factor": factor, "total_loss": 1.0 - factor}


def combine_rss(sigmas: Sequence[float]) -> float:
    """Root-sum-square of independent 1-sigma values."""
    s = np.asarray(list(sigmas), dtype=float)
    return float(np.sqrt(np.sum(s ** 2)))


def combine_correlated(sigmas: Sequence[float], corr) -> float:
    """sigma = sqrt(s^T R s) for a correlation matrix ``R`` (symmetric, unit diagonal, PSD)."""
    s = np.asarray(list(sigmas), dtype=float)
    r = np.asarray(corr, dtype=float)
    n = len(s)
    if r.shape != (n, n):
        raise ValueError(f"corr must be {n}x{n}, got {r.shape}")
    if not np.allclose(r, r.T, atol=1e-9) or not np.allclose(np.diag(r), 1.0, atol=1e-9):
        raise ValueError("corr must be symmetric with unit diagonal")
    if np.min(np.linalg.eigvalsh((r + r.T) / 2)) < -1e-9:
        raise ValueError("corr must be positive semi-definite")
    return float(np.sqrt(max(s @ r @ s, 0.0)))


def horizon_sigma(sigma_long_term: float, sigma_interannual: float, years: float) -> float:
    """sqrt(sigma_lt^2 + sigma_iav^2 / years). ``years`` >= 1 (1 = single year)."""
    if years < 1:
        raise ValueError("years must be >= 1")
    return float(np.sqrt(sigma_long_term ** 2 + sigma_interannual ** 2 / years))


def total_sigma(components: Iterable[Component], years: float = 1.0, corr=None) -> float:
    """Combine components for an ``years``-year horizon.

    Long-term components are combined by RSS (or with ``corr``, a correlation matrix over the
    long-term components in the order given). Interannual components are RSS-combined
    among themselves, then divided by sqrt(years) before joining.
    """
    comps = list(components)
    lt = [c.sigma for c in comps if c.kind == "long_term"]
    iav = [c.sigma for c in comps if c.kind == "interannual"]
    s_lt = combine_correlated(lt, corr) if (corr is not None and lt) else combine_rss(lt)
    return horizon_sigma(s_lt, combine_rss(iav), years)


def speed_to_energy_sigma(sigma_speed_fraction: float, sensitivity: float) -> float:
    """Convert a wind-speed uncertainty (fraction of mean speed) to energy (fraction of energy).

    ``sensitivity`` = d ln(E) / d ln(v) at the site; it depends on the power curve and the
    speed distribution and is a user input (values around 1.5 to 2.5 are common in
    screening; that is a rule of thumb, not a recommendation).
    """
    return sensitivity * sigma_speed_fraction


def exceedance_value(p50: float, sigma: float, level: float) -> float:
    """Value exceeded with probability ``level`` percent (90 -> P90) under a normal distribution."""
    if not (0 < level < 100):
        raise ValueError("level must be in (0, 100)")
    z = norm.ppf(level / 100.0)
    val = p50 * (1.0 - z * sigma)
    if val <= 0:
        raise ValueError(f"P{level:g} is non-positive (sigma={sigma}); normal approximation invalid")
    return float(val)


def exceedance_table(p50: float, sigma: float, levels: Sequence[float] = DEFAULT_LEVELS) -> Dict[str, float]:
    """``{"P50": ..., "P75": ..., ...}`` for the given levels."""
    return {f"P{lv:g}": exceedance_value(p50, sigma, lv) for lv in levels}
