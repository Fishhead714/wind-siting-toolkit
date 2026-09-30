"""Evidence tier: how much of the decision may rest on on-site measurement.

Rules this module implements (the tier is computed from facts; the engine reports, but does not
block, measured fields found in a model-only tier):

* A partial campaign (< 12 months) or one without long-term correction does **not** enter the
  decision path; it can only calibrate or cross-check the model. A few months of data replacing
  the model would turn a seasonal bias into a fake annual truth.
* The verdict wording depends on the tier. Model-only tiers can never say "fit" without
  qualification.
"""
from __future__ import annotations

from typing import Optional

from .site import SiteConditions

MIN_MONTHS_FOR_MEASURED_TIER = 12.0

_EXCEEDS = "exceeds class design assumptions - needs turbine-maker site-specific load check"


def _wording(pass_txt, marginal_txt, tail):
    return {"pass": pass_txt, "marginal": marginal_txt,
            "insufficient_data": "insufficient data - cannot decide",
            "fail": "not suitable (exceeds a hard turbine limit)",
            "exceeds_class_assumption": _EXCEEDS + tail}


VERDICT_WORDING = {
    "A": _wording("preliminary fit (model level)", "borderline - manual review (model level)", " (model level)"),
    "A_short": _wording("preliminary fit (model level, short on-site record as cross-check)",
                        "borderline - manual review (model level)", " (model level)"),
    "A_neigh": _wording("preliminary fit (model level, nearby-mast cross-check)",
                        "borderline - manual review (model level)", " (model level)"),
    "B": _wording("preliminary fit (measurement-supported)", "borderline - manual review (measurement-supported)",
                  " (measurement-supported)"),
}


def decide_tier(onsite_months: Optional[float] = None, long_term_corrected: bool = False,
                neighbour_mast: bool = False) -> str:
    """Tier from facts. ``onsite_months=None`` means no on-site campaign."""
    if onsite_months is not None:
        if onsite_months >= MIN_MONTHS_FOR_MEASURED_TIER and long_term_corrected:
            return "B"
        return "A_short"
    return "A_neigh" if neighbour_mast else "A"


def wording(tier: str, overall: str) -> str:
    return VERDICT_WORDING[tier][overall]


def measurement_leaks(tier: str, site: SiteConditions):
    """Field names with 'measured' evidence in a model-only tier (should be empty)."""
    return [] if tier == "B" else site.measured_keys()
