"""Turbine-class parameters, expressed as data you can override.

**This is not a copy of the IEC 61400-1 tables.** The standard is a paid document and none of
its text or table layout is reproduced here. The numbers below are the widely quoted
headline values (reference wind speed, reference turbulence intensity and a few fixed ratios),
gathered from public secondary summaries of the
class scheme. They are provided only so the screening logic can run out of the box.

**Verify every number against the edition of the standard you hold** and, if they differ,
build your own table with :meth:`ClassTable.from_dict`. Nothing in the engine hard-codes a
class limit: every comparison goes through a ``ClassTable``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class WindClass:
    """One wind-speed class. ``None`` means the class does not define that limit."""
    name: str
    vref_ms: Optional[float]
    vave_ms: Optional[float] = None


@dataclass(frozen=True)
class ClassTable:
    wind_classes: Mapping[str, WindClass]
    # turbulence category name -> reference turbulence intensity I_ref (expected value at 15 m/s)
    turbulence_iref: Mapping[str, float]
    # fixed ratios used to derive limits a datasheet does not state
    vave_over_vref: float
    ve50_over_vref: float
    # exponent of the fixed power law used to extrapolate extreme winds to hub height
    ewm_shear_exponent: float
    # additive constant b (m/s) of the normal-turbulence sigma1 line: sigma1 = I_ref * (0.75 U + b)
    ntm_b_ms: float
    # fallbacks when a turbine gives no limit of its own
    default_flow_inclination_deg: float
    default_shear_alpha: float
    reference_air_density: float = 1.225
    # False until YOU have checked the numbers against the standard you hold and set it True.
    verified: bool = False

    def ntm_p90_factor(self, u_ref: float = 15.0) -> float:
        """Ratio (90th percentile sigma1 line) / (I_ref * U) at ``u_ref``.

        I_ref is an *expected value*; the site-side turbulence we estimate is a 90th percentile.
        Comparing them directly counts the percentile inflation twice, so limits are converted
        to the 90th percentile basis with this factor.
        """
        return (0.75 * u_ref + self.ntm_b_ms) / u_ref

    def ntm_limit_curve(self, iref: float, u: float) -> float:
        """Turbulence-intensity limit at wind speed ``u`` (90th percentile basis)."""
        return iref * (0.75 * u + self.ntm_b_ms) / u

    def required_turbulence_category(self, ti_p90: float) -> str:
        """Lowest category whose 15 m/s limit covers ``ti_p90``; ``'S'`` if none does.

        Uses the 15 m/s reference speed only; the engine's turbulence check is stricter (tightest bin)."""
        for cat in sorted(self.turbulence_iref, key=self.turbulence_iref.get):
            if ti_p90 <= self.turbulence_iref[cat] * self.ntm_p90_factor():
                return cat
        return "S"

    def vref_of(self, class_name: str) -> Optional[float]:
        wc = self.wind_classes.get(class_name)
        return None if wc is None else wc.vref_ms

    @classmethod
    def from_dict(cls, d: Mapping) -> "ClassTable":
        wc = {k: WindClass(k, v.get("vref_ms"), v.get("vave_ms"))
              for k, v in d["wind_classes"].items()}
        return cls(
            wind_classes=wc,
            turbulence_iref=dict(d["turbulence_iref"]),
            vave_over_vref=d["vave_over_vref"],
            ve50_over_vref=d["ve50_over_vref"],
            ewm_shear_exponent=d["ewm_shear_exponent"],
            ntm_b_ms=d["ntm_b_ms"],
            default_flow_inclination_deg=d["default_flow_inclination_deg"],
            default_shear_alpha=d["default_shear_alpha"],
            reference_air_density=d.get("reference_air_density", 1.225),
            verified=bool(d.get("verified", False)),
        )


def default_table() -> ClassTable:
    """Illustrative defaults. UNVERIFIED against the standard - see module docstring.

    Mixed editions: turbulence category A+ exists only in the newer edition, older editions have
    A/B/C. Class T's annual-mean limit is left undefined here; the fixed Vave = 0.2 x Vref ratio
    is applied to T only as an assumption, not confirmed from the standard.
    """
    return ClassTable.from_dict({
        "wind_classes": {
            "I": {"vref_ms": 50.0, "vave_ms": 10.0},
            "II": {"vref_ms": 42.5, "vave_ms": 8.5},
            "III": {"vref_ms": 37.5, "vave_ms": 7.5},
            "T": {"vref_ms": 57.0, "vave_ms": None},   # tropical-cyclone class
            "S": {"vref_ms": None, "vave_ms": None},   # special: designer-defined
        },
        "turbulence_iref": {"A+": 0.18, "A": 0.16, "B": 0.14, "C": 0.12},
        "vave_over_vref": 0.2,
        "ve50_over_vref": 1.4,
        "ewm_shear_exponent": 0.11,
        "ntm_b_ms": 5.6,
        "default_flow_inclination_deg": 8.0,
        "default_shear_alpha": 0.2,
    })
