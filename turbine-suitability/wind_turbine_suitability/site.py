"""Site condition card: each field is a scalar or an interval plus its evidence level."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

Number = Union[int, float]
Value = Union[Number, Tuple[float, float], Sequence[float], None]

EVIDENCE_LEVELS = ("measured", "model", "assumption", "unavailable")


def is_finite_value(v) -> bool:
    """True for a finite scalar, or a non-empty sequence of finite numbers.

    Needed because NaN makes every comparison False, so an unchecked ``if x > limit`` falls
    through to the "pass" branch. NaN inputs must become *unknown*, never *pass*.
    """
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    return False


@dataclass
class SiteField:
    value: Value
    evidence: str = "model"
    unit: str = ""
    note: str = ""

    def __post_init__(self):
        if self.evidence not in EVIDENCE_LEVELS:
            raise ValueError(f"evidence must be one of {EVIDENCE_LEVELS}, got {self.evidence!r}")
        if isinstance(self.value, list):
            self.value = tuple(self.value)


@dataclass
class SiteConditions:
    """Conditions at one hub height. Field names used by the engine:

    hub_height_m, v50_ms, vave_ms, ve50_ms, ti_p90_range (interval, 90th percentile TI),
    air_density_kgm3, air_density_cold_extreme_kgm3, elevation_max_m,
    temp_max_abs_c (interval), temp_min_abs_c (interval), shear_exponent,
    flow_inclination_deg, and optional integer flags v50_uncertainty_level,
    ti_uncertainty_level, shear_neutral_only, terrain_complexity_level (verdict caps; the scale and
    thresholds are set in ``ScreeningConfig.cap_thresholds`` and are arbitrary examples).
    Missing fields are allowed; the engine reports them as unknown.
    """
    name: str
    fields: Dict[str, SiteField] = field(default_factory=dict)

    def set(self, key: str, value: Value, evidence: str = "model", unit: str = "", note: str = ""):
        self.fields[key] = SiteField(value, evidence, unit, note)
        return self

    def get(self, key: str) -> Optional[SiteField]:
        return self.fields.get(key)

    def value(self, key: str):
        f = self.fields.get(key)
        return None if f is None else f.value

    def measured_keys(self):
        return [k for k, f in self.fields.items() if f.evidence == "measured"]
