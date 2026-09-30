"""Generic turbine description. Every limit is optional; a missing limit means *unknown*."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Sequence, Tuple


@dataclass
class TurbineSpec:
    model: str
    iec_class: Optional[str] = None
    rotor_diameter_m: Optional[float] = None
    hub_height_options_m: Optional[Sequence[float]] = None
    vref_max_ms: Optional[float] = None
    vave_max_ms: Optional[float] = None
    ve50_max_ms: Optional[float] = None
    ref_turbulence_iref: Optional[float] = None
    air_density_min_kgm3: Optional[float] = None
    air_density_max_kgm3: Optional[float] = None
    extreme_air_density_max_kgm3: Optional[float] = None
    max_altitude_m: Optional[float] = None
    operating_temp_min_c: Optional[float] = None
    operating_temp_max_c: Optional[float] = None
    survival_temp_min_c: Optional[float] = None
    derating_temp_c: Optional[float] = None
    max_shear_exponent: Optional[float] = None
    max_flow_inclination_deg: Optional[float] = None
    min_tip_clearance_m: Optional[float] = None
    cutout_wind_speed_ms: Optional[float] = None
    source: str = "user-supplied"

    @classmethod
    def from_dict(cls, d: dict) -> "TurbineSpec":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown turbine fields: {sorted(unknown)}")
        return cls(**d)
