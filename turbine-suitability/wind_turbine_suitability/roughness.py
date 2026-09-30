"""Surface roughness length z0 and the neutral turbulence intensity it implies.

The Davenport-Wieringa classification (Wieringa 1992) is a published, widely cited scale.
The values below are the commonly quoted representative z0 per class, kept as an *example
default*; mapping a land-cover product onto these classes is an engineering judgement and is
left to the caller, who supplies (z0_low, z0_typical, z0_high) per surface type.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

DAVENPORT_WIERINGA_Z0_M = {
    "sea": 0.0002,
    "smooth": 0.005,
    "open": 0.03,
    "roughly_open": 0.10,
    "rough": 0.25,
    "very_rough": 0.5,
    "closed": 1.0,
    "chaotic": 2.0,
}


def neutral_ti(z0_m: float, height_m: float) -> float:
    """Neutral log-law turbulence intensity, TI = 1 / ln(z / z0). Treated as a mean TI."""
    if not (math.isfinite(z0_m) and math.isfinite(height_m)) or z0_m <= 0 or height_m <= z0_m:
        raise ValueError("need 0 < z0 < height")
    return 1.0 / math.log(height_m / z0_m)


def equivalent_z0(parts: Iterable[Tuple[float, float]], height_m: float) -> float:
    """Area-weighted equivalent z0 from ``(z0, weight)`` pairs.

    z0 spans orders of magnitude, so an arithmetic mean is dominated by a small patch of
    forest. Averaging is done in TI space (the quantity we need next) and inverted.
    """
    parts = [(z, w) for z, w in parts if w > 0]
    if not parts:
        raise ValueError("no positive weights")
    total = sum(w for _, w in parts)
    ti = sum(w * neutral_ti(z, height_m) for z, w in parts) / total
    return height_m * math.exp(-1.0 / ti)


def z0_interval_ti(z0_low: float, z0_high: float, height_m: float) -> Tuple[float, float]:
    """Neutral TI interval implied by a z0 interval (TI grows with z0)."""
    return neutral_ti(z0_low, height_m), neutral_ti(z0_high, height_m)
