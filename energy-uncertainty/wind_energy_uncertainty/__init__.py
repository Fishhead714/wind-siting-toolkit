"""wind_energy_uncertainty: loss chain and uncertainty combination to exceedance values.

Screening-level arithmetic under a normal-distribution assumption. Not a bankable
energy assessment; see ``README.md``.
"""
from .core import (  # noqa: F401
    DEFAULT_LEVELS,
    EXAMPLE_COMPONENTS,
    Component,
    apply_loss_chain,
    combine_correlated,
    combine_rss,
    exceedance_table,
    exceedance_value,
    horizon_sigma,
    speed_to_energy_sigma,
    total_sigma,
)

__version__ = "0.1.0"
