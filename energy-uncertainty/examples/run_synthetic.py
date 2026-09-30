"""Synthetic demo. Every number here is an illustrative placeholder, not data from any site."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wind_energy_uncertainty import (EXAMPLE_COMPONENTS, apply_loss_chain,  # noqa: E402
                                     exceedance_table, total_sigma)

gross_gwh = 100.0
losses = {"wake": 0.08, "availability": 0.03, "electrical": 0.02, "other": 0.01}  # example values
net = apply_loss_chain(gross_gwh, losses)
print(f"gross {gross_gwh:.1f} -> net P50 {net['net']:.2f} (total loss {net['total_loss']:.1%})")
for years in (1, 10):
    s = total_sigma(EXAMPLE_COMPONENTS, years=years)
    tab = exceedance_table(net["net"], s)
    print(f"{years:>2}-year horizon, sigma {s:.1%}: " + ", ".join(f"{k} {v:.2f}" for k, v in tab.items()))
