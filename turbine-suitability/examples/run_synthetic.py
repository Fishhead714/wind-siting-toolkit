"""Runnable demo with entirely synthetic inputs (no real turbine, no real place).

    python examples/run_synthetic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wind_turbine_suitability import (SiteConditions, SuitabilityEngine, TurbineSpec,  # noqa: E402
                                      decide_tier, render_markdown)
from wind_turbine_suitability import extremes, roughness, temperature, turbulence  # noqa: E402

HUB = 110.0

# 1. Estimate site inputs from synthetic records.
annual_max_1min_10m = [24 + (i * 7 % 9) for i in range(30)]          # synthetic annual maxima, m/s
v50 = extremes.v50_hub_from_annual_maxima(annual_max_1min_10m, HUB, exponent=0.11)
z0 = roughness.equivalent_z0([(roughness.DAVENPORT_WIERINGA_Z0_M["open"], 0.7),
                              (roughness.DAVENPORT_WIERINGA_Z0_M["rough"], 0.3)], HUB)
ti = turbulence.ti_p90_interval(z0 * 0.5, z0 * 2.0, HUB)
tmax = temperature.temperature_extreme_interval([26 + (i % 6) for i in range(30)], "max", 100, 120)
tmin = temperature.temperature_extreme_interval([-4 - (i % 6) for i in range(30)], "min", 100, 120)

site = SiteConditions("Synthetic site")
site.set("hub_height_m", HUB, "assumption", "m")
site.set("v50_ms", round(v50, 2), "model", "m/s")
site.set("vave_ms", round(v50 / 5.0, 2), "model", "m/s")
site.set("ve50_ms", round(v50 * 1.35, 2), "assumption", "m/s")   # synthetic gust ratio
site.set("ti_p90_range", ti, "assumption")
site.set("air_density_kgm3", 1.22, "model")
site.set("air_density_cold_extreme_kgm3", 1.27, "model")
site.set("elevation_max_m", 120.0, "model", "m")
site.set("temp_max_abs_c", tmax["interval_c"], "model", "C")
site.set("temp_min_abs_c", tmin["interval_c"], "model", "C")
site.set("shear_exponent", 0.15, "model")
site.set("flow_inclination_deg", 2.0, "model")

# 2. Synthetic, generic turbine descriptions (not any real product).
catalog = [
    TurbineSpec("Generic 3 MW class (synthetic)", "II", 120.0, [110.0], vref_max_ms=42.5,
                ref_turbulence_iref=0.16, air_density_min_kgm3=1.0, air_density_max_kgm3=1.3,
                max_altitude_m=2000.0, operating_temp_max_c=45.0, survival_temp_min_c=-30.0,
                cutout_wind_speed_ms=25.0, source="synthetic"),
    TurbineSpec("Generic low-wind class (synthetic)", "III", 150.0, [110.0], vref_max_ms=37.5,
                ref_turbulence_iref=0.12, air_density_min_kgm3=1.0, air_density_max_kgm3=1.3,
                max_altitude_m=2000.0, operating_temp_max_c=45.0, survival_temp_min_c=-30.0,
                cutout_wind_speed_ms=25.0, source="synthetic"),
]

tier = decide_tier()      # no on-site measurement: model-level wording only
results = SuitabilityEngine().screen({HUB: site}, catalog, tier=tier)
print(render_markdown(results))
