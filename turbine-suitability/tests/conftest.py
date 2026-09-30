"""Synthetic fixtures. Nothing here describes a real turbine or a real place."""
import pytest

from wind_turbine_suitability import SiteConditions, TurbineSpec


def generic_turbine(**over):
    d = dict(model="Generic 3 MW class (synthetic)", iec_class="II", rotor_diameter_m=120.0,
             hub_height_options_m=[100.0], vref_max_ms=42.5, ref_turbulence_iref=0.16,
             air_density_min_kgm3=1.0, air_density_max_kgm3=1.3, max_altitude_m=2000.0,
             operating_temp_max_c=45.0, survival_temp_min_c=-30.0, operating_temp_min_c=-15.0,
             cutout_wind_speed_ms=25.0, source="synthetic")
    d.update(over)
    return TurbineSpec(**d)


def benign_site(hub=100.0):
    s = SiteConditions("Synthetic site A")
    s.set("hub_height_m", hub, "assumption", "m")
    s.set("v50_ms", 30.0, "model", "m/s")
    s.set("vave_ms", 6.0, "model", "m/s")
    s.set("ve50_ms", 40.0, "model", "m/s")
    s.set("ti_p90_range", (0.10, 0.13), "model")
    s.set("air_density_kgm3", 1.22, "model")
    s.set("air_density_cold_extreme_kgm3", 1.27, "model")
    s.set("elevation_max_m", 120.0, "model", "m")
    s.set("temp_max_abs_c", (28.0, 32.0), "model", "C")
    s.set("temp_min_abs_c", (-12.0, -8.0), "model", "C")
    s.set("shear_exponent", 0.14, "model")
    s.set("flow_inclination_deg", 3.0, "model")
    return s


@pytest.fixture
def turbine():
    return generic_turbine()


@pytest.fixture
def site():
    return benign_site()
