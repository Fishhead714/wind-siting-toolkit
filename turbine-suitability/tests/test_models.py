import math

import pytest

from wind_turbine_suitability import ClassTable, decide_tier, default_table, wording
from wind_turbine_suitability import extremes, roughness, temperature, turbulence
from wind_turbine_suitability.turbine import TurbineSpec


def test_table_override_and_no_hardcoded_limits():
    d = {"wind_classes": {"X": {"vref_ms": 40.0, "vave_ms": 8.0}}, "turbulence_iref": {"Q": 0.2},
         "vave_over_vref": 0.25, "ve50_over_vref": 1.5, "ewm_shear_exponent": 0.1,
         "ntm_b_ms": 5.0, "default_flow_inclination_deg": 7.0, "default_shear_alpha": 0.25}
    t = ClassTable.from_dict(d)
    assert t.vref_of("X") == 40.0
    assert t.required_turbulence_category(0.2 * t.ntm_p90_factor() * 0.99) == "Q"
    assert t.required_turbulence_category(0.5) == "S"


def test_p90_factor_and_curve_monotone():
    t = default_table()
    assert t.ntm_limit_curve(0.16, 25) < t.ntm_limit_curve(0.16, 15) < t.ntm_limit_curve(0.16, 5)
    # hand values: (0.75*15 + 5.6)/15 = 1.12333; 0.16 * (0.75*10 + 5.6)/10 = 0.2096
    assert t.ntm_p90_factor() == pytest.approx(1.123333, abs=1e-5)
    assert t.ntm_limit_curve(0.16, 10) == pytest.approx(0.2096, abs=1e-6)


def test_roughness_ti_space_average_not_arithmetic():
    z = roughness.equivalent_z0([(0.03, 0.95), (1.0, 0.05)], 100.0)
    arithmetic = 0.95 * 0.03 + 0.05 * 1.0
    assert 0.03 < z < arithmetic
    assert roughness.neutral_ti(0.03, 100.0) == pytest.approx(1 / math.log(100 / 0.03))
    with pytest.raises(ValueError):
        roughness.neutral_ti(0.0, 100.0)


def test_ti_interval_orders_and_verdict():
    lo, hi = turbulence.ti_p90_interval(0.03, 0.25, 100.0)
    assert lo < hi
    t = default_table()
    assert turbulence.verdict_against_iref((0.05, 0.06), 0.16, t) == "pass"
    assert turbulence.verdict_against_iref((0.30, 0.35), 0.12, t) == "fail"
    assert turbulence.verdict_against_iref((0.10, 0.30), 0.14, t) == "marginal"


def test_gumbel_recovers_parameters():
    # deterministic Gumbel quantile sample (mu=30, beta=3)
    n = 400
    xs = [30 - 3 * math.log(-math.log((i + 0.5) / n)) for i in range(n)]
    fit = extremes.gumbel_fit(xs)
    assert fit.mu == pytest.approx(30, abs=0.6) and fit.beta == pytest.approx(3, abs=0.3)
    assert fit.return_level(50) > fit.return_level(10) > fit.mu


def test_gumbel_short_sample_returns_none():
    assert extremes.gumbel_fit([30, 31, 29, 33]) is None
    assert extremes.gumbel_fit([25.0] * 20) is None
    assert extremes.v50_hub_from_annual_maxima([30, 31], 100.0, 0.11) is None


def test_nan_inputs_raise_not_silent():
    with pytest.raises(ValueError):
        roughness.neutral_ti(float("nan"), 100.0)
    with pytest.raises(ValueError):
        extremes.extrapolate_to_hub(30.0, 10.0, float("nan"), 0.11)
    with pytest.raises(ValueError):
        extremes.GumbelFit(30.0, 3.0, 20).return_level(float("nan"))
    with pytest.raises(ValueError):
        turbulence.verdict_against_iref((float("nan"), 0.1), 0.16, default_table())
    assert extremes.gumbel_fit([float("nan")] * 30) is None


def test_extrapolation_and_conversion():
    assert extremes.extrapolate_to_hub(30.0, 10.0, 100.0, 0.11) == pytest.approx(30 * 10 ** 0.11)
    xs = [25 + (i % 7) for i in range(30)]
    v = extremes.v50_hub_from_annual_maxima(xs, 100.0, 0.11)
    assert v is not None and v > max(xs) * extremes.ONE_MIN_TO_TEN_MIN


def test_temperature_interval_and_lapse():
    tmax = [34 + (i % 5) * 0.5 for i in range(25)]
    same = temperature.temperature_extreme_interval(tmax, "max", 100.0, 100.0)
    higher = temperature.temperature_extreme_interval(tmax, "max", 100.0, 1100.0)
    assert same["interval_c"][0] <= same["interval_c"][1]
    assert higher["interval_c"][1] == pytest.approx(same["interval_c"][1] - 6.5, abs=0.02)
    assert higher["elevation_correction_unreliable"] and not same["elevation_correction_unreliable"]
    tmin = [-5 - (i % 6) for i in range(25)]
    cold = temperature.temperature_extreme_interval(tmin, "min", 0.0, 0.0)
    assert cold["interval_c"][0] <= min(tmin) + 1e-9
    assert temperature.temperature_extreme_interval(tmin[:5], "min", 0, 0) is None


def test_tier_rules():
    assert decide_tier() == "A"
    assert decide_tier(neighbour_mast=True) == "A_neigh"
    assert decide_tier(onsite_months=6, long_term_corrected=True) == "A_short"
    assert decide_tier(onsite_months=14, long_term_corrected=False) == "A_short"
    assert decide_tier(onsite_months=14, long_term_corrected=True) == "B"
    assert "model level" in wording("A", "pass")
    assert "measurement-supported" in wording("B", "pass")


def test_turbine_unknown_field_rejected():
    with pytest.raises(ValueError):
        TurbineSpec.from_dict({"model": "x", "power_curve": [1, 2]})
