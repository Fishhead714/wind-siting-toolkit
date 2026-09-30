import math

import pytest

from wind_turbine_suitability import SuitabilityEngine, render_markdown
from wind_turbine_suitability.engine import DISCLAIMER

from conftest import benign_site, generic_turbine


def by_id(res, cid):
    return next(c for c in res["constraints"] if c["id"] == cid)


def test_benign_site_passes(site, turbine):
    r = SuitabilityEngine().evaluate(site, turbine)
    assert r["overall"] == "pass", [(c["id"], c["verdict"], c["reason"]) for c in r["constraints"]]
    assert "model level" in r["overall_label"]
    assert r["disclaimer"] == DISCLAIMER


def test_hard_limit_fails_and_not_compensated(site, turbine):
    site.set("v50_ms", 45.0)
    r = SuitabilityEngine().evaluate(site, turbine)
    assert r["overall"] == "fail"
    assert "vref" in r["blocking_hard_limits"]


def test_marginal_band(site, turbine):
    site.set("v50_ms", 41.0)  # inside limit but within 5 % band
    r = SuitabilityEngine().evaluate(site, turbine)
    assert by_id(r, "vref")["verdict"] == "marginal"
    assert r["overall"] == "marginal"


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nan_site_value_is_unknown_never_pass(site, turbine, bad):
    site.set("v50_ms", bad)
    r = SuitabilityEngine().evaluate(site, turbine)
    assert by_id(r, "vref")["verdict"] == "unknown"
    assert r["overall"] == "insufficient_data"


def test_all_required_nan_never_fit(turbine):
    s = benign_site()
    for k in list(s.fields):
        if k != "hub_height_m":
            s.set(k, float("nan"))
    r = SuitabilityEngine().evaluate(s, turbine)
    assert r["overall"] == "insufficient_data"


def test_missing_turbine_limit_is_unknown(site):
    t = generic_turbine(vref_max_ms=None)
    r = SuitabilityEngine().evaluate(site, t)
    assert by_id(r, "vref")["verdict"] == "unknown"
    assert "vref" in r["missing"]
    assert r["overall"] == "insufficient_data"


def test_vave_derived_from_vref_is_flagged(site, turbine):
    r = SuitabilityEngine().evaluate(site, turbine)
    c = by_id(r, "vave")
    assert c["limit_derived"] and "vave" in r["derived_limits"]


def test_class_assumption_exceedance_is_not_a_hard_fail(site, turbine):
    site.set("ti_p90_range", (0.30, 0.34))
    r = SuitabilityEngine().evaluate(site, turbine)
    assert r["overall"] == "exceeds_class_assumption"
    assert "turbulence" in r["exceeds_class_assumptions"]
    assert "not suitable" not in r["overall_label"]


def test_turbulence_interval_straddle_is_marginal(site, turbine):
    site.set("ti_p90_range", (0.10, 0.40))
    r = SuitabilityEngine().evaluate(site, turbine)
    assert by_id(r, "turbulence")["verdict"] == "marginal"


def test_turbulence_limit_uses_tightest_bin(site):
    """Independent hand computation: 0.16 * (0.75*25 + 5.6) / 25 = 0.15584; cut-out changes the bin."""
    eng = SuitabilityEngine()
    assert by_id(eng.evaluate(site, generic_turbine()), "turbulence")["limit"] == pytest.approx(0.15584, abs=1e-5)
    lim20 = by_id(eng.evaluate(site, generic_turbine(cutout_wind_speed_ms=20.0)), "turbulence")["limit"]
    assert lim20 == pytest.approx(0.16 * (0.75 * 20 + 5.6) / 20, abs=1e-5)
    # a site interval above the 25 m/s limit but below the 15 m/s limit: the engine must be the stricter one
    site.set("ti_p90_range", (0.16, 0.17))
    assert 0.17 <= 0.16 * (0.75 * 15 + 5.6) / 15
    assert by_id(eng.evaluate(site, generic_turbine()), "turbulence")["verdict"] == "fail"


def test_tip_clearance_geometry(site):
    t = generic_turbine(rotor_diameter_m=181.0)
    r = SuitabilityEngine().evaluate(site, t)
    assert by_id(r, "tip_clearance")["verdict"] == "fail"
    assert r["overall"] == "fail"


def test_v50_vave_ratio_is_flag_only(site, turbine):
    site.set("v50_ms", 12.0)  # ratio 2.0, outside the default window
    r = SuitabilityEngine().evaluate(site, turbine)
    assert "v50_consistency" in r["data_quality_flags"]
    assert r["overall"] != "fail" and "data-quality flag" in r["overall_label"]
    site.set("v50_ms", 15.0)  # ratio 2.5: inside the default window
    assert SuitabilityEngine().evaluate(site, turbine)["data_quality_flags"] == []


def test_interval_in_scalar_fields_is_unknown_not_crash(site, turbine):
    site.set("hub_height_m", (99.0, 101.0))
    site.set("air_density_kgm3", (1.10, 1.15))
    r = SuitabilityEngine().evaluate(site, turbine)
    assert by_id(r, "tip_clearance")["verdict"] == "unknown"
    assert by_id(r, "air_density")["verdict"] == "unknown"
    assert r["overall"] == "insufficient_data"


def test_zero_tip_clearance_override_is_respected(site):
    t = generic_turbine(min_tip_clearance_m=0.0, rotor_diameter_m=200.0)  # hub 100, radius 100 -> clearance 0
    assert by_id(SuitabilityEngine().evaluate(site, t), "tip_clearance")["verdict"] == "pass"
    t2 = generic_turbine(rotor_diameter_m=200.0)  # default 25 m required
    assert by_id(SuitabilityEngine().evaluate(site, t2), "tip_clearance")["verdict"] == "fail"


def test_screen_empty_raises_clear_error(turbine):
    with pytest.raises(ValueError):
        SuitabilityEngine().screen({}, [turbine])


def test_cold_stop_is_yield_flag_not_class_exceedance(site):
    t = generic_turbine(operating_temp_min_c=-5.0)   # site minimum interval (-12, -8) entirely below
    r = SuitabilityEngine().evaluate(site, t)
    assert "temp_low_operating" in r["yield_flags"]
    assert r["overall"] == "pass" and "exceeds class" not in r["overall_label"]


def test_tier_b_wording_is_preliminary(site, turbine):
    r = SuitabilityEngine().evaluate(site, turbine, tier="B")
    assert r["overall_label"] == "preliminary fit (measurement-supported)"


def test_unverified_table_is_reported(site, turbine):
    from wind_turbine_suitability import ClassTable, default_table
    r = SuitabilityEngine().evaluate(site, turbine)
    assert r["table_verified"] is False and "unverified" in r["disclaimer"]
    assert "not verified" in render_markdown([r])
    d = default_table()
    ok = ClassTable.from_dict({"wind_classes": {"II": {"vref_ms": 42.5}}, "turbulence_iref": {"A": 0.16},
                               "vave_over_vref": 0.2, "ve50_over_vref": 1.4, "ewm_shear_exponent": 0.11,
                               "ntm_b_ms": 5.6, "default_flow_inclination_deg": 8.0,
                               "default_shear_alpha": 0.2, "verified": True})
    assert SuitabilityEngine(table=ok).evaluate(site, turbine)["table_verified"] is True


def test_cap_thresholds_are_configurable(site, turbine):
    from wind_turbine_suitability import ScreeningConfig
    site.set("v50_uncertainty_level", 3)
    assert "vref" in SuitabilityEngine().evaluate(site, turbine)["verdict_capped"]
    cfg = ScreeningConfig(cap_thresholds=(("vref", "v50_uncertainty_level", 4),))
    assert SuitabilityEngine(config=cfg).evaluate(site, turbine)["verdict_capped"] == []


def test_verdict_cap_downgrades_pass(site, turbine):
    site.set("shear_neutral_only", 1)
    r = SuitabilityEngine().evaluate(site, turbine)
    assert "shear" in r["verdict_capped"]
    assert r["overall"] == "marginal"


def test_extreme_density_proxy_only_marginal(site):
    t = generic_turbine(air_density_max_kgm3=1.2)  # no extreme value given -> proxy
    site.set("air_density_cold_extreme_kgm3", 1.30)
    site.set("air_density_kgm3", 1.10)
    r = SuitabilityEngine().evaluate(site, t)
    assert by_id(r, "air_density_extreme")["verdict"] == "marginal"


def test_derating_flag_does_not_change_verdict(site):
    t = generic_turbine(derating_temp_c=25.0)
    r = SuitabilityEngine().evaluate(site, t)
    assert r["derating_flag"] and r["overall"] == "pass"


def test_screen_needs_hub_heights_and_sorts(site):
    eng = SuitabilityEngine()
    ok = generic_turbine(model="Generic A (synthetic)")
    nohub = generic_turbine(model="Generic B (synthetic)", hub_height_options_m=None)
    bad = generic_turbine(model="Generic C (synthetic)", vref_max_ms=25.0)
    out = eng.screen({100.0: site}, [nohub, bad, ok])
    assert [r["turbine"] for r in out][0] == "Generic A (synthetic)"
    assert out[-1]["overall"] == "fail"
    b = next(r for r in out if r["turbine"] == "Generic B (synthetic)")
    assert b["overall"] == "insufficient_data" and "hub_height_options_m" in b["missing"]


def test_screen_missing_hub_card_is_insufficient(site):
    t = generic_turbine(hub_height_options_m=[100.0, 140.0])
    out = SuitabilityEngine().screen({100.0: site}, [t])
    h140 = next(r for r in out if r["hub_height_m"] == 140.0)
    assert h140["overall"] == "insufficient_data"


def test_measured_field_flagged_in_model_tier(site, turbine):
    site.set("shear_exponent", 0.14, "measured")
    r = SuitabilityEngine().evaluate(site, turbine, tier="A")
    assert r["tier_policy_violation"]
    r2 = SuitabilityEngine().evaluate(site, turbine, tier="B")
    assert r2["tier_policy_violation"] is None


def test_report_renders(site, turbine):
    out = SuitabilityEngine().screen({100.0: site}, [turbine])
    md = render_markdown(out)
    assert "Screening-level" in md and "Generic 3 MW class" in md
