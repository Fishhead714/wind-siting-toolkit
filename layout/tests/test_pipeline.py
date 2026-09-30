"""End-to-end runs on synthetic terrain: routing, output tables, idempotency."""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

from wind_layout.pipeline import OUTPUT_DISCLAIMER, run_pipeline
from conftest import UTM_EPSG, zone_polygon

RATED_KW = 2400.0
DIAMETER = 100.0


def _turbine(rho=1.225):
    return GenericWindTurbine(name="Synthetic2p4MW", diameter=DIAMETER, hub_height=80,
                              power_norm=RATED_KW, turbulence_intensity=0.10, air_density=rho)


def _zones(geom=None):
    return gpd.GeoDataFrame({"id": ["zA"]}, geometry=[geom or zone_polygon()], crs=4326)


def _land(row):
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([row.geometry], crs=4326).to_crs(UTM_EPSG))


def _run(dem, out_dir, windrose, n_wt=3, zones=None, **kw):
    return run_pipeline(
        zones or _zones(), dem, _land, _turbine(), lambda zid: windrose, lambda row: n_wt,
        out_dir, turbine_factory=_turbine, turbine_rated_mw=RATED_KW / 1000,
        wake_model_sensitivity_check=False, shear_alpha=0.14, **kw)


def test_complex_terrain_goes_through_mountain_branch(dem_complex, windrose, tmp_path):
    (res,) = _run(dem_complex, tmp_path / "o", windrose)
    assert res["route"] == "mountain_branch"
    assert res["n_turbines"] >= 3
    assert res["disclaimer"] == OUTPUT_DISCLAIMER
    assert "not_a_bankable_energy_assessment" in res["disclaimer"]
    assert (tmp_path / "o" / "zA_turbines.gpkg").exists()
    assert res["n_spacing_conflict"] == 0
    assert res["site_conditions"]["hub_height_shear_applied"] is True


def test_site_suitability_table_sanity(dem_complex, windrose, tmp_path):
    (res,) = _run(dem_complex, tmp_path / "o", windrose)
    df = pd.read_csv(res["site_suitability_csv"], encoding="utf-8-sig")
    assert len(df) == res["n_turbines"]
    assert df["wtg_id"].is_unique
    # elevations inside the synthetic surface range (150..450 m)
    assert df["elev_m"].between(150, 450).all()
    assert df["slope_deg"].between(0, 90).all()
    # air density at those elevations: physically plausible band, below sea level value
    assert df["air_density_kg_m3"].between(1.0, 1.225).all()
    assert df["weibull_A_mean"].between(5, 15).all() and df["weibull_k_mean"].between(1, 4).all()
    # one turbine cannot produce more than rated power x 8760 h
    max_gwh = RATED_KW / 1e6 * 8760
    assert df["aep_net_gwh"].between(0, max_gwh).all()
    assert (df["inflow_angle_deg"].dropna() >= 0).all()
    assert set(df["inflow_angle_flag"]) <= {"OK", "EXCEEDS_8DEG_SCREEN", "UNKNOWN"}
    assert set(df["ti_verdict"]) <= {"PASS", "FAIL", "NOT_CHECKED"}
    assert df["ti_verdict_is_lenient_vs_iec"].astype(str).eq("True").all()
    assert (df["design_ti"] > 0).all() and (df["effective_ti_max"] >= 0.0).all()


def test_flat_terrain_goes_through_flat_branch(dem_flat, windrose, tmp_path):
    (res,) = _run(dem_flat, tmp_path / "o", windrose, n_wt=3)
    assert res["route"] == "flat_branch"
    assert res["n_turbines"] == 3
    bi = res["branch_info"]
    assert bi["feasible"] is True
    assert 0 <= bi["wake_loss_pct"] < 40
    assert bi["aep_net_gwh"] <= bi["aep_gross_gwh"]


def test_idempotent_second_run_returns_same_result(dem_complex, windrose, tmp_path):
    out = tmp_path / "o"
    first = _run(dem_complex, out, windrose)
    files = sorted(p.name for p in out.iterdir())
    gpkg_bytes = (out / "zA_turbines.gpkg").read_bytes()
    second = _run(dem_complex, out, windrose)
    assert second == first                                       # identical JSON round-trip
    assert sorted(p.name for p in out.iterdir()) == files        # no extra outputs
    assert (out / "zA_turbines.gpkg").read_bytes() == gpkg_bytes  # not rewritten


def test_same_input_two_dirs_gives_same_layout(dem_complex, windrose, tmp_path):
    a = _run(dem_complex, tmp_path / "a", windrose)[0]
    b = _run(dem_complex, tmp_path / "b", windrose)[0]
    ga = gpd.read_file(tmp_path / "a" / "zA_turbines.gpkg", layer="turbines_utm")
    gb = gpd.read_file(tmp_path / "b" / "zA_turbines.gpkg", layer="turbines_utm")
    assert np.allclose(ga.geometry.x, gb.geometry.x) and np.allclose(ga.geometry.y, gb.geometry.y)
    assert a["branch_info"]["aep_net_gwh"] == pytest.approx(b["branch_info"]["aep_net_gwh"])


def test_changed_parameters_refuse_to_skip(dem_complex, windrose, tmp_path):
    out = tmp_path / "o"
    _run(dem_complex, out, windrose)
    with pytest.raises(RuntimeError, match="parameters differ"):
        _run(dem_complex, out, windrose, downwind_d=6.0)


def test_zones_must_be_wgs84(dem_complex, windrose, tmp_path):
    z = _zones().to_crs(UTM_EPSG)
    with pytest.raises(ValueError):
        _run(dem_complex, tmp_path / "o", windrose, zones=z)


def test_low_wind_filter_recomputes_energy(dem_complex, windrose, tmp_path):
    from wind_layout.terrainclassify import load_dem
    elev, _ = load_dem(dem_complex)
    ws = np.where(elev > 350, 8.0, 5.0).astype(np.float32)     # only high ground is windy
    (res,) = _run(dem_complex, tmp_path / "o", windrose, wind_speed_raster=ws, min_wind_speed_ms=7.0)
    df = pd.read_csv(res["site_suitability_csv"], encoding="utf-8-sig")
    assert (df["mean_ws_ms"] >= 7.0).all()
    assert res["branch_info"]["wind_speed_filter"]["min_wind_speed_ms"] == 7.0
