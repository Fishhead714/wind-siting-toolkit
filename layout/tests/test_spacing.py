import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from wind_layout.spacingcheck import (
    check_spacing, ellipse_conflict, ellipse_required_distance,
    dominant_energy_azimuth, high_energy_azimuths,
)

D = 100.0
A, B = 5 * D / 2, 3 * D / 2      # half-axes for 5D downwind / 3D crosswind


def _gdf(coords):
    return gpd.GeoDataFrame(
        {"wtg_id": [f"T{i}" for i in range(len(coords))]},
        geometry=[Point(*c) for c in coords], crs="EPSG:32707")


def test_required_distance_ellipse_extremes():
    # wind from north-south axis (azimuth 0): along-axis needs 5D, across needs 3D
    assert ellipse_required_distance(A, B, 0.0, 0.0) == pytest.approx(5 * D)
    assert ellipse_required_distance(A, B, 0.0, 90.0) == pytest.approx(3 * D)


def test_conflict_is_direction_dependent():
    # 400 m apart: fine across the wind (needs 300), a conflict along it (needs 500)
    assert bool(ellipse_conflict(0.0, 400.0, A, B, 0.0)) is True
    assert bool(ellipse_conflict(400.0, 0.0, A, B, 0.0)) is False


def test_check_spacing_statuses():
    gdf = _gdf([(0, 0), (0, 400), (2000, 0), (2320, 0)])
    r = check_spacing(gdf, D, azimuth_deg=0.0, downwind_d=5, crosswind_d=3)
    assert r["status"]["T0"] == "CONFLICT" and r["status"]["T1"] == "CONFLICT"
    assert r["status"]["T2"] in ("OK", "NEAR")
    assert len(r["conflicts"]) == 1
    assert len(r["ellipses_gdf"]) == 4


def test_check_spacing_ok_far_apart():
    gdf = _gdf([(0, 0), (0, 2000), (2000, 0)])
    r = check_spacing(gdf, D, azimuth_deg=45.0)
    assert set(r["status"].values()) == {"OK"}


def test_check_spacing_rejects_geographic_crs():
    gdf = _gdf([(0, 0), (0, 1)]).set_crs("EPSG:4326", allow_override=True)
    with pytest.raises(ValueError):
        check_spacing(gdf, D, azimuth_deg=0.0)


def test_energy_azimuth_prefers_strong_sector(windrose):
    wr = dict(windrose)
    wr["sector_freq_pct"] = [100 / 12] * 12
    wr["sector_weibull_A"] = [6.0] * 12
    wr["sector_weibull_A"][3] = 12.0     # windy sector centred at 90 deg
    assert dominant_energy_azimuth(wr) == pytest.approx(90.0)
    assert 90.0 in high_energy_azimuths(wr, 0.5)
