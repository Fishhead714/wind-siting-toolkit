import numpy as np
import pytest

from wind_layout.terrainclassify import (
    classify_zone, compute_rix, dem_pixel_size_m, load_dem,
    RIX_CRITICAL_SLOPE_DEG,
)
from wind_layout.terrainlayout import compute_slope
from conftest import zone_polygon


def _classify(path, geom, **kw):
    elev, tr = load_dem(path)
    return classify_zone(geom, elev, tr, slope_deg=compute_slope(elev, tr), **kw), elev, tr


def test_flat_routes_to_flat_branch(dem_flat, zone):
    r, _, _ = _classify(dem_flat, zone)
    assert r["tier"] == "flat"
    assert r["route"] == "flat_branch"
    assert r["relief_m"] < 100


def test_rolling_gentle_overturned_to_flat_by_rix(dem_rolling, zone):
    r, _, _ = _classify(dem_rolling, zone)
    assert r["relief_m"] >= 100           # relief alone would say "mountain"
    assert r["tier"] == "rolling"
    assert r["rix_pct"] <= 1.0
    assert r["route"] == "flat_branch"    # fine DEM + RIX ~ 0 overturns the relief verdict


def test_complex_routes_to_mountain_branch(dem_complex, zone):
    r, _, _ = _classify(dem_complex, zone)
    assert r["rix_pct"] >= 5.0
    assert r["route"] == "mountain_branch"
    assert r["tier"] in ("rolling", "mountainous")


def test_relief_only_fallback_without_slope_raster(dem_rolling, zone):
    elev, tr = load_dem(dem_rolling)
    r = classify_zone(zone, elev, tr)     # no slope raster -> no RIX
    assert r["rix_pct"] is None
    assert r["route"] == "mountain_branch"


def test_dem_resolution_gate_blocks_rix_overturn(dem_rolling_coarse, zone):
    """On a DEM coarser than the gate a low RIX must NOT overturn the relief verdict."""
    elev, tr = load_dem(dem_rolling_coarse)
    assert max(dem_pixel_size_m(tr)) > 60.0
    r = classify_zone(zone, elev, tr, slope_deg=compute_slope(elev, tr))
    assert r["relief_m"] >= 100
    assert r["route"] == "mountain_branch"
    # raising the gate lets the same low RIX overturn: proves the gate is what decided it
    r2 = classify_zone(zone, elev, tr, slope_deg=compute_slope(elev, tr),
                       rix_max_pixel_size_m=500.0)
    assert r2["rix_pct"] <= 1.0
    assert r2["route"] == "flat_branch"


def test_rix_is_share_of_steep_pixels(tmp_path):
    """RIX on a hand-built slope raster: 25% of pixels steeper than the critical slope."""
    from rasterio.transform import from_origin
    slope = np.zeros((40, 40), dtype=np.float32)
    slope[:, :10] = 30.0                  # first quarter of the columns is steep
    tr = from_origin(-140.0, -19.8, 0.0004, 0.0004)
    geom = zone_polygon(frac=(0.0, 1.0), n=40)
    r = compute_rix(slope, tr, geom)
    assert r["rix_pct"] == pytest.approx(25.0, abs=3.0)
    assert r["critical_slope_deg"] == pytest.approx(np.degrees(np.arctan(0.3)), abs=0.05)
    assert RIX_CRITICAL_SLOPE_DEG == pytest.approx(16.7, abs=0.05)


def test_rix_critical_slope_parameter_matters(tmp_path):
    from rasterio.transform import from_origin
    slope = np.full((20, 20), 10.0, dtype=np.float32)
    tr = from_origin(-140.0, -19.8, 0.0004, 0.0004)
    geom = zone_polygon(frac=(0.0, 1.0), n=20)
    assert compute_rix(slope, tr, geom)["rix_pct"] == 0.0
    assert compute_rix(slope, tr, geom, critical_slope_deg=5.0)["rix_pct"] == 100.0


def test_high_rix_overrides_low_relief(dem_steep_small, zone):
    """Relief alone would say flat; RIX >= threshold sends the zone to the mountain branch."""
    r, _, _ = _classify(dem_steep_small, zone)
    assert r["relief_m"] < 100
    assert r["rix_pct"] >= 5.0
    assert r["route"] == "mountain_branch"
    # with the RIX threshold raised above the computed value the same zone is flat
    r2, _, _ = _classify(dem_steep_small, zone, rix_threshold_pct=r["rix_pct"] + 1.0)
    assert r2["route"] == "flat_branch"
