"""Candidate generation on complex terrain: snapping to buildable ground and exclusions."""
import geopandas as gpd
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Point, box

from wind_layout.rasterutils import sample_raster
from wind_layout.spacingcheck import check_spacing
from wind_layout.terrainclassify import load_dem
from wind_layout.terrainlayout import (
    compute_slope, compute_valley_mask, optimize_layout_mountain,
    greedy_min_spacing_select,
)
from conftest import UTM_EPSG, zone_polygon

D = 100.0


def _land(zone_poly, minus=None):
    g = gpd.GeoSeries([zone_poly], crs=4326).to_crs(UTM_EPSG)
    if minus is not None:
        g = gpd.GeoSeries([g.iloc[0].difference(minus)], crs=UTM_EPSG)
    return gpd.GeoDataFrame(geometry=g)


def _run(dem, land, **kw):
    elev, tr = load_dem(dem)
    slope = compute_slope(elev, tr)
    valley = compute_valley_mask(elev, tr)
    out = optimize_layout_mountain(land, elev, slope, valley, tr, D, **kw)
    return out, elev, slope, valley, tr


def test_candidates_inside_land_on_buildable_ground(dem_complex, zone):
    land = _land(zone)
    out, elev, slope, valley, tr = _run(dem_complex, land, buildability_slope_deg=17.0)
    pts = out["points_gdf"]
    assert len(pts) >= 3
    poly = land.geometry.union_all()
    assert all(poly.covers(p) for p in pts.geometry)
    lon, lat = Transformer.from_crs(UTM_EPSG, 4326, always_xy=True).transform(
        pts.geometry.x.values, pts.geometry.y.values)
    s = sample_raster(slope, tr, lon, lat)
    assert np.all(s <= 17.0 + 1e-6)             # slope hard-exclusion honoured
    v = sample_raster(valley.astype(float), tr, lon, lat)
    assert np.all(v < 0.5)                      # never in a TPI valley


def test_exclusion_zone_is_respected(dem_complex, zone):
    land = _land(zone)
    base, *_ = _run(dem_complex, land)
    pts = base["points_gdf"]
    # carve a hole around the first candidate; it must disappear from the result
    victim = pts.geometry.iloc[0]
    hole = victim.buffer(400.0)
    out, *_ = _run(dem_complex, _land(zone, minus=hole))
    assert len(out["points_gdf"]) >= 1
    assert not any(hole.covers(p) for p in out["points_gdf"].geometry)


def test_ellipse_selection_passes_ellipse_check(dem_complex, zone):
    land = _land(zone)
    az = 90.0
    out, *_ = _run(dem_complex, land, spacing_ellipse=(5 * D / 2, 3 * D / 2, az))
    r = check_spacing(out["points_gdf"], D, azimuth_deg=az, downwind_d=5, crosswind_d=3)
    assert r["conflicts"] == []


def test_min_spacing_greedy_respects_distance():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 3000, size=(400, 2))
    prio = rng.uniform(size=400)
    sel = greedy_min_spacing_select(pts, prio, 300.0)
    d = np.hypot(sel[:, None, 0] - sel[None, :, 0], sel[:, None, 1] - sel[None, :, 1])
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 300.0 - 1e-9
    # the highest-priority point is always taken first
    assert any(np.allclose(q, pts[int(np.argmax(prio))]) for q in sel)


def test_snap_moves_boundary_point_to_safe_side():
    from wind_layout.layoutoptimize import _snap_to_safe_side
    usable = box(0, 0, 1000, 1000)
    excl = [box(400, 400, 600, 600)]
    pts = [Point(500.0, 1000.3),      # 0.3 m outside usable land: numerical residual
           Point(500.0, 599.7),       # 0.3 m inside exclusion
           Point(500.0, 800.0),       # fine
           Point(500.0, 1200.0)]      # 200 m outside: real violation, must not be moved
    out, rec = _snap_to_safe_side(pts, usable, excl, margin_m=5.0, max_penetration_m=1.0)
    ok = lambda q: usable.covers(q) and not any(z.covers(q) for z in excl)
    assert ok(out[0]) and ok(out[1])
    assert out[2].equals(pts[2])
    assert out[3].equals(pts[3])
    flags = {r["index"]: r["snapped"] for r in rec}
    assert flags == {0: True, 1: True, 3: False}
