import numpy as np
import pytest
import geopandas as gpd
import shapely
from shapely.geometry import box, Polygon
from pyproj import CRS

from wind_micrositing import (OUTPUT_DISCLAIMER, Raster, disk_fraction_filter, estimate_capacity,
                              max_raster_filter, min_raster_filter, oriented_grid, prepare_area,
                              site_turbines, slope_degrees)
from conftest import KW, METRIC_EPSG, X0, Y0


def _nn_min_dist(x, y):
    p = np.c_[x, y]
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(d, np.inf)
    return d.min()


def test_axis_aligned_grid_spacing():
    x, y = oriented_grid((0, 0, 3000, 3000), dx=500, dy=300, azimuth_deg=0)
    ys = np.unique(np.round(y, 6))
    xs = np.unique(np.round(x, 6))
    assert np.allclose(np.diff(ys), 500)      # downwind axis = north at azimuth 0
    assert np.allclose(np.diff(xs), 300)      # crosswind axis = east


def test_rotated_grid_projects_to_regular_lattice():
    az = 37.0
    x, y = oriented_grid((0, 0, 5000, 5000), 500, 300, az)
    a = np.radians(az)
    u = x * np.sin(a) + y * np.cos(a)
    v = x * np.cos(a) - y * np.sin(a)
    assert np.allclose(np.unique(np.round(u / 500, 6)) % 1, 0, atol=1e-6)
    assert np.allclose(np.unique(np.round(v / 300, 6)) % 1, 0, atol=1e-6)


def test_fixed_azimuth_count_on_rectangle(rect_gdf):
    # margin 0.5 D = 50 m -> inner 9900 x 5900. Aligned grid, offset 0.
    res = site_turbines(rect_gdf, azimuth_deg=0, **KW)
    inner = box(X0 + 50, Y0 + 50, X0 + 9950, Y0 + 5950)
    gx, gy = oriented_grid(inner.bounds, 500, 300, 0)
    expected = int(shapely.contains(inner, shapely.points(gx, gy)).sum())
    assert res.n_turbines == expected > 0
    assert res.crs == CRS.from_user_input(METRIC_EPSG)


def test_points_inside_and_spacing_respected(rect_gdf):
    res = site_turbines(rect_gdf, azimuth_deg=25, **KW)
    inner = rect_gdf.geometry.iloc[0].buffer(-50)
    assert shapely.contains(inner, shapely.points(res.x, res.y)).all()
    assert _nn_min_dist(res.x, res.y) >= 300 - 1e-6   # crosswind spacing is the tightest
    assert res.n_turbines > 0


def test_azimuth_axis_symmetry(rect_gdf):
    a = site_turbines(rect_gdf, azimuth_deg=30, **KW)
    b = site_turbines(rect_gdf, azimuth_deg=210, **KW)
    assert a.n_turbines == b.n_turbines
    pa = sorted(map(tuple, np.round(np.c_[a.x, a.y], 4)))
    pb = sorted(map(tuple, np.round(np.c_[b.x, b.y], 4)))
    assert pa == pb


def test_orientation_search_beats_or_equals_every_tried_and_is_deterministic(rect_gdf):
    r1 = site_turbines(rect_gdf, search_step_deg=15, **KW)
    r2 = site_turbines(rect_gdf, search_step_deg=15, **KW)
    assert r1.n_turbines == max(c for _, _, c in r1.search)
    assert (r1.azimuth_deg, r1.n_turbines) == (r2.azimuth_deg, r2.n_turbines)
    first_best = next(az for az, _, c in r1.search if c == r1.n_turbines)
    assert r1.azimuth_deg == first_best            # ties resolve to smallest azimuth
    assert len(r1.search) == 12                    # 0..165 step 15
    fixed = site_turbines(rect_gdf, azimuth_deg=0, **KW)
    assert r1.n_turbines >= fixed.n_turbines       # 0 deg is in the search set


def test_offset_search_finds_shifted_lattice_and_moves_points():
    # Inner strip is y in [50, 450] above Y0 (a multiple of 500): the unshifted
    # lattice has rows at 0 and 500 (both outside); a half-step shift puts a row at 250.
    strip = box(X0, Y0, X0 + 2000, Y0 + 500)
    none = site_turbines(strip, crs=METRIC_EPSG, azimuth_deg=0, offset_steps=1, **KW)
    two = site_turbines(strip, crs=METRIC_EPSG, azimuth_deg=0, offset_steps=2, **KW)
    assert none.n_turbines == 0
    assert two.n_turbines > 0 and two.offset[0] == 0.5
    assert np.allclose(two.y, Y0 + 250)


def test_exact_positions_with_distinct_spacings():
    # dx=500 (downwind = north at az 0), dy=300 (crosswind = east). X0 is not a
    # multiple of 300, so x rows sit at 500100, 500400, 500700.
    rect = box(X0, Y0, X0 + 1000, Y0 + 1200)
    res = site_turbines(rect, crs=METRIC_EPSG, azimuth_deg=0, **KW)
    got = sorted(zip(np.round(res.x, 6), np.round(res.y, 6)))
    want = sorted((x, y) for x in (500_100.0, 500_400.0, 500_700.0)
                  for y in (Y0 + 500, Y0 + 1000))
    assert res.n_turbines == 6
    assert np.allclose(got, want)
    # swapping the spacings would give 4 turbines: the test distinguishes them
    swapped = site_turbines(rect, crs=METRIC_EPSG, azimuth_deg=0, rotor_diameter_m=100.0,
                            spacing_downwind_d=3.0, spacing_crosswind_d=5.0)
    assert swapped.n_turbines == 4


def test_hole_is_respected():
    poly = Polygon(box(X0, Y0, X0 + 8000, Y0 + 8000).exterior.coords,
                   [box(X0 + 3000, Y0 + 3000, X0 + 5000, Y0 + 5000).exterior.coords])
    res = site_turbines(poly, crs=METRIC_EPSG, azimuth_deg=0, **KW)
    hole = box(X0 + 3000, Y0 + 3000, X0 + 5000, Y0 + 5000)
    assert not shapely.intersects(hole, shapely.points(res.x, res.y)).any()
    assert res.n_turbines > 0


def test_multi_feature_gdf_is_dissolved():
    g = gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 3000, Y0 + 3000),
                                   box(X0 + 6000, Y0, X0 + 9000, Y0 + 3000)], crs=METRIC_EPSG)
    res = site_turbines(g, azimuth_deg=0, **KW)
    assert res.x.min() < X0 + 3000 and res.x.max() > X0 + 6000
    assert abs(res.area_km2 - 18.0) < 1e-9


def test_area_too_small_gives_zero_turbines():
    res = site_turbines(box(X0, Y0, X0 + 80, Y0 + 80), crs=METRIC_EPSG, azimuth_deg=0, **KW)
    assert res.n_turbines == 0 and res.to_geodataframe().empty


def test_edge_margin_override(rect_gdf):
    big = site_turbines(rect_gdf, azimuth_deg=0, edge_margin_m=1000, **KW)
    small = site_turbines(rect_gdf, azimuth_deg=0, edge_margin_m=0, **KW)
    assert small.n_turbines >= big.n_turbines
    assert big.x.min() >= X0 + 1000


# ---- CRS policy -------------------------------------------------------------------------

def test_geographic_crs_rejected_by_default():
    ll = gpd.GeoDataFrame(geometry=[box(-177.1, 9.0, -177.0, 9.1)], crs=4326)
    with pytest.raises(ValueError, match="geographic"):
        site_turbines(ll, azimuth_deg=0, **KW)


def test_geographic_crs_reprojected_on_request_matches_metric():
    ll = gpd.GeoDataFrame(geometry=[box(-177.1, 9.0, -177.0, 9.1)], crs=4326)
    res = site_turbines(ll, azimuth_deg=0, on_geographic="reproject", **KW)
    assert CRS.from_user_input(res.crs).is_projected
    utm_geom = ll.to_crs(ll.estimate_utm_crs())
    ref = site_turbines(utm_geom, azimuth_deg=0, **KW)
    assert res.n_turbines == ref.n_turbines > 10
    assert abs(res.area_km2 - utm_geom.geometry.area.iloc[0] / 1e6) < 1e-9
    assert 100 < res.area_km2 < 130           # ~0.1 deg square near 9 N is ~120 km2


def test_degree_numbers_are_not_silently_treated_as_metres():
    geom = box(-177.1, 9.0, -177.0, 9.1)
    with pytest.raises(ValueError):
        site_turbines(geom, crs=4326, azimuth_deg=0, **KW)


def test_non_metre_projected_crs_rejected():
    # US survey feet state-plane style CRS
    ft = gpd.GeoDataFrame(geometry=[box(0, 0, 30000, 20000)], crs=2263)
    with pytest.raises(ValueError, match="metre"):
        site_turbines(ft, azimuth_deg=0, **KW)


def test_missing_crs_errors():
    with pytest.raises(ValueError, match="crs"):
        site_turbines(box(0, 0, 1000, 1000), azimuth_deg=0, **KW)
    with pytest.raises(ValueError, match="no CRS"):
        site_turbines(gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)]), azimuth_deg=0, **KW)


def test_conflicting_crs_errors(rect_gdf):
    with pytest.raises(ValueError, match="conflicts"):
        prepare_area(rect_gdf, crs=32602)


def test_argument_validation(rect_gdf):
    with pytest.raises(ValueError):
        site_turbines(rect_gdf, **KW)                       # neither azimuth nor step
    with pytest.raises(ValueError):
        site_turbines(rect_gdf, azimuth_deg=0, rotor_diameter_m=0,
                      spacing_downwind_d=5, spacing_crosswind_d=3)
    with pytest.raises(ValueError):
        site_turbines(rect_gdf, search_step_deg=0, **KW)


# ---- rasters / filters ------------------------------------------------------------------

def _mask_raster(fill=1.0):
    return Raster(np.full((300, 400), fill), X0 - 500, Y0 + 6500, 25.0)


def test_slope_of_known_plane():
    rows, cols = np.mgrid[0:50, 0:50]
    dem = Raster(0.1 * 25.0 * cols.astype(float), 0, 1250, 25.0)   # rise 0.1 per metre east
    s = slope_degrees(dem)
    assert np.allclose(s.array, np.degrees(np.arctan(0.1)), atol=1e-9)


def test_sample_off_raster_is_nan_and_fails_filters():
    r = _mask_raster()
    v = r.sample([X0 - 10_000], [Y0])
    assert np.isnan(v[0])
    assert not min_raster_filter(r, 0.5)(np.array([X0 - 10_000]), np.array([Y0]))[0]
    assert not max_raster_filter(r, 0.5)(np.array([X0 - 10_000]), np.array([Y0]))[0]


def test_disk_fraction_values_and_edge_rule():
    arr = np.zeros((200, 200))
    arr[:, 100:] = 1.0                       # right half usable
    r = Raster(arr, 0, 5000, 25.0)
    f = r.disk_fraction([2500.0, 4000.0, 100.0], [2500.0, 2500.0, 2500.0], 300.0)
    assert abs(f[0] - 0.5) < 0.05            # on the boundary: about half
    assert f[1] == pytest.approx(1.0)
    assert f[2] == 0.0                       # disk leaves the raster -> conservative 0


def test_filters_reduce_count_and_all_must_pass(rect_gdf):
    base = site_turbines(rect_gdf, azimuth_deg=0, **KW)
    cols = 400
    arr = np.ones((300, cols))
    arr[:, : cols // 2] = 0.0                # west half fails
    mask = Raster(arr, X0 - 500, Y0 + 6500, 25.0)
    filt = site_turbines(rect_gdf, azimuth_deg=0, filters=[min_raster_filter(mask, 1.0)], **KW)
    assert 0 < filt.n_turbines < base.n_turbines
    assert filt.x.min() >= X0 - 500 + (cols // 2) * 25.0 - 1e-6
    none = site_turbines(rect_gdf, azimuth_deg=0,
                         filters=[min_raster_filter(mask, 1.0), max_raster_filter(mask, 0.0)], **KW)
    assert none.n_turbines == 0
    disk = site_turbines(rect_gdf, azimuth_deg=0,
                         filters=[disk_fraction_filter(mask, 300, 0.999)], **KW)
    assert 0 < disk.n_turbines <= filt.n_turbines


def test_search_uses_filters(rect_gdf):
    arr = np.ones((300, 400))
    arr[:, :200] = 0.0
    mask = Raster(arr, X0 - 500, Y0 + 6500, 25.0)
    r = site_turbines(rect_gdf, search_step_deg=30,
                      filters=[min_raster_filter(mask, 1.0)], **KW)
    assert r.x.min() >= X0 + 4500 - 1e-6


# ---- outputs ----------------------------------------------------------------------------

def test_capacity_and_summary_carry_disclaimer(rect_gdf):
    res = site_turbines(rect_gdf, azimuth_deg=0, **KW)
    s = res.summary(mw_per_turbine=4.0)
    assert s["capacity_mw"] == pytest.approx(res.n_turbines * 4.0)
    assert s["disclaimer"] == OUTPUT_DISCLAIMER and "Not a bankable" in s["disclaimer"]
    assert s["mw_per_km2"] == pytest.approx(s["capacity_mw"] / 60.0)
    e = estimate_capacity(10, 3.0, 20.0)
    assert e["capacity_mw"] == 30.0 and e["mw_per_km2"] == 1.5 and "disclaimer" in e
    with pytest.raises(ValueError):
        estimate_capacity(1, 0)


def test_to_geodataframe_roundtrip(rect_gdf):
    res = site_turbines(rect_gdf, azimuth_deg=0, **KW)
    g = res.to_geodataframe()
    assert len(g) == res.n_turbines and g.crs == CRS.from_user_input(METRIC_EPSG)
    assert list(g.turbine_id) == list(range(1, len(g) + 1))
