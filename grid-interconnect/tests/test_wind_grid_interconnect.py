import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from wind_grid_interconnect import DISCLAIMER, to_metric_geoms
from wind_grid_interconnect.costsurface import (build_cost_surface, corridor_reuse_ratio,
                                                least_cost_path, tap_point)
from wind_grid_interconnect.crs import is_metric
from wind_grid_interconnect.network import HybridProblem, solve_network
from wind_grid_interconnect.roadgraph import build_road_graph, graph_report
from wind_grid_interconnect.sensitivity import (critical_multiplier_histogram, detect_regime_changes,
                                                stability_band, sweep_offroad_mult)
from wind_grid_interconnect.synthetic import SYNTHETIC_CRS, make_scenario
from wind_grid_interconnect.water import build_water_barrier, crosses_water

ROOT = Path(__file__).resolve().parents[1]


# --- CRS --------------------------------------------------------------------------------
def test_is_metric():
    assert is_metric(SYNTHETIC_CRS)
    assert is_metric("EPSG:32633")
    assert not is_metric("EPSG:4326")
    assert not is_metric(None)


def test_to_metric_geoms_rejects_geographic_target_and_missing_crs():
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError):
        to_metric_geoms(gdf, "EPSG:4326")
    with pytest.raises(ValueError):
        to_metric_geoms(gpd.GeoDataFrame(geometry=[Point(1, 1)]), "EPSG:32633")


def test_to_metric_geoms_reprojects():
    gdf = gpd.GeoDataFrame(geometry=[Point(15.0, 0.0)], crs="EPSG:4326")
    (g,) = to_metric_geoms(gdf, "EPSG:32633")   # zone 33 central meridian is 15 E
    assert abs(g.x - 500000) < 1


def test_disclaimer_mentions_scope():
    assert "Screening-level" in DISCLAIMER and "modelling assumption" in DISCLAIMER


# --- water ------------------------------------------------------------------------------
def test_barrier_and_crossing(scenario, barrier):
    assert crosses_water((5000, 8000), (7000, 8000), barrier)
    assert not crosses_water((1000, 8000), (2000, 8000), barrier)
    assert not crosses_water((1, 1), (2, 2), None)
    assert build_water_barrier() is None


# --- road graph -------------------------------------------------------------------------
def test_road_graph_connected(graph, scenario):
    rep = graph_report(graph)
    assert rep["components"] == 1
    total = sum(g.length for g in scenario.roads)
    assert sum(d["weight"] for *_, d in graph.edges(data=True)) == pytest.approx(total, rel=1e-6)


def test_road_graph_snaps_endpoints():
    a = LineString([(1000, 1000), (1100, 1000)])
    b = LineString([(1100.4, 1000), (1100, 1100)])
    G = build_road_graph([a, b], snap_m=1.0)
    assert graph_report(G)["components"] == 1


def test_road_graph_grid_straddle_does_not_merge():
    """Endpoints 0.02 m apart but on opposite sides of a rounding boundary stay separate."""
    a = LineString([(1000, 1000), (1100.49, 1000)])
    b = LineString([(1100.51, 1000), (1100, 1100)])
    assert graph_report(build_road_graph([a, b], snap_m=1.0))["components"] == 2
    # a coarser grid whose boundary is elsewhere merges them
    assert graph_report(build_road_graph([a, b], snap_m=10.0))["components"] == 1


def test_degrees_rejected_at_entry_points():
    deg_line = LineString([(10.0, 50.0), (10.1, 50.0)])
    with pytest.raises(ValueError, match="degrees"):
        build_road_graph([deg_line])
    with pytest.raises(ValueError, match="degrees"):
        build_water_barrier([deg_line], line_buffer_m=0.001)
    with pytest.raises(ValueError, match="degrees"):
        build_cost_surface((10.0, 50.0, 11.0, 51.0), cell=0.01)
    G = build_road_graph([LineString([(1000, 1000), (2000, 1000)])])
    with pytest.raises(ValueError, match="degrees"):
        HybridProblem([(10.0, 50.0), (10.1, 50.1)], G)


# --- hybrid network ---------------------------------------------------------------------
def test_mst_structure(problem):
    r = problem.solve(5.0)
    assert len(r.edges) == problem.n - 1
    assert r.n_infeasible_edges == 0
    assert r.total_len_m > 0


def test_new_construction_avoids_barrier(problem, barrier):
    for M in (1.0, 3.0, 5.0, 20.0):
        r = problem.solve(M)
        assert r.new_water_overlap_m == pytest.approx(0.0, abs=1e-6)
        for ln in r.new_lines:
            assert not ln.intersects(barrier)


def test_high_multiplier_more_reuse(problem):
    lo, hi = problem.solve(1.0), problem.solve(30.0)
    assert hi.reuse_pct >= lo.reuse_pct
    assert 0.0 <= lo.reuse_pct <= 100.0


def test_solve_network_matches_problem(problem, graph, barrier, scenario):
    a = solve_network(scenario.points, graph, barrier, offroad_mult=3.7)
    assert a.reuse_pct == pytest.approx(problem.solve(3.7).reuse_pct)


def test_critical_multiplier_is_exact_flip_point(problem):
    """For every pair with a finite M*: direct < via below M*, equal at M*, direct > via above."""
    n_checked = 0
    for i in range(problem.n):
        for j in range(i + 1, problem.n):
            if problem.direct_blocked[i, j] or not np.isfinite(problem.road_len[i, j]):
                continue
            gap = problem.euclid[i, j] - problem.snap_dist[i] - problem.snap_dist[j]
            if gap <= 0:
                continue
            mstar = problem.road_len[i, j] / gap
            d, v = problem.pair_costs(mstar)
            assert d[i, j] == pytest.approx(v[i, j], rel=1e-9)
            d, v = problem.pair_costs(mstar * 0.9)
            assert d[i, j] < v[i, j]
            d, v = problem.pair_costs(mstar * 1.1)
            assert d[i, j] > v[i, j]
            n_checked += 1
    assert n_checked == int(np.isfinite(problem.critical_multipliers()).sum()) > 10


def test_direct_vs_road_toy():
    """Two points 1000 m apart, each 100 m from a road running between them."""
    road = LineString([(0, 0), (1000, 0)])
    G = build_road_graph([road])
    pts = [(0, 100), (1000, 100)]
    p = HybridProblem(pts, G, None, k_snap=5)
    # direct = 1000*M ; via road = 200*M + 1000 -> flips at M = 1000/800 = 1.25
    assert p.solve(1.1).existing_len_m == pytest.approx(0.0)
    assert p.solve(1.5).existing_len_m == pytest.approx(1000.0)
    assert p.critical_multipliers()[0] == pytest.approx(1.25)
    d, v = p.pair_costs(1.25)
    assert d[0, 1] == pytest.approx(1250.0) and v[0, 1] == pytest.approx(1250.0)


def test_forced_snap_flagged():
    road = LineString([(0, 0), (1000, 0)])
    wall = box(-500, 30, 1500, 60)   # water between the point and the road
    G = build_road_graph([road])
    p = HybridProblem([(500, 100), (900, 200)], G, wall, k_snap=5)
    assert p.snaps[0][2] is True


# --- sensitivity ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sweep(problem):
    return sweep_offroad_mult(problem, np.geomspace(1, 30, 60))


def test_sweep_is_a_staircase(sweep):
    steps = detect_regime_changes(sweep, jump_pp=5.0)
    assert len(steps) >= 1                       # at least one threshold jump on this scenario
    vals = [r["reuse_pct"] for r in sweep]
    assert len(set(round(v, 3) for v in vals)) < len(vals) // 2   # plateaus, not a smooth curve


def test_stability_band_exact():
    rows = [{"offroad_mult": m, "reuse_pct": r} for m, r in
            zip([1, 2, 3, 4, 5, 6], [10.0, 11.0, 30.0, 31.0, 32.0, 60.0])]
    bands = stability_band(rows, tol_pp=3.0)
    assert [(b["m_lo"], b["m_hi"]) for b in bands] == [(1, 2), (3, 5), (6, 6)]
    steps = detect_regime_changes(rows, jump_pp=5.0)
    assert [(x["m_lo"], x["m_hi"], x["delta_pp"]) for x in steps] == [(2, 3, 19.0), (5, 6, 28.0)]


def test_stability_band_on_sweep_covers_all_points(sweep):
    bands = stability_band(sweep, tol_pp=3.0)
    assert bands[0]["m_lo"] == sweep[0]["offroad_mult"] and bands[-1]["m_hi"] == sweep[-1]["offroad_mult"]
    # within every band the reuse ratio spread is at most tol
    for b in bands:
        vals = [r["reuse_pct"] for r in sweep if b["m_lo"] <= r["offroad_mult"] <= b["m_hi"]]
        assert max(vals) - min(vals) <= 3.0 + 1e-9


def test_histogram_counts(problem):
    h = critical_multiplier_histogram(problem)
    assert sum(h["counts"]) == h["n_pairs"] == len(problem.critical_multipliers())


def test_ensemble_spread_is_large():
    vals = []
    for seed in range(6):
        sc = make_scenario(seed=seed)
        pr = HybridProblem(sc.points, build_road_graph(sc.roads),
                           build_water_barrier(sc.river_lines, sc.lakes))
        vals.append(pr.solve(5.0).reuse_pct)
    assert max(vals) - min(vals) > 5.0           # placement matters as much as M


# --- cost surface -----------------------------------------------------------------------
def test_least_cost_avoids_hard_cells_and_prefers_roads():
    road = LineString([(0, 500), (2000, 500)])
    wall = box(900, 0, 1100, 400)                # blocks the direct low line, leaves a gap above
    s = build_cost_surface((0, 0, 2000, 1000), cell=20, roads=[road], hard=[wall])
    out = least_cost_path(s, (10, 100), (1990, 100))
    assert out["hard_cells_crossed"] == 0
    assert out["road_cell_share"] > 0.3
    assert out["length_m"] > 1980


def test_barrier_narrower_than_cell_is_not_leaky():
    """A 24 m wide barrier on 100 m cells must still block: every touched cell is hard."""
    river = LineString([(1000, 0), (1000, 2000)]).buffer(12)   # x in [988, 1012]: no cell centre inside
    s = build_cost_surface((0, 0, 2000, 2000), cell=100, hard=[river])
    assert s.hard_mask.sum() >= 20                              # one column of 20 cells (or two)
    out = least_cost_path(s, (50, 1000), (1950, 1000))
    # the barrier spans the whole domain, so a crossing is unavoidable and must be reported
    assert out["hard_cells_crossed"] >= 1
    assert out["hard_vector_overlap_m"] == pytest.approx(24.0, abs=1.0)
    assert out["line"].intersects(river)


def test_path_around_narrow_barrier_has_no_vector_overlap():
    wall = LineString([(1000, 0), (1000, 1500)]).buffer(12)     # leaves a gap at y > 1500
    s = build_cost_surface((0, 0, 2000, 2000), cell=100, hard=[wall])
    out = least_cost_path(s, (50, 500), (1950, 500))
    assert out["hard_cells_crossed"] == 0
    assert out["hard_vector_overlap_m"] == 0.0
    assert not out["line"].intersects(wall)
    assert out["length_m"] > 1900 + 500                          # forced detour


def test_hard_crossing_reported_when_unavoidable():
    wall = box(900, 0, 1100, 1000)
    s = build_cost_surface((0, 0, 2000, 1000), cell=20, hard=[wall])
    assert least_cost_path(s, (10, 500), (1990, 500))["hard_cells_crossed"] > 0


def test_cost_surface_validation():
    with pytest.raises(ValueError):
        build_cost_surface((0, 0, 1000, 1000), cell=100, terrain_mult=np.ones((3, 3)))
    s = build_cost_surface((0, 0, 1000, 1000), cell=100)
    with pytest.raises(ValueError):
        s.rc(5000, 5000)


def test_tap_and_corridor_reuse():
    line = LineString([(0, 0), (1000, 0)])
    assert tap_point((300, 200), line) == pytest.approx((300, 0))
    route = LineString([(0, 5), (500, 5), (500, 300)])
    r = corridor_reuse_ratio(route, [line], tol_m=25)
    assert 0.5 < r < 0.7
    assert corridor_reuse_ratio(route, []) == 0.0


# --- example ----------------------------------------------------------------------------
def test_example_runs(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "run_synthetic.py"), str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT, timeout=600,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert "Screening-level" in r.stdout
    assert len(summ["sweep_seed0"]) == 60
    assert len(summ["steps_ge_5pp"]) >= 1
    assert summ["collector_example_M"]["new_water_overlap_m"] == pytest.approx(0.0, abs=1e-6)
    assert summ["collector_example_M"]["infeasible_edges"] == 0
    for name, line in summ["least_cost_lines"].items():
        # the vector overlap is only allowed where the path crossed a hard cell
        if line["hard_cells_crossed"] == 0:
            assert line["hard_vector_overlap_m"] == pytest.approx(0.0, abs=1e-6), name
        assert 0.0 <= line["corridor_reuse"] <= 1.0
    assert (tmp_path / "sweep_seed0.csv").exists()
