"""End-to-end demonstration on invented data (no external data needed).

    python examples/run_synthetic.py [output_dir]

Everything below is synthetic: the road network, river, lake, terrain field and
turbine positions are generated in ``wind_grid_interconnect.synthetic``. The
numbers illustrate the *behaviour* of the method (in particular the step-like
dependence of the reuse ratio on the off-road multiplier); they say nothing
about any real site. Screening level only.
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from wind_grid_interconnect import DISCLAIMER
from wind_grid_interconnect.costsurface import build_cost_surface, corridor_reuse_ratio, least_cost_path, tap_point
from wind_grid_interconnect.network import HybridProblem
from wind_grid_interconnect.roadgraph import build_road_graph, graph_report
from wind_grid_interconnect.sensitivity import (critical_multiplier_histogram, detect_regime_changes,
                                                stability_band, sweep_offroad_mult)
from wind_grid_interconnect.synthetic import make_scenario
from wind_grid_interconnect.water import build_water_barrier

WATER_BUFFER_M = 12.0                    # example default channel-width proxy
FINE_GRID = np.round(np.geomspace(1.0, 30.0, 60), 3)
ENSEMBLE_SEEDS = range(20)
ENSEMBLE_MULTS = (2.5, 4.5, 7.0, 11.0)
DEMO_MULT = 3.7                          # arbitrary example value, not a recommendation

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="wgi_"))
out.mkdir(parents=True, exist_ok=True)


def problem_for(seed):
    sc = make_scenario(seed=seed)
    barrier = build_water_barrier(sc.river_lines, sc.lakes, WATER_BUFFER_M)
    graph = build_road_graph(sc.roads)
    return sc, barrier, graph, HybridProblem(sc.points, graph, barrier)


# 1. sensitivity sweep on one scenario ---------------------------------------------------
sc, barrier, graph, prob = problem_for(0)
print("road graph:", graph_report(graph))
rows = sweep_offroad_mult(prob, FINE_GRID)
with open(out / "sweep_seed0.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\nseed 0: reuse ratio vs off-road multiplier (every 6th point)")
for r in rows[::6]:
    print(f"  M={r['offroad_mult']:7.2f}  reuse={r['reuse_pct']:5.1f}%  total={r['total_km']:6.2f} km")
steps = detect_regime_changes(rows, jump_pp=5.0)
print("steps >= 5 pp:", [(s['m_lo'], s['m_hi'], round(s['delta_pp'], 1)) for s in steps])
print("plateaus (within 3 pp):", [(b['m_lo'], b['m_hi']) for b in stability_band(rows, 3.0)])
print("pair switch multipliers:", critical_multiplier_histogram(prob))

# 2. how much does the answer depend on the (invented) geometry? -------------------------
ens = {m: [] for m in ENSEMBLE_MULTS}
for seed in ENSEMBLE_SEEDS:
    _, _, _, p = problem_for(seed)
    for m in ENSEMBLE_MULTS:
        ens[m].append(p.solve(m).reuse_pct)
print(f"\n{len(list(ENSEMBLE_SEEDS))} random turbine placements, reuse % at fixed M:")
for m, v in ens.items():
    print(f"  M={m:>4}: mean {np.mean(v):5.1f}  min {np.min(v):5.1f}  max {np.max(v):5.1f}")

# 3. collector network at an example default multiplier ---------------------------------
res = prob.solve(DEMO_MULT)
print(f"\ncollector network at M={DEMO_MULT} (arbitrary example value):", {k: round(v, 3) if isinstance(v, float) else v
                                                        for k, v in res.summary().items()})

# 4. least-cost lines on the cost surface ------------------------------------------------
surface = build_cost_surface(sc.bounds, cell=sc.cell, roads=sc.roads, hard=[barrier],
                             terrain_mult=sc.terrain_mult)
tap = tap_point(tuple(sc.substation), sc.transmission_line)
west = sc.turbines[np.argmin(np.hypot(*(sc.turbines - np.array([3000.0, 7000.0])).T))]
lines = {}
for name, a, b in (("substation_to_line_tap", tuple(sc.substation), tap),
                   ("west_cluster_to_substation", tuple(west), tuple(sc.substation))):
    lc = least_cost_path(surface, a, b)
    lc["corridor_reuse"] = corridor_reuse_ratio(lc["line"], sc.roads, 2 * sc.cell)
    lines[name] = {k: v for k, v in lc.items() if k != "line"}
    print(f"\nleast-cost line {name}: length {lc['length_m']/1000:.2f} km, "
          f"road-cell share {lc['road_cell_share']:.2f}, hard cells crossed {lc['hard_cells_crossed']}, "
          f"overlap with vector barrier {lc['hard_vector_overlap_m']:.1f} m, "
          f"corridor reuse {lc['corridor_reuse']:.2f}")

json.dump({"disclaimer": DISCLAIMER, "sweep_seed0": rows, "steps_ge_5pp": steps,
           "ensemble_reuse_pct": {str(m): v for m, v in ens.items()},
           "collector_example_M": res.summary(), "least_cost_lines": lines},
          open(out / "summary.json", "w"), indent=2)
print("\nwrote", out)
print(DISCLAIMER)
