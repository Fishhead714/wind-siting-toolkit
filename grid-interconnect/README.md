# wind-grid-interconnect

Screening-level tools for **collector-road and grid-interconnection routing** around a wind
farm cluster, using open geospatial libraries only (numpy, scipy, shapely, geopandas,
networkx, scikit-image).

> **Screening-level result only.** Not a route survey or an engineering design.
> Right-of-way, permits, land tenure, crossings of third-party assets and geotechnical
> conditions are not considered. The off-road cost multiplier `M` is a modelling
> assumption, not a measured constant.

License: Apache-2.0. All example data in this repository is synthetic (invented geometry in
a made-up local CRS); no real site, project or result appears anywhere in the package.

## What it does

| Module | Purpose |
|---|---|
| `roadgraph` | Build a routable graph from road centre-lines (endpoint snapping; no noding of mid-segment crossings, see Limitations). |
| `water` | Build a water barrier (buffered waterway lines + water polygons) and test segments against it. |
| `network` | Hybrid network over turbines + collection point: each pair is linked either by a straight new-build segment (cost = length x `M`, forbidden across water) or through the existing road graph (cost = road length + snap connectors x `M`). A minimum spanning tree gives the network and the **existing-road reuse ratio** (length on existing roads / total length). |
| `costsurface` | Raster cost surface (base cost, terrain multiplier, soft multipliers, cheap road cells, hard cells), least-cost path via scikit-image, tap point on an existing line, corridor reuse ratio. |
| `sensitivity` | Sweep `M`, detect regime changes, find stability bands, histogram the per-pair switch multipliers. |
| `crs` | Metric CRS handling: `to_metric_geoms` rejects non-metric target CRSs; the other entry points reject coordinates that all lie within +-360 (a heuristic for degrees). |
| `synthetic` | Deterministic synthetic scenario used by the tests and example. |

## Install and run

```bash
pip install -r requirements.txt
pip install pytest && pytest
python examples/run_synthetic.py out/
```

## Minimal example

```python
import numpy as np
from wind_grid_interconnect.synthetic import make_scenario
from wind_grid_interconnect.roadgraph import build_road_graph
from wind_grid_interconnect.water import build_water_barrier
from wind_grid_interconnect.network import HybridProblem
from wind_grid_interconnect.sensitivity import sweep_offroad_mult, detect_regime_changes

sc = make_scenario(seed=0)                      # replace with your own metric-CRS data
graph = build_road_graph(sc.roads)
barrier = build_water_barrier(sc.river_lines, sc.lakes)
problem = HybridProblem(sc.points, graph, barrier)   # last point = collection point

rows = sweep_offroad_mult(problem, np.geomspace(1, 30, 60))
print(detect_regime_changes(rows, jump_pp=5))
print(problem.solve(3.7).summary())             # 3.7 is an arbitrary example value
```

For your own data, load layers with geopandas and convert with
`to_metric_geoms(gdf, metric_crs)`, where `metric_crs` is a projected CRS in metres that suits
your area (for example a UTM zone). Every length, buffer and raster cell is in metres.

## Sensitivity to the off-road multiplier `M`

`M` is the relative cost of one metre of new construction versus one metre of existing road.
It is an input you must choose (`solve_network` has no default for it), and the reuse ratio
can depend on it in a step-like way.

**Mechanism.** Each node pair has a critical multiplier `M* = R / (D - s_i - s_j)` (`R` road
path length, `D` straight distance, `s` snap connector lengths). Below `M*` the pair is linked
directly, above it through the roads. The spanning tree changes only when a pair crosses its
`M*`, so the reuse ratio is piecewise constant in `M` and jumps when one or more pairs flip.

**Behaviour of the bundled toy (seed 0; 16 turbines + 1 collection point; 60 log-spaced
values of `M` from 1 to 30; reproduce with `examples/run_synthetic.py`).** This describes the
toy geometry only and says nothing about any real layout:

| `M` range | reuse ratio |
|---|---|
| 1.0 - 1.78 | 14.8 % |
| 1.885 - 2.99 | 27.1 % |
| 3.17 - 3.77 | 35.4 % |
| 3.99 - 12.6 | 52.0 % |
| >= 15.9 | 52.7 % |

Steps of +12, +8 and +17 percentage points fall inside brackets of `M` narrower than 0.25.
Where the steps fall depends on the geometry: over 20 random turbine placements on the same
roads and water, the ratio at `M = 4.5` had mean 55 %, min 45 %, max 63 %, and at `M = 2.5`
mean 45 %, min 27 %, max 62 %. A step location found on one layout therefore says nothing
about another.

**Working with your own data.** `sweep_offroad_mult` solves for a list of `M` values cheaply
(everything independent of `M` is precomputed in `HybridProblem`); `detect_regime_changes`
lists brackets with large jumps, `stability_band` lists runs of `M` with a nearly constant
ratio, and `critical_multiplier_histogram` shows in which ranges of `M` pairs can flip.
A ratio is easiest to interpret when quoted with the range of `M` it was computed over.
The choice of that range is a modelling decision that depends on your own cost assumptions;
the package does not supply one. The metric is length-based; it says nothing about cost,
upgrade needs, land access or road condition.

## Limitations

- Screening only (see the disclaimer above). Nothing here replaces a route survey.
- Road graph: vertices are rounded to a `snap_m` grid, so vertices closer than `snap_m` but on
  opposite sides of a cell boundary do not merge. Roads that cross without a shared vertex are
  not connected unless `node_intersections=True`, which makes every crossing a junction. Check
  `graph_report` for the number of components.
- Water barrier: it constrains new-build links and node connectors only. Segments of the
  input road graph are not tested against it (modelling assumption). New-build links that
  cross water are treated as infeasible; if no feasible link exists the result reports
  `infeasible_edges`.
- Snapping: a node snaps to the nearest road vertex whose connector avoids water; if none
  does, it uses the nearest vertex and the snap is flagged `forced`.
- Cost surface: grid quantised (8-connected); expect jagged paths and a length error of a few
  percent. Hard geometries are rasterised all-touched (any cell touching them is hard), and
  `least_cost_path` also reports `hard_vector_overlap_m`, the length of the path inside the
  unrasterised geometry (diagonal steps can still clip a corner). `cell` has no default;
  `road_cost`, `hard_cost` and `road_buffer_m` are example values with no calibration.
- Reuse is measured on lengths after deduplicating overlaps, not on cost.
- Water buffers (`line_buffer_m`, example default 12 m) are channel-width proxies.

## Tests

`pytest` runs 28 synthetic-data tests: CRS and degree-coordinate rejection, barrier logic, graph
construction (including a grid-straddle case), MST invariants, no new build across water, the
exact flip point `M*` for every eligible pair, staircase and ensemble-spread checks, a barrier
narrower than a cell, cost-surface behaviour and the example script.
