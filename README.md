# wind-siting-toolkit

Screening-level tools for onshore wind siting studies:

| Directory | What it does | Licence |
|---|---|---|
| [`noise-flicker/`](noise-flicker/) | Wind turbine noise (ISO 9613-2) and shadow flicker (astronomical maximum and expected case) at receptors and on a grid; exports contour lines and bands to GeoPackage. Pure Python. | Apache-2.0 |
| [`layout/`](layout/) | Turbine layout for screening studies: terrain classification (relief and RIX) that routes a zone to an optimiser, layout optimisation over PyWake/TOPFARM, slope- and valley-aware layout for complex terrain, directional spacing check, and per-turbine site-suitability inputs (air density, effective turbulence, inflow angle). Not bankable energy assessment. | Apache-2.0 |
| [`micrositing/`](micrositing/) | Oriented-grid turbine siting inside a developable area: searches grid orientation and lattice offset, applies spacing, slope and usable-land filters, and reports capacity. Checks the CRS is metric. Screening only. | Apache-2.0 |
| [`grid-interconnect/`](grid-interconnect/) | Screening-level interconnection routing: road-network graph, least-cost path over a cost surface, water-body avoidance, existing-corridor reuse ratio, and a sweep of the off-road cost multiplier that shows how sharply the reuse ratio depends on it. Not a route survey. | Apache-2.0 |
| [`turbine-suitability/`](turbine-suitability/) | Turbine-class suitability screening from site wind, turbulence, extreme-wind and temperature inputs, with the class table as an overridable parameter set. Standard library only. Not an energy yield calculation. | Apache-2.0 |
| [`energy-uncertainty/`](energy-uncertainty/) | Loss chain and uncertainty combination (RSS or correlated) to P50/P75/P90/P99 exceedance under a normal assumption, one-year and N-year horizons. Not a bankable energy assessment. | Apache-2.0 |
| [`maps/`](maps/) | Headless map renderer built on PyQGIS for siting figures: constraint maps, developable-area maps, noise and shadow-flicker maps, with legend, scale bar, north arrow, locator inset and credit line. Includes a script that derives the developable area from a site boundary and exclusion layers. | GPL-3.0-or-later |

The parts work on their own. In a full study, `layout` produces turbine positions, `noise-flicker`
writes contours to a GeoPackage, and `maps` draws them. The demo below covers the last two steps.

Nothing site-specific is built in. Turbine parameters, noise spectra and assessment criteria are
inputs; the files under `noise-flicker/examples/` are examples from public sources (an open reference
turbine and international guidance) and must be replaced with values for your turbine and
jurisdiction.

## End-to-end demo (synthetic data)

```bash
PY_GEO=python3 PY_QGIS=python3 bash demo/run_all.sh
```

- `PY_GEO`: a Python with numpy, scipy, shapely, pyproj, geopandas, pyogrio and contourpy.
- `PY_QGIS`: a Python that can `import qgis` (on Windows, the OSGeo4W `python-qgis` launcher).

The demo builds a synthetic 12-turbine layout in open sea, computes noise and shadow-flicker contours
with `noise-flicker`, and renders two maps with `maps` into `demo/out/`. It takes a few minutes.

See [`noise-flicker/README.md`](noise-flicker/README.md) and [`maps/README.md`](maps/README.md) for
details, inputs and limitations.

## Scope and limitations

These are screening tools. Results depend on the inputs you supply and on the simplifications
documented in each directory (for example flat terrain and downwind propagation). They are not a
substitute for a site-specific assessment by a qualified consultant or for the procedure required
by your permitting authority.

## Background

These tools started as scripts for real onshore wind siting studies: checking noise and flicker at
houses, drawing constraint maps for review meetings, testing layouts, estimating how much of a cable
route could follow existing roads. Each one was rebuilt as a generic engine: no project data, no
site-specific rules, example defaults labelled as such. Most of the code was written with AI agents
and then reviewed, tested on synthetic data, and read line by line before release.
The author is a wind-energy engineer by trade; see the [profile](https://github.com/Fishhead714).

## Licence

Each directory is licensed separately: `maps/` is GPL-3.0-or-later and every other directory is Apache-2.0
(`maps/` imports QGIS, hence the GPL). See the root `LICENSE` and each directory's `LICENSE`.
Third-party credits (colour schemes, data sources) are in each directory's `THIRD_PARTY_NOTICES.md`.
