# wind-siting-toolkit

Screening-level tools for onshore wind siting studies:

| Directory | What it does | Licence |
|---|---|---|
| [`noise-flicker/`](noise-flicker/) | Wind turbine noise (ISO 9613-2) and shadow flicker (astronomical maximum and expected case) at receptors and on a grid; exports contour lines and bands to GeoPackage. Pure Python. | Apache-2.0 |
| [`maps/`](maps/) | Headless map renderer built on PyQGIS for siting figures: constraint maps, developable-area maps, noise and shadow-flicker maps, with legend, scale bar, north arrow, locator inset and credit line. Includes a script that derives the developable area from a site boundary and exclusion layers. | GPL-3.0-or-later |

The two parts work on their own. Together they go from a turbine layout to finished figures:
`noise-flicker` writes contours to a GeoPackage, and `maps` draws them.

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

## Licence

Each directory is licensed separately: `noise-flicker/` under Apache-2.0 and `maps/` under
GPL-3.0-or-later (the renderer imports QGIS). See the root `LICENSE` and each directory's `LICENSE`.
Third-party credits (colour schemes, data sources) are in each directory's `THIRD_PARTY_NOTICES.md`.
