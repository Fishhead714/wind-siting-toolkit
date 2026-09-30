# wind-micrositing

> **Scope and disclaimer.** This package gives a **screening-level** turbine count and
> installed-capacity estimate. It is **not** a bankable layout or energy-yield assessment:
> no wake modelling, no wind-resource or long-term correction, no noise or shadow-flicker,
> no access roads, no grid connection, no ownership or permitting checks. The text is
> embedded as `OUTPUT_DISCLAIMER` in every summary.

## Problem

Early in siting you often have a polygon of "land that might be usable" and need a fast
answer to "roughly how many turbines fit, and how many MW is that?". A regular grid with
spacing expressed in rotor diameters is the usual first cut. Where the grid is anchored
and how it is rotated changes the count, so this package searches the orientation
instead of guessing it, and applies point filters (slope, wind class, land-use mask, ...)
that you supply.

## What it does

1. Takes a developable-area polygon (for example one produced by
   `wind-siting-toolkit/maps/scripts/derive_developable_area.py`, or any polygon you have).
2. Shrinks it inward by an edge margin (default 0.5 D) so the rotor circle stays inside.
3. Lays a rectangular grid with `spacing_downwind_d x D` along one axis and
   `spacing_crosswind_d x D` across it.
4. Either uses the azimuth you give, or tries every azimuth in `[0, 180)` at your step and
   keeps the one with most turbines (ties go to the smallest azimuth, so runs are
   reproducible). Optional `offset_steps` also shifts the lattice.
5. Keeps only points that pass every filter you pass in.
6. Multiplies the count by your MW per turbine.

## Install and use

```bash
pip install -r requirements.txt        # numpy, shapely, geopandas, pyproj
pip install pytest && pytest           # synthetic-data tests
python examples/run_synthetic.py
```

```python
import geopandas as gpd
from wind_micrositing import site_turbines, max_raster_filter, Raster, slope_degrees

area = gpd.read_file("developable_area.gpkg")          # must be in a metric projected CRS
dem = Raster(array, x0, y0, res)                       # your DEM, same CRS, square pixels
result = site_turbines(
    area,
    rotor_diameter_m=130, spacing_downwind_d=5, spacing_crosswind_d=3,
    search_step_deg=10,                                # or azimuth_deg=...
    filters=[max_raster_filter(slope_degrees(dem), 15)],
)
print(result.summary(mw_per_turbine=3.4))
result.to_geodataframe().to_file("turbines.gpkg")
```

Every number above is an example. Rotor diameter, spacing multiples, azimuth search step,
slope limit and MW per turbine are all arguments; nothing is read from a hidden config.
`EXAMPLE_DEFAULTS` holds arbitrary values (5D x 3D is a common rule-of-thumb pair) and is
**not** a recommendation; the example script uses 6D x 3.5D.

## Coordinate systems

Spacing is in metres, so the area must be in a **metre-based projected CRS** (any UTM
zone or national grid is fine). Verified behaviour, covered by tests:

- Geographic input (lon/lat degrees) is **rejected** with a `ValueError` by default.
- `on_geographic="reproject"` converts to the UTM zone estimated from the area centre
  and returns **everything in that UTM CRS** (`result.crs`, the turbine coordinates and
  `to_geodataframe()`), not in the input CRS; reproject the output yourself if you need
  lon/lat back. Rasters you pass to filters must then be in that UTM CRS too.
- A projected CRS whose unit is not the metre (for example a US-feet grid) is rejected.
- A bare shapely geometry needs an explicit `crs=`; a missing CRS is an error, not a guess.

Rasters used by filters must be in the same CRS as the area; the package does no
reprojection of rasters.

## Limits

- One regular lattice. Real layouts adapt to terrain, roads and wakes; this does not.
  Density from a 5D x 3D grid is high; real wind farms are usually sparser after
  wake and constraint work.
- The spacing rule is your input. This package does not tell you what spacing is right
  for your turbine, site or regulation.
- Filters see only the turbine point. A filter that needs the rotor area or a setback
  around infrastructure has to be built into the polygon or a disk filter.
- Off-raster points fail raster filters, and disks that leave a raster count as 0 usable;
  both are conservative choices.
- The orientation search is brute force over the step you pick. Fine steps on big areas
  with slow filters take time.

## Licence

Apache-2.0 (see the repository root). Dependency licences are in `THIRD_PARTY_NOTICES.md`.
