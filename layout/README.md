# wind-layout

> **Scope and disclaimer.** This package produces **screening / pre-feasibility level**
> turbine layouts and energy estimates. It is **not a bankable energy yield assessment**:
> it has no measured wind data or long-term correction (MCP), no full loss cascade, no
> uncertainty combination and no P50/P90. Wakes use engineering models with a flat-terrain
> flow assumption; the mountain branch returns terrain-constrained *candidate* positions,
> not wake-optimal ones. The text is embedded as `OUTPUT_DISCLAIMER` in every summary JSON.

## Problem

Early wind siting often has many candidate zones with very different terrain, and one
layout method does not fit all of them. On flat ground you can let an optimiser maximise
energy under a spacing constraint. On steep ground a flat-terrain wake model says little
about position quality, and what matters first is where a turbine can physically stand.
This package routes each zone to a suitable method, checks the result with a directional
spacing test, and tabulates per-turbine site conditions so that a reviewer can see what
the numbers rest on.

## Flow

```
zone polygon (WGS84) + DEM
        |
        v
terrain classification (relief, RIX on the zone's own pixels)
        |
        +-- flat / gently rolling ----> flat branch
        |                                 TOPFARM + PyWake, fixed turbine count,
        |                                 land + exclusion constraints, snap to safe side
        |
        +-- steep / complex ----------> mountain branch
                                          local terrain maxima on buildable, non-valley
                                          pixels, greedy pick with elliptical spacing
        |
        v
directional-ellipse spacing check (CONFLICT / NEAR / OK)
        |
        v
per-turbine site table (elevation, slope, air density, inflow angle,
effective turbulence vs. IEC class) + GPKG + summary JSON
```

Routing rule (all thresholds are example defaults, see below): RIX at or above 5 % goes
to the mountain branch; relief of 100 m or more with RIX at or below 1 % on a DEM of at
most 60 m pixel goes to the flat branch; any other relief of 100 m or more goes to the
mountain branch. RIX is the share of pixels steeper than arctan(0.3), about 16.7 degrees.
It depends strongly on DEM resolution, hence the pixel-size gate.

## Install

Python 3.11 to 3.13 (a TOPFARM constraint).

```bash
pip install -r requirements.txt
pip install pytest        # only to run the tests
python -m pytest
```

## Minimal example

`examples/run_synthetic.py` builds a synthetic ridge DEM at an arbitrary open-ocean
location, an invented wind rose and a generic turbine, and runs the whole pipeline:

```bash
PYTHONPATH=. python examples/run_synthetic.py /tmp/wind_layout_demo
```

The core call is:

```python
from wind_layout.pipeline import run_pipeline

results = run_pipeline(
    zones_gdf,                # GeoDataFrame, EPSG:4326, one row per candidate zone
    "dem.tif",                # GeoTIFF in degrees
    usable_land_fn,           # (zone_row) -> GeoDataFrame of usable land in a metric CRS
    turbine,                  # py_wake WindTurbine
    windrose_fn,              # (zone_id) -> dict of 12-sector frequency / Weibull A / k
    n_wt_fn,                  # (zone_row) -> turbine count for the flat branch
    "out/",
    turbine_factory=make_turbine,   # optional: rebuild the turbine with site air density
    turbine_rated_mw=2.4,
    shear_alpha=0.14,               # optional: convert the wind rose to hub height
)
```

## Inputs

| Input | Form |
|---|---|
| Zones | GeoDataFrame, WGS84, unique id column |
| DEM | GeoTIFF, geographic degrees, nodata respected |
| Usable land | callable returning a metric-CRS GeoDataFrame per zone (setbacks and roads are your upstream logic) |
| Wind rose | dict per zone: `sector_freq_pct`, `sector_weibull_A`, `sector_weibull_k` (12 sectors), optional `height_used_m` |
| Turbine | py_wake `WindTurbine` (and optionally a factory taking air density) |
| Optional | wind-speed raster on the DEM grid, shear exponent scalar or raster, exclusion layers |

## Outputs

Per zone, in the output directory:

- `<zone>_turbines.gpkg`: layers `turbines_utm`, `turbines_wgs84`, `spacing_ellipses`
- `<zone>_site_suitability.csv`: one row per turbine (elevation, slope, air density,
  Weibull mean, inflow angle per main direction and energy-weighted, effective
  turbulence versus the IEC class, net energy)
- `<zone>_pipeline_summary.json`: routing reason, branch metrics, spacing verdict,
  wind rose, all run parameters and a fingerprint, and the disclaimer

Re-running is idempotent: a zone with an existing summary is skipped when the parameter
fingerprint matches, and raises `RuntimeError` when it does not, so old results are never
returned silently under new parameters.

## Modules

| Module | Role |
|---|---|
| `pipeline` | orchestration and site table |
| `terrainclassify` | relief, RIX, three-tier routing, DEM resolution gate |
| `terrainlayout` | slope, TPI valley mask, mountain-branch candidate selection |
| `layoutoptimize` | TOPFARM/PyWake layout optimisation, boundary snapping, wake-model sensitivity |
| `spacingcheck` | ellipse spacing check, energy-weighted dominant direction |
| `siteconditions` | air density, IEC 61400-1 turbulence screening, inflow angle, shear |
| `rasterutils` | raster sampling and masked statistics |

## Parameters are examples, not recommendations

Every numeric default (60 m DEM gate, 5 % / 1 % RIX thresholds, 17 degree buildability
slope, 5D / 3D spacing, wake constants, 200 to 350 W/m2 specific-power window, and so on)
is an **example default** that the author has not calibrated for your terrain. The wake
decay constant in particular (`WAKE_CONSTANTS["turbopark_A_onshore"] = 0.064`) comes from a
single reported tuning and is unverified; calibrate it or use `wake_model_sensitivity`
and report a range. Sources and their verification status are listed in
`THIRD_PARTY_NOTICES.md`.

## Limitations

- Flat-terrain wake model: no terrain speed-up, lee separation or terrain-induced
  turbulence. Complex-terrain results need microscale (CFD-type) flow modelling before
  any decision.
- No measured wind data, no MCP, no P50/P90, no loss cascade. The wind rose is an input
  you must justify.
- The mountain branch ranks candidates by a raster you supply (wind speed); without one it
  falls back to elevation, which is not energy density.
- The IEC turbulence check omits the 1.28 sigma_sigma_hat term of eq. (35), so it is
  looser than the standard. Constants follow Edition 3 as quoted and were not checked
  against Edition 4.
- Inflow angle is estimated from DEM slope along the wind direction, not from flow modelling.
- Flat-branch output has no hard slope or valley constraint; violations are counted and
  warned about.
- Not covered: noise, shadow flicker, ice throw, aviation and radar, extreme wind,
  collection-line and road cost.

## License

Apache-2.0. See `THIRD_PARTY_NOTICES.md` for dependencies and references.
