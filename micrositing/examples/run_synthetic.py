"""Run the siting on a synthetic polygon. No real data, no network.

    python examples/run_synthetic.py

Coordinates are round numbers in UTM zone 1 (EPSG:32601), whose central
meridian lies in open ocean. All parameter values are EXAMPLE values.
"""
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wind_micrositing import (EXAMPLE_DEFAULTS as EX, Raster, disk_fraction_filter,  # noqa: E402
                              max_raster_filter, site_turbines, slope_degrees)

CRS = 32601
X0, Y0 = 500_000.0, 1_000_000.0

# Developable area: an L-shaped polygon with a rectangular exclusion hole.
outer = [(X0, Y0), (X0 + 12_000, Y0), (X0 + 12_000, Y0 + 5_000),
         (X0 + 6_000, Y0 + 5_000), (X0 + 6_000, Y0 + 9_000), (X0, Y0 + 9_000)]
hole = box(X0 + 2_000, Y0 + 2_000, X0 + 3_500, Y0 + 3_500)
area = Polygon(outer, [hole.exterior.coords])

# Synthetic DEM (a smooth ridge) and a synthetic "usable land" mask, 30 m pixels.
res, nrow, ncol = 30.0, 360, 480
rr, cc = np.mgrid[0:nrow, 0:ncol]
dem = Raster(150 * np.exp(-((cc - 300) / 60.0) ** 2) + 0.02 * rr, X0 - 300, Y0 + 9_500, res)
slope = slope_degrees(dem)
usable = Raster((np.random.default_rng(1).random((nrow, ncol)) > 0.05).astype(float),
                X0 - 300, Y0 + 9_500, res)

result = site_turbines(
    area, crs=CRS,
    rotor_diameter_m=EX["rotor_diameter_m"],
    spacing_downwind_d=6.0,                                  # example value
    spacing_crosswind_d=3.5,                                 # example value
    search_step_deg=EX["search_step_deg"],
    filters=[max_raster_filter(slope, 15.0),                 # example slope limit
             disk_fraction_filter(usable, 300.0, 0.6)],      # example usable-land rule
)
summary = result.summary(EX["mw_per_turbine"])
print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2))
print(f"turbines: {result.n_turbines}, best orientation {result.azimuth_deg:.0f} deg, "
      f"{len(result.search)} orientations tried")
assert result.n_turbines > 0
