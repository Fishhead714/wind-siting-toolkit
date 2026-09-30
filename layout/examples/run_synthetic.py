"""Minimal end-to-end run on a synthetic ridge DEM (no external data needed).

    python examples/run_synthetic.py [output_dir]

The terrain, wind rose and turbine below are invented for illustration. The
location is an arbitrary point over open ocean. The result is a screening-level
demonstration, not an energy assessment.
"""
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from rasterio.transform import from_origin
from shapely.geometry import box

from wind_layout.pipeline import OUTPUT_DISCLAIMER, run_pipeline

LON0, LAT_TOP, RES, N = -140.0, -19.8, 0.0004, 250     # ~44 m pixels, ~11 km square
UTM = 32707                                            # metric CRS covering that longitude

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="wind_layout_"))
out.mkdir(parents=True, exist_ok=True)

# 1. synthetic terrain: crossed sinusoidal ridges, 150 m amplitude, 2.5 km wavelength
yy, xx = np.mgrid[0:N, 0:N].astype(float)
wl = 2500.0 / (RES * 111320.0)
elev = 300 + 150 * np.sin(2 * np.pi * xx / wl) * np.sin(2 * np.pi * yy / (0.8 * wl))
dem = out / "synthetic_dem.tif"
with rasterio.open(dem, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(LON0, LAT_TOP, RES, RES),
                   nodata=-9999.0) as dst:
    dst.write(elev.astype("float32"), 1)

# 2. one candidate zone (WGS84) and how to obtain its usable land (metric CRS)
zone = box(LON0 + 0.2 * N * RES, LAT_TOP - 0.8 * N * RES, LON0 + 0.8 * N * RES, LAT_TOP - 0.2 * N * RES)
zones = gpd.GeoDataFrame({"id": ["demo"]}, geometry=[zone], crs=4326)
usable_land = lambda row: gpd.GeoDataFrame(geometry=gpd.GeoSeries([row.geometry], crs=4326).to_crs(UTM))

# 3. invented wind rose (12 sectors) and turbine
freq = np.array([4, 5, 6, 8, 10, 14, 16, 12, 9, 7, 5, 4], float)
windrose = {"sector_freq_pct": list(freq / freq.sum() * 100),
            "sector_weibull_A": [7.0, 7.2, 7.5, 8.0, 8.5, 9.0, 9.2, 8.8, 8.2, 7.8, 7.4, 7.1],
            "sector_weibull_k": [2.0] * 12, "dominant_sector_deg": 180.0, "height_used_m": 100.0}
make_turbine = lambda rho=1.225: GenericWindTurbine(
    name="Synthetic2p4MW", diameter=100, hub_height=80, power_norm=2400,
    turbulence_intensity=0.10, air_density=rho)

results = run_pipeline(zones, str(dem), usable_land, make_turbine(), lambda zone_id: windrose,
                       lambda row: 4, out / "results", turbine_factory=make_turbine,
                       turbine_rated_mw=2.4, shear_alpha=0.14)

r = results[0]
print(f"\nroute={r['route']}  turbines={r['n_turbines']}  spacing conflicts={r['n_spacing_conflict']}")
print("site table:", r["site_suitability_csv"])
print("\nDISCLAIMER:", OUTPUT_DISCLAIMER["not_a_bankable_energy_assessment"])
