#!/usr/bin/env python3
"""Generate a SYNTHETIC wind-farm layout for the noise/flicker engine demo.

Nothing here corresponds to a real site. The layout sits at an arbitrary
offset in EPSG:32625 (UTM 25N) chosen to fall in the open Atlantic (about 9 N, 35 W);
the latitude only matters for the sun-position model. Output is deterministic
(fixed seed) so re-running produces identical files.

Usage: python make_demo_data.py [output.gpkg]   (default: demo_site.gpkg next to this file)

Layers written (all EPSG:32625):
  wtg        12 turbine points   (wtg_id)
  buildings  synthetic building footprints (bid, kind)
  receptors  building centroids  (bid)  -> used as noise/flicker receptors
  boundary   site boundary polygon
"""
from pathlib import Path

import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from shapely import affinity

HERE = Path(__file__).resolve().parent
CRS = "EPSG:32625"
E0, N0 = 280_000.0, 1_000_000.0          # arbitrary synthetic origin (open sea)
rng = np.random.default_rng(20260929)

# 12 turbines: 3 staggered rows x 4, 520 m in-row, 900 m between rows
wtg = []
for r in range(3):
    for c in range(4):
        x = E0 + c * 520 + (260 if r % 2 else 0)
        y = N0 + r * 900
        wtg.append({"wtg_id": f"T{r * 4 + c + 1:02d}", "geometry": Point(x, y)})
wtg = gpd.GeoDataFrame(wtg, crs=CRS)

# three synthetic hamlets + a few scattered farmsteads
clusters = [((E0 - 1100, N0 + 700), 14, 90),    # west of the array
            ((E0 + 2900, N0 + 1300), 18, 110),  # east of the array
            ((E0 + 900, N0 - 1000), 10, 70)]    # below the array
b = []
for (cx, cy), n, spread in clusters:
    for _ in range(n):
        x, y = cx + rng.normal(0, spread), cy + rng.normal(0, spread)
        b.append((x, y, "house"))
for x, y in [(E0 + 1300, N0 + 450), (E0 - 600, N0 + 1900), (E0 + 2300, N0 + 2500),
             (E0 + 600, N0 + 2700), (E0 + 2500, N0 - 500)]:
    b.append((x, y, "farmstead"))

rows = []
for i, (x, y, kind) in enumerate(b, 1):
    w, h = rng.uniform(8, 16), rng.uniform(7, 12)
    poly = affinity.rotate(box(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
                           rng.uniform(0, 90), origin="centroid")
    rows.append({"bid": f"B{i:03d}", "kind": kind, "geometry": poly})
bld = gpd.GeoDataFrame(rows, crs=CRS)
rec = bld.copy()
rec["geometry"] = bld.geometry.centroid
rec = rec[["bid", "geometry"]]

boundary = gpd.GeoDataFrame(
    {"name": ["synthetic site"]},
    geometry=[wtg.union_all().convex_hull.buffer(450, join_style=2)], crs=CRS)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "demo_site.gpkg"
if out.exists():
    out.unlink()                                 # idempotent: rewrite whole file
for name, g in [("wtg", wtg), ("buildings", bld), ("receptors", rec), ("boundary", boundary)]:
    g.to_file(out, layer=name, driver="GPKG")
print(f"wrote {out}: {len(wtg)} turbines, {len(bld)} buildings")
