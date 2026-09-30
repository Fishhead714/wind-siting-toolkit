#!/usr/bin/env python3
"""Generate SYNTHETIC input data for the render_map.py demo (geopandas only, no QGIS).

Nothing here corresponds to a real site or a real calculation. The layout sits at an
arbitrary offset in EPSG:32625 (UTM zone 25N) that falls in open sea. The "noise contours"
and "shadow-flicker" layers are plain geometric buffers around the turbines; they only
exist to exercise the map styles and carry no physical meaning. Output is deterministic
(fixed seed, fixed geometry), so re-running rewrites identical files.

Files written next to this script (all EPSG:32625):
  demo_site.gpkg
    boundary         site boundary polygon (name)
    wtg              12 turbine points (wtg_id)
    buildings        building footprints (bid, kind)
    receptors        building centroids (receptor_id, noise_db, flicker_h, exceeds 0/1)
  demo_contours.gpkg
    noise_contour    closed contour lines (level_db = 35 / 40 / 45, metric)
    flicker_isoline  isolines (hours_per_year = 8 / 30 / 100, case)
    flicker_band     hour bands (hours_min, hours_max, case)
  demo_exclusions.gpkg
    dwelling_setback buildings buffered by 500 m (example value)
    road_setback     a synthetic road buffered by 150 m (example value)
"""
import random
from pathlib import Path

import geopandas as gpd
from shapely import affinity
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
CRS = "EPSG:32625"
E0, N0 = 280_000.0, 1_000_000.0        # arbitrary synthetic origin (open sea)
rng = random.Random(20260929)

# Example criteria used only to populate the demo attributes.
NOISE_LEVELS = {45: 350.0, 40: 550.0, 35: 850.0}      # level_db -> buffer distance (m)
NOISE_CRITERION_DB = 40
FLICKER_LEVELS = {100: 300.0, 30: 750.0, 8: 1000.0}   # hours/year -> ellipse radius (m)
FLICKER_CRITERION_H = 30


def write(path, layers):
    if path.exists():
        path.unlink()                                 # idempotent: rewrite the whole file
    for name, gdf in layers:
        gdf.to_file(path, layer=name, driver="GPKG")


# 12 turbines: 3 staggered rows of 4, 520 m in-row, 900 m between rows
wtg = []
for r in range(3):
    for c in range(4):
        x = E0 + c * 520 + (260 if r % 2 else 0)
        y = N0 + r * 900
        wtg.append({"wtg_id": f"T{r * 4 + c + 1:02d}", "geometry": Point(x, y)})
wtg = gpd.GeoDataFrame(wtg, crs=CRS)
turbines = unary_union(list(wtg.geometry))

# three synthetic hamlets plus a few scattered farmsteads
clusters = [((E0 - 1100, N0 + 700), 14, 90),
            ((E0 + 2900, N0 + 1300), 18, 110),
            ((E0 + 900, N0 - 1000), 10, 70)]
pts = []
for (cx, cy), n, spread in clusters:
    for _ in range(n):
        pts.append((cx + rng.gauss(0, spread), cy + rng.gauss(0, spread), "house"))
for x, y in [(E0 + 1300, N0 + 450), (E0 - 600, N0 + 1900), (E0 + 2300, N0 + 2500),
             (E0 + 600, N0 + 2700), (E0 + 2500, N0 - 500)]:
    pts.append((x, y, "farmstead"))

rows = []
for i, (x, y, kind) in enumerate(pts, 1):
    w, h = rng.uniform(8, 16), rng.uniform(7, 12)
    poly = affinity.rotate(box(x - w / 2, y - h / 2, x + w / 2, y + h / 2), rng.uniform(0, 90), origin="centroid")
    rows.append({"bid": f"B{i:03d}", "kind": kind, "geometry": poly})
bld = gpd.GeoDataFrame(rows, crs=CRS)

boundary = gpd.GeoDataFrame({"name": ["synthetic site"]},
                            geometry=[turbines.convex_hull.buffer(450, join_style=2)], crs=CRS)

# "noise": concentric buffers of the turbine union; higher level = closer to the turbines
noise_polys = {lvl: turbines.buffer(d) for lvl, d in NOISE_LEVELS.items()}
noise = gpd.GeoDataFrame(
    [{"level_db": lvl, "metric": "LAeq night (synthetic)", "geometry": poly.boundary}
     for lvl, poly in sorted(noise_polys.items())], crs=CRS)

# "shadow flicker": union of east-west stretched ellipses around each turbine
def flicker_zone(radius):
    return unary_union([affinity.scale(p.buffer(radius), xfact=1.5, yfact=0.7) for p in wtg.geometry])


fz = {h: flicker_zone(r) for h, r in FLICKER_LEVELS.items()}
case = "astronomical_max"
iso = gpd.GeoDataFrame(
    [{"hours_per_year": h, "case": case, "geometry": g.boundary} for h, g in sorted(fz.items())], crs=CRS)
bands = gpd.GeoDataFrame(
    [{"hours_min": 8, "hours_max": 30, "case": case, "geometry": fz[8].difference(fz[30])},
     {"hours_min": 30, "hours_max": 100, "case": case, "geometry": fz[30].difference(fz[100])},
     {"hours_min": 100, "hours_max": None, "case": case, "geometry": fz[100]}], crs=CRS)


def level_at(pt, polys, outside):
    """Highest class whose zone contains the point (classes are nested)."""
    hit = [lvl for lvl, poly in polys.items() if poly.contains(pt)]
    return max(hit) if hit else outside


rec_rows = []
for b in bld.itertuples():
    c = b.geometry.centroid
    ndb = level_at(c, noise_polys, 30)
    fh = level_at(c, fz, 0)
    rec_rows.append({"receptor_id": f"R{b.bid[1:]}", "noise_db": ndb, "flicker_h": fh,
                     "exceeds": int(ndb >= NOISE_CRITERION_DB or fh >= FLICKER_CRITERION_H), "geometry": c})
rec = gpd.GeoDataFrame(rec_rows, crs=CRS)

setback = gpd.GeoDataFrame({"rule": ["dwelling setback 500 m (example)"]},
                           geometry=[bld.buffer(500).union_all()], crs=CRS)
road = gpd.GeoDataFrame({"rule": ["road setback 150 m (example)"]},
                        geometry=[LineString([(E0 - 1500, N0 + 1350), (E0 + 3500, N0 + 1150)]).buffer(150)],
                        crs=CRS)

write(HERE / "demo_site.gpkg", [("boundary", boundary), ("wtg", wtg), ("buildings", bld), ("receptors", rec)])
write(HERE / "demo_contours.gpkg", [("noise_contour", noise), ("flicker_isoline", iso), ("flicker_band", bands)])
write(HERE / "demo_exclusions.gpkg", [("dwelling_setback", setback), ("road_setback", road)])
print(f"wrote demo data to {HERE}: {len(wtg)} turbines, {len(bld)} buildings, "
      f"{int(rec.exceeds.sum())} receptors flagged")
