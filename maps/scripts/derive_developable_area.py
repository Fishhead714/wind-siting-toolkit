#!/usr/bin/env python3
"""Derive the developable area of a site: site boundary minus the union of all exclusion layers.

Produces a single polygon layer plus area statistics. The approach follows the common
two-step presentation of constraint mapping (first all constraints overlaid, then a single
flat "potential developable area" map) and borrows the idea of constraint exclusivity from
land-eligibility analysis (Ryberg, Robinius & Stolten, "Evaluating Land Eligibility
Constraints of Renewable Energy Sources in Europe", Energies, 2018): how much area is
excluded by one constraint alone and by no other. Doing the union on the data side
keeps rendering simple and makes the areas directly measurable; render_map.py draws the
result as an ordinary polygon layer (categories `developable_area` and `exclusion_overlap`).

This is data preparation, not rendering. It never modifies its inputs.

Usage (needs geopandas >= 0.14 and shapely >= 2; no QGIS):
    derive_developable_area.py \\
        --boundary boundary.gpkg:site_boundary \\
        --exclude constraints.gpkg:dwelling_setback:"Dwelling setback" \\
        --exclude constraints.gpkg:road_setback:"Road setback" \\
        --out developable.gpkg [--min-area-ha 0.5] [--crs EPSG:32625] \\
        [--check-points layout.gpkg:turbines] [--caption-note "Setbacks measured from building footprints"]

Output layers:
  developable_area    developable polygons (area_ha)
  exclusion_union     excluded part of the site; n_constraints = how many constraints overlap there
  developable_summary table: excluded area and exclusive area (area excluded by this constraint
                      only) per constraint, plus site area and developable area. A constraint with a
                      small exclusive area changes the result little if it is dropped.
A JSON summary is printed to stdout.
"""
import argparse
import json
import sys

import geopandas as gpd
import pandas as pd
import shapely
from shapely.ops import unary_union


def parse_src(spec):
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"expected <file>:<layer>[:<label>], got {spec!r}")
    path, layer = parts[0], parts[1]
    label = ":".join(parts[2:]) if len(parts) > 2 else layer
    return path, layer, label


def load_clipped(path, layer, site_gs, crs):
    # Reading with a mask only fetches features that intersect the site (geopandas reprojects
    # the mask to the data CRS itself). Constraint layers often cover a whole region, and
    # reading everything before clipping is an order of magnitude slower.
    gdf = gpd.read_file(path, layer=layer, mask=site_gs)
    if gdf.empty:
        return shapely.Polygon()
    gdf = gdf.to_crs(crs)
    geoms = shapely.force_2d(shapely.make_valid(gdf.geometry.values))
    polys = [g for g in geoms if g is not None and not g.is_empty and g.area > 0]
    if not polys:
        return shapely.Polygon()
    return unary_union(polys).intersection(site_gs.to_crs(crs).iloc[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boundary", required=True,
                    help="<file>:<layer> with the site boundary polygon(s), or turbine points (see --site-buffer-m)")
    ap.add_argument("--exclude", action="append", required=True,
                    help="<file>:<layer>[:<label>] exclusion layer; repeat for each constraint")
    ap.add_argument("--out", required=True, help="output GeoPackage")
    ap.add_argument("--crs", help="metric CRS for the calculation; default: CRS of the boundary layer "
                                  "(must be a projected CRS)")
    ap.add_argument("--site-buffer-m", type=float, default=1000.0,
                    help="when --boundary contains points: site = convex hull of the points buffered by "
                         "this distance in metres (default 1000). Useful early on, before a formal site "
                         "boundary exists")
    ap.add_argument("--check-points", help="<file>:<layer> with existing turbine points: report how many "
                                           "fall inside each exclusion layer. If existing or planned "
                                           "positions sit inside a layer, that layer is probably not a "
                                           "true exclusion for this site; overlay it as a review layer "
                                           "instead of subtracting it")
    ap.add_argument("--min-area-ha", type=float, default=0.5,
                    help="drop developable fragments smaller than this (default 0.5 ha, roughly too small "
                         "for one turbine foundation and crane pad)")
    ap.add_argument("--caption-note", metavar="TEXT",
                    help="free text copied verbatim into the summary as caption_note, e.g. how setbacks "
                         "were measured; put it in the figure caption. Omitted when not given")
    a = ap.parse_args()

    bpath, blayer, _ = parse_src(a.boundary)
    bnd = gpd.read_file(bpath, layer=blayer)
    if bnd.empty or bnd.geometry.is_empty.all():
        # An empty boundary gives a site area of 0, so every constraint excludes 0 ha and the
        # result looks like "no constraints" when in fact no site was read.
        raise SystemExit(f"boundary layer {bpath}:{blayer} has no features; use a layer that contains the site")
    crs = a.crs or bnd.crs
    if not gpd.GeoSeries([], crs=crs).crs.is_projected:
        raise SystemExit(f"calculation CRS {crs} is not projected, areas would be wrong; pass --crs (e.g. a UTM zone)")
    geoms = shapely.force_2d(shapely.make_valid(bnd.to_crs(crs).geometry.values))
    if all(g.geom_type in ("Point", "MultiPoint") for g in geoms if g is not None):
        site = unary_union(geoms).convex_hull.buffer(a.site_buffer_m)
        print(f"  site = convex hull of {len(geoms)} points buffered by {a.site_buffer_m:g} m")
    else:
        site = unary_union(geoms)
    site_gs = gpd.GeoSeries([site], crs=crs)

    excl = []
    for spec in a.exclude:
        path, layer, label = parse_src(spec)
        g = load_clipped(path, layer, site_gs, crs)
        if g.is_empty:
            print(f"  WARNING {label}: does not intersect the site (0 ha); check the layer and its coverage",
                  file=sys.stderr)
        excl.append((label, layer, g))
        print(f"  {label:<24s} {g.area / 1e4:10.1f} ha inside the site")

    union = unary_union([g for _, _, g in excl if not g.is_empty]) if excl else shapely.Polygon()
    dev = site.difference(union)
    parts = [p for p in shapely.get_parts(dev) if p.area >= a.min_area_ha * 1e4]
    dropped_ha = (dev.area - sum(p.area for p in parts)) / 1e4

    rows = []
    for i, (label, layer, g) in enumerate(excl):
        others = unary_union([h for j, (_, _, h) in enumerate(excl) if j != i and not h.is_empty]) \
            if len(excl) > 1 else shapely.Polygon()
        rows.append({"constraint": label, "layer": layer,
                     "excluded_ha": round(g.area / 1e4, 1),
                     "exclusive_ha": round(g.difference(others).area / 1e4, 1)})
    if a.check_points:
        ppath, player, _ = parse_src(a.check_points)
        pts = shapely.force_2d(gpd.read_file(ppath, layer=player).to_crs(crs).geometry.values)
        for row, (label, _, g) in zip(rows, excl):
            n_in = int(sum(1 for q in pts if q is not None and not g.is_empty and g.contains(q)))
            row["points_inside"] = n_in
            if n_in:
                print(f"  WARNING {label}: {n_in}/{len(pts)} existing points fall inside this layer; it may "
                      f"not be a true exclusion here. Consider removing it from --exclude and overlaying "
                      f"it as a review layer", file=sys.stderr)

    # Overlap count: how many constraints exclude each piece of the site (for an overlap-count map).
    pieces = shapely.get_parts(shapely.polygonize(shapely.get_parts(
        unary_union([shapely.boundary(g) for _, _, g in excl if not g.is_empty] + [shapely.boundary(site)]))))
    cnt_rows = []
    for p in pieces:
        if not p.within(site.buffer(1e-6)) or p.area < 1:
            continue
        pt = p.representative_point()
        n = sum(1 for _, _, g in excl if not g.is_empty and g.contains(pt))
        if n:
            cnt_rows.append({"n_constraints": n, "geometry": p})

    site_ha = site.area / 1e4
    dev_ha = sum(p.area for p in parts) / 1e4
    summary = {"site_ha": round(site_ha, 1), "developable_ha": round(dev_ha, 1),
               "developable_share": round(dev_ha / site_ha, 4) if site_ha else None,
               "dropped_slivers_ha": round(dropped_ha, 1), "min_area_ha": a.min_area_ha,
               "crs": str(crs), "constraints": rows,
               "site_from_points_buffer_m": a.site_buffer_m if not bnd.geom_type.isin(["Polygon", "MultiPolygon"]).any() else None}
    if a.caption_note:
        summary["caption_note"] = a.caption_note

    gpd.GeoDataFrame({"area_ha": [round(p.area / 1e4, 2) for p in parts]}, geometry=parts, crs=crs) \
        .to_file(a.out, layer="developable_area", driver="GPKG")
    if cnt_rows:
        gpd.GeoDataFrame(cnt_rows, crs=crs).dissolve(by="n_constraints", as_index=False) \
            .to_file(a.out, layer="exclusion_union", driver="GPKG")
    tbl = pd.DataFrame(rows + [{"constraint": "__site__", "layer": "", "excluded_ha": round(site_ha, 1),
                                "exclusive_ha": None},
                               {"constraint": "__developable__", "layer": "", "excluded_ha": round(dev_ha, 1),
                                "exclusive_ha": None}])
    # The summary is a plain table; the empty geometry column carries the calculation CRS only so
    # that the writer does not warn about a missing CRS.
    gpd.GeoDataFrame(tbl, geometry=[None] * len(tbl), crs=crs).to_file(a.out, layer="developable_summary",
                                                                       driver="GPKG")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
