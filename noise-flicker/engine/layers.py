#!/usr/bin/env python3
"""GPKG/CSV -> engine objects. **Its only reason to exist is to prevent CRS errors.**

Both engines compute geometry in **metres**; turbines and receptors must share one
projected CRS (e.g. the site's UTM zone). Passing WGS84 lon/lat as metres raises no
error - it silently produces wrong results. This module forces a reprojection and
rejects a CRS that is still geographic.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd

import flicker as _f
import noise as _n


def _to_metric(gdf, epsg: int | None):
    if gdf.crs is None:
        raise ValueError("layer has no CRS. Set one first - guessing a CRS is the most common "
                         "source of silent errors.")
    if epsg is not None:
        gdf = gdf.to_crs(epsg=epsg)
    elif gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    if gdf.crs.is_geographic:
        raise ValueError(f"target CRS {gdf.crs} is still geographic (degrees). "
                         f"The engines work in metres; give a projected CRS.")
    return gdf


def _pts(gdf):
    """Representative points. Polygons (building footprints) use representative_point so
    the point is guaranteed to lie inside the shape."""
    g = gdf.geometry
    return g.representative_point() if (g.geom_type != "Point").any() else g


def receptors_from_gpkg(path, layer=None, *, epsg=None, id_field=None,
                        kind_field=None, la90_field=None, height_m=1.5,
                        where=None):
    """Dwelling/sensitive receptor points or building polygons -> receptors. Returns the
    receptor lists **shared** by the noise and flicker engines."""
    gdf = gpd.read_file(path, layer=layer)
    if where:
        gdf = gdf.query(where)
    if gdf.empty:
        raise ValueError(f"{path}:{layer} is empty after filtering (where={where!r})")
    gdf = _to_metric(gdf, epsg)
    pts = _pts(gdf)
    out_n, out_f = [], []
    for i, (idx, row) in enumerate(gdf.iterrows()):
        p = pts.iloc[i]
        # Use item access: attribute access such as `row.project` can collide with a
        # GeoSeries method name and silently return the wrong thing
        rid = str(row[id_field]) if id_field else str(idx)
        kind = str(row[kind_field]) if kind_field else ""
        la90 = float(row[la90_field]) if (la90_field and row[la90_field] is not None
                                          and row[la90_field] == row[la90_field]) else None
        out_n.append(_n.Receptor(p.x, p.y, rid, background_la90_dBA=la90, kind=kind,
                                 crs=gdf.crs))
        out_f.append(_f.FlickerReceptor(p.x, p.y, rid, height_m=height_m, kind=kind,
                                        crs=gdf.crs))
    return out_n, out_f, gdf.crs


def turbines_from_gpkg(path, layer=None, *, epsg=None, id_field=None,
                       hub_height_m=None, rotor_diameter_m=None, lwa_dBA=None,
                       hub_field=None, rotor_field=None, lwa_field=None, where=None):
    """Turbines -> ([noise.Turbine], [flicker.FlickerTurbine]).

    Turbine parameters may be given as a scalar (one model for the whole site) or a field
    name (per position). Giving neither raises - **there is no default turbine**: a default
    L_WA would suggest the result is based on a specific turbine model.
    """
    gdf = gpd.read_file(path, layer=layer)
    if where:                      # symmetric with the receptor loader
        gdf = gdf.query(where)
    # Without this check an empty layer (and no turbine parameters) silently returns
    # ([], [], crs), and downstream code would still produce an "all comply" report.
    if gdf.empty:
        raise ValueError(f"{path}:{layer} turbine layer is empty"
                         + (f" (where={where!r})" if where else ""))
    gdf = _to_metric(gdf, epsg)
    pts = _pts(gdf)

    def val(row, scalar, field, name):
        if field is not None:
            return float(row[field])
        if scalar is not None:
            return float(scalar)
        raise ValueError(f"{name} is required (scalar or field name). There is deliberately "
                         f"no default: an invented turbine parameter would make the "
                         f"screening result look evidence-based.")

    tn, tf = [], []
    for i, (idx, row) in enumerate(gdf.iterrows()):
        p = pts.iloc[i]
        wid = str(row[id_field]) if id_field else f"WTG{i+1}"
        hub = val(row, hub_height_m, hub_field, "hub_height_m")
        tn.append(_n.Turbine(p.x, p.y, hub, val(row, lwa_dBA, lwa_field, "lwa_dBA"),
                             wid, crs=gdf.crs))
        tf.append(_f.FlickerTurbine(p.x, p.y, hub,
                                    val(row, rotor_diameter_m, rotor_field,
                                        "rotor_diameter_m"), wid, crs=gdf.crs))
    return tn, tf, gdf.crs


def to_csv(result: dict, path):
    """Per-receptor CSV (standard format) + a sidecar `<csv>.meta.txt` (limits of validity,
    criterion provenance, model parameters).

    Returns `(csv_path, meta_path)`.

    Dict-valued columns are flattened, not dropped: `contributing_wtg_hours` points
    directly at the turbine to move and `hour_of_day_minutes` feeds the time-of-day
    argument. Filtering them out with `extrasaction="ignore"` would lose them silently.
    The disclaimer travels with the CSV via the sidecar file.
    """
    import csv
    rows = result["receptors"]
    if not rows:
        raise ValueError("result contains no receptors")
    # Internal pipeline keys such as `_resolved` / `_table_value_dBA` must not reach the
    # deliverable (they duplicate public columns).
    keys = [k for k in rows[0] if not k.startswith("_")]
    flat = [{k: ("; ".join(f"{a}={b}" for a, b in (v or {}).items())
                 if isinstance(v, dict) else v)
             for k, v in r.items()} for r in rows]
    # The disclaimer must travel with the data, but not by breaking the CSV format:
    # `#` header lines make pd.read_csv() fail and, worse, make csv.DictReader silently
    # treat the comment as the header. The CSV stays standard; the limits of validity go
    # into a sidecar .meta.txt whose path is returned as well.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)
    meta = Path(str(path) + ".meta.txt")
    meta.write_text(
        f"{result.get('_what', '')}\n\n"
        f"[Limits of validity] {result.get('_disclaimer', '')}\n\n"
        f"[Criterion provenance] {json.dumps(result.get('criterion', {}), ensure_ascii=False, indent=2)}\n\n"
        f"[Model parameters] {json.dumps(result.get('model', {}), ensure_ascii=False, indent=2)}\n\n"
        f"Note: rows with status=INVALID_UNDER_ROTOR have empty (not 0) numeric columns - "
        f"a filter such as `df[df.astro_hours_per_year > 30]` silently drops them, and "
        f"they are exactly the rows that most need a human look. Group by status first.\n",
        encoding="utf-8")
    return path, meta


def load_site(turbine_path, receptor_path, *, turbine_layer=None, receptor_layer=None,
              epsg=None, **kw):
    """**Recommended entry point**: load turbines and receptors together, forcing both
    layers into the same CRS.

    Calling `turbines_from_gpkg` / `receptors_from_gpkg` separately lets each layer pick
    its own `estimate_utm_crs()`; when a site straddles a UTM zone boundary they land in
    different zones and the engines compute Euclidean distances in metres across them -
    a few km can come out as hundreds of km. Here the turbine layer fixes the CRS and the
    receptor layer follows it.

    kw is passed through to the two loaders (prefix `t_` for turbines, `r_` for receptors).
    """
    t_kw = {k[2:]: v for k, v in kw.items() if k.startswith("t_")}
    r_kw = {k[2:]: v for k, v in kw.items() if k.startswith("r_")}
    tn, tf, crs = turbines_from_gpkg(turbine_path, turbine_layer, epsg=epsg, **t_kw)
    rn, rf, crs_r = receptors_from_gpkg(receptor_path, receptor_layer,
                                        epsg=crs.to_epsg(), **r_kw)
    _n.assert_same_crs(tn, rn)
    return tn, tf, rn, rf, crs
