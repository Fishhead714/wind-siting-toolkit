"""Terrain classification: route a zone to the flat/rolling branch or the mountain branch.

Zones are routed by DEM relief (max - min elevation inside the zone geometry),
optionally corrected by RIX (Ruggedness Index, area fraction of steep pixels).
The flat branch uses AEP optimisation (``layoutoptimize``); the mountain branch
uses terrain-constrained candidate generation (``terrainlayout``). A three-level
severity label (flat / rolling / mountainous) is attached for report wording only.

Single source of truth: import this module rather than copying the criteria.

All thresholds below are **engineering assumptions with example defaults**, not
standard-mandated values. Calibrate them for your own terrain and DEM product.
"""
import numpy as np
import rasterio

from .rasterutils import masked_stats, sample_raster


#: Critical slope for RIX. RIX (Ruggedness Index) is the *fraction of the area
#: around a site where terrain slope exceeds a critical value*; the customary
#: critical value is tan = 0.3, i.e. arctan(0.3) ~ 16.7 deg (Bowen & Mortensen,
#: 1996, "Exploring the limits of WAsP", Risoe-R-995; see THIRD_PARTY_NOTICES.md).
#: This is an area-level statistic, NOT a per-pixel "cannot build here" rule:
#: a high RIX says linear-flow models such as WAsP are unreliable on that terrain.
#: It is independent of construction platform slope and of IEC inflow-angle limits;
#: the three criteria must not substitute for one another.
RIX_CRITICAL_SLOPE_DEG = np.degrees(np.arctan(0.3))   # ~16.699 deg

#: Coarsest DEM pixel size (metres) at which RIX is allowed to **override relief**.
#:
#: RIX counts pixels steeper than the critical slope, and the slope comes from a
#: finite difference across pixel centres, so RIX is a property of the DEM
#: resolution as much as of the terrain: it collapses towards zero as the DEM
#: gets coarser. "RIX near zero overrides relief" therefore only holds on a
#: sufficiently fine DEM; on a coarse resampled DEM the same rugged area would
#: silently be labelled a gentle plateau.
#:
#: Example default, not a standard value: it separates ~30 m global DEM products
#: from ~90 m and coarser ones. Calibrate for your own DEM sources.
RIX_MAX_PIXEL_SIZE_FOR_OVERTURN_M = 60.0


def dem_pixel_size_m(transform, mean_lat_deg=None):
    """Approximate ground size of a DEM pixel (metres). Returns (dx_m, dy_m).

    For a geographic (degree) raster: north-south is 111320 * |e|; east-west
    additionally scales by cos(lat).
    """
    dy = abs(transform.e) * 111320.0
    if mean_lat_deg is None:
        dx = abs(transform.a) * 111320.0
    else:
        dx = abs(transform.a) * 111320.0 * np.cos(np.radians(mean_lat_deg))
    return float(dx), float(dy)


def load_dem(path):
    """Read a GeoTIFF DEM, nodata -> NaN; returns (elev, transform).

    Assumes a geographic (WGS84 degree) CRS, the usual distribution CRS of public
    DEM products.
    """
    with rasterio.open(path) as src:
        elev = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)
    return elev, transform


def compute_rix(slope_deg, transform, geom, critical_slope_deg=None):
    """RIX (%) **inside** the given geometry: share of pixels steeper than the critical slope.

    :param slope_deg: 2D ndarray, slope in degrees, on the same grid as ``transform``
        (as produced by ``terrainlayout.compute_slope()``)
    :param transform: rasterio Affine, degree coordinates
    :param geom: shapely (Multi)Polygon in the raster's CRS
    :param critical_slope_deg: critical slope; default RIX_CRITICAL_SLOPE_DEG (~16.7 deg)
    :return: {'rix_pct', 'n_pixels', 'critical_slope_deg', 'bbox_fill_pct'};
        ``rix_pct`` is None when no valid pixel falls inside the geometry

    The statistic is masked by the geometry itself (not its bounding box), and
    ``bbox_fill_pct`` is reported alongside.
    """
    from .rasterutils import geometry_mask_window

    if critical_slope_deg is None:
        critical_slope_deg = RIX_CRITICAL_SLOPE_DEG

    window, inside, _cov = geometry_mask_window(slope_deg, transform, geom)
    if window is None:
        return {"rix_pct": None, "n_pixels": 0,
                "critical_slope_deg": float(critical_slope_deg), "bbox_fill_pct": None}
    sel = inside & ~np.isnan(window)
    n = int(sel.sum())
    if n == 0:
        return {"rix_pct": None, "n_pixels": 0,
                "critical_slope_deg": float(critical_slope_deg), "bbox_fill_pct": 0.0}
    return {
        "rix_pct": float((window[sel] > critical_slope_deg).sum() / n * 100),
        "n_pixels": n,
        "critical_slope_deg": float(critical_slope_deg),
        "bbox_fill_pct": float(inside.sum() / inside.size * 100),
    }


def _sample_grid(geom, elev, transform, n_grid):
    """n_grid x n_grid grid sampling over the bbox. **Fallback only**; the main path is masked_stats().

    Kept so that a geometry entirely outside the DEM produces an explainable
    failure instead of a crash. It has two known weaknesses (bbox, sparse
    sampling); do not use it as the primary path.
    """
    minx, miny, maxx, maxy = geom.bounds
    lons = np.linspace(minx, maxx, n_grid)
    lats = np.linspace(miny, maxy, n_grid)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return sample_raster(elev, transform, lon_grid.ravel(), lat_grid.ravel())


_RECOMMENDATIONS = {
    "flat": "Flat: linear flow extrapolation (WAsP-type) is broadly usable; lay out "
            "with the flat/rolling branch (AEP optimisation).",
    "rolling": "Rolling: local terrain effects are limited but not negligible; lay out "
               "with the mountain branch (terrain-constrained candidate generation) "
               "and verify key positions with CFD.",
    "mountainous": "Mountainous: local speed-up, lee-side and turbulence effects are "
                   "significant and linear extrapolation is unreliable; lay out with "
                   "the mountain branch and strongly consider micro-scale CFD "
                   "(e.g. a RANS-based flow model) before fixing positions.",
}


def _recommendation_for(tier, route, rix_pct, rix_overturned):
    """Recommendation text follows the **actual route**, not only the tier."""
    base = _RECOMMENDATIONS[tier]
    if route == "mountain_branch":
        return base
    if rix_overturned:
        return (f"Gentle plateau: relief falls in the '{tier}' tier but measured "
                f"RIX={rix_pct:.2f}% is near zero (almost no steep pixels inside the "
                "zone); use the flat branch for AEP optimisation. "
                "Still verify the local flow at individual positions: a low RIX only "
                "says slopes are generally gentle and does not rule out isolated "
                "terrain breaks.")
    return base


def classify_zone(geom, elev, transform, flat_threshold_m=100.0, mountain_threshold_m=300.0,
                  rix_gentle_pct=1.0,
                  n_grid=11, slope_deg=None, rix_threshold_pct=5.0, rix_critical_slope_deg=None,
                  rix_max_pixel_size_m=RIX_MAX_PIXEL_SIZE_FOR_OVERTURN_M,
                  min_dem_coverage_pct=95.0):
    """Terrain classification for a single zone geometry (degree coordinates).

    All numeric thresholds are example defaults (engineering assumptions), not
    standard values; calibrate them for your terrain and DEM.

    :param geom: shapely (Multi)Polygon in the DEM's geographic CRS (e.g. EPSG:4326)
    :param elev: 2D ndarray, DEM elevation (NaN = nodata)
    :param transform: rasterio Affine on the same grid as ``elev``
    :param flat_threshold_m: relief below this routes to the flat/rolling branch
        (``layoutoptimize``), otherwise to the mountain branch (``terrainlayout``).
        Example default 100 m.
    :param mountain_threshold_m: boundary between the rolling and mountainous
        severity labels. Affects report wording only, not the route.
    :param n_grid: density of the fallback bbox grid sampling (n_grid x n_grid points)
    :param slope_deg: optional 2D slope raster (same grid). RIX is computed only if
        it is given. **Without it the classifier falls back to relief only**, which
        is scale-insensitive: a 120 m gentle slope over 20 km and a 120 m cliff
        over 2 km get the same number although their flow is entirely different.
        Pass a slope raster whenever you have one.
    :param rix_gentle_pct: RIX at or below this value means "measured gentle
        everywhere" and can **override** an over-threshold relief, routing to the
        flat branch. Example default 1%. Only active if ``slope_deg`` was given
        and the DEM is fine enough (see ``rix_max_pixel_size_m``).
    :param rix_threshold_pct: RIX at or above this value means complex terrain
        (mountain branch). Example default 5%. The RIX literature gives a
        continuous "higher RIX, less reliable WAsP" relationship and a delta-RIX
        correction method, not an official binary threshold.
    :param rix_critical_slope_deg: RIX critical slope, default ~16.7 deg (tan = 0.3)
    :param rix_max_pixel_size_m: coarsest DEM pixel at which RIX may override relief
    :param min_dem_coverage_pct: warn when less of the geometry than this lies on the DEM
    :return: {
        'relief_m', 'elev_min_m', 'elev_max_m', 'elev_mean_m',
        'tier': 'flat'|'rolling'|'mountainous',
        'route': 'flat_branch'|'mountain_branch',
        'route_reason': str, which criterion triggered the route (for traceability),
        'rix_pct': float|None, None when ``slope_deg`` was not given,
        'rix_critical_slope_deg': float|None,
        'recommendation': str,
        'n_valid_samples': int,
        ... plus DEM pixel size, coverage and sampling-method diagnostics
    }
    """
    # Statistics are computed inside the geometry at full resolution, not on a
    # bbox grid: the bbox contains terrain outside the zone (and sea surface for
    # coastal zones), and a sparse grid can miss true extremes on km-scale zones.
    # `_sample_grid` remains as a fallback so a geometry entirely outside the
    # raster yields an explainable error.
    stats = masked_stats(elev, transform, geom)
    if stats["n_pixels"]:
        elev_min, elev_max = stats["min"], stats["max"]
        elev_mean = stats["mean"]
        relief = stats["relief"]
        n_valid = stats["n_pixels"]
        bbox_fill_pct = stats["bbox_fill_pct"]
        sampling = "geometry_masked_full_resolution"
        dem_coverage_pct = stats["dem_coverage_pct"]
    else:
        vals = _sample_grid(geom, elev, transform, n_grid)
        if np.all(np.isnan(vals)):
            raise ValueError(
                "The zone has no valid elevation pixel on the DEM; check that geom "
                "overlaps the DEM footprint and is in the DEM's geographic CRS"
            )
        elev_min, elev_max = float(np.nanmin(vals)), float(np.nanmax(vals))
        elev_mean = float(np.nanmean(vals))
        relief = float(elev_max - elev_min)
        n_valid = int(np.sum(~np.isnan(vals)))
        bbox_fill_pct = None
        sampling = "bbox_grid_fallback"
        dem_coverage_pct = None

    if relief < flat_threshold_m:
        tier = "flat"
    elif relief < mountain_threshold_m:
        tier = "rolling"
    else:
        tier = "mountainous"

    if dem_coverage_pct is not None and dem_coverage_pct < min_dem_coverage_pct:
        print(f"  [WARN] Only {dem_coverage_pct:.1f}% of this zone's area lies on the DEM "
              f"(below {min_dem_coverage_pct:.0f}%): relief / RIX / representative "
              "elevation describe only that corner, not the whole zone. Use a DEM "
              "with fuller coverage, or clip the zone to the DEM before re-running")

    # Routing criterion, three levels (not a plain OR of relief and RIX):
    #
    # A plain OR can only tighten, never correct. It catches "small total relief
    # but locally steep" (corrugated terrain) but not the mirror case: "large
    # total relief but gentle everywhere" (a tilted plateau). Relief is a
    # scale-insensitive coarse proxy: the same 200 m drop is a steep slope over
    # 2 km but a ~0.3% gradient over a 60 km^2 plateau. RIX measures directly
    # "how much of the area is really steep", the natural indicator for "is
    # linear extrapolation trustworthy". So when RIX has been measured and is
    # confirmed near zero, it should be able to overrule relief.
    #
    #   RIX >= rix_threshold_pct       -> mountain branch (RIX tightens)
    #   RIX <= rix_gentle_pct          -> flat branch, even if relief is over the limit
    #   in between + relief over limit -> mountain branch (conservative)
    # Without a slope raster RIX is None and the classifier reverts entirely to
    # relief only: no measurement, no override.
    rix = compute_rix(slope_deg, transform, geom, rix_critical_slope_deg) if slope_deg is not None else None
    rix_pct = rix["rix_pct"] if (rix and rix["rix_pct"] is not None) else None
    relief_complex = relief >= flat_threshold_m
    rix_complex = bool(rix_pct is not None and rix_pct >= rix_threshold_pct)

    # Resolution gate: RIX collapses monotonically towards 0 as the DEM coarsens,
    # so "RIX near zero => override relief" only holds on a fine enough DEM. The
    # tightening direction (high RIX => mountain) is unaffected: a coarse DEM can
    # only under-estimate RIX, never inflate it.
    lat_c = (transform.f + transform.e * elev.shape[0] / 2) if transform is not None else None
    px_dx, px_dy = dem_pixel_size_m(transform, lat_c)
    dem_px_m = max(px_dx, px_dy)
    rix_may_overturn = dem_px_m <= rix_max_pixel_size_m
    rix_gentle = bool(rix_pct is not None and rix_pct <= rix_gentle_pct and rix_may_overturn)
    rix_overturn_blocked = bool(
        rix_pct is not None and rix_pct <= rix_gentle_pct and not rix_may_overturn)

    if rix_complex:
        route = "mountain_branch"
    elif relief_complex and rix_gentle:
        route = "flat_branch"          # measured gentle everywhere; the relief proxy is overruled
    elif relief_complex:
        route = "mountain_branch"
    else:
        route = "flat_branch"

    route_reason = []
    if rix_complex:
        route_reason.append(f"RIX {rix_pct:.1f}% >= {rix_threshold_pct}%")
    elif relief_complex and rix_gentle:
        route_reason.append(
            f"relief {relief:.0f}m >= {flat_threshold_m}m BUT RIX {rix_pct:.2f}% <= {rix_gentle_pct}%: "
            "measured gentle everywhere, classed as a gentle plateau rather than "
            "mountain; the scale-insensitivity of relief is overruled by RIX")
    elif relief_complex and rix_overturn_blocked:
        route_reason.append(
            f"relief {relief:.0f}m >= {flat_threshold_m}m; RIX {rix_pct:.2f}% is near zero but "
            f"DEM pixel {dem_px_m:.0f}m is coarser than {rix_max_pixel_size_m:.0f}m, so override is NOT allowed: "
            "a coarse DEM pushes RIX towards 0, so a low RIX cannot prove the terrain is gentle")
    elif relief_complex:
        route_reason.append(f"relief {relief:.0f}m >= {flat_threshold_m}m"
                            + (f"; RIX {rix_pct:.1f}% lies between {rix_gentle_pct} and {rix_threshold_pct}%, not enough to override"
                               if rix_pct is not None else "; no slope raster given, so RIX could not take part"))
    if not route_reason:
        route_reason.append(
            f"relief {relief:.0f}m < {flat_threshold_m}m"
            + (f" and RIX {rix['rix_pct']:.1f}% < {rix_threshold_pct}%" if rix and rix["rix_pct"] is not None
               else "; no slope raster given, RIX not computed, complexity judged from relief only")
        )

    return {
        "relief_m": round(relief, 1),
        "elev_min_m": elev_min,
        "elev_max_m": elev_max,
        "elev_mean_m": elev_mean,
        "tier": tier,
        "route": route,
        "route_reason": "; ".join(route_reason),
        "rix_pct": (rix["rix_pct"] if rix else None),
        "rix_critical_slope_deg": (rix["critical_slope_deg"] if rix else None),
        # RIX is a property of DEM resolution: the pixel size must travel with the number
        "dem_pixel_size_m": round(dem_px_m, 2),
        # How much of the geometry really lies on the DEM. Below
        # min_dem_coverage_pct every statistic above describes only a corner of it.
        "dem_coverage_pct": (round(dem_coverage_pct, 2)
                             if dem_coverage_pct is not None else None),
        "dem_coverage_ok": (None if dem_coverage_pct is None
                            else dem_coverage_pct >= min_dem_coverage_pct),
        "rix_may_overturn_relief": rix_may_overturn,
        "rix_max_pixel_size_m": rix_max_pixel_size_m,
        # The recommendation follows the actual route; tier is only a severity label.
        "recommendation": _recommendation_for(tier, route, rix_pct, rix_gentle),
        "tier_note": "tier is a terrain-severity label (relief only); the algorithm route is in 'route'",
        "n_valid_samples": n_valid,
        "sampling_method": sampling,
        "bbox_fill_pct": bbox_fill_pct,
    }


def classify_zones(zones_gdf, elev, transform, flat_threshold_m=100.0, mountain_threshold_m=300.0,
                    n_grid=11, id_col=None, slope_deg=None, rix_threshold_pct=5.0,
                    rix_critical_slope_deg=None, rix_gentle_pct=1.0,
                    rix_max_pixel_size_m=RIX_MAX_PIXEL_SIZE_FOR_OVERTURN_M,
                    min_dem_coverage_pct=95.0):
    """Batch version: ``zones_gdf`` is a GeoDataFrame (EPSG:4326); calls classify_zone() per row.

    :return: list[dict]; each dict is the classify_zone() result plus a 'zone_id'
        key (value of ``id_col``, or the row index when not given)

    All arguments are passed to ``classify_zone`` **by keyword** so that
    inserting a parameter there can never silently shift positional arguments.
    """
    if zones_gdf.crs is None or zones_gdf.crs.to_epsg() != 4326:
        raise ValueError(
            "zones_gdf must be in WGS84 (EPSG:4326): DEM sampling works on the "
            "lon/lat grid, so call to_crs(4326) first"
        )

    results = []
    for idx, row in zones_gdf.iterrows():
        r = classify_zone(
            row.geometry, elev, transform,
            flat_threshold_m=flat_threshold_m,
            mountain_threshold_m=mountain_threshold_m,
            rix_gentle_pct=rix_gentle_pct,
            n_grid=n_grid,
            slope_deg=slope_deg,
            rix_threshold_pct=rix_threshold_pct,
            rix_critical_slope_deg=rix_critical_slope_deg,
            rix_max_pixel_size_m=rix_max_pixel_size_m,
            min_dem_coverage_pct=min_dem_coverage_pct,
        )
        r["zone_id"] = row[id_col] if id_col else idx
        results.append(r)
    return results
