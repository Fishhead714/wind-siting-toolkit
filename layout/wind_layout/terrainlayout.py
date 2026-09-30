"""Complex-terrain turbine layout: terrain-constrained candidate generation.

Single source of truth: import this module rather than copying the algorithm.

The flat-flow wake model used by ``layoutoptimize`` (TOPFARM + PyWake) assumes flat
terrain and cannot represent local speed-up, lee-side or turbulence effects in
complex terrain. This module instead generates discrete, terrain-constrained
candidates: hard exclusion by slope, hard exclusion of valleys by TPI, and
priority-greedy selection among local terrain maxima (an approximation of ridge
lines). It is the "mountain branch" of the pipeline; its output must then go
through the directional-ellipse spacing check (``spacingcheck``):

    terrain classification -> route == 'mountain_branch' -> this module
    -> directional ellipse spacing check

**The output is "terrain-constrained candidate positions", not "wake-optimal
positions".** Reliable wake assessment in complex terrain needs CFD, which this
module does not do; it only applies terrain-level hard constraints, and any report
must state that limit.

Notes on the method:
  1. The rotor diameter is a parameter (``rotor_diameter_m``), not a hard-coded
     turbine.
  2. The greedy priority is configurable via ``priority_raster``. By default it is
     elevation itself (follow ridge lines), but an external raster (for example a
     global wind-resource raster whose underlying model already includes terrain
     downscaling) can be passed as the ranking criterion. The shape criterion
     (local maximum / ridge) still uses elevation; only "which ridge candidate
     wins" uses the energy-related metric.
  3. The valley TPI is computed directly as a boolean raster
     (``compute_valley_mask``) with no vectorise-then-rasterise round trip.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from shapely.geometry import Point


def compute_slope(elev, transform, mean_lat_deg=None):
    """Slope by standard central differences (np.gradient), not the Horn 3x3 kernel.

    Accuracy is adequate for screening.

    :param elev: 2D ndarray, elevation on a geographic-degree raster (nodata = NaN)
    :param transform: rasterio Affine in degrees
    :param mean_lat_deg: representative latitude for the cos correction; default
        is the latitude of the raster centre
    :return: slope_deg, 2D ndarray, degrees
    """
    nrow, ncol = elev.shape
    if mean_lat_deg is None:
        lat_top = transform.f
        lat_bottom = transform.f + transform.e * nrow
        mean_lat_deg = (lat_top + lat_bottom) / 2
    dy_m = abs(transform.e) * 111320
    dx_m = abs(transform.a) * 111320 * math.cos(math.radians(mean_lat_deg))
    dzdy, dzdx = np.gradient(elev, dy_m, dx_m)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def compute_valley_mask(elev, transform, radius_m=500.0, tpi_std_threshold=-1.0):
    """Valley / depression mask from the Topographic Position Index (Weiss, 2001).

    TPI = point elevation - neighbourhood mean, standardised by the neighbourhood
    standard deviation. A standardised TPI below the threshold (default -1.0, i.e.
    more than one standard deviation below the neighbourhood mean) is a valley.

    The neighbourhood sums (sum, sum of squares, count) are computed by FFT
    convolution instead of a per-pixel Python callback with
    ``scipy.ndimage.generic_filter``; results are equivalent (Var = E[X^2] - E[X]^2)
    and the runtime is orders of magnitude lower on large rasters.

    :param radius_m: TPI neighbourhood radius, default 500 m. Engineering
        assumption (example default, not a standard value); about 17 pixels for
        a ~30 m DEM. Tune for your terrain.
    :param tpi_std_threshold: standardised TPI threshold, default -1.0 (example default)
    :return: valley_mask, 2D bool ndarray
    """
    nrow, ncol = elev.shape
    # On a degree grid the metric length of one pixel in x is scaled by cos(lat).
    # Radii in pixels are therefore computed per direction so the neighbourhood is
    # a geographic circle, consistent with compute_slope (which also applies cos).
    lat_top = transform.f
    lat_bottom = transform.f + transform.e * nrow
    mean_lat_deg = (lat_top + lat_bottom) / 2
    py_m = abs(transform.e) * 111320
    px_m = abs(transform.a) * 111320 * math.cos(math.radians(mean_lat_deg))
    ry_px = max(1, round(radius_m / py_m))
    rx_px = max(1, round(radius_m / px_m))

    cy, cx = ry_px, rx_px
    yy, xx = np.ogrid[:2 * ry_px + 1, :2 * rx_px + 1]
    # Elliptical pixel kernel = geographic circular neighbourhood
    kernel = (((yy - cy) / ry_px) ** 2 + ((xx - cx) / rx_px) ** 2 <= 1.0).astype(np.float32)

    valid = (~np.isnan(elev)).astype(np.float32)
    elev_filled = np.where(valid.astype(bool), elev, 0.0).astype(np.float32)

    # Var = E[X^2] - E[X]^2 is the classic catastrophic-cancellation form: two
    # large numbers subtracted to give a small one. In float32 this bites at high
    # elevations (a uniformly raised terrain must give an identical TPI in theory).
    # Fix: subtract the global mean BEFORE convolving to bring magnitudes back
    # near zero. The mean is a constant and does not change TPI (point minus
    # neighbourhood mean) or the neighbourhood standard deviation.
    # Keep float32: float64 doubles the complex FFT intermediates, which can
    # exhaust memory on very large rasters. What removes the cancellation is the
    # centring, not higher precision.
    offset = float(np.nanmean(elev)) if np.any(valid.astype(bool)) else 0.0
    elev_centered = np.where(valid.astype(bool), elev - offset, 0.0).astype(np.float32)

    count = fftconvolve(valid, kernel, mode="same")
    sum1 = fftconvolve(elev_centered, kernel, mode="same")
    sum2 = fftconvolve(elev_centered ** 2, kernel, mode="same")

    count = np.clip(count, 1e-6, None)
    neighbor_mean_c = sum1 / count          # neighbourhood mean of the centred data
    neighbor_mean_sq = sum2 / count
    neighbor_var = np.clip(neighbor_mean_sq - neighbor_mean_c ** 2, 0, None)
    neighbor_std = np.sqrt(neighbor_var)

    neighbor_mean = neighbor_mean_c + offset
    tpi = elev - neighbor_mean
    tpi_std = np.divide(tpi, neighbor_std, out=np.zeros_like(tpi), where=neighbor_std > 1e-6)
    return (tpi_std < tpi_std_threshold) & ~np.isnan(elev)


def greedy_min_spacing_select(points, priority, min_dist):
    """Greedy selection by descending priority with a minimum pairwise distance.

    A spatial hash (cell edge = min_dist) does the neighbour check in amortised
    O(N), in place of a true dynamic KD-tree.

    :param points: (N, 2) ndarray, metric projected coordinates
    :param priority: (N,) ndarray, larger = higher priority (elevation, wind speed, ...)
    :param min_dist: minimum distance, metres
    :return: (M, 2) ndarray of accepted points (a subset, original coordinates)
    """
    n = len(points)
    if n == 0:
        return points
    order = np.argsort(-priority)
    buckets = {}
    accepted_idx = []

    def cell_of(pt):
        return (int(pt[0] // min_dist), int(pt[1] // min_dist))

    for idx in order:
        pt = points[idx]
        cx, cy = cell_of(pt)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), ()):
                    if np.hypot(*(points[j] - pt)) < min_dist:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            accepted_idx.append(idx)
            buckets.setdefault((cx, cy), []).append(idx)
    return points[accepted_idx]


def greedy_elliptical_select(points, priority, a, b, azimuth_deg, min_dist=None):
    """Greedy selection by descending priority using a **directional ellipse** constraint.

    Using only an isotropic circle whose radius equals the larger spacing
    (``downwind_d * D``) makes the downstream ellipse check a tautology (the
    circle is stricter than the ellipse) and over-constrains the crosswind
    direction, so the turbine count is systematically low. Elliptical packing
    fits up to a/b times more than circular packing at the same larger spacing.

    The spatial-hash cell edge is 2a (the maximum required centre distance), so a
    3 x 3 neighbourhood covers every possible conflict.

    :param a: downwind semi-axis = downwind_d * D / 2
    :param b: crosswind semi-axis = crosswind_d * D / 2
    :param azimuth_deg: major-axis azimuth (compass bearing, clockwise from north)
    :param min_dist: optional extra isotropic lower bound (e.g. an absolute minimum
        distance required by construction or roads). None = ellipse criterion only.
    :return: (M, 2) ndarray of accepted points
    """
    from .spacingcheck import ellipse_conflict

    n = len(points)
    if n == 0:
        return points
    cell = 2.0 * max(a, b)
    order = np.argsort(-priority)
    buckets = {}
    accepted = []

    for idx in order:
        pt = points[idx]
        cx, cy = int(pt[0] // cell), int(pt[1] // cell)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), ()):
                    o = points[j]
                    ddx, ddy = pt[0] - o[0], pt[1] - o[1]
                    if bool(ellipse_conflict(ddx, ddy, a, b, azimuth_deg)):
                        ok = False
                        break
                    if min_dist and math.hypot(ddx, ddy) < min_dist:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            accepted.append(idx)
            buckets.setdefault((cx, cy), []).append(idx)
    return points[accepted]


def _largest_polygon(geom):
    if hasattr(geom, "geometry"):
        geom = geom.geometry
    if hasattr(geom, "union_all"):
        geom = geom.union_all()
    elif hasattr(geom, "unary_union"):
        geom = geom.unary_union
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def optimize_layout_mountain(
    usable_land_gdf,
    elev,
    slope_deg,
    valley_mask,
    dem_transform,
    rotor_diameter_m,
    min_spacing_d=3.0,
    buildability_slope_deg=17.0,
    search_window_d=1.0,
    priority_raster=None,
    id_prefix="MTN",
    spacing_ellipse=None,
    min_local_relief_m=4.0,
):
    """Complex-terrain layout: local terrain maxima among buildable, non-valley pixels.

    Candidates are local terrain maxima (an approximation of ridge lines) among
    pixels whose slope is acceptable and which are not in a valley. They are
    selected greedily from highest to lowest priority, with any two turbines at
    least ``min_spacing_d * D`` apart (isotropic circle; pass ``spacing_ellipse``
    for the directional criterion). The result must still go through the
    directional ellipse check in ``spacingcheck``.

    :param usable_land_gdf: geopandas GeoDataFrame/GeoSeries in a projected (metre)
        CRS; usable-land polygons (unioned, largest part kept)
    :param elev: 2D ndarray, DEM elevation (geographic raster, nodata = NaN), same
        grid as ``dem_transform``
    :param slope_deg: 2D ndarray, output of ``compute_slope()``, same grid
    :param valley_mask: 2D bool ndarray, output of ``compute_valley_mask()``, same grid
    :param dem_transform: rasterio Affine in degrees
    :param rotor_diameter_m: rotor diameter D
    :param min_spacing_d: minimum spacing multiple (isotropic circle). Example
        default 3.0; set your own.
    :param buildability_slope_deg: **per-pixel** hard slope exclusion. The
        semantics are "can a turbine be built at this point" (crane pad, road
        access). It is independent of two other angles that are easy to confuse
        with it:

          | criterion | meaning |
          |---|---|
          | RIX critical slope, ~16.7 deg | can the flow model be trusted on this terrain (an area statistic, not a per-point rule) |
          | inflow angle (IEC 61400-1 site assessment) | does the load assumption hold for this turbine |
          | this parameter | can this turbine be built |

        The default 17 deg is an **unsourced screening assumption** (example
        default). It must not be read as a construction slope limit: the natural
        slope before earthworks and the finished platform / road gradients are
        different quantities, and how steep a natural slope can still be levelled
        is a cost question, not a hard limit. Set it from the crane manual of
        your turbine and your earthwork budget.
    :param min_local_relief_m: minimum relief (metres) inside the search window
        for a pixel to count as a "local terrain maximum". Example default 4.0,
        chosen on the order of the absolute vertical accuracy of public ~30 m
        DEM products. A bump below the vertical accuracy of the DEM is noise, not
        a ridge. Adjust when you change DEM product.
    :param spacing_ellipse: optional ``(a, b, azimuth_deg)``. If given, selection
        uses the **directional ellipse** criterion and ``min_spacing_d`` becomes an
        isotropic lower bound on top of it (may be None). Without it a circle is
        used; note that a circle of radius ``downwind_d * D`` is exactly the
        maximum distance the ellipse criterion can require, which makes a later
        ellipse check always pass and gives too few turbines.
    :param search_window_d: local-maximum search window as a multiple of the
        rotor diameter, default 1.0 D
    :param priority_raster: 2D ndarray on the elevation grid, used to rank
        candidates. **Strongly recommended: pass a wind-resource raster** (e.g. a
        hub-height mean wind speed grid). Elevation is not energy density, and ranking
        by height alone puts turbines on high but poorly exposed positions. With
        None, elevation itself is used and the returned ``priority_source`` is
        ``"elevation_fallback"`` so a report can show the ranking basis is height,
        not energy.
    :param id_prefix: prefix of the output wtg_id
    :return: {
        'points_gdf': GeoDataFrame(wtg_id, geometry=Point, crs=usable_land_gdf.crs),
        'n_peak_candidates': int, candidates after slope + valley filtering, before spacing,
        'n_final': int, number of selected positions,
        'priority_source': 'wind_resource_raster' | 'elevation_fallback',
        'buildability_slope_deg': slope threshold actually used,
        ...
    }
    """
    import geopandas as gpd
    import pyproj

    if usable_land_gdf.crs is None or not usable_land_gdf.crs.is_projected:
        raise ValueError("usable_land_gdf must be in a projected (metre) CRS; "
                         "call to_crs(estimate_utm_crs()) first")

    # "Same grid as the DEM" is a hard precondition. Without a check, a
    # priority_raster of a different shape would be silently accepted and sampled
    # at completely wrong positions through dem_transform, with no error.
    for _name, _arr in (("slope_deg", slope_deg), ("valley_mask", valley_mask),
                        ("priority_raster", priority_raster)):
        if _arr is not None and np.shape(_arr) != np.shape(elev):
            raise ValueError(
                f"shape of {_name} {np.shape(_arr)} does not match elev {np.shape(elev)}. "
                "This module indexes pixels through a single dem_transform; a different "
                "grid would silently sample the wrong positions. Resample it to the DEM "
                "grid first")

    crs = usable_land_gdf.crs
    usable = _largest_polygon(usable_land_gdf)
    to_wgs = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    to_utm = pyproj.Transformer.from_crs(4326, crs, always_xy=True)

    if priority_raster is None:
        priority_raster = elev
        priority_source = "elevation_fallback"
        print("  [WARN] No priority_raster given for the mountain branch; falling back to "
              "ranking by ELEVATION. Elevation is not energy density; pass a wind-speed raster")
    else:
        priority_source = "wind_resource_raster"

    search_window_m = search_window_d * rotor_diameter_m
    min_spacing_m = min_spacing_d * rotor_diameter_m

    minx, miny, maxx, maxy = usable.bounds
    margin_m = search_window_m
    corner_lon, corner_lat = to_wgs.transform(
        [minx - margin_m, maxx + margin_m], [miny - margin_m, maxy + margin_m]
    )
    inv = ~dem_transform
    cols, rows = inv * (np.array(corner_lon), np.array(corner_lat))
    r1, r2 = max(0, int(np.floor(rows.min()))), min(elev.shape[0], int(np.ceil(rows.max())) + 1)
    c1, c2 = max(0, int(np.floor(cols.min()))), min(elev.shape[1], int(np.ceil(cols.max())) + 1)
    if r2 <= r1 or c2 <= c1:
        empty = gpd.GeoDataFrame({"wtg_id": []}, geometry=[], crs=crs)
        return {"points_gdf": empty, "n_peak_candidates": 0, "n_final": 0,
                "priority_source": priority_source,
                "buildability_slope_deg": buildability_slope_deg}

    elev_c = elev[r1:r2, c1:c2]
    slope_c = slope_deg[r1:r2, c1:c2]
    valley_c = valley_mask[r1:r2, c1:c2]
    priority_c = priority_raster[r1:r2, c1:c2]
    if elev_c.size == 0 or np.all(np.isnan(elev_c)):
        empty = gpd.GeoDataFrame({"wtg_id": []}, geometry=[], crs=crs)
        return {"points_gdf": empty, "n_peak_candidates": 0, "n_final": 0,
                "priority_source": priority_source,
                "buildability_slope_deg": buildability_slope_deg}

    pixel_size_m = abs(dem_transform.e) * 111320
    window_px = max(1, round(search_window_m / pixel_size_m))
    size = window_px * 2 + 1
    elev_filled = np.where(np.isnan(elev_c), -np.inf, elev_c)
    local_max = ndimage.maximum_filter(elev_filled, size=size, mode="nearest")

    # Notes on the peak criterion:
    # (a) `elev >= local_max` holds for EVERY pixel on perfectly level ground (a
    #     flat plateau, water surface, levelled area would make every pixel a
    #     candidate). So the neighbourhood must contain relief: the pixel must be
    #     strictly higher than the neighbourhood minimum.
    # (b) A pixel whose slope cannot be computed is NOT eligible (conservative).
    #     np.gradient spreads a nodata NaN to adjacent pixels, so cliff edges and
    #     shorelines would otherwise be exempt from the slope exclusion, exactly
    #     where they should not be.
    # (c) local_min must be computed with NaN filled as +inf (the minimum filter
    #     then ignores it), and any nodata in the window disqualifies the pixel;
    #     otherwise a level platform next to nodata (tile seams, water masks,
    #     clip borders) grows a ring of fake ridges.
    # (d) The relief threshold is min_local_relief_m, tied to the DEM vertical
    #     accuracy: below it, "local maximum ~ ridge line" is pure surface noise.
    elev_for_min = np.where(np.isnan(elev_c), np.inf, elev_c)
    local_min = ndimage.minimum_filter(elev_for_min, size=size, mode="nearest")
    nodata_in_window = ndimage.maximum_filter(
        np.isnan(elev_c).astype(np.uint8), size=size, mode="nearest") > 0
    has_relief = ((local_max - local_min) > min_local_relief_m) & ~nodata_in_window
    eligible = (slope_c <= buildability_slope_deg) & ~valley_c & ~np.isnan(elev_c)
    is_peak = (elev_filled >= local_max) & has_relief & eligible
    rr, cc = np.nonzero(is_peak)

    if len(rr) == 0:
        empty = gpd.GeoDataFrame({"wtg_id": []}, geometry=[], crs=crs)
        return {"points_gdf": empty, "n_peak_candidates": 0, "n_final": 0,
                "priority_source": priority_source,
                "buildability_slope_deg": buildability_slope_deg}

    lon, lat = dem_transform * (cc + c1 + 0.5, rr + r1 + 0.5)
    x, y = to_utm.transform(lon, lat)
    pts = np.column_stack([x, y])
    priorities = priority_c[rr, cc]

    import shapely

    inside = shapely.contains(usable, shapely.points(pts))
    pts, priorities = pts[inside], priorities[inside]
    n_peak_candidates = len(pts)

    if spacing_ellipse is not None:
        _a, _b, _az = spacing_ellipse
        kept = greedy_elliptical_select(pts, priorities, _a, _b, _az,
                                        min_dist=min_spacing_m if min_spacing_d else None)
        spacing_criterion = (f"directional_ellipse(a={_a:.0f}m,b={_b:.0f}m,az={_az:.0f}deg)"
                             + (f"+min_dist={min_spacing_m:.0f}m" if min_spacing_d else ""))
    else:
        kept = greedy_min_spacing_select(pts, priorities, min_spacing_m)
        spacing_criterion = f"isotropic_circle(r={min_spacing_m:.0f}m={min_spacing_d}D)"

    points_gdf = gpd.GeoDataFrame(
        {"wtg_id": [f"{id_prefix}_{i}" for i in range(len(kept))]},
        geometry=[Point(x, y) for x, y in kept],
        crs=crs,
    )
    return {
        "points_gdf": points_gdf,
        "n_peak_candidates": n_peak_candidates,
        "n_final": len(kept),
        "priority_source": priority_source,
        "buildability_slope_deg": buildability_slope_deg,
        "spacing_criterion": spacing_criterion,
        "min_local_relief_m": min_local_relief_m,
    }
