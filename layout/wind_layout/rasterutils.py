"""Shared raster sampling and geometry-masked statistics.

Single source of truth: the other modules import from here rather than
re-implementing these helpers.

Two pitfalls this module exists to avoid:

1. **Pixel lookup must use floor, not round.** ``~transform * (x, y)`` returns
   fractional coordinates where the pixel's top-left corner is an integer, so the
   pixel containing a point is ``floor`` of that value. ``np.round`` shifts the
   lookup by half a pixel and, because numpy rounds half to even, which neighbour
   gets sampled depends on index parity for points that sit exactly on pixel
   centres.
2. **Terrain statistics must be computed inside the geometry, not in its bounding
   box.** For irregular areas the bounding box can be mostly outside the polygon,
   and terrain from outside the polygon then leaks into relief / RIX.
"""
import numpy as np


def lonlat_to_rowcol(transform, lon, lat):
    """Map x/y (raster CRS) to raster (rows, cols) using floor, not round."""
    inv = ~transform
    cols, rows = inv * (np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return (np.floor(np.asarray(rows)).astype(int),
            np.floor(np.asarray(cols)).astype(int))


def sample_raster(raster, transform, lon, lat):
    """Sample a raster at points; out-of-bounds points return NaN.

    Returning NaN is deliberate: the caller must decide how to treat it. Never
    read NaN as "no problem" (a failed sample silently passing a check is a
    classic way to get a wrong "OK").
    """
    rows, cols = lonlat_to_rowcol(transform, lon, lat)
    nrow, ncol = raster.shape
    valid = (rows >= 0) & (rows < nrow) & (cols >= 0) & (cols < ncol)
    out = np.full(np.shape(rows), np.nan, dtype=float)
    out[valid] = raster[rows[valid], cols[valid]]
    return out


def raster_extent_polygon(raster, transform):
    """Polygon of the raster footprint (in the raster's CRS units)."""
    from shapely.geometry import box as _box

    nrow, ncol = raster.shape
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (ncol, nrow)
    return _box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def geometry_dem_coverage(raster, transform, geom):
    """Percentage (0-100) of the geometry's area that lies inside the raster.

    ``geometry_mask_window`` clips its window to the raster bounds before
    computing the inside fraction, so it cannot reveal that most of the geometry
    lies outside the DEM. This function intersects the geometries directly, so
    clipping cannot hide the shortfall. The ratio is taken in the raster's own
    coordinate units; only the ratio is used, never the absolute area.
    """
    if geom.is_empty or geom.area <= 0:
        return 0.0
    inter = geom.intersection(raster_extent_polygon(raster, transform))
    return float(inter.area / geom.area * 100)


def geometry_mask_window(raster, transform, geom):
    """Take a window and mask by the geometry itself (not its bounding box).

    :return: (window_array, mask, coverage_pct)
        ``coverage_pct`` is the percentage of the geometry's area inside the
        raster footprint. If the geometry lies fully outside the raster the
        result is ``(None, None, 0.0)``.

    Statistics (relief / RIX / representative elevation) must be computed inside
    the mask. ``coverage_pct`` answers a different question: whether the geometry
    extends beyond the DEM. A low value of both means the geometry is both
    partly outside the DEM and irregular; looking at only one of them misses
    cases.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    coverage = geometry_dem_coverage(raster, transform, geom)

    minx, miny, maxx, maxy = geom.bounds
    rows, cols = lonlat_to_rowcol(transform, [minx, maxx], [miny, maxy])
    r1, r2 = int(min(rows)), int(max(rows)) + 1
    c1, c2 = int(min(cols)), int(max(cols)) + 1
    r1, c1 = max(0, r1), max(0, c1)
    r2, c2 = min(raster.shape[0], r2), min(raster.shape[1], c2)
    if r2 <= r1 or c2 <= c1:
        return None, None, coverage

    window = raster[r1:r2, c1:c2]
    win_transform = transform * Affine.translation(c1, r1)
    inside = ~geometry_mask([geom], out_shape=window.shape, transform=win_transform,
                            invert=False)
    return window, inside, coverage


def masked_stats(raster, transform, geom):
    """Elevation statistics inside a geometry; all fields None if no valid pixel.

    :return: {'min','max','mean','relief','n_pixels','bbox_fill_pct','dem_coverage_pct'}

        - ``bbox_fill_pct``: share of the fetched window that is inside the
          geometry. Low means the shape is irregular and bounding-box statistics
          would be distorted. It cannot reveal a geometry extending past the DEM
          (the window has already been clipped).
        - ``dem_coverage_pct``: share of the geometry's area that lies inside the
          DEM. Low means the statistics describe only a small corner of the
          geometry, so ``relief`` / RIX / mean elevation do not represent it.

        Read both together.
    """
    window, inside, coverage = geometry_mask_window(raster, transform, geom)
    empty = {"min": None, "max": None, "mean": None, "relief": None,
             "n_pixels": 0, "bbox_fill_pct": None, "dem_coverage_pct": coverage}
    if window is None:
        return empty
    sel = inside & ~np.isnan(window)
    n = int(sel.sum())
    if n == 0:
        return {**empty, "bbox_fill_pct": 0.0}
    vals = window[sel]
    return {
        "min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()),
        "relief": float(vals.max() - vals.min()), "n_pixels": n,
        "bbox_fill_pct": float(inside.sum() / inside.size * 100),
        "dem_coverage_pct": coverage,
    }
