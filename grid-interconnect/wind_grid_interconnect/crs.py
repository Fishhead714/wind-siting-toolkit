"""Metric-CRS handling.

Every distance, buffer and raster cell in this package is in metres, so all
geometry must be in a projected CRS with metre units. ``to_metric_geoms``
rejects a non-metric target CRS. The other entry points only receive bare
coordinates (no CRS attached), so they apply ``assert_not_degrees``, a
plausibility heuristic: coordinates that all lie within +-360 are assumed to be
degrees and rejected. It cannot catch every mistake (for example a geographic
CRS shifted by an offset), so keep track of your CRS yourself.
"""
from __future__ import annotations

from typing import List

import geopandas as gpd
import pyproj

DISCLAIMER = (
    "Screening-level result only. Not a route survey or an engineering design. "
    "Right-of-way, permits, land tenure, crossings of third-party assets and "
    "geotechnical conditions are not considered. The off-road cost multiplier is a "
    "modelling assumption, not a measured constant."
)


def is_metric(crs) -> bool:
    """True if ``crs`` is projected and its axis unit is the metre."""
    if crs is None:
        return False
    crs = pyproj.CRS.from_user_input(crs)
    if not crs.is_projected:
        return False
    return all(a.unit_name in ("metre", "meter") for a in crs.axis_info)


def to_metric_geoms(gdf: gpd.GeoDataFrame, metric_crs) -> List:
    """Return the geometries of ``gdf`` in ``metric_crs`` (list of shapely objects).

    ``gdf`` must carry a CRS. It is reprojected when it differs from ``metric_crs``.
    Raises ``ValueError`` if ``metric_crs`` is not a metre-based projected CRS or
    if ``gdf`` has no CRS.
    """
    if not is_metric(metric_crs):
        raise ValueError(f"metric_crs must be a projected CRS in metres, got {metric_crs!r}")
    if gdf.crs is None:
        raise ValueError("input GeoDataFrame has no CRS; set it explicitly (gdf.set_crs)")
    if pyproj.CRS.from_user_input(gdf.crs) != pyproj.CRS.from_user_input(metric_crs):
        gdf = gdf.to_crs(metric_crs)
    return list(gdf.geometry.values)


def assert_not_degrees(bounds, what: str = "input") -> None:
    """Raise ``ValueError`` if ``bounds`` = (xmin, ymin, xmax, ymax) look like degrees.

    Heuristic: every coordinate within +-360. Metre coordinates that small are
    unusual; offset your local grid (e.g. add a false easting) if you really use one.
    """
    if bounds is None or len(bounds) != 4:
        return
    if max(abs(float(v)) for v in bounds) <= 360.0:
        raise ValueError(
            f"{what}: all coordinates lie within +-360, which looks like lon/lat degrees. "
            "Reproject to a metre-based CRS first (see to_metric_geoms)."
        )
