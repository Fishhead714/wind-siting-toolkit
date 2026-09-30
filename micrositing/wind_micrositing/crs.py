"""Metric-CRS handling.

All spacing arithmetic is in metres, so the developable area must be in a
projected CRS whose unit is the metre. Longitude/latitude input is either
rejected (default) or reprojected to the local UTM zone.
"""
from __future__ import annotations

import geopandas as gpd
import shapely
from pyproj import CRS


def _crs_is_metric(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    for axis in crs.axis_info:
        if axis.unit_name.lower() not in ("metre", "meter"):
            return False
    return True


def prepare_area(area, crs=None, on_geographic: str = "raise"):
    """Return ``(geometry, crs)`` with the geometry in a metre-based projected CRS.

    Parameters
    ----------
    area : GeoDataFrame, GeoSeries or shapely (Multi)Polygon
        Developable area. Multiple features are dissolved into one geometry.
    crs : anything pyproj accepts, optional
        Required when ``area`` is a bare shapely geometry; ignored (must not
        conflict) when ``area`` carries its own CRS.
    on_geographic : {"raise", "reproject"}
        What to do when the CRS is geographic (degrees). "raise" (default)
        refuses; "reproject" converts to the UTM zone estimated from the
        area's centre. A CRS that is projected but not in metres is always
        rejected.
    """
    if on_geographic not in ("raise", "reproject"):
        raise ValueError("on_geographic must be 'raise' or 'reproject'")
    if isinstance(area, (gpd.GeoDataFrame, gpd.GeoSeries)):
        series = area.geometry if isinstance(area, gpd.GeoDataFrame) else area
        own = series.crs
        if own is None and crs is None:
            raise ValueError("area has no CRS and no crs argument was given")
        if own is not None and crs is not None and CRS.from_user_input(crs) != own:
            raise ValueError("crs argument conflicts with the CRS of area")
        the_crs = own if own is not None else CRS.from_user_input(crs)
        geom = shapely.union_all(series.values)
    else:
        if crs is None:
            raise ValueError("crs is required when area is a bare shapely geometry")
        the_crs = CRS.from_user_input(crs)
        geom = area
    the_crs = CRS.from_user_input(the_crs)

    if the_crs.is_geographic:
        if on_geographic == "raise":
            raise ValueError(
                "area is in a geographic CRS (degrees); spacing needs metres. "
                "Reproject to a projected CRS (e.g. UTM) or pass on_geographic='reproject'."
            )
        ser = gpd.GeoSeries([geom], crs=the_crs)
        utm = ser.estimate_utm_crs()
        geom = ser.to_crs(utm).iloc[0]
        the_crs = CRS.from_user_input(utm)
    elif not _crs_is_metric(the_crs):
        raise ValueError(f"CRS {the_crs.to_string()} is not metre-based projected")

    if geom is None or geom.is_empty:
        raise ValueError("area geometry is empty")
    return geom, the_crs
