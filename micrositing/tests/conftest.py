"""Shared synthetic fixtures. Nothing here refers to a real place.

Coordinates sit in UTM zone 1 (EPSG:32601), whose central meridian is in open
ocean; the numbers are round on purpose.
"""
import geopandas as gpd
import pytest
from shapely.geometry import box

METRIC_EPSG = 32601
X0, Y0 = 500_000.0, 1_000_000.0


@pytest.fixture
def rect_gdf():
    """10 km x 6 km rectangle in a metric CRS."""
    return gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 10_000, Y0 + 6_000)], crs=METRIC_EPSG)


KW = dict(rotor_diameter_m=100.0, spacing_downwind_d=5.0, spacing_crosswind_d=3.0)
