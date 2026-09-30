"""Shared fixtures: synthetic DEMs written to temporary GeoTIFFs.

All coordinates are placed over open ocean (mid-Pacific); nothing here refers
to a real site. Elevations are synthetic surfaces.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

# Arbitrary open-ocean location; UTM zone 7 south (EPSG:32707) covers it.
LON0, LAT0 = -140.0, -20.0
UTM_EPSG = 32707
FINE_RES = 0.0004     # deg, about 44 m: finer than the 60 m RIX gate
COARSE_RES = 0.0012   # deg, about 133 m: coarser than the gate
N = 250               # pixels per side for fine grids (about 11 km)


def write_dem(path, elev, res=FINE_RES, lon0=LON0, lat_top=LAT0 + 0.2):
    transform = from_origin(lon0, lat_top, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", height=elev.shape[0], width=elev.shape[1],
        count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(elev.astype("float32"), 1)
    return str(path)


def surface(kind, n=N, res=FINE_RES):
    """Synthetic terrain surfaces, elevation in metres."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    px_m = res * 111320.0
    if kind == "flat":
        return 20.0 + 5.0 * np.sin(x / 40.0) * np.cos(y / 55.0)          # relief ~10 m
    if kind == "rolling":
        # long gentle swell: relief ~ 200 m, max slope well under the RIX critical slope
        wavelength_px = 4000.0 / px_m * 4
        return 150.0 + 100.0 * np.sin(2 * np.pi * x / wavelength_px) * np.cos(2 * np.pi * y / (1.3 * wavelength_px))
    if kind == "complex":
        # ridges: amplitude 150 m over 2.5 km wavelength (max slope ~ 20 deg)
        wavelength_px = 2500.0 / px_m
        return 300.0 + 150.0 * np.sin(2 * np.pi * x / wavelength_px) * np.sin(2 * np.pi * y / (0.8 * wavelength_px))
    if kind == "steep_small":
        # low relief (< 100 m) but short, steep bumps: 40 m amplitude over 400 m wavelength
        wavelength_px = 400.0 / px_m
        return 50.0 + 40.0 * np.sin(2 * np.pi * x / wavelength_px) * np.sin(2 * np.pi * y / wavelength_px)
    raise ValueError(kind)


def zone_polygon(frac=(0.2, 0.8), n=N, res=FINE_RES, lon0=LON0, lat_top=LAT0 + 0.2):
    lo, hi = frac
    return box(lon0 + lo * n * res, lat_top - hi * n * res,
               lon0 + hi * n * res, lat_top - lo * n * res)


@pytest.fixture
def dem_flat(tmp_path):
    return write_dem(tmp_path / "flat.tif", surface("flat"))


@pytest.fixture
def dem_rolling(tmp_path):
    return write_dem(tmp_path / "rolling.tif", surface("rolling"))


@pytest.fixture
def dem_complex(tmp_path):
    return write_dem(tmp_path / "complex.tif", surface("complex"))


@pytest.fixture
def dem_rolling_coarse(tmp_path):
    # same physical relief but sampled at ~133 m: RIX collapses, gate must engage
    n = int(N * FINE_RES / COARSE_RES)
    return write_dem(tmp_path / "rolling_coarse.tif", surface("rolling", n=n, res=COARSE_RES),
                     res=COARSE_RES)


@pytest.fixture
def zone():
    return zone_polygon()


@pytest.fixture
def windrose():
    """Synthetic 12-sector wind rose (not a measurement)."""
    freq = np.array([4, 5, 6, 8, 10, 14, 16, 12, 9, 7, 5, 4], dtype=float)
    return {
        "sector_freq_pct": list(freq / freq.sum() * 100),
        "sector_weibull_A": [7.0, 7.2, 7.5, 8.0, 8.5, 9.0, 9.2, 8.8, 8.2, 7.8, 7.4, 7.1],
        "sector_weibull_k": [2.0] * 12,
        "dominant_sector_deg": 180.0,
        "height_used_m": 100.0,
    }


@pytest.fixture
def dem_steep_small(tmp_path):
    return write_dem(tmp_path / "steep_small.tif", surface("steep_small"))
