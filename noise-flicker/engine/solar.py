#!/usr/bin/env python3
"""Solar position - NOAA Solar Calculator algorithm, vectorised with numpy.

**Why not pvlib**: implemented here to avoid a pvlib dependency (wheels are not
guaranteed for every Python version). The NOAA algorithm is accurate to about 0.01 deg
for 1900-2100; the angular scales that matter for flicker are the rotor's subtended
angle (order 0.1 deg) and the 3 deg elevation threshold, so 0.01 deg is ample.
Checked against reference values in the test suite.

Angle conventions (local ENU frame throughout):
  azimuth is measured clockwise in degrees (0=N, 90=E, 180=S, 270=W)
  sun unit vector s = (sin A * cos g, cos A * cos g, sin g), i.e. ENU components
"""
from __future__ import annotations

import numpy as np

SUN_ANGULAR_DIAMETER_DEG = 0.53   # solar angular diameter, for the 'blade covers x % of the solar disc' rule


def julian_day(dt64: np.ndarray) -> np.ndarray:
    """numpy datetime64[s] (**UTC**) -> Julian day."""
    secs = dt64.astype("datetime64[s]").astype(np.int64)
    return secs / 86400.0 + 2440587.5


def position(dt64_utc: np.ndarray, lat_deg: float, lon_deg: float,
             *, refraction: bool = True) -> tuple:
    """Return (elevation_deg, azimuth_deg) as numpy arrays.

    lon_deg is positive east. refraction=True applies the atmospheric refraction
    correction to the apparent elevation (about 0.16 deg above the 3 deg flicker
    threshold - small, but cheap to include).
    """
    jd = julian_day(np.asarray(dt64_utc))
    t = (jd - 2451545.0) / 36525.0                      # Julian centuries

    L0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0        # geometric mean longitude
    M = 357.52911 + t * (35999.05029 - 0.0001537 * t)                   # mean anomaly
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)              # orbital eccentricity
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * t)
         + np.sin(3 * Mr) * 0.000289)                                   # equation of centre
    true_long = L0 + C
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega))  # apparent longitude

    eps0 = (23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813)))
                    / 60.0) / 60.0)                                     # mean obliquity of the ecliptic
    eps = eps0 + 0.00256 * np.cos(np.radians(omega))                    # corrected obliquity

    decl = np.degrees(np.arcsin(np.sin(np.radians(eps))
                                * np.sin(np.radians(app_long))))        # declination

    # equation of time (minutes)
    y = np.tan(np.radians(eps / 2.0)) ** 2
    L0r = np.radians(L0)
    eot = 4.0 * np.degrees(
        y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
        - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))

    # true solar time -> hour angle
    secs = np.asarray(dt64_utc).astype("datetime64[s]").astype(np.int64)
    minutes_utc = (secs % 86400) / 60.0
    true_solar_min = (minutes_utc + eot + 4.0 * lon_deg) % 1440.0
    ha = true_solar_min / 4.0 - 180.0                     # degrees, 0 at solar noon
    ha = np.where(ha < -180.0, ha + 360.0, ha)

    latr, declr, har = np.radians(lat_deg), np.radians(decl), np.radians(ha)
    cos_zen = (np.sin(latr) * np.sin(declr)
               + np.cos(latr) * np.cos(declr) * np.cos(har))
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zen = np.degrees(np.arccos(cos_zen))
    elev = 90.0 - zen

    if refraction:
        elev = elev + _refraction(elev)

    # azimuth: clockwise, 0 deg = N
    denom = np.cos(latr) * np.sin(np.radians(zen))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = (np.sin(latr) * cos_zen - np.sin(declr)) / denom
    cos_az = np.clip(np.where(np.isfinite(cos_az), cos_az, 0.0), -1.0, 1.0)
    acos_val = np.degrees(np.arccos(cos_az))
    # NOAA branches: morning az = 540 - acos, afternoon az = acos + 180 (both mod 360).
    # Writing `360 - acos` is a common mistake that puts the noon sun on the wrong side
    # of the sky; a regression test guards against it.
    az = np.where(har > 0.0, (acos_val + 180.0) % 360.0, (540.0 - acos_val) % 360.0)
    return elev, az % 360.0


def _refraction(elev_deg: np.ndarray) -> np.ndarray:
    """Atmospheric refraction correction (NOAA piecewise formula), degrees."""
    e = np.asarray(elev_deg, dtype=float)
    te = np.tan(np.radians(np.where(np.abs(e) < 1e-9, 1e-9, e)))
    r = np.zeros_like(e)
    m1 = e > 85.0
    m2 = (e > 5.0) & ~m1
    m3 = (e > -0.575) & ~m1 & ~m2
    m4 = ~m1 & ~m2 & ~m3
    r = np.where(m2, 58.1 / te - 0.07 / te**3 + 0.000086 / te**5, r)
    r = np.where(m3, 1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711))), r)
    r = np.where(m4, -20.774 / te, r)
    return r / 3600.0


def unit_vectors(elev_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    """(N,) elevation/azimuth -> (N,3) ENU unit vectors."""
    g, a = np.radians(elev_deg), np.radians(az_deg)
    cg = np.cos(g)
    return np.stack([np.sin(a) * cg, np.cos(a) * cg, np.sin(g)], axis=-1)
