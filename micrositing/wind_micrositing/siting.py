"""Oriented-grid turbine siting inside a developable-area polygon."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import geopandas as gpd
import numpy as np
import shapely

from .crs import prepare_area
from .grid import oriented_grid
from .rasters import Raster

OUTPUT_DISCLAIMER = (
    "Screening-level estimate only. Turbine count and capacity come from a "
    "regular grid laid over a polygon plus simple point filters. There is no "
    "wake, wind-resource, noise, shadow-flicker, access-road or grid-connection "
    "analysis. Not a bankable energy yield or layout assessment."
)

# Arbitrary illustrative values so the examples run. They are NOT recommendations,
# have no cited source and are not tuned to any site: choose your own rotor
# diameter, spacing rule and rating.
EXAMPLE_DEFAULTS = {
    "rotor_diameter_m": 130.0,     # arbitrary example value
    "spacing_downwind_d": 5.0,     # common rule-of-thumb example (5D x 3D); choose your own
    "spacing_crosswind_d": 3.0,    # common rule-of-thumb example (5D x 3D); choose your own
    "edge_margin_d": 0.5,          # example: keep the rotor circle inside the area
    "search_step_deg": 10.0,       # example: coarser is faster, finer finds more
    "mw_per_turbine": 3.4,         # arbitrary example value
}

PointFilter = Callable[[np.ndarray, np.ndarray], np.ndarray]


def min_raster_filter(raster: Raster, minimum: float) -> PointFilter:
    """Keep points where the raster value is >= ``minimum`` (NaN/off-raster fails)."""
    return lambda x, y: raster.sample(x, y) >= minimum


def max_raster_filter(raster: Raster, maximum: float) -> PointFilter:
    """Keep points where the raster value is <= ``maximum`` (NaN/off-raster fails)."""
    return lambda x, y: raster.sample(x, y) <= maximum


def disk_fraction_filter(raster: Raster, radius_m: float, minimum: float) -> PointFilter:
    """Keep points whose surrounding disk has mean raster value >= ``minimum``."""
    return lambda x, y: raster.disk_fraction(x, y, radius_m) >= minimum


@dataclass
class SitingResult:
    x: np.ndarray
    y: np.ndarray
    crs: object
    azimuth_deg: float
    offset: tuple
    params: dict
    area_km2: float
    n_candidates_in_area: int
    search: list = field(default_factory=list)   # (azimuth, offset, count) tried
    disclaimer: str = OUTPUT_DISCLAIMER

    @property
    def n_turbines(self) -> int:
        return int(self.x.size)

    def capacity_mw(self, mw_per_turbine: float) -> float:
        return self.n_turbines * mw_per_turbine

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"turbine_id": np.arange(1, self.n_turbines + 1)},
            geometry=shapely.points(self.x, self.y), crs=self.crs)

    def summary(self, mw_per_turbine: float) -> dict:
        cap = self.capacity_mw(mw_per_turbine)
        return {
            "n_turbines": self.n_turbines,
            "mw_per_turbine": mw_per_turbine,
            "capacity_mw": cap,
            "area_km2": self.area_km2,
            "mw_per_km2": cap / self.area_km2 if self.area_km2 > 0 else None,
            "azimuth_deg": self.azimuth_deg,
            "params": self.params,
            "disclaimer": self.disclaimer,
        }


def _place(inner, bounds, dx, dy, az, offset, filters):
    gx, gy = oriented_grid(bounds, dx, dy, az, offset)
    keep = shapely.contains(inner, shapely.points(gx, gy))
    gx, gy = gx[keep], gy[keep]
    n_in = gx.size
    for f in filters:
        if gx.size == 0:
            break
        ok = np.asarray(f(gx, gy), dtype=bool)
        gx, gy = gx[ok], gy[ok]
    return gx, gy, n_in


def site_turbines(
    area,
    *,
    rotor_diameter_m: float,
    spacing_downwind_d: float,
    spacing_crosswind_d: float,
    azimuth_deg: float | None = None,
    search_step_deg: float | None = None,
    offset_steps: int = 1,
    edge_margin_m: float | None = None,
    filters: Sequence[PointFilter] = (),
    crs=None,
    on_geographic: str = "raise",
) -> SitingResult:
    """Place turbines on a regular oriented grid inside ``area``.

    Either fix the grid orientation with ``azimuth_deg`` or let the function
    search 0 <= azimuth < 180 in steps of ``search_step_deg`` and keep the
    orientation that yields most turbines (ties go to the smallest azimuth, so
    results are deterministic). ``offset_steps`` = n also tries an n x n set of
    lattice shifts per orientation. Given both, the fixed azimuth wins for the
    orientation but offsets are still searched.

    Spacings are multiples of the rotor diameter D: ``spacing_downwind_d`` along
    the grid axis, ``spacing_crosswind_d`` across it. ``edge_margin_m`` shrinks
    the area inward (default 0.5 D). ``filters`` are callables ``f(x, y) -> bool
    mask``; a turbine must pass all of them. See the raster filter helpers.

    Coordinates are metres: ``area`` must be in a metre-based projected CRS
    (see ``prepare_area`` for the geographic-CRS policy).
    """
    if rotor_diameter_m <= 0 or spacing_downwind_d <= 0 or spacing_crosswind_d <= 0:
        raise ValueError("rotor diameter and spacing multiples must be positive")
    if azimuth_deg is None and search_step_deg is None:
        raise ValueError("give azimuth_deg or search_step_deg")
    if search_step_deg is not None and search_step_deg <= 0:
        raise ValueError("search_step_deg must be positive")
    if offset_steps < 1:
        raise ValueError("offset_steps must be >= 1")

    geom, the_crs = prepare_area(area, crs=crs, on_geographic=on_geographic)
    dx = spacing_downwind_d * rotor_diameter_m
    dy = spacing_crosswind_d * rotor_diameter_m
    margin = 0.5 * rotor_diameter_m if edge_margin_m is None else edge_margin_m
    if margin < 0:
        raise ValueError("edge_margin_m must be >= 0")
    inner = geom.buffer(-margin) if margin > 0 else geom
    area_km2 = geom.area / 1e6
    params = {
        "rotor_diameter_m": rotor_diameter_m,
        "spacing_downwind_d": spacing_downwind_d,
        "spacing_crosswind_d": spacing_crosswind_d,
        "edge_margin_m": margin,
        "offset_steps": offset_steps,
        "azimuth_fixed": azimuth_deg is not None,
    }
    empty = SitingResult(np.array([]), np.array([]), the_crs,
                         float(azimuth_deg or 0.0), (0.0, 0.0), params, area_km2, 0)
    if inner.is_empty:
        return empty

    azimuths = ([float(azimuth_deg)] if azimuth_deg is not None
                else list(np.arange(0.0, 180.0, search_step_deg)))
    offsets = [(i / offset_steps, j / offset_steps)
               for i in range(offset_steps) for j in range(offset_steps)]
    bounds = inner.bounds
    best, tried = None, []
    for az in azimuths:
        for off in offsets:
            gx, gy, n_in = _place(inner, bounds, dx, dy, az, off, filters)
            tried.append((az, off, int(gx.size)))
            if best is None or gx.size > best[0].size:   # strict > keeps earliest tie
                best = (gx, gy, az, off, n_in)
    gx, gy, az, off, n_in = best
    return SitingResult(gx, gy, the_crs, float(az), off, params, area_km2, int(n_in), tried)


def estimate_capacity(n_turbines: int, mw_per_turbine: float, area_km2: float | None = None) -> dict:
    """Installed-capacity arithmetic with the screening disclaimer attached."""
    if n_turbines < 0 or mw_per_turbine <= 0:
        raise ValueError("n_turbines must be >= 0 and mw_per_turbine > 0")
    cap = n_turbines * mw_per_turbine
    out = {"n_turbines": n_turbines, "mw_per_turbine": mw_per_turbine, "capacity_mw": cap,
           "disclaimer": OUTPUT_DISCLAIMER}
    if area_km2:
        out["area_km2"] = area_km2
        out["mw_per_km2"] = cap / area_km2
    return out
