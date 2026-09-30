"""Cost surface and least-cost path (e.g. turbine cluster -> substation or line tap).

The surface is a regular grid in a metre CRS. Cell cost = base cost, multiplied
or replaced by:

* roads:   cells within ``road_buffer_m`` of a road centre-line take ``road_cost``
           (cheaper than ``base_cost``: reuse / upgrade of an existing corridor)
* soft:    cells inside soft-constraint polygons are multiplied (avoid if possible)
* hard:    cells inside hard-constraint polygons (incl. the water barrier) get
           ``hard_cost`` (avoid unless nothing else connects)

All cost values are relative weights and **example defaults**, not calibrated
costs. ``hard_cost`` is finite, so a path can still cross a hard cell if that is
the only way; ``least_cost_path`` reports how many hard cells it crossed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union
from skimage.graph import route_through_array

from .crs import assert_not_degrees


@dataclass
class CostSurface:
    cost: np.ndarray          # (rows, cols); row 0 is the top (max y)
    x0: float                 # left edge
    y_top: float              # top edge
    cell: float
    hard_mask: np.ndarray
    road_mask: np.ndarray
    hard_geom: Optional[object] = None   # vector union of the hard geometries (unbuffered)

    def rc(self, x: float, y: float) -> Tuple[int, int]:
        r = int((self.y_top - y) // self.cell)
        c = int((x - self.x0) // self.cell)
        if not (0 <= r < self.cost.shape[0] and 0 <= c < self.cost.shape[1]):
            raise ValueError("point lies outside the cost surface")
        return r, c

    def xy(self, r: int, c: int) -> Tuple[float, float]:
        return self.x0 + (c + 0.5) * self.cell, self.y_top - (r + 0.5) * self.cell


def _mask(geom, xs, ys) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.zeros(xs.shape, dtype=bool)
    return shapely.contains_xy(geom, xs, ys)


def build_cost_surface(bounds: Sequence[float], cell: float, *, base_cost: float = 1.0,
                       roads: Iterable = (), road_cost: float = 0.47, road_buffer_m: float = 17.0,
                       hard: Iterable = (), hard_cost: float = 650.0,
                       soft: Iterable[Tuple[object, float]] = (),
                       terrain_mult: Optional[np.ndarray] = None) -> CostSurface:
    """Rasterise the constraints into a cost grid.

    ``bounds`` = (xmin, ymin, xmax, ymax) in metres. ``soft`` is an iterable of
    ``(geometry, multiplier)``. ``terrain_mult`` is an optional array of the grid's
    shape (e.g. derived from slope) that multiplies the base cost. Defaults are
    examples; calibrate them to your own cost ratios.

    ``hard`` geometries are rasterised *all-touched*: every cell whose square
    intersects a hard geometry is a hard cell, so a barrier narrower than a cell
    cannot slip between cell centres. Because paths run through cell centres and
    diagonal steps cut corners, ``least_cost_path`` additionally reports the
    length of the returned line that intersects the vector geometry.
    """
    xmin, ymin, xmax, ymax = bounds
    assert_not_degrees(tuple(bounds), "cost-surface bounds")
    if cell <= 0:
        raise ValueError("cell must be positive")
    cols = int(np.ceil((xmax - xmin) / cell))
    rows = int(np.ceil((ymax - ymin) / cell))
    xs = xmin + (np.arange(cols) + 0.5) * cell
    ys = ymax - (np.arange(rows) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    cost = np.full((rows, cols), float(base_cost))
    if terrain_mult is not None:
        if terrain_mult.shape != cost.shape:
            raise ValueError(f"terrain_mult shape {terrain_mult.shape} != grid {cost.shape}")
        cost *= terrain_mult
    for geom, mult in soft:
        cost = np.where(_mask(geom, X, Y), cost * mult, cost)
    road_geoms = [g.buffer(road_buffer_m) for g in roads if g is not None and not g.is_empty]
    road_mask = _mask(unary_union(road_geoms) if road_geoms else None, X, Y)
    cost = np.where(road_mask, np.minimum(cost, road_cost), cost)
    hard_geoms = [g for g in hard if g is not None and not g.is_empty]
    hard_geom = unary_union(hard_geoms) if hard_geoms else None
    if hard_geom is None:
        hard_mask = np.zeros(cost.shape, dtype=bool)
    else:
        shapely.prepare(hard_geom)
        h = cell / 2.0
        squares = shapely.box(X - h, Y - h, X + h, Y + h)
        hard_mask = shapely.intersects(hard_geom, squares)
    cost = np.where(hard_mask, hard_cost, cost)
    return CostSurface(cost, xmin, ymax, cell, hard_mask, road_mask, hard_geom)


def least_cost_path(surface: CostSurface, start: Tuple[float, float], end: Tuple[float, float]) -> dict:
    """8-connected least-cost path (scikit-image ``route_through_array``, geometric).

    Returns the path as a ``LineString`` (cell centres, metre CRS), the accumulated
    cost, its length, the share of path cells lying on road cells, the number of
    hard cells crossed and ``hard_vector_overlap_m`` (length of the line inside the
    unrasterised hard geometry). The path is grid-quantised; expect a jagged line and a
    length error of a few percent versus a smooth route.
    """
    s, e = surface.rc(*start), surface.rc(*end)
    idx, total = route_through_array(surface.cost, s, e, fully_connected=True, geometric=True)
    pts = [surface.xy(r, c) for r, c in idx]
    line = LineString(pts) if len(pts) > 1 else LineString([pts[0], pts[0]])
    on_road = float(np.mean([surface.road_mask[r, c] for r, c in idx]))
    hard = int(sum(surface.hard_mask[r, c] for r, c in idx))
    in_vec = 0.0
    if surface.hard_geom is not None:
        in_vec = float(line.intersection(surface.hard_geom).length)
    return {"line": line, "cost": float(total), "length_m": line.length,
            "road_cell_share": on_road, "hard_cells_crossed": hard,
            "hard_vector_overlap_m": in_vec}


def tap_point(point: Tuple[float, float], line) -> Tuple[float, float]:
    """Nearest point on an existing line (e.g. a transmission line) to ``point``: a candidate tap."""
    p = nearest_points(line, Point(point))[0]
    return p.x, p.y


def corridor_reuse_ratio(route, roads: Iterable, tol_m: float = 25.0) -> float:
    """Share (0-1) of ``route`` length within ``tol_m`` of an existing road/corridor centre-line."""
    road_geoms = [g for g in roads if g is not None and not g.is_empty]
    if not road_geoms or route.length == 0:
        return 0.0
    return float(route.intersection(unary_union(road_geoms).buffer(tol_m)).length / route.length)
