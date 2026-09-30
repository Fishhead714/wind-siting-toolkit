"""Water-body avoidance barrier.

The barrier is the union of buffered waterway lines and water polygons. It is
used as a hard constraint on *new* construction: straight connectors and
off-road links may not intersect it. Modelling assumption: segments of the
input road graph are not tested against the barrier (they are taken as already
usable where they meet water); supply roads accordingly.
"""
from __future__ import annotations

from typing import Iterable, Optional

from shapely.geometry import LineString
from shapely.ops import unary_union

from .crs import assert_not_degrees


def build_water_barrier(lines: Iterable = (), polygons: Iterable = (), line_buffer_m: float = 12.0):
    """Union of waterway lines (buffered by ``line_buffer_m``) and water polygons.

    ``line_buffer_m`` is a channel-width proxy (example default 12 m); replace it
    with widths that fit your data. Returns ``None`` when there is no input.
    All geometry must be in a metre-based CRS.
    """
    geoms = [g.buffer(line_buffer_m) for g in lines] + list(polygons)
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    out = unary_union(geoms)
    assert_not_degrees(out.bounds, "water barrier")
    return out


def crosses_water(p1, p2, barrier: Optional[object]) -> bool:
    """True if the straight segment p1-p2 intersects the barrier."""
    if barrier is None:
        return False
    return LineString([p1, p2]).intersects(barrier)
