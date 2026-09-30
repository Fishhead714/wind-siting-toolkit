"""Road graph.

Road centre-lines are converted into an undirected weighted graph (weight =
segment length in metres). Vertex coordinates are rounded to a ``snap_m`` grid,
so vertices that fall in the same grid cell share a node. Two vertices closer
than ``snap_m`` but on opposite sides of a grid-cell boundary do NOT merge.

Two lines that cross without a shared vertex are not connected unless
``node_intersections=True``, which splits lines at every crossing (every
crossing then becomes a junction). Check the connected components
(``graph_report``) on your own data.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
from shapely.ops import unary_union

from .crs import assert_not_degrees


def _round_pt(p, r):
    return (round(p[0] / r) * r, round(p[1] / r) * r)


def build_road_graph(lines: Iterable, snap_m: float = 1.0, node_intersections: bool = False) -> nx.Graph:
    """Build the road graph from centre-lines (metre CRS). Parallel duplicates keep the shorter weight."""
    lines = [g for g in lines if g is not None and not g.is_empty]
    if lines:
        assert_not_degrees(unary_union(lines).bounds, "road lines")
    if node_intersections and lines:
        lines = [unary_union(lines)]
    G = nx.Graph()
    for geom in lines:
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            coords = list(part.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                ra, rb = _round_pt(a[:2], snap_m), _round_pt(b[:2], snap_m)
                if ra == rb:
                    continue
                d = float(np.hypot(ra[0] - rb[0], ra[1] - rb[1]))
                if G.has_edge(ra, rb):
                    if G[ra][rb]["weight"] > d:
                        G[ra][rb]["weight"] = d
                else:
                    G.add_edge(ra, rb, weight=d)
    return G


def graph_report(G: nx.Graph) -> dict:
    """Node/edge counts and connected components, for sanity-checking the input."""
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "components": len(comps), "largest_component_nodes": comps[0] if comps else 0}
