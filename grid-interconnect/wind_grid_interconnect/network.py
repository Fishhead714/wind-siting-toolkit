"""Hybrid road-reuse / new-build network over a set of nodes.

Nodes are turbines plus one or more collection points (substation or line tap).
Each node is snapped to the road graph; the pairwise "hybrid distance" is the
cheaper of

* a straight off-road link:           ``euclid(i, j) * M``
* a link via the road graph:          ``(snap_i + snap_j) * M + road_path(i, j)``

where ``M`` is the *off-road cost multiplier* (cost of one metre of new
construction relative to one metre of using/upgrading an existing road).
A minimum spanning tree over the hybrid distances gives the network; the
length that runs on existing roads versus new construction gives the reuse
ratio.

``M`` is a modelling assumption. Use ``wind_grid_interconnect.sensitivity`` to
see how strongly the reuse ratio depends on it before quoting a number.

Straight off-road links and node connectors that intersect the water barrier are
infeasible. Modelling assumption: segments of the road graph are not tested
against the barrier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString
from shapely.ops import unary_union

from .crs import assert_not_degrees
from .water import crosses_water

_BIG = 1e12          # finite stand-in for "infeasible" so the MST stays well defined
_MIN_CONNECTOR_M = 0.5


@dataclass
class NetworkResult:
    offroad_mult: float
    existing_len_m: float
    new_len_m: float
    edges: List[tuple]
    existing_lines: list = field(repr=False)
    new_lines: list = field(repr=False)
    new_water_overlap_m: float = 0.0
    n_forced_snaps: int = 0
    n_direct_blocked: int = 0
    n_infeasible_edges: int = 0

    @property
    def total_len_m(self) -> float:
        return self.existing_len_m + self.new_len_m

    @property
    def reuse_pct(self) -> float:
        return 100.0 * self.existing_len_m / self.total_len_m if self.total_len_m else 0.0

    def summary(self) -> dict:
        return {"offroad_mult": self.offroad_mult, "total_km": self.total_len_m / 1000,
                "existing_km": self.existing_len_m / 1000, "new_km": self.new_len_m / 1000,
                "reuse_pct": self.reuse_pct, "new_water_overlap_m": self.new_water_overlap_m,
                "forced_snaps": self.n_forced_snaps, "direct_blocked": self.n_direct_blocked,
                "infeasible_edges": self.n_infeasible_edges}


class HybridProblem:
    """Everything that does not depend on ``M`` is computed once here.

    ``solve(M)`` is then cheap, which is what makes dense sensitivity sweeps practical.

    Parameters
    ----------
    points : (n, 2) array-like, node coordinates in a metre CRS.
    graph : road graph from ``roadgraph.build_road_graph`` (same CRS).
    barrier : water barrier geometry or None.
    k_snap : number of nearest road vertices tried when snapping a node.
        The first whose straight connector avoids the barrier is used; if none
        does, the nearest is used and the snap is flagged ``forced``.
    """

    def __init__(self, points: Sequence, graph: nx.Graph, barrier=None, k_snap: int = 60):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2:
            raise ValueError("points must be an (n>=2, 2) array of metre coordinates")
        if graph.number_of_nodes() == 0:
            raise ValueError("road graph is empty")
        for what, xy in (("points", self.points), ("road graph", np.array(list(graph.nodes()), dtype=float))):
            assert_not_degrees((*xy.min(axis=0), *xy.max(axis=0)), what)
        self.graph, self.barrier, self.k_snap = graph, barrier, k_snap
        n = len(self.points)
        self.n = n
        gnodes = np.array(list(graph.nodes()))
        self.snaps = []      # (snap_point, distance_m, forced)
        for k in range(n):
            d = np.hypot(*(gnodes - self.points[k]).T)
            order = np.argsort(d)[:k_snap]
            chosen = None
            for idx in order:
                sp = tuple(gnodes[idx])
                if not crosses_water(tuple(self.points[k]), sp, barrier):
                    chosen = (sp, float(d[idx]), False)
                    break
            if chosen is None:
                idx = order[0]
                chosen = (tuple(gnodes[idx]), float(d[idx]), True)
            self.snaps.append(chosen)
        self._sp_len, self._sp_path = [], []
        for k in range(n):
            lengths, paths = nx.single_source_dijkstra(graph, self.snaps[k][0], weight="weight")
            self._sp_len.append(lengths)
            self._sp_path.append(paths)
        self.euclid = np.hypot(*(self.points[:, None, :] - self.points[None, :, :]).transpose(2, 0, 1))
        self.direct_blocked = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                b = crosses_water(tuple(self.points[i]), tuple(self.points[j]), barrier)
                self.direct_blocked[i, j] = self.direct_blocked[j, i] = b
        self.road_len = np.full((n, n), np.inf)   # road-graph path length between snap points
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.road_len[i, j] = self._sp_len[i].get(self.snaps[j][0], np.inf)
        self.snap_dist = np.array([s[1] for s in self.snaps])

    # -- pairwise costs -------------------------------------------------------
    def pair_costs(self, offroad_mult: float):
        M = float(offroad_mult)
        direct = np.where(self.direct_blocked, np.inf, self.euclid * M)
        via = (self.snap_dist[:, None] + self.snap_dist[None, :]) * M + self.road_len
        return direct, via

    def critical_multipliers(self) -> np.ndarray:
        """Per unordered node pair: the multiplier above which the road route wins.

        ``direct <= via`` iff ``M * (D - s_i - s_j) <= R`` (D = straight distance,
        s = snap distances, R = road path length), so the pair flips at
        ``M* = R / (D - s_i - s_j)``. ``inf`` means the direct link always wins
        (D <= s_i + s_j); pairs with a blocked direct link or no road path never flip
        and are omitted. These flips are the elementary events behind any jump in
        the reuse ratio.
        """
        out = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.direct_blocked[i, j] or not np.isfinite(self.road_len[i, j]):
                    continue
                gap = self.euclid[i, j] - self.snap_dist[i] - self.snap_dist[j]
                out.append(np.inf if gap <= 0 else self.road_len[i, j] / gap)
        return np.array(out)

    # -- solve ---------------------------------------------------------------
    def solve(self, offroad_mult: float) -> NetworkResult:
        direct, via = self.pair_costs(offroad_mult)
        best = np.minimum(direct, via)
        best = np.where(np.isfinite(best), best, _BIG)
        np.fill_diagonal(best, 0.0)
        best = np.maximum(best, 1e-9) * (1 - np.eye(self.n))
        mst = minimum_spanning_tree(csr_matrix(best)).tocoo()
        edges = list(zip(mst.row.tolist(), mst.col.tolist()))
        if len(edges) != self.n - 1:
            raise RuntimeError(f"spanning tree has {len(edges)} edges, expected {self.n - 1}")
        existing, new = [], []
        n_inf = 0
        for i, j in edges:
            if best[i, j] >= _BIG:
                n_inf += 1
            use_road = via[i, j] <= direct[i, j] and np.isfinite(via[i, j])
            path = self._sp_path[i].get(self.snaps[j][0]) if use_road else None
            if not use_road or path is None:
                new.append(LineString([tuple(self.points[i]), tuple(self.points[j])]))
                continue
            si, sj = self.snaps[i], self.snaps[j]
            if si[1] > _MIN_CONNECTOR_M:
                new.append(LineString([tuple(self.points[i]), si[0]]))
            if sj[1] > _MIN_CONNECTOR_M:
                new.append(LineString([tuple(self.points[j]), sj[0]]))
            if len(path) >= 2:
                existing.append(LineString(path))
        ex_u = unary_union(existing) if existing else None
        new_u = unary_union(new) if new else None
        overlap = 0.0
        if new_u is not None and self.barrier is not None:
            overlap = new_u.intersection(self.barrier).length
        return NetworkResult(
            offroad_mult=float(offroad_mult),
            existing_len_m=ex_u.length if ex_u is not None else 0.0,
            new_len_m=new_u.length if new_u is not None else 0.0,
            edges=edges, existing_lines=existing, new_lines=new,
            new_water_overlap_m=overlap,
            n_forced_snaps=sum(1 for s in self.snaps if s[2]),
            n_direct_blocked=int(self.direct_blocked[np.triu_indices(self.n, 1)].sum()),
            n_infeasible_edges=n_inf)


def solve_network(points, graph, barrier=None, *, offroad_mult: float, k_snap: int = 60) -> NetworkResult:
    """One-shot convenience wrapper. ``offroad_mult`` is required: it is a modelling assumption, see README."""
    return HybridProblem(points, graph, barrier, k_snap).solve(offroad_mult)
