"""Sensitivity of the reuse ratio to the off-road cost multiplier ``M``.

The multiplier is a modelling assumption (relative cost of a metre of new
construction versus a metre of existing road). The reuse ratio it produces is
not a smooth function of ``M``: the network is a minimum spanning tree over
pairwise hybrid distances, and each pair switches between "straight off-road"
and "via road" at its own critical multiplier. Many switches close together
produce a step in the reuse ratio. Sweeping ``M`` densely on your own inputs shows where the steps are; a reuse
ratio is best quoted together with the range of ``M`` over which it was computed.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .network import HybridProblem

#: Example sweep grid (arbitrary log-spaced values; choose a grid that covers your own range).
EXAMPLE_MULTS = (1.2, 1.7, 2.4, 3.3, 4.6, 6.5, 9.1, 12.8, 18.0)


def sweep_offroad_mult(problem: HybridProblem, mults: Sequence[float] = EXAMPLE_MULTS) -> List[dict]:
    """Solve the network for each multiplier; returns one summary dict per value."""
    return [problem.solve(m).summary() for m in mults]


def detect_regime_changes(rows: Sequence[dict], jump_pp: float = 5.0) -> List[dict]:
    """Adjacent sweep points whose reuse ratio differs by at least ``jump_pp`` percentage points.

    Each hit is a bracket ``(m_lo, m_hi)`` in which the ratio changes by more than
    ``jump_pp``; the true step lies inside it (refine with a finer grid).
    """
    rows = sorted(rows, key=lambda r: r["offroad_mult"])
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        d = b["reuse_pct"] - a["reuse_pct"]
        if abs(d) >= jump_pp:
            out.append({"m_lo": a["offroad_mult"], "m_hi": b["offroad_mult"],
                        "reuse_lo_pct": a["reuse_pct"], "reuse_hi_pct": b["reuse_pct"],
                        "delta_pp": d})
    return out


def stability_band(rows: Sequence[dict], tol_pp: float = 3.0) -> List[dict]:
    """Maximal runs of consecutive sweep points whose reuse ratio stays within ``tol_pp`` of each other.

    Long runs are plateaus where the choice of ``M`` does not matter much;
    the boundaries between runs are the thresholds.
    """
    rows = sorted(rows, key=lambda r: r["offroad_mult"])
    bands, run = [], []
    for r in rows:
        vals = [x["reuse_pct"] for x in run] + [r["reuse_pct"]]
        if run and max(vals) - min(vals) > tol_pp:
            bands.append(_band(run))
            run = []
        run.append(r)
    if run:
        bands.append(_band(run))
    return bands


def _band(run):
    v = [x["reuse_pct"] for x in run]
    return {"m_lo": run[0]["offroad_mult"], "m_hi": run[-1]["offroad_mult"],
            "reuse_min_pct": min(v), "reuse_max_pct": max(v), "n_points": len(run)}


def critical_multiplier_histogram(problem: HybridProblem, bins: Sequence[float] = (0, 2, 4, 6, 8, 12, 20, 50, np.inf)):
    """Counts of node pairs by the multiplier at which they switch from off-road to road (see ``HybridProblem.critical_multipliers``)."""
    crit = problem.critical_multipliers()
    hist, _ = np.histogram(crit, bins=bins)
    return {"bins": list(bins), "counts": hist.tolist(), "n_pairs": int(len(crit))}
