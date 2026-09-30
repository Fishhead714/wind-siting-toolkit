#!/usr/bin/env python3
"""Grid computation + contour export -> GPKG (input for noise / shadow-flicker maps).

**This module contains no physical model.** It does three things only:
  1. build a grid around the turbines and treat grid nodes as virtual receptors;
  2. evaluate them with the **existing functions** of noise.py / flicker.py (see "Reuse");
  3. extract contours (contourpy, the kernel behind matplotlib contour) and write a GPKG
     with a fixed schema.

## Reuse (why not call assess_* for every grid node)

Noise: with L_WA, hub height and meteorology fixed, the Lp of one turbine depends only on
  horizontal distance. So for each (L_WA, hub height) pair an Lp(d) table is built with
  `noise.lp_single` at **1 m steps**, grid nodes interpolate linearly in distance, and the
  whole array is summed **by energy** exactly as `noise.lp_at_point` does. Interpolation
  error is below 0.001 dB; the test suite compares against `noise.lp_at_point` point by
  point. Distances below 1 m are clamped to 1 m, as in lp_at_point.

Flicker: on flat terrain the flicker field of one turbine depends only on the receptor's
  offset from the turbine (the whole site uses one lat/lon, as `flicker.assess_flicker`
  itself does). So for each (hub height, rotor diameter) pair `flicker.assess_flicker` is
  called **unchanged** to compute a single-turbine kernel (hours/year on a relative offset
  grid, astronomical + expected), which is then shifted, bilinearly interpolated and summed
  per turbine onto the output grid.
  Note: **summing over turbines != the per-minute OR of assess_flicker**: two turbines
  shading the same point in the same minute are counted twice (conservative / too high).
  The same holds for the expected value (the engine takes the max probability over
  turbines, the grid sums). Therefore **receptor-layer values are always computed exactly
  with assess_flicker on the real receptors**; the grid is only used to draw lines, and the
  maximum grid-vs-exact deviation at receptors is written to the params table
  (`flicker_grid_vs_exact_*`).

## Limits of validity

Identical to the two engines (screening level, see noise.OUTPUT_DISCLAIMER /
flicker.OUTPUT_DISCLAIMER). The text is written verbatim into the params table; the
parameter box of any map should quote it unchanged.

## CRS

Inputs must be in a **projected CRS in metres**, turbines and receptors in the same CRS.
**No implicit reprojection is done**: layers._to_metric would auto-estimate a UTM zone for
geographic data, so this module rejects such input before calling layers - the contours
must overlay the caller's other layers, so the caller chooses the CRS, not us.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np   # noqa: E402

import criteria      # noqa: E402
import flicker       # noqa: E402
import noise         # noqa: E402

#: There are no built-in contour levels. Levels are the criterion limit from the criteria
#: registry (noise limit / flicker hours-per-year limit) plus any levels the user gives with
#: --noise-levels / --flicker-levels or --levels-file (see examples/contour_levels.example.json).


def resolve_levels(user_levels, criterion_level, what):
    """Union of the user's levels and the criterion level; error if both are missing."""
    lv = set(float(v) for v in (user_levels or []))
    if criterion_level is not None:
        lv.add(round(float(criterion_level), 1))
    if not lv:
        raise ValueError(
            f"no {what} contour levels: the criteria registry gives no limit for this "
            f"jurisdiction/period and none were supplied. Pass --{what}-levels "
            f"(comma-separated) or --levels-file (see examples/contour_levels.example.json).")
    return sorted(lv)

CASE_ASTRO = "astronomical_max"
CASE_EXPECTED = "expected"

CASE_DEFINITIONS = {
    CASE_ASTRO: "astronomical maximum (worst case): cloudless year + rotor always turning + "
                "rotor plane always facing the receptor (industry convention, threshold "
                "atan(R/L))",
    CASE_EXPECTED: "expected: nacelle orientation weighted by the wind rose, exact ray / "
                   "rotor-plane intersection per wind-direction bin, x sunshine probability "
                   "x operating probability",
}


# ======================================================================= CRS
def check_metric_crs(crs, what: str = "layer"):
    """Require a projected CRS in metres, otherwise raise. Returns a pyproj.CRS."""
    from pyproj import CRS
    if crs is None:
        raise ValueError(f"{what} has no CRS. Set one first - guessing a CRS is the most "
                         f"common source of silent errors.")
    c = CRS.from_user_input(crs)
    if c.is_geographic or not c.is_projected:
        raise ValueError(
            f"{what} has CRS {c.to_string()} (not projected / units are degrees). This module "
            f"does **no implicit reprojection**:\nreproject the data to the site's UTM zone "
            f"(or another metric projected CRS) first.")
    units = {(a.unit_name or "").lower() for a in c.axis_info[:2]}
    if not units <= {"metre", "meter", "m"}:
        raise ValueError(f"{what} CRS {c.to_string()} has units {units}, not metres.")
    return c


def _layer_crs(path, layer):
    import pyogrio
    return pyogrio.read_info(str(path), layer=layer)["crs"]


def load_inputs(turbine_path, turbine_layer, receptor_path=None, receptor_layer=None,
                **kw):
    """Read turbines (+ optional receptors), checking the CRS before loading.
    Returns (tn, tf, rn, rf, crs).

    kw as for layers.load_site (t_* for turbines, r_* for receptors).
    """
    import layers
    ct = check_metric_crs(_layer_crs(turbine_path, turbine_layer),
                          f"turbine layer {turbine_path}:{turbine_layer}")
    if receptor_path is None:
        t_kw = {k[2:]: v for k, v in kw.items() if k.startswith("t_")}
        tn, tf, crs = layers.turbines_from_gpkg(turbine_path, turbine_layer, **t_kw)
        return tn, tf, [], [], crs
    cr = check_metric_crs(_layer_crs(receptor_path, receptor_layer),
                          f"receptor layer {receptor_path}:{receptor_layer}")
    if ct != cr:
        raise ValueError(f"turbine layer CRS {ct.to_string()} != receptor layer CRS "
                         f"{cr.to_string()}. This module does no implicit reprojection; "
                         f"bring both into the same CRS first.")
    return layers.load_site(turbine_path, receptor_path, turbine_layer=turbine_layer,
                            receptor_layer=receptor_layer, **kw)


# ================================================================= grid tools
def _grid_axes(xmin, ymin, xmax, ymax, res):
    """Grid axes aligned to integer multiples of res (grid nodes do not drift between runs
    on the same site -> idempotent)."""
    x0, y0 = math.floor(xmin / res) * res, math.floor(ymin / res) * res
    x1, y1 = math.ceil(xmax / res) * res, math.ceil(ymax / res) * res
    return (np.arange(x0, x1 + res * 0.5, res), np.arange(y0, y1 + res * 0.5, res))


def _chaikin(c, n_iter):
    """Chaikin corner-cutting. Closed rings stay closed, open lines keep their end points.
    Each pass takes the 1/4 and 3/4 points of every edge; after n passes the maximum
    displacement from the original polyline is <= 1/4 of the longest edge. Grid contour
    edges are <= one cell diagonal, so the displacement is <= 0.36 cell."""
    closed = len(c) > 3 and np.allclose(c[0], c[-1])
    for _ in range(n_iter):
        if len(c) < 3:
            break
        p0, p1 = c[:-1], c[1:]
        q, r = 0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1
        mid = np.empty((2 * len(q), 2)); mid[0::2], mid[1::2] = q, r
        c = np.vstack([mid, mid[:1]]) if closed else np.vstack([c[:1], mid, c[-1:]])
    return c


def extract_lines(xs, ys, z, levels, smooth_iter=0):
    """[(level, shapely.LineString)]. NaN cells are treated as missing. smooth_iter>0 applies
    Chaikin smoothing."""
    import contourpy
    from shapely.geometry import LineString
    zz = np.ma.masked_invalid(z)
    gen = contourpy.contour_generator(x=xs, y=ys, z=zz,
                                      line_type=contourpy.LineType.Separate)
    out = []
    for lv in levels:
        for seg in gen.lines(float(lv)):
            if len(seg) >= 2:
                if smooth_iter:
                    seg = _chaikin(np.asarray(seg, dtype=float), smooth_iter)
                out.append((float(lv), LineString(seg)))
    return out


def extract_bands(xs, ys, z, levels, smooth_iter=0):
    """Filled bands [(lo, hi or None, shapely.Polygon)]: one polygon set between each pair of
    adjacent levels, hi=None above the top level.
    Banded fills with distinct hues are easier to read than light/dark steps of a single
    hue, which are hard to tell apart. Uses the same grid and the same Chaikin smoothing as
    extract_lines, so band edges coincide with the contour lines (the boundaries of
    contourpy's filled output are the contours of the same levels)."""
    import contourpy
    from shapely.geometry import Polygon
    zz = np.ma.masked_invalid(z)
    gen = contourpy.contour_generator(x=xs, y=ys, z=zz, fill_type=contourpy.FillType.OuterOffset)
    top = float(np.nanmax(z)) + 1.0 if np.isfinite(np.nanmax(z)) else None
    lv = sorted(float(v) for v in levels)
    out = []
    for i, lo in enumerate(lv):
        hi = lv[i + 1] if i + 1 < len(lv) else None
        upper = hi if hi is not None else top
        if upper is None or upper <= lo:
            continue
        points, offsets = gen.filled(lo, upper)
        for pts, offs in zip(points, offsets):
            rings = [np.asarray(pts[offs[k]:offs[k + 1]], dtype=float) for k in range(len(offs) - 1)]
            if smooth_iter:
                rings = [_chaikin(r, smooth_iter) for r in rings]
            rings = [r for r in rings if len(r) >= 4]
            if not rings:
                continue
            poly = Polygon(rings[0], rings[1:])
            if not poly.is_valid:
                poly = poly.buffer(0)
            for g in getattr(poly, "geoms", [poly]):
                if g.area > 0:
                    out.append((lo, hi, g))
    return out


# ================================================================== noise grid
def _lp_table(lwa, hub, atm, d_max):
    """Lp(d) table built with noise.lp_single unchanged (1 m steps). d<1 m is clamped to
    1 m, as in lp_at_point."""
    d = np.arange(0.0, d_max + 2.0, 1.0)
    lp = np.array([noise.lp_single(lwa, max(v, 1.0), h_src=hub, atm=atm) for v in d])
    return d, lp


def noise_grid(turbines, atm, *, min_level, res=25.0, buffer_m=None,
               bbox=None, max_buffer_m=15000.0):
    """Energy-summed Lp grid of the whole array. Returns (xs, ys, Z[dB(A)], info).

    Without buffer_m the buffer is chosen automatically: first the distance at which the
    loudest single turbine decays to (lowest level - 3 dB). Array summation raises the far
    field (in dense arrays the grid edge can remain above the lowest level), so while the
    edge maximum is >= the lowest level the buffer is enlarged x1.5 and the grid recomputed,
    up to max_buffer_m. Remaining truncation is reported in info.
    The 1000 m minimum buffer, the x1.5 growth factor and max_buffer_m are numerical
    grid-extent parameters, not physical criteria.
    """
    tx = np.array([t.x for t in turbines]); ty = np.array([t.y for t in turbines])
    auto = buffer_m is None and bbox is None
    if buffer_m is None:
        lw = max(t.lwa_dBA for t in turbines)
        h = min(t.hub_height_m for t in turbines)
        try:
            buffer_m = noise.required_distance(lw, min_level - 3.0, h_src=h, atm=atm)
        except ValueError:
            buffer_m = max_buffer_m
        buffer_m = float(min(max(buffer_m, 1000.0), max_buffer_m))
    tables = {}
    while True:
        bb = bbox or (tx.min() - buffer_m, ty.min() - buffer_m,
                      tx.max() + buffer_m, ty.max() + buffer_m)
        xs, ys = _grid_axes(*bb, res)
        X, Y = np.meshgrid(xs, ys)
        d_max = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) + 10.0
        E = np.zeros_like(X)
        for t in turbines:
            key = (t.lwa_dBA, t.hub_height_m)
            if key not in tables or tables[key][0][-1] < d_max:
                tables[key] = _lp_table(t.lwa_dBA, t.hub_height_m, atm, d_max)
            td, tl = tables[key]
            d = np.hypot(X - t.x, Y - t.y)
            E += 10.0 ** (np.interp(np.maximum(d, 1.0), td, tl) / 10.0)
        Z = 10.0 * np.log10(E)
        edge = max(Z[0].max(), Z[-1].max(), Z[:, 0].max(), Z[:, -1].max())
        if not auto or edge < min_level or buffer_m >= max_buffer_m:
            break
        buffer_m = min(buffer_m * 1.5, max_buffer_m)
    return xs, ys, Z, {"bbox": bb, "buffer_m": buffer_m, "res": res,
                       "shape": Z.shape, "edge_max_db": float(edge),
                       "n_tables": len(tables)}


# ================================================================== flicker grid
def _flicker_worker(args):
    """Worker process: call assess_flicker unchanged on a batch of receptors; return
    condensed rows + the model block."""
    tspec, rspec, kw = args
    T = [flicker.FlickerTurbine(x, y, h, d, i) for (x, y, h, d, i) in tspec]
    R = [flicker.FlickerReceptor(x, y, i, height_m=hm) for (x, y, i, hm) in rspec]
    # The (N,3)@(3,) products inside assess_flicker let the BLAS library grab every core;
    # even a single process becomes several times slower, and parallel workers fight over
    # cores. Limiting BLAS to 1 thread affects speed only, not values; without threadpoolctl
    # the code still runs, just slower.
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            out = flicker.assess_flicker(T, R, **kw)
    except ImportError:
        out = flicker.assess_flicker(T, R, **kw)
    rows = [(r["receptor_id"], r["astro_hours_per_year"], r["expected_hours_per_year"],
             r["status"], r["astro_max_minutes_per_day"]) for r in out["receptors"]]
    return rows, out["model"], out["criterion"], out["_disclaimer"]


def run_flicker(turbines, receptors, kw, *, workers=1, chunk=400):
    """Chunked parallel wrapper around assess_flicker (receptors are independent; results
    are bit-identical to a single process).

    Returns ({receptor_id: (astro_h, expected_h, status, max_min_day)}, model, criterion,
    disclaimer).
    """
    tspec = [(t.x, t.y, t.hub_height_m, t.rotor_diameter_m, t.id) for t in turbines]
    rspec = [(r.x, r.y, r.id, r.height_m) for r in receptors]
    jobs = [(tspec, rspec[i:i + chunk], kw) for i in range(0, len(rspec), chunk)]
    if workers <= 1 or len(jobs) == 1:
        res = [_flicker_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_flicker_worker, jobs))
    table = {}
    for rows, *_ in res:
        for rid, a, e, s, m in rows:
            table[rid] = (a, e, s, m)
    _, model, crit, disc = res[0]
    return table, model, crit, disc


def flicker_kernel(hub, rotor_d, kw, *, res, observer_height_m=1.5, workers=1):
    """Single-turbine flicker kernel: (astro, expected) hours/year on a relative offset grid.

    Offsets beyond d_max are set to 0 (assess_flicker skips them as well). Cells inside the
    rotor projection (INVALID_UNDER_ROTOR) are filled with the nearest valid cell, only to
    keep the interpolation continuous.
    """
    d_max = flicker.max_distance_m(kw["blade_chord_m"])
    K = int(math.ceil(d_max / res)) + 1
    off = np.arange(-K, K + 1) * res
    OX, OY = np.meshgrid(off, off)
    inside = np.hypot(OX, OY) <= d_max + res
    iy, ix = np.nonzero(inside)
    recs = [flicker.FlickerReceptor(OX[a, b], OY[a, b], f"{a}_{b}",
                                    height_m=observer_height_m)
            for a, b in zip(iy, ix)]
    T = [flicker.FlickerTurbine(0.0, 0.0, hub, rotor_d, "K")]
    table, model, crit, disc = run_flicker(T, recs, kw, workers=workers)
    A = np.zeros(OX.shape); E = np.zeros(OX.shape)
    has_exp = kw.get("wind_rose") is not None
    for a, b in zip(iy, ix):
        ah, eh, st, _ = table[f"{a}_{b}"]
        A[a, b] = np.nan if ah is None else ah
        E[a, b] = np.nan if (eh is None and st == "INVALID_UNDER_ROTOR") else (eh or 0.0)
    # Fill the rotor projection (INVALID) with the **nearest valid cell**, only to keep the
    # interpolation continuous. Filling with the kernel maximum (~1000 h) would let bilinear
    # interpolation carry that value into points within one cell outside the rotor, grossly
    # inflating the drawn value there.
    from scipy.ndimage import distance_transform_edt
    bad = np.isnan(A)
    if bad.any():
        _, (ny_, nx_) = distance_transform_edt(bad, return_indices=True)
        A[bad] = A[ny_[bad], nx_[bad]]
        E[bad] = E[ny_[bad], nx_[bad]]
    return {"K": K, "res": res, "astro": A, "expected": E if has_exp else None,
            "d_max": d_max, "n_points": int(inside.sum()), "model": model,
            "criterion": crit, "disclaimer": disc}


def flicker_grid(turbines, kw, *, res=20.0, observer_height_m=1.5, workers=1,
                 levels=None, refine=True, kernel_oversample=2):
    """Flicker field on the output grid. Returns (xs, ys, {case: Z}, info).

    Two steps:
      1. shift + bilinearly interpolate the single-turbine kernels and **sum** per turbine
         to get an upper bound S, recording the single-turbine maximum M and the number of
         contributing turbines C;
      2. **overlap refinement** (refine=True): the true multi-turbine value V always lies in
         [M, S] - astro is a per-minute OR, so max_i <= V <= sum_i; expected is a sum of
         per-minute max probabilities, likewise. Grid nodes whose interval [M, S] straddles
         any level (with a 10 % + 0.5 h margin) and with C >= 2 are recomputed with
         assess_flicker (all turbines) unchanged, replacing the grid value.
         Every other node either has a single contributing turbine (the sum is exact up to
         interpolation error) or has its whole interval between the same pair of adjacent
         levels (overlap changes the value but not the contour topology).
      Without step 2, dense arrays can be substantially overestimated and receptors drawn
      into the wrong band - simultaneous shading by several turbines is not a small effect.
      The 10 % + 0.5 h margin is a numerical tolerance, not a physical criterion.
    """
    from scipy.ndimage import map_coordinates
    kernels = {}
    for t in turbines:
        key = (t.hub_height_m, t.rotor_diameter_m)
        if key not in kernels:
            # Kernels are sampled at res/kernel_oversample: the flicker field has very steep
            # near-field edges (at low latitudes almost zero on the poleward side and
            # hundreds of hours to the east and west). With kernel and output grid at the
            # same resolution the value is interpolated twice and edges are smeared by about
            # one cell, visibly overstating low values next to a steep edge.
            kernels[key] = flicker_kernel(*key, kw, res=res / kernel_oversample,
                                          observer_height_m=observer_height_m,
                                          workers=workers)
    d_max = max(k["d_max"] for k in kernels.values())
    tx = np.array([t.x for t in turbines]); ty = np.array([t.y for t in turbines])
    xs, ys = _grid_axes(tx.min() - d_max - res, ty.min() - d_max - res,
                        tx.max() + d_max + res, ty.max() + d_max + res, res)
    cases = [CASE_ASTRO] + ([CASE_EXPECTED] if kw.get("wind_rose") is not None else [])
    shape = (len(ys), len(xs))
    Z = {c: np.zeros(shape) for c in cases}
    Mx = {c: np.zeros(shape) for c in cases}
    C = np.zeros(shape, dtype=np.int32)
    for t in turbines:
        k = kernels[(t.hub_height_m, t.rotor_diameter_m)]
        kres = k["res"]
        span = k["K"] * kres
        i0, i1 = np.searchsorted(xs, t.x - span), np.searchsorted(xs, t.x + span, "right")
        j0, j1 = np.searchsorted(ys, t.y - span), np.searchsorted(ys, t.y + span, "right")
        GX, GY = np.meshgrid(xs[i0:i1], ys[j0:j1])
        coords = [(GY - t.y) / kres + k["K"], (GX - t.x) / kres + k["K"]]
        for c in cases:
            src = k["astro"] if c == CASE_ASTRO else k["expected"]
            v = map_coordinates(src, coords, order=1, mode="constant", cval=0.0)
            Z[c][j0:j1, i0:i1] += v
            np.maximum(Mx[c][j0:j1, i0:i1], v, out=Mx[c][j0:j1, i0:i1])
            if c == CASE_ASTRO:
                C[j0:j1, i0:i1] += (v > 0)
    n_ref, t_ref, exact_mask = 0, 0.0, np.zeros(shape, dtype=bool)
    if refine and levels and len(turbines) > 1:
        ts = time.time()
        n_ref, exact_mask = _refine_overlap(turbines, kw, xs, ys, Z, Mx, C, levels,
                                observer_height_m, workers)
        t_ref = time.time() - ts
    k0 = next(iter(kernels.values()))
    return xs, ys, Z, {"res": res, "d_max": d_max, "n_kernels": len(kernels),
                       "kernel_points": sum(k["n_points"] for k in kernels.values()),
                       "n_refined_nodes": n_ref, "refine_s": round(t_ref, 2),
                       "exact_mask": exact_mask,
                       "n_nodes": int(np.prod(shape)),
                       "shape": shape, "model": k0["model"],
                       "criterion": k0["criterion"], "disclaimer": k0["disclaimer"]}


def _refine_overlap(turbines, kw, xs, ys, Z, Mx, C, levels, observer_height_m, workers):
    """Replace grid nodes whose interval [single-turbine max, sum] straddles a level with the
    exact assess_flicker value. Modifies Z in place.

    Two-stage speed-up (interpolated values are only kept where no contour passes; every
    node a contour passes through is computed exactly):
      a. first compute exactly the checkerboard half of the suspect nodes ((even, even)
         coarse nodes + (odd, odd) block centres);
      b. for the remaining suspect nodes: if the 4 cross neighbours (1 cell away, all either
         exact or with an interval that does not straddle) lie on the same side of **every**
         level, take the mean of the 4 neighbours; otherwise compute exactly.
    Returns (number of exactly computed nodes, exact mask).
    """
    cases = list(Z)
    shape = Z[cases[0]].shape
    amb = np.zeros(shape, dtype=bool)
    for c in cases:
        lo, hi = Mx[c] * 0.9 - 0.5, Z[c] * 1.1 + 0.5
        for lv in levels:
            amb |= (lo <= lv) & (hi >= lv)
    amb &= C >= 2
    done = np.zeros(shape, dtype=bool)

    def exact(mask):
        jj, ii = np.nonzero(mask)
        if not len(jj):
            return 0
        recs = [flicker.FlickerReceptor(xs[i], ys[j], f"{j}_{i}", height_m=observer_height_m)
                for j, i in zip(jj, ii)]
        table, *_ = run_flicker(turbines, recs, kw, workers=workers)
        for j, i in zip(jj, ii):
            a, e, st, _ = table[f"{j}_{i}"]
            if st != "INVALID_UNDER_ROTOR":       # inside the rotor projection: keep the upper bound (far above the top level)
                Z[CASE_ASTRO][j, i] = a
                if CASE_EXPECTED in Z:
                    Z[CASE_EXPECTED][j, i] = e
            done[j, i] = True
        return len(jj)

    # Stage 1: coarse nodes (even, even) + block centres (odd, odd) are all computed
    # exactly - about 1/4 of the suspect nodes each. Computing only the coarse nodes and
    # interpolating the rest of each block from 4 corners 2 cells away would erase small
    # flicker lobes narrower than 2 cells between the corners. With the block centres added,
    # every remaining node (edge midpoint) is 1 cell from its 4 cross neighbours, so the
    # detection limit drops below 1 cell (the limit of any grid).
    jg, ig = np.indices(shape)
    anchor = ((jg % 2) == (ig % 2))              # (even, even) and (odd, odd)
    n = exact(amb & anchor)
    ok = anchor & (done | ~amb)                  # exact, or interval does not straddle (same side as truth)
    side = np.stack([Z[c] >= lv for c in cases for lv in levels], axis=-1)
    rest = amb & ~anchor
    jj, ii = np.nonzero(rest)
    need = np.zeros(shape, dtype=bool)
    ny, nx = shape
    for j, i in zip(jj, ii):
        nb = [(j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)]   # cross neighbours, all anchors
        if any(a < 0 or b < 0 or a >= ny or b >= nx for a, b in nb) or \
                not all(ok[a, b] for a, b in nb) or \
                any((side[a, b] != side[nb[0]]).any() for a, b in nb[1:]):
            need[j, i] = True
            continue
        for c in cases:
            z = Z[c]
            z[j, i] = 0.25 * sum(z[a, b] for a, b in nb)
    n += exact(need)
    return n, done


def final_line_side_check(rows, pts, exact, levels, margin=lambda lv: max(2.0, 0.1 * lv)):
    """Point-in-polygon test on the **lines actually written** (after smoothing), compared
    with the exact receptor values.

    rows: flicker_isoline rows (hours_per_year/case/geometry); pts: {id: (x, y)};
    exact: {case: {id: exact value}}. Even-odd rule: a receptor enclosed an odd number of
    times by closed rings of one level is above that level.
    Returns a dict: wrong-side count (excluding receptors within margin of the level), total
    wrong-side count, number of open lines, examples.
    """
    import shapely
    from shapely.geometry import Polygon
    ids = list(pts)
    X = np.array([pts[i][0] for i in ids]); Y = np.array([pts[i][1] for i in ids])
    out = {"beyond_margin": 0, "all": 0, "n_open_lines": 0, "n_checked": 0, "examples": []}
    for case, ex in exact.items():
        for lv in levels:
            rings = [r["geometry"] for r in rows
                     if r["case"] == case and r["hours_per_year"] == lv]
            out["n_open_lines"] += sum(1 for g in rings if not g.is_closed)
            cnt = np.zeros(len(ids), dtype=int)
            for g in rings:
                if g.is_closed and len(g.coords) >= 4:
                    cnt += shapely.contains_xy(Polygon(g.coords), X, Y)
            inside = (cnt % 2) == 1
            for k, rid in enumerate(ids):
                v = ex.get(rid)
                if v is None:
                    continue
                out["n_checked"] += 1
                if inside[k] != (v >= lv):
                    out["all"] += 1
                    if abs(v - lv) > margin(lv):
                        out["beyond_margin"] += 1
                        out.setdefault("_mism", []).append((rid, case, lv, v,
                                                            "inside line" if inside[k]
                                                            else "outside line"))
    return out


def _bilinear(xs, ys, Z, x, y):
    from scipy.ndimage import map_coordinates
    res = xs[1] - xs[0]
    return float(map_coordinates(Z, [[(y - ys[0]) / res], [(x - xs[0]) / res]],
                                 order=1, mode="nearest")[0])


# ================================================================== main flow
def noise_criterion_level(country, period, background_la90):
    """Criterion limit dB(A), or (None, reason)."""
    try:
        lim, info = criteria.noise_limit(country, period=period,
                                         background_la90=background_la90,
                                         background_is_measured=False)
        return float(lim), info
    except (ValueError, NotImplementedError, KeyError) as e:
        return None, {"_skipped": str(e).splitlines()[0]}


def build(tn, tf, rn, rf, crs, *, country, atm=None, period="night",
          background_la90=None, noise_res=25.0, noise_levels=None,
          noise_buffer_m=None, wind_speed_ms=None,
          flicker_kw=None, flicker_res=20.0, flicker_levels=None,
          observer_height_m=1.5, workers=1, do_noise=True, do_flicker=True,
          wind_rose_source=None, inputs_desc=None, flicker_smooth_iter=2):
    """Compute every layer. Returns dict(noise_contour, flicker_isoline, flicker_band,
    receptor, params) (GeoDataFrame/DataFrame).

    ``country`` is the jurisdiction key in the criteria registry.
    flicker_kw: passed unchanged to flicker.assess_flicker (lat_deg/lon_deg/utc_offset_hours/
    blade_chord_m/blade_chord_source/wind_rose/sunshine_probability/... without country).
    """
    import geopandas as gpd
    import pandas as pd
    t0 = time.time()
    crs = check_metric_crs(crs, "input data")
    params = []

    def P(k, v):
        params.append({"key": k, "value": v if isinstance(v, str)
                       else json.dumps(v, ensure_ascii=False, default=str)})

    P("generated_at", _dt.datetime.now().isoformat(timespec="seconds"))
    P("engine", "noise_flicker contours.py (reuses noise.py / flicker.py)")
    P("crs", crs.to_string())
    P("country", country)
    if inputs_desc:
        for k, v in inputs_desc.items():
            P(f"input_{k}", v)
    P("observer_height_m", observer_height_m)
    P("n_turbines", len(tn or tf))

    # Receptor results are stored in a dict keyed by id and layers does not check
    # uniqueness. Duplicate ids would merge silently - the point takes the last entry and
    # flicker_h may come from another building, putting exceeds=0 at the wrong place.
    # No automatic renaming or de-duplication: the id is the key back to the source data.
    # Turbine ids are not checked: this module never joins results back by turbine id, and
    # real layouts with duplicate or garbled turbine ids would otherwise be blocked.
    for tag, objs in (("receptor", rn), ("receptor", rf)):
        seen, dup = set(), []
        for o in objs or []:
            (dup.append(o.id) if o.id in seen else seen.add(o.id))
        if dup:
            raise ValueError(f"{tag} ids are not unique: {sorted(set(dup))[:20]}"
                             f"{' ...' if len(set(dup)) > 20 else ''} ({len(set(dup))} "
                             f"duplicated). Results are joined back by id, so duplicates "
                             f"would silently swap values. Use a unique field (--r-id-field).")
    rec = {r.id: {"receptor_id": r.id, "x": r.x, "y": r.y, "noise_db": None,
                  "flicker_h": None, "n_ex": None, "f_ex": None, "invalid": False}
           for r in (rn or rf)}
    noise_rows, flick_rows, band_rows = [], [], []

    # ---------------------------------------------------------------- noise
    if do_noise:
        if atm is None:
            raise ValueError("noise needs atm=noise.Atmosphere(...) (site night-time "
                             "meteorology and spectrum; deliberately no default)")
        lim, lim_info = noise_criterion_level(country, period, background_la90)
        levels = resolve_levels(noise_levels, lim, "noise")
        ts = time.time()
        xs, ys, Z, ninfo = noise_grid(tn, atm, res=noise_res, buffer_m=noise_buffer_m,
                                      min_level=min(levels))
        for lv, g in extract_lines(xs, ys, Z, levels):
            noise_rows.append({"level_db": lv, "metric": "LAeq",
                               "wind_speed_ms": wind_speed_ms, "geometry": g})
        tn_s = time.time() - ts
        P("noise_model", "ISO 9613-2 downwind + ISO 9613-1 atmospheric absorption; A_bar=0; "
                         "D_c=0; grid nodes = virtual receptors, whole-array energy sum "
                         "(as noise.lp_at_point)")
        P("noise_metric", "LAeq (ISO 9613-2 downwind specific level, i.e. the turbines' own "
                          "level; for increment-over-background criteria this is the level "
                          "compared with background + allowance)")
        P("noise_lwa_dBA", sorted({t.lwa_dBA for t in tn}))
        P("noise_hub_height_m", sorted({t.hub_height_m for t in tn}))
        P("noise_wind_speed_ms", wind_speed_ms)
        P("noise_spectrum", atm.spectrum)
        P("noise_spectrum_shape_dB", {str(k): v for k, v in noise.SPECTRA[atm.spectrum].items()})
        P("noise_atmosphere", atm.as_dict())
        P("noise_period", period)
        P("noise_background_la90_dBA", background_la90)
        P("noise_background_is_measured", False if background_la90 is not None else None)
        P("noise_criterion_limit_dBA", lim)
        P("noise_criterion_info", {k: v for k, v in lim_info.items()
                                   if k in ("mode", "rule", "resolved", "allowance_dB",
                                            "limit_dBA", "enforceability",
                                            "verified_date", "registry", "_skipped")})
        P("noise_levels_db", levels)
        P("noise_levels_rule", "criterion limit (criteria registry, per jurisdiction / "
          "period) plus user-supplied levels" + ("" if noise_levels else " (none supplied)"))
        P("noise_grid_res_m", noise_res)
        P("noise_study_extent", {"bbox": [round(v, 1) for v in ninfo["bbox"]],
                                 "buffer_m": round(ninfo["buffer_m"], 1),
                                 "shape": list(ninfo["shape"])})
        P("noise_grid_edge_max_db", round(ninfo["edge_max_db"], 2))
        if ninfo["edge_max_db"] >= min(levels):
            P("noise_WARNING_truncated", f"grid edge maximum {ninfo['edge_max_db']:.1f} dB >= "
              f"lowest level {min(levels)}; that contour is truncated at the edge of the study "
              f"extent (not closed). Increase the buffer to fix.")
        P("noise_disclaimer", noise.OUTPUT_DISCLAIMER)
        P("noise_compute_s", round(tn_s, 2))
        # Receptors: exact values via assess_receptors / lp_at_point, not grid interpolation
        if rn:
            # Skipping assess_receptors whenever there is no site-wide background value
            # (lim=None) would drop per-receptor measured LA90 values and leave every noise
            # exceedance empty. Instead: for jurisdictions that need a background level,
            # every receptor that has one (its own measured value or the site-wide assumed
            # value) is judged against **its own limit**; receptors with neither get
            # noise_db only, without a judgement.
            try:
                needs_bg = criteria.needs_background(country)
            except (KeyError, NotImplementedError):
                needs_bg, lim = True, None
            if not needs_bg:
                judged = rn if lim is not None else []
            else:
                judged = [r for r in rn if r.background_la90_dBA is not None
                          or background_la90 is not None]
            if judged:
                out = noise.assess_receptors(tn, judged, atm, country=country, period=period,
                                             default_background_la90=background_la90)
                for r in out["receptors"]:
                    rec[r["receptor_id"]]["noise_db"] = r["Lp_total_dBA"]
                    rec[r["receptor_id"]]["n_ex"] = r["status"] == "EXCEED"
            jid = {r.id for r in judged}
            for r in rn:
                if r.id not in jid:
                    rec[r.id]["noise_db"] = round(noise.lp_at_point(tn, r.x, r.y, atm)[0], 1)
            n_meas = sum(1 for r in rn if r.background_la90_dBA is not None)
            P("noise_receptor_judgement", {
                "n_judged": len(judged), "n_not_judged": len(rn) - len(judged),
                "n_with_measured_LA90": n_meas})
            if needs_bg and n_meas:
                P("noise_limit_line_note",
                  (f"the criterion contour {lim} dB(A) uses the site-wide background "
                   f"{background_la90} (assumed); receptor exceedance is judged against each "
                   f"receptor's own measured background ({n_meas} receptors have measured "
                   f"values), so a receptor outside the criterion line may have exceeds=1, "
                   f"or one inside it exceeds=0.")
                  if lim is not None else
                  "no site-wide background given -> **no criterion-limit contour is drawn** "
                  "(per-receptor limits differ, so no single line exists); receptor "
                  f"exceedance is judged against each receptor's own measured background "
                  f"({n_meas} receptors have measured values).")
            dev = max(abs(_bilinear(xs, ys, Z, r.x, r.y) - rec[r.id]["noise_db"]) for r in rn
                      if xs[0] <= r.x <= xs[-1] and ys[0] <= r.y <= ys[-1]) \
                if any(xs[0] <= r.x <= xs[-1] and ys[0] <= r.y <= ys[-1] for r in rn) else None
            P("noise_grid_vs_exact_max_abs_db", None if dev is None else round(dev, 2))

    # ---------------------------------------------------------------- flicker
    if do_flicker:
        if not flicker_kw or "blade_chord_m" not in flicker_kw:
            raise ValueError("flicker needs flicker_kw (at least lat_deg/lon_deg/"
                             "utc_offset_hours/blade_chord_m). The blade chord deliberately "
                             "has no default, see flicker.assess_flicker.")
        kw = {k: v for k, v in flicker_kw.items() if not k.startswith("_")}
        kw["country"] = country
        flim, _, _ = criteria.flicker_limits(country)
        levels = resolve_levels(flicker_levels, flim, "flicker")
        ts = time.time()
        xs, ys, Z, finfo = flicker_grid(tf, kw, res=flicker_res,
                                        observer_height_m=observer_height_m, workers=workers,
                                        levels=levels)
        for case, zz in Z.items():
            for lv, g in extract_lines(xs, ys, zz, levels, smooth_iter=flicker_smooth_iter):
                flick_rows.append({"hours_per_year": lv, "case": case, "geometry": g})
            for lo, hi, g in extract_bands(xs, ys, zz, levels, smooth_iter=flicker_smooth_iter):
                band_rows.append({"hours_min": lo, "hours_max": hi, "case": case, "geometry": g})
        tf_s = time.time() - ts
        m = finfo["model"]
        P("flicker_model", "exact ray / rotor-plane intersection + NOAA solar position "
                           "(flicker.assess_flicker); grid = shifted single-turbine kernels "
                           "summed per turbine")
        P("flicker_cases", {c: CASE_DEFINITIONS[c] for c in Z})
        if CASE_EXPECTED not in Z:
            P("flicker_expected_skipped", "no wind rose given -> astronomical maximum only "
                                          "(a default wind rose is deliberately not invented)")
        P("flicker_rotor_diameter_m", sorted({t.rotor_diameter_m for t in tf}))
        P("flicker_hub_height_m", sorted({t.hub_height_m for t in tf}))
        P("flicker_min_sun_elevation_deg", m["min_sun_elevation_deg"])
        P("flicker_blade_chord_m", m["blade_chord_m"])
        P("flicker_blade_chord_source", m["blade_chord_source"])
        P("flicker_max_distance_m", m["max_distance_m"])
        P("flicker_max_distance_basis", m["max_distance_basis"])
        for k in ("lat_deg", "lon_deg", "utc_offset_hours", "year", "step_minutes",
                  "sunshine_probability", "operating_probability"):
            P(f"flicker_{k}", m.get(k))
        P("flicker_wind_rose", flicker_kw.get("wind_rose"))
        P("flicker_wind_rose_source", wind_rose_source)
        P("flicker_criterion", {k: finfo["criterion"].get(k) for k in
                                ("limit_hours_per_year", "limit_minutes_per_day",
                                 "is_hard_veto", "enforceability", "verified_date")})
        P("flicker_levels_h", levels)
        P("flicker_levels_rule", "criterion hours-per-year limit (criteria registry) plus "
          "user-supplied levels" + ("" if flicker_levels else " (none supplied)"))
        P("flicker_grid_res_m", flicker_res)
        P("flicker_kernel_res_m", flicker_res / 2)
        P("flicker_line_smoothing", f"Chaikin corner-cutting, {flicker_smooth_iter} passes "
          f"(line shape only, displacement from the raw polyline <= 0.36 x {flicker_res:g} m; "
          f"no grid value is changed)"
          if flicker_smooth_iter else "none")
        P("flicker_study_extent", {"bbox": [round(float(xs[0]), 1), round(float(ys[0]), 1),
                                            round(float(xs[-1]), 1), round(float(ys[-1]), 1)],
                                   "buffer_m": round(finfo["d_max"] + flicker_res, 1),
                                   "shape": list(finfo["shape"])})
        P("flicker_grid_method",
          "single-turbine kernels (computed with assess_flicker unchanged), shifted, "
          "bilinearly interpolated and summed per turbine to an upper bound; nodes whose "
          "bounds [single-turbine max, sum] straddle a level with >= 2 contributing turbines "
          "are recomputed with assess_flicker over all turbines (per-minute OR, as in the "
          "engine). Remaining nodes carry interpolation error only. Receptor-layer values are "
          "exact.")
        P("flicker_refined_nodes", {"n": finfo["n_refined_nodes"], "of": finfo["n_nodes"],
                                    "seconds": finfo["refine_s"]})
        P("flicker_disclaimer", finfo["disclaimer"])
        if rf:
            table, _, _, _ = run_flicker(tf, rf, kw, workers=workers)
            devs, mism = [], []
            for r in rf:
                a, e, st, _m = table[r.id]
                rec[r.id]["flicker_h"] = a
                if st == "INVALID_UNDER_ROTOR":
                    rec[r.id]["invalid"] = True
                rec[r.id]["f_ex"] = {"EXCEED": True, "OK": False}.get(st)
                if a is None or not (xs[0] <= r.x <= xs[-1] and ys[0] <= r.y <= ys[-1]):
                    continue
                devs.append((_bilinear(xs, ys, Z[CASE_ASTRO], r.x, r.y) - a, a))
                for case, ex_v in ((CASE_ASTRO, a), (CASE_EXPECTED, e)):
                    if case not in Z or ex_v is None:
                        continue
                    g = _bilinear(xs, ys, Z[case], r.x, r.y)
                    for lv in levels:
                        if (g >= lv) != (ex_v >= lv) and abs(ex_v - lv) > max(2.0, 0.1 * lv):
                            mism.append((r.id, case, lv, ex_v, round(g, 1)))
            # Receptors that appear to be on the wrong side: recompute exactly on the 8
            # neighbours one cell away. If the exact values within one cell already straddle
            # the level, the line is less than one cell from the receptor - a positioning
            # error inherent to any gridded contour, not a fault of this module. Only
            # otherwise is it a genuine wrong side.
            fl = final_line_side_check(
                flick_rows, {r.id: (r.x, r.y) for r in rf},
                {c: {r.id: table[r.id][0 if c == CASE_ASTRO else 1] for r in rf
                     if table[r.id][2] != "INVALID_UNDER_ROTOR"} for c in Z},
                levels)
            fmism = fl.pop("_mism", [])
            genuine, within, fgen, fwith = [], [], [], []
            if mism or fmism:
                probe = {}
                rpos = {r.id: r for r in rf}
                for rid in {m_[0] for m_ in mism + fmism}:
                    r0 = rpos[rid]
                    for k_, (dx, dy) in enumerate([(a_, b_) for a_ in (-1, 0, 1)
                                                   for b_ in (-1, 0, 1) if a_ or b_]):
                        probe[f"{rid}#{k_}"] = flicker.FlickerReceptor(
                            r0.x + dx * flicker_res, r0.y + dy * flicker_res, f"{rid}#{k_}",
                            height_m=observer_height_m)
                ptab, *_ = run_flicker(tf, list(probe.values()), kw, workers=workers)
                for src, g_, w_ in ((mism, genuine, within), (fmism, fgen, fwith)):
                    for m_ in src:
                        rid, case, lvv = m_[:3]
                        ci = 0 if case == CASE_ASTRO else 1
                        vals = [ptab[f"{rid}#{k_}"][ci] for k_ in range(8)] + \
                               [table[rid][ci]]
                        vals = [v for v in vals if v is not None]
                        (w_ if (min(vals) < lvv <= max(vals)) else g_).append(m_)
            fl["genuine"] = len(fgen)
            fl["within_one_cell"] = len(fwith)
            fl["examples"] = [f"{a_}:{b_}:{c_:g}h exact={d_} {e_}"
                              for a_, b_, c_, d_, e_ in (fgen or fwith)[:5]]
            P("flicker_final_line_side_check", {
                **fl, "_note": "point-in-polygon test (even-odd rule) on the **written "
                               "(smoothed) closed lines**, compared with exact receptor "
                               "values. beyond_margin = receptor/level pairs more than "
                               "max(2 h, 10%) (a numerical tolerance) from the level yet on the wrong side of the "
                               "line; all includes those close to the level. With "
                               "n_open_lines > 0 open lines cannot be tested for inside/"
                               "outside and that level's result is incomplete. beyond_margin "
                               "is further split by an exact check of the 8 neighbours one "
                               "cell away into within_one_cell (exact values within one cell "
                               "already straddle the level: the line is < 1 cell from the "
                               "receptor, an inherent grid positioning error) and genuine "
                               "(should be 0)."})
            if devs:
                P("flicker_grid_vs_exact", {
                    "side_mismatch_genuine": len(genuine),
                    "side_mismatch_within_one_cell": len(within),
                    "side_mismatch_examples": [
                        f"{a_}:{b_}:{c_:g}h exact={d_} grid={e_}"
                        for a_, b_, c_, d_, e_ in (genuine or within)[:5]],
                    "astro_dev_h_all": {"max_over": round(max(d for d, _ in devs), 2),
                                        "max_under": round(min(d for d, _ in devs), 2),
                                        "n": len(devs)},
                    "_note": "grid interpolation minus exact value at receptors. **Contours "
                             "only depend on which side of a level a value falls**: "
                             "side_mismatch = receptor more than max(2 h, 10%) (a numerical tolerance) from the level "
                             "yet drawn on the other side of the line by the grid; "
                             "within_one_cell = exact values within one cell of the receptor "
                             "already straddle the level (line < 1 cell away, inherent grid "
                             "positioning error); genuine should be 0. Numeric deviations can "
                             "be large - nodes whose interval [single-turbine max, sum] "
                             "straddles no level are not recomputed and keep the summed upper "
                             "bound (simultaneous shading counted twice), which changes no "
                             "line."})
        P("flicker_compute_s", round(tf_s, 2))
        P("flicker_kernel_points", finfo["kernel_points"])

    # ---------------------------------------------------------------- receptors
    rrows = []
    for v in rec.values():
        flags = [f for f in (v["n_ex"], v["f_ex"]) if f is not None]
        ex = None if (v["invalid"] or not flags) else int(any(flags))
        from shapely.geometry import Point
        rrows.append({"receptor_id": str(v["receptor_id"]),
                      "noise_db": v["noise_db"], "flicker_h": v["flicker_h"],
                      "exceeds": ex, "geometry": Point(v["x"], v["y"])})
    P("receptor_fields", {
        "noise_db": "LAeq dB(A), exact value from assess_receptors/lp_at_point (not grid "
                    "interpolation)",
        "flicker_h": "astronomical maximum h/year (exact assess_flicker value); empty inside "
                     "a rotor projection",
        "exceeds": "1 = at least one **decidable** criterion exceeded (noise limit / flicker "
                   "hours or minutes threshold from the registry); 0 = no decidable criterion "
                   "exceeded; empty = no decidable criterion (missing background LA90, no "
                   "numeric flicker threshold) or receptor inside a rotor projection"})
    P("total_compute_s", round(time.time() - t0, 2))

    def gdf(rows, cols, geom):
        if rows:
            return gpd.GeoDataFrame(rows, columns=cols + ["geometry"], geometry="geometry", crs=crs)
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)

    out = {
        "noise_contour": gdf(noise_rows, ["level_db", "metric", "wind_speed_ms"], "LineString"),
        "flicker_isoline": gdf(flick_rows, ["hours_per_year", "case"], "LineString"),
        "flicker_band": gdf(band_rows, ["hours_min", "hours_max", "case"], "Polygon"),
        "receptor": gdf(rrows, ["receptor_id", "noise_db", "flicker_h", "exceeds"], "Point"),
        "params": pd.DataFrame(params, columns=["key", "value"]),
    }
    # Fixed dtypes (empty values must still have the right type: map renderers build
    # categories from the schema)
    out["noise_contour"] = out["noise_contour"].astype(
        {"level_db": "float64", "metric": "object", "wind_speed_ms": "float64"})
    out["flicker_isoline"] = out["flicker_isoline"].astype(
        {"hours_per_year": "float64", "case": "object"})
    out["flicker_band"] = out["flicker_band"].astype(
        {"hours_min": "float64", "hours_max": "float64", "case": "object"})
    out["receptor"] = out["receptor"].astype(
        {"receptor_id": "object", "noise_db": "float64", "flicker_h": "float64",
         "exceeds": "Int64"})
    return out


GEOM_TYPES = {"noise_contour": "LineString", "flicker_isoline": "LineString",
              "flicker_band": "Polygon", "receptor": "Point"}


def write_gpkg(layers_out: dict, path) -> Path:
    """Rewrite the whole file (idempotent: a re-run yields the same set of layers, no
    duplicated features)."""
    import pyogrio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for name, gt in GEOM_TYPES.items():
        pyogrio.write_dataframe(layers_out[name], str(path), layer=name, driver="GPKG",
                                geometry_type=gt, promote_to_multi=False)
    pyogrio.write_dataframe(layers_out["params"], str(path), layer="params", driver="GPKG")
    return path


# ======================================================================= CLI
def _floats(s):
    return [float(v) for v in s.split(",") if v.strip()] if s else None


def _rose(s):
    if not s:
        return None
    p = Path(s)
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        return {float(k): float(v) for k, v in d.items()}
    return {float(a): float(b) for a, b in (kv.split(":") for kv in s.split(","))}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Noise / shadow-flicker grids + contours -> GPKG (noise_contour / "
                    "flicker_isoline / flicker_band / receptor / params). Inputs must be in a "
                    "projected CRS (metres); no implicit reprojection is done.")
    g = ap.add_argument_group("inputs")
    g.add_argument("--turbines", required=True); g.add_argument("--turbine-layer")
    g.add_argument("--t-id-field"); g.add_argument("--t-where")
    g.add_argument("--receptors"); g.add_argument("--receptor-layer")
    g.add_argument("--r-id-field"); g.add_argument("--r-where")
    g.add_argument("--r-la90-field",
                   help="per-receptor measured background LA90 field (for criteria that "
                        "depend on background noise)")
    m = ap.add_argument_group("turbine (scalar or field, one of the two; deliberately no default)")
    m.add_argument("--hub-height-m", type=float); m.add_argument("--hub-field")
    m.add_argument("--rotor-diameter-m", type=float); m.add_argument("--rotor-field")
    m.add_argument("--lwa-dba", type=float); m.add_argument("--lwa-field")
    m.add_argument("--wind-speed-ms", type=float,
                   help="wind speed the L_WA refers to (written to the wind_speed_ms field "
                        "only, not used in the calculation)")
    c = ap.add_argument_group("criteria / noise")
    c.add_argument("--criteria",
                   help="criteria registry JSON (otherwise the WTG_NOISE_FLICKER_CRITERIA "
                        "environment variable is used)")
    c.add_argument("--country", required=True,
                   help="jurisdiction key in the criteria registry")
    c.add_argument("--period", default="night", choices=["night", "daytime"])
    c.add_argument("--background-la90", type=float,
                   help="assumed background LA90 (used to resolve the limit of increment-"
                        "over-background or background-adjusted criteria; labelled as an "
                        "assumption)")
    c.add_argument("--t-c", type=float, help="site night-time air temperature, deg C "
                                             "(required for noise)")
    c.add_argument("--rh", type=float, help="relative humidity %% (required for noise)")
    c.add_argument("--spectrum-file",
                   help="JSON file with octave-band spectral shapes (see "
                        "examples/spectrum.example.json); required for noise")
    c.add_argument("--spectrum",
                   help="name of the spectrum to use from --spectrum-file (optional if the "
                        "file defines exactly one)")
    c.add_argument("--ground", default="hard", choices=["hard", "iso_alt"])
    c.add_argument("--observer-height-m", type=float, default=1.5)
    c.add_argument("--noise-res", type=float, default=25.0)
    c.add_argument("--noise-buffer-m", type=float)
    c.add_argument("--noise-levels", help="comma-separated dB(A) contour levels, added to "
                                          "the criterion limit (no built-in levels)")
    f = ap.add_argument_group("shadow flicker")
    f.add_argument("--utc-offset", type=float, help="UTC offset in hours (required for flicker)")
    f.add_argument("--lat", type=float); f.add_argument("--lon", type=float)
    f.add_argument("--blade-chord-m", type=float,
                   help="effective blade chord, m (required for flicker; deliberately no "
                        "default)")
    f.add_argument("--blade-chord-source", help="source of the blade chord (written to params)")
    f.add_argument("--wind-rose", help="'0:5,30:4,...' or a JSON file {azimuth: frequency} "
                                       "(direction the wind blows from)")
    f.add_argument("--wind-rose-source", help="source of the wind rose (written to params)")
    f.add_argument("--sunshine-prob", type=float); f.add_argument("--operating-prob",
                                                                  type=float, default=1.0)
    f.add_argument("--year", type=int, default=2026)
    f.add_argument("--flicker-step-min", type=int, default=1)
    f.add_argument("--min-sun-elev", type=float, default=3.0,
                   help="minimum solar elevation, degrees (default 3, the LAI "
                        "WKA-Schattenwurfhinweise convention; see flicker.py)")
    f.add_argument("--flicker-res", type=float, default=20.0)
    f.add_argument("--flicker-levels", help="comma-separated h/year contour levels, added to "
                                            "the criterion limit (no built-in levels)")
    f.add_argument("--flicker-smooth", type=int, default=2,
                   help="Chaikin smoothing passes for flicker lines (0 = none, default 2)")
    o = ap.add_argument_group("run")
    o.add_argument("--skip-noise", action="store_true")
    o.add_argument("--skip-flicker", action="store_true")
    o.add_argument("--levels-file",
                   help="JSON {\"noise_dB\": [...], \"flicker_h\": [...]} with contour levels "
                        "(see examples/contour_levels.example.json); merged with the flags")
    # Run-time parameter only: cap of 24 worker processes to keep memory bounded.
    o.add_argument("--workers", type=int,
                   default=max(1, min(24, (os.cpu_count() or 2) - 2)))
    o.add_argument("-o", "--out", required=True, help="output GPKG (whole file rewritten)")
    a = ap.parse_args(argv)

    if a.criteria:
        criteria.set_path(a.criteria)
        # set_path() only affects this process. Flicker workers run in separate processes
        # (spawn/forkserver start methods do not inherit module state), so the resolved path
        # is also exported through the environment variable they read.
        os.environ[criteria.ENV] = str(criteria.criteria_path().resolve())

    out_p = Path(a.out).resolve()
    for p in (a.turbines, a.receptors):
        if p and Path(p).resolve() == out_p:
            raise SystemExit("output path equals an input path - refusing to overwrite input data.")

    kw = dict(t_id_field=a.t_id_field, t_where=a.t_where,
              t_hub_height_m=a.hub_height_m, t_hub_field=a.hub_field,
              t_rotor_diameter_m=a.rotor_diameter_m, t_rotor_field=a.rotor_field,
              t_lwa_dBA=a.lwa_dba, t_lwa_field=a.lwa_field)
    if a.skip_noise and a.lwa_dba is None and a.lwa_field is None:
        kw["t_lwa_dBA"] = float("nan")          # flicker only: L_WA is not used
    if a.skip_flicker and a.rotor_diameter_m is None and a.rotor_field is None:
        kw["t_rotor_diameter_m"] = float("nan")
    if a.receptors:
        kw.update(r_id_field=a.r_id_field, r_where=a.r_where, r_la90_field=a.r_la90_field,
                  r_height_m=a.observer_height_m)
    tn, tf, rn, rf, crs = load_inputs(a.turbines, a.turbine_layer, a.receptors,
                                      a.receptor_layer, **kw)

    atm = None
    spectrum_name = None
    if not a.skip_noise:
        if a.t_c is None or a.rh is None:
            raise SystemExit("noise needs --t-c and --rh (site night-time meteorology; "
                             "deliberately no default)")
        if not a.spectrum_file:
            raise SystemExit("noise needs --spectrum-file (octave-band spectral shape of the "
                             "turbine; there is deliberately no built-in spectrum)")
        names = noise.load_spectra(a.spectrum_file)
        if a.spectrum is None:
            if len(names) != 1:
                raise SystemExit(f"{a.spectrum_file} defines {len(names)} spectra {names}; "
                                 f"choose one with --spectrum")
            spectrum_name = names[0]
        else:
            if a.spectrum not in names:
                raise SystemExit(f"spectrum {a.spectrum!r} is not defined in "
                                 f"{a.spectrum_file}; available: {names}")
            spectrum_name = a.spectrum
        atm = noise.Atmosphere(a.t_c, a.rh, spectrum=spectrum_name, ground=a.ground,
                               receiver_height_m=a.observer_height_m)

    fkw = None
    if not a.skip_flicker:
        if a.utc_offset is None:
            raise SystemExit("flicker needs --utc-offset")
        lat, lon = a.lat, a.lon
        if lat is None or lon is None:           # site lon/lat from the turbine centroid (solar position only)
            from pyproj import Transformer
            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = tr.transform(float(np.mean([t.x for t in tf])),
                                    float(np.mean([t.y for t in tf])))
        if a.blade_chord_m is None:
            raise SystemExit("flicker needs --blade-chord-m (effective blade chord, m; "
                             "deliberately no default)")
        fkw = dict(lat_deg=round(lat, 4), lon_deg=round(lon, 4), utc_offset_hours=a.utc_offset,
                   year=a.year, step_minutes=a.flicker_step_min,
                   min_sun_elevation_deg=a.min_sun_elev, blade_chord_m=a.blade_chord_m,
                   blade_chord_source=a.blade_chord_source, wind_rose=_rose(a.wind_rose),
                   sunshine_probability=a.sunshine_prob,
                   operating_probability=a.operating_prob)

    nlev, flev = list(_floats(a.noise_levels) or []), list(_floats(a.flicker_levels) or [])
    if a.levels_file:
        lf = json.loads(Path(a.levels_file).read_text(encoding="utf-8"))
        nlev += [float(v) for v in lf.get("noise_dB", [])]
        flev += [float(v) for v in lf.get("flicker_h", [])]
    nlev, flev = nlev or None, flev or None

    t0 = time.time()
    res = build(tn, tf, rn, rf, crs, country=a.country, atm=atm, period=a.period,
                background_la90=a.background_la90, noise_res=a.noise_res,
                noise_levels=nlev, noise_buffer_m=a.noise_buffer_m,
                wind_speed_ms=a.wind_speed_ms,
                flicker_kw=fkw, wind_rose_source=a.wind_rose_source,
                flicker_res=a.flicker_res, flicker_levels=flev,
                flicker_smooth_iter=a.flicker_smooth,
                observer_height_m=a.observer_height_m, workers=a.workers,
                do_noise=not a.skip_noise, do_flicker=not a.skip_flicker,
                inputs_desc={"turbines": f"{a.turbines}:{a.turbine_layer}"
                                         + (f" where {a.t_where}" if a.t_where else ""),
                             "receptors": (f"{a.receptors}:{a.receptor_layer}"
                                           + (f" where {a.r_where}" if a.r_where else ""))
                             if a.receptors else None,
                             "criteria": str(criteria.criteria_path()),
                             "spectrum_file": a.spectrum_file})
    p = write_gpkg(res, a.out)
    print(f"OK {p}  ({time.time() - t0:.1f} s)")
    for k in ("noise_contour", "flicker_isoline", "receptor"):
        print(f"   {k}: {len(res[k])} features")
    for _, r in res["params"].iterrows():
        if "WARNING" in r["key"] or r["key"].endswith("_vs_exact_max_abs_db") \
                or r["key"].startswith("flicker_grid_vs_exact"):
            print(f"   {r['key']}: {r['value']}")


if __name__ == "__main__":
    main()
