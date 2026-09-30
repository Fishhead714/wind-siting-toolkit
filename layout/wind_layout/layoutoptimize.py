"""TOPFARM (TopFarm2) wrapper: optimise turbine coordinates for maximum AEP.

Single source of truth: import this module rather than copying the call pattern.

Given a fixed number of turbines and usable-land constraints, optimise the
coordinates. TOPFARM does not plan the turbine count from scratch and does not
support directional (different downwind / crosswind multiples) spacing
constraints; the count is given by the caller and spacing is a simple circular
threshold. Typical pipeline:

    usable land polygon -> optimize_layout() with a fixed count (this module)
    -> directional-ellipse spacing check (``spacingcheck``)

The wake models are engineering models that assume flat, homogeneous inflow.
Results are screening / pre-feasibility level; see ``pipeline.OUTPUT_DISCLAIMER``.
"""
import math

import numpy as np
import shapely.geometry as sg

from shapely.ops import nearest_points

from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import ExclusionZone, InclusionZone, XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
from topfarm.easy_drivers import EasyScipyOptimizeDriver


def evaluate_layout(wfm, x, y, **kwargs):
    """Compute energy once for **fixed coordinates**: gross / net / wake loss. No optimisation.

    Two uses:
      1. ``optimize_layout`` uses it to attach a true wake-loss figure to its result;
      2. re-computing energy after positions were changed downstream (dropping
         low-wind positions, manual moves, boundary snapping).

    **Definition warning**: the ``aep_before/after_gwh`` values returned by
    ``optimize_layout`` compare "initial layout vs optimised layout" net energy;
    they are NOT a wake loss. A wake loss requires re-running with the wake turned
    off, which is what this function does
    (py_wake ``SimulationResult.aep(with_wake_loss=False)``).

    :return: {'aep_gross_gwh', 'aep_net_gwh', 'wake_loss_pct'}
    """
    sim_res = wfm(np.asarray(x, dtype=float), np.asarray(y, dtype=float), **kwargs)
    aep_net = float(sim_res.aep().sum())
    aep_gross = float(sim_res.aep(with_wake_loss=False).sum())
    wake_loss_pct = (1 - aep_net / aep_gross) * 100 if aep_gross > 0 else float("nan")
    return {
        "aep_gross_gwh": aep_gross,
        "aep_net_gwh": aep_net,
        "wake_loss_pct": wake_loss_pct,
        # The validity boundary must travel with the numbers: modules are often
        # imported on their own, bypassing run_pipeline's summary, so the
        # module-level return value carries a pointer to the disclaimer too.
        "disclaimer_ref": (
            "Screening / pre-feasibility estimate, not a bankable energy yield "
            "assessment (EYA); engineering wake model, complex terrain needs CFD "
            "review. Full boundary: pipeline.OUTPUT_DISCLAIMER"
        ),
    }


def _snap_candidates(target_poly, p, margin_m):
    """For a constraint polygon and a violating point, generate candidate positions
    pushed ``margin_m`` from the nearest boundary point, outward or inward.

    The direction is not predicted: **it is found by trial and verified against the
    real constraints, not derived from a sign convention.** Deriving the push
    direction from the vector ``p - nearest_boundary_point`` with a sign flips the
    wrong way when the point sits deep inside a forbidden zone (the direction then
    points further inside), and concave polygons have more degenerate cases. So
    up to four candidate directions are generated and the caller verifies each
    against the real constraints, taking the first that satisfies all of them.
    """
    b = nearest_points(target_poly.boundary, p)[0]
    dirs = []
    for vx, vy in ((p.x - b.x, p.y - b.y),
                   (target_poly.representative_point().x - b.x,
                    target_poly.representative_point().y - b.y)):
        d = math.hypot(vx, vy)
        if d > 1e-12:
            dirs.append((vx / d, vy / d))
            dirs.append((-vx / d, -vy / d))
    return b, [sg.Point(b.x + ux * margin_m, b.y + uy * margin_m) for ux, uy in dirs]


def _snap_to_safe_side(pts, usable_poly, exclusion_polys, margin_m, max_penetration_m=1.0):
    """Push positions that sit mathematically on a constraint boundary ``margin_m`` to the safe side.

    Returns (new point list, correction records).

    Why it is needed: the optimum of a constrained problem often lies exactly on
    the boundary of an active constraint, and SLSQP stops once the residual is
    around 1e-3 m within finite iterations. That is a numerical effect, not a
    layout problem, but a zero-tolerance shapely ``covers()`` calls it infeasible.
    Raising maxiter does not help.

    The fix is to push the point a few metres to the safe side; the margin should
    be much smaller than the spacing requirement so no neighbour's spacing verdict
    is affected.

    **Both failure cases are recorded as ``snapped: False`` and the coordinates
    are left untouched**; the function never silently claims a fix:
      1. none of the candidate directions satisfies both "inside usable land" and
         "outside all exclusion zones";
      2. **the penetration exceeds ``max_penetration_m``**. That is not a numerical
         residual but a genuinely misplaced turbine. Silently moving such a point
         by tens or hundreds of metres is more dangerous than leaving it: the
         displacement changes the wind resource, spacing and setback relations
         while the report shows only ``feasible=True``. Such points must be
         visible, so they are only recorded and the upper-level feasible flag
         stays False.

    :param max_penetration_m: largest penetration (metres) that may be corrected
        automatically. Example default 1.0, chosen well above typical numerical
        residuals (millimetre scale).
    """

    def ok(q):
        return usable_poly.covers(q) and not any(z.covers(q) for z in exclusion_polys)

    corrections = []
    out = list(pts)
    for i, p in enumerate(out):
        if ok(p):
            continue
        if not usable_poly.covers(p):
            target, reason = usable_poly, "outside_usable_land"
        else:
            target = next(z for z in exclusion_polys if z.covers(p))
            reason = "inside_exclusion_zone"

        b, candidates = _snap_candidates(target, p, margin_m)
        penetration = p.distance(b)
        rec = {"index": i, "reason": reason, "penetration_m": round(penetration, 6)}

        if penetration > max_penetration_m:
            rec["snapped"] = False
            rec["note"] = (f"Penetration {penetration:.3f}m exceeds max_penetration_m="
                           f"{max_penetration_m}m: not a numerical residual but a real "
                           "violation; coordinates unchanged - please review this position")
            corrections.append(rec)
            continue

        chosen = next((q for q in candidates if ok(q)), None)
        rec["snapped"] = chosen is not None
        if chosen is not None:
            rec["moved_m"] = round(p.distance(chosen), 3)
            out[i] = chosen
        else:
            rec["note"] = (f"None of the four directions at margin_m={margin_m} satisfies "
                           "'inside usable land AND outside all exclusion zones'; "
                           "coordinates unchanged")
        corrections.append(rec)
    return out, corrections


def _largest_polygon(geom):
    """Union the geometries of a GeoDataFrame/GeoSeries and keep the largest polygon part.

    TOPFARM's InclusionZone/ExclusionZone take a single polygon coordinate array.
    The usable land is normally one polygon, so no multi-part trimming happens; it
    only discards the smaller parts if the input itself is a fragmented
    MultiPolygon (inspect that before calling).
    """
    if hasattr(geom, "geometry"):
        geom = geom.geometry
    if hasattr(geom, "union_all"):
        geom = geom.union_all()
    elif hasattr(geom, "unary_union"):
        geom = geom.unary_union
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def specific_power_w_m2(rated_kw, rotor_diameter_m):
    """Specific power = rated power / swept area, W/m^2. Catches unrealistic turbine definitions.

    A generic turbine definition with an implausibly low specific power gives an
    implausibly high net capacity factor, and nothing else in the chain would flag it.
    """
    import math as _m
    return rated_kw * 1000.0 / (_m.pi / 4 * rotor_diameter_m ** 2)


#: Heuristic plausibility window for onshore specific power (W/m^2). Example
#: default, not a standard: adjust to the turbine classes you actually use.
#: Outside the window the check only warns (special turbines exist, but they
#: must not pass silently).
ONSHORE_SPECIFIC_POWER_RANGE = (200.0, 350.0)


def check_specific_power(rated_kw, rotor_diameter_m, label=""):
    """Specific-power plausibility gate. Returns (sp, ok); prints a warning, never raises."""
    sp = specific_power_w_m2(rated_kw, rotor_diameter_m)
    lo, hi = ONSHORE_SPECIFIC_POWER_RANGE
    ok = lo <= sp <= hi
    if not ok:
        print(f"  [WARN] {label}specific power {sp:.0f} W/m2 is outside the assumed onshore "
              f"window {lo:.0f}-{hi:.0f} ({rated_kw:.0f}kW / D={rotor_diameter_m:.0f}m). "
              + ("Low specific power gives a high capacity factor; check the turbine definition."
                 if sp < lo else "High specific power under-estimates energy."))
    return sp, ok


#: Wake-model constants for the onshore / offshore cases.
#:
#: Library defaults are often offshore-calibrated; using them onshore tends to
#: over-estimate wake loss. **Evidence quality differs per entry and none of these
#: is a universal constant; calibrate against your own data whenever you can.**
#:   - ``turbopark_A_onshore``: EXAMPLE DEFAULT. A single-case tuning value that
#:     has been reported in a conference poster (WindEurope 2025, poster PO110);
#:     the poster itself states it needs generalising. Not peer reviewed, not
#:     independently reproduced, and not re-verified by this package's author (see
#:     THIRD_PARTY_NOTICES.md). Treat as "an assumption", not a finding.
#:   - ``turbopark_A_offshore``: the default ``A`` of PyWake's ``TurboGaussianDeficit``
#:     (0.04; verified in PyWake's source), described as similar to Orsted's
#:     TurbOPark model.
#:   - ``noj_k_onshore``: 0.075, the commonly quoted WAsP land default for the
#:     Jensen / N.O. Jensen wake decay constant (WAsP documentation; page not
#:     re-verified here).
#:   - ``noj_k_offshore``: 0.04, a commonly quoted offshore value (slower wake
#:     recovery). WAsP documentation words it as a recommended lower limit.
WAKE_CONSTANTS = {
    "turbopark_A_onshore": 0.064,   # example default, single-case value - calibrate yourself
    "turbopark_A_offshore": 0.04,   # PyWake TurboGaussianDeficit default
    "noj_k_onshore": 0.075,         # WAsP land default (documentation)
    "noj_k_offshore": 0.04,         # offshore, slow wake recovery
}


def recommended_wake_model(offshore=False, terrain_complexity=None):
    """Return a wake-model instance for the scenario plus an explanatory note.

    **Library defaults are calibrated offshore, and that is the easiest trap.**
    PyWake's ``TurboGaussianDeficit`` default ``A=0.04`` and the classic Jensen
    ``k=0.04`` are offshore values. Onshore turbulence is higher and wake recovery
    faster, so offshore constants tend to **over-estimate wake loss** onshore.

    | model | onshore | offshore | source / evidence |
    |---|---|---|---|
    | TurbOPark / TurboGaussian ``A`` | 0.064 (example default) | 0.04 | onshore: single-case conference poster value, see ``WAKE_CONSTANTS``; offshore: PyWake default |
    | Jensen ``k`` | 0.075 | 0.04 | WAsP documentation (land default; offshore lower limit) |

    ### Validity boundary of the onshore ``A``

    ``A=0.064`` is a single-case tuning value reported in a conference poster.
    Using it as a default for all onshore projects borrows a single-case
    calibration as a general parameter. Until you have your own measured / calibrated data:
      - treat AEP numbers as "an estimate under this assumption", not a
        generalisable finding;
      - if AEP drives a decision, deliver this note with it, or use a sensitivity
        range (see ``wake_model_sensitivity``).

    **Always report model sensitivity.** Swapping the wake model on identical
    coordinates can change the wake loss by a large factor.

    **In complex terrain every engineering wake model is an extrapolation.** Their
    derivation assumes flat, homogeneous inflow; local speed-up, lee-side
    separation and terrain-induced turbulence are outside the model. With
    ``terrain_complexity='complex'`` the model is not swapped (another would be
    equally wrong); a warning is added to the note: wake losses at such a site
    are only good for **ranking** options, and quantitative statements need CFD.

    :param offshore: True to use the offshore constant
    :param terrain_complexity: None | 'simple' | 'complex' (affects the note only)
    :return: (deficit_model, note)
    """
    from py_wake.deficit_models.gaussian import TurboGaussianDeficit

    A = WAKE_CONSTANTS["turbopark_A_offshore" if offshore else "turbopark_A_onshore"]
    note = (f"TurboGaussianDeficit(A={A})"
            + (" (PyWake default, offshore-calibrated)" if offshore
               else " (onshore example default; single-case value, calibrate yourself)"))
    if terrain_complexity == "complex":
        note += ("; WARNING complex terrain: engineering wake models assume flat, "
                 "homogeneous inflow; local speed-up, lee-side separation and "
                 "terrain-induced turbulence are not in the model. Wake losses here "
                 "are only for **ranking** alternatives, not quantitative results; "
                 "quantitative work needs CFD")
    return TurboGaussianDeficit(A=A), note


def wake_model_sensitivity(site, turbine, x, y, models=None, offshore=False):
    """Run several wake models on the same coordinates and return each wake loss.

    The default ``wake_deficit_model`` is invisible in a result, yet swapping the
    model on identical coordinates can move the wake loss a lot. The offshore
    calibrations of TurbOPark / Jensen in particular need justification when used
    onshore. **A single wake-loss number should not appear alone in a deliverable.**

    :return: {model name: {'aep_gross_gwh','aep_net_gwh','wake_loss_pct'}}
    """
    from py_wake.deficit_models.gaussian import (
        BastankhahGaussianDeficit, TurboGaussianDeficit)
    from py_wake.deficit_models.noj import NOJDeficit
    from py_wake.wind_farm_models import PropagateDownwind

    if models is None:
        # Both an "onshore" and an offshore-calibrated TurboGaussian are run, so
        # that "wrong constant" and "wrong model" can be told apart.
        A_on = WAKE_CONSTANTS["turbopark_A_onshore"]
        A_off = WAKE_CONSTANTS["turbopark_A_offshore"]
        k_on = WAKE_CONSTANTS["noj_k_onshore"]
        models = {
            f"TurboGaussian(A={A_on},onshore example)": TurboGaussianDeficit(A=A_on),
            f"TurboGaussian(A={A_off},offshore reference)": TurboGaussianDeficit(A=A_off),
            "BastankhahGaussian": BastankhahGaussianDeficit(),
            f"NOJ(k={k_on},onshore)": NOJDeficit(k=k_on),
        }
    out = {}
    for name, dm in models.items():
        try:
            out[name] = evaluate_layout(
                PropagateDownwind(site, turbine, wake_deficitModel=dm), x, y)
        except Exception as e:   # noqa: BLE001 - one model failing must not sink the rest
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _explode_polygons(gdfs):
    """Flatten GeoDataFrames/GeoSeries/geometries into a polygon list, **dropping none**.

    Unlike ``_largest_polygon`` ("keep the biggest part", valid only for usable
    land that really is one block), this keeps everything: exclusion zones are
    naturally multi-feature layers, and applying the former to them silently drops
    most of their area.
    """
    out = []
    for g in (gdfs or []):
        geom = g.geometry if hasattr(g, "geometry") else g
        if hasattr(geom, "union_all"):
            geom = geom.union_all()
        elif hasattr(geom, "unary_union"):
            geom = geom.unary_union
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        out.extend(p for p in parts if not p.is_empty and p.geom_type == "Polygon")
    return out


def _sample_initial_layout(usable_poly, exclusion_polys, n_wt, rng, max_tries=20000,
                           min_spacing_m=None):
    """Rejection-sample ``n_wt`` initial points inside usable land, outside all exclusions, **respecting minimum spacing**.

    If the initial layout ignored spacing, ``aep_before`` would come from an
    infeasible layout that is not comparable to the feasible optimised one, and
    the resulting "AEP gain" percentage could be negative for no meaningful reason.
    """
    minx, miny, maxx, maxy = usable_poly.bounds
    pts = []
    tries = 0
    while len(pts) < n_wt and tries < max_tries:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        p = sg.Point(x, y)
        if not usable_poly.covers(p):
            continue
        if any(z.covers(p) for z in exclusion_polys):
            continue
        if min_spacing_m and any(
                math.hypot(x - px, y - py) < min_spacing_m for px, py in pts):
            continue
        pts.append((x, y))
    spacing_respected = bool(min_spacing_m)
    if len(pts) < n_wt and min_spacing_m:
        # If sampling with the spacing constraint cannot fill the count, degrade
        # rather than crash: these points are only placeholder design variables and
        # smart_start will overwrite them (it has its own min_space). A hard
        # failure would kill a zone that could be solved normally. But record that
        # spacing is not guaranteed, because the comparability of aep_before
        # depends on it.
        print(f"  [WARN] With the {min_spacing_m:.0f}m spacing constraint only {len(pts)}/{n_wt} "
              "initial points were sampled; falling back to a placeholder layout without "
              "spacing. If smart_start also fails, aep_vs_initial_pct is not comparable "
              "(flagged in the return value)")
        pts, spacing_respected = [], False
        tries = 0
        while len(pts) < n_wt and tries < max_tries:
            tries += 1
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            p = sg.Point(x, y)
            if not usable_poly.covers(p) or any(z.covers(p) for z in exclusion_polys):
                continue
            pts.append((x, y))

    if len(pts) < n_wt:
        raise RuntimeError(
            f"Only {len(pts)}/{n_wt} feasible initial points found in {max_tries} tries "
            "(spacing already relaxed); the usable land is too small or too fragmented "
            "for this turbine count. Check n_wt first"
        )
    xs, ys = zip(*pts)
    return np.array(xs), np.array(ys), spacing_respected


def capacity_density_mw_km2(n_wt, rated_mw, footprint_area_km2):
    """Installed-capacity density diagnostic, MW/km^2.

    An absurdly low value (say < 1) means turbines were spread over a far larger
    area than needed: the typical behaviour of pure AEP maximisation in a big
    boundary, where turbines are pushed to spacings at which wakes do not interact.
    That is mathematically better but unbuildable in practice (land, collection
    lines, roads and land-acquisition extent all blow up).

    Reference: Denholm et al., "Land-Use Requirements of Modern Wind Power Plants in
    the United States", NREL/TP-6A2-45834 (2009), which reports total-site-area
    densities of roughly 3 MW/km^2 for the projects surveyed (exact statistics not
    re-verified by this package's author; see THIRD_PARTY_NOTICES.md).

    The number is very sensitive to the area definition: the same wind farm gives
    single-digit MW/km^2 on a total-site boundary but several times more on a
    tight hull around the turbines. **Confirm both sides use the same area
    definition before comparing.**
    """
    if not footprint_area_km2:
        return None
    return n_wt * rated_mw / footprint_area_km2


def compact_footprint(
    usable_land_gdf, target_area_km2, priority_raster=None, dem_transform=None,
    n_coarse=20, n_fine=9, min_inside_fraction=0.6,
):
    """Choose a **compact** square sub-area of a large usable land as the real layout extent.

    The problem it solves: a candidate boundary can be much larger than the
    area the turbines really need. Feeding the whole area to an AEP optimiser makes it
    spread the turbines as far as it can. The right fix is not a mystery penalty
    term but to **tighten the layout extent to an engineering-reasonable size
    first**, then optimise.

    Method: a coarse then a fine grid search over square-window centres, scoring
    each candidate window:
      1. the intersection of window and usable land must be >= min_inside_fraction
         of the window (excludes windows cut to pieces at the edge);
      2. the intersection area must be >= 0.8 * target_area_km2 (room for the turbines);
      3. among candidates satisfying both, pick the highest mean of ``priority_raster``
         (usually wind speed).

    :param usable_land_gdf: usable land in a projected (metre) CRS
    :param target_area_km2: target footprint area; can be derived from
        n_wt * rated_mw / target capacity density
    :param priority_raster: optional geographic raster (e.g. a wind-speed grid) on
        the grid of ``dem_transform``. Without it the choice degrades to "the
        window with the largest intersection" and the wind resource is no longer
        considered
    :param dem_transform: rasterio Affine of ``priority_raster``
    :return: (single-polygon GeoDataFrame, info dict); if no window qualifies,
        (original usable land, info)
    """
    import geopandas as gpd

    crs = usable_land_gdf.crs
    usable = _largest_polygon(usable_land_gdf)
    full_area_km2 = usable.area / 1e6
    info = {"target_area_km2": target_area_km2, "original_area_km2": full_area_km2}
    if full_area_km2 <= target_area_km2:
        info["shrunk"] = False
        info["reason"] = "usable land is not larger than the target area; no shrinking needed"
        return usable_land_gdf, info

    half = math.sqrt(target_area_km2 * 1e6) / 2
    minx, miny, maxx, maxy = usable.bounds

    def score(cx, cy):
        win = sg.box(cx - half, cy - half, cx + half, cy + half)
        inter = win.intersection(usable)
        if inter.is_empty:
            return None
        inside_frac = inter.area / win.area
        if inside_frac < min_inside_fraction or inter.area / 1e6 < target_area_km2 * 0.8:
            return None
        if priority_raster is None or dem_transform is None:
            return inter.area, win, inter
        vals = _zonal_mean(priority_raster, dem_transform, win, crs)
        if vals is None:
            return None
        return vals, win, inter

    best = None
    for stage, n in (("coarse", n_coarse), ("fine", n_fine)):
        if stage == "coarse":
            xs = np.linspace(minx + half, maxx - half, n)
            ys = np.linspace(miny + half, maxy - half, n)
        else:
            if best is None:
                break
            step = (maxx - minx) / max(n_coarse - 1, 1)
            bx, by = best[1].centroid.x, best[1].centroid.y
            xs = np.linspace(bx - step, bx + step, n)
            ys = np.linspace(by - step, by + step, n)
        for cx in xs:
            for cy in ys:
                s = score(cx, cy)
                if s and (best is None or s[0] > best[0]):
                    best = s

    if best is None:
        info["shrunk"] = False
        info["reason"] = (f"No window satisfies both inside_fraction>={min_inside_fraction} and "
                          f"intersection area>={target_area_km2 * 0.8:.1f}km2; usable land left unchanged")
        return usable_land_gdf, info

    _, _, inter = best
    # `inter` may be a MultiPolygon, and optimize_layout later keeps only the
    # largest part WITHOUT warning. So the reported final_area_km2 would not be the
    # constraint area the optimiser really uses. Report both.
    _eff = _largest_polygon(gpd.GeoDataFrame(geometry=[inter], crs=crs))
    _dropped = (1 - _eff.area / inter.area) * 100 if inter.area else 0.0
    _nparts = len(getattr(inter, "geoms", [inter]))
    if _dropped > 1e-6:
        print(f"  [WARN] The footprint intersection has {_nparts} parts and the optimiser uses "
              f"only the largest: {_dropped:.2f}% of the area is dropped. Report "
              "effective_area_km2 rather than final_area_km2")
    info.update({
        "shrunk": True,
        "final_area_km2": inter.area / 1e6,
        "effective_area_km2": _eff.area / 1e6,
        "dropped_by_largest_polygon_pct": round(_dropped, 4),
        "n_parts": _nparts,
        "shrink_ratio": _eff.area / usable.area,
        "selected_by": "priority_raster_zonal_mean" if priority_raster is not None else "intersection_area",
    })
    return gpd.GeoDataFrame(geometry=[inter], crs=crs), info


def _zonal_mean(raster, transform, geom_projected, crs):
    """Transform a projected geometry to WGS84 and take the mean of the raster over its bbox."""
    import pyproj

    to_wgs = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    minx, miny, maxx, maxy = geom_projected.bounds
    lons, lats = to_wgs.transform([minx, maxx], [miny, maxy])
    cols, rows = (~transform) * (np.array(lons), np.array(lats))
    r1, r2 = int(np.floor(min(rows))), int(np.ceil(max(rows))) + 1
    c1, c2 = int(np.floor(min(cols))), int(np.ceil(max(cols))) + 1
    r1, c1 = max(0, r1), max(0, c1)
    r2, c2 = min(raster.shape[0], r2), min(raster.shape[1], c2)
    if r2 <= r1 or c2 <= c1:
        return None
    win = raster[r1:r2, c1:c2]
    valid = ~np.isnan(win)
    return float(win[valid].mean()) if valid.any() else None


def optimize_layout(
    wfm,
    n_wt,
    usable_land_gdf,
    exclusion_gdfs=None,
    rotor_diameter_m=None,
    min_spacing_d=3.0,
    maxiter=300,
    seed=None,
    id_prefix="OPT",
    use_smart_start=True,
    smart_start_wd_step=10,
    smart_start_grid_n=60,
    boundary_tol_m=None,
    snap_margin_m=5.0,
    snap_max_penetration_m=1.0,
):
    """Optimise turbine coordinates for maximum AEP with a fixed count under land constraints.

    :param wfm: a configured py_wake WindFarmModel (real site + turbine + wake
        model). AEP is computed entirely by it; this module has no simplified AEP
        engine. **Set the turbine's air_density before building ``wfm``**, from the
        site elevation: PyWake's default 1.225 kg/m^3 is a sea-level value and, at
        high-elevation sites, systematically over-estimates power (see
        ``siteconditions.air_density_from_elevation``).
    :param n_wt: number of turbines; must be fixed (TOPFARM does not do "how many
        fit" count optimisation)
    :param usable_land_gdf: geopandas GeoDataFrame/GeoSeries in a projected
        (metre) CRS; usable-land polygon (unioned, largest part kept)
    :param exclusion_gdfs: optional list of GeoDataFrame/GeoSeries of exclusion
        zones inside the usable land (e.g. community setback buffers). Use it when
        they have not already been subtracted from the usable land
    :param rotor_diameter_m: rotor diameter; default read from ``wfm.windTurbines.diameter()``
    :param min_spacing_d: minimum spacing multiple (circular, non-directional).
        Example default 3D. This is looser than a 3D crosswind / 5D downwind
        ellipse, so always re-check with ``spacingcheck`` afterwards
    :param maxiter: maximum SLSQP iterations. Too few (e.g. 60) can end with
        "Iteration limit reached" and coordinates stuck halfway; 300 is a safer
        starting point
    :param seed: random seed of the initial layout, for reproducibility
    :param id_prefix: prefix of the output wtg_id, e.g. "OPT_0", "OPT_1", ...
    :param use_smart_start: use TOPFARM's smart_start to place initial positions
        greedily on the AEP map instead of pure random rejection sampling. This is
        the right lever for "Iteration limit reached": raising maxiter does not
        help, the random start is the root cause. If smart_start fails it falls
        back to random sampling (recorded in the return value ``init_method``).
    :param smart_start_wd_step: wind-direction step (deg) of smart_start's AEP map,
        default 10 (36 sectors). TOPFARM's default of 1 degree steps is wasteful
        for screening accuracy.
    :param smart_start_grid_n: side length of smart_start's candidate grid,
        default 60 (=3600 candidate points)
    :param boundary_tol_m: geometric tolerance of the feasibility verdict, metres.
        None means 1e-3 * D. **Zero tolerance is wrong**: the optimum of a
        constrained problem often lies on the boundary of an active constraint,
        and the millimetre residual SLSQP leaves is numerical, not a violation
        (see ``_snap_to_safe_side``).
    :param snap_margin_m: how far a boundary-hugging position beyond tolerance is
        pushed to the safe side, default 5 m (example default). 0 = report only.
    :param snap_max_penetration_m: largest penetration corrected automatically,
        default 1 m (example default). Beyond it the position is only recorded
        (``snapped=False``) and ``feasible`` stays False: a deep penetration is not
        a numerical residual, and silently moving a turbine tens of metres changes
        wind resource / spacing / setback in a way the report cannot show.
    :return: {
        'points_gdf': GeoDataFrame(wtg_id, geometry=Point, crs=usable_land_gdf.crs),
        'aep_before_gwh', 'aep_after_gwh', 'aep_vs_initial_pct',   # before/after comparison, NOT wake loss
        'aep_gross_gwh', 'aep_net_gwh', 'wake_loss_pct',     # the true wake-loss definition
        'feasible': bool, independent geometric re-check after optimisation (does
            not rely on scipy's own convergence status): all points inside usable
            land (with boundary_tol_m tolerance) + outside all exclusions + minimum
            spacing satisfied
        'min_spacing_actual_m': float,
        'init_method': 'smart_start' | 'random',
        'boundary_snap_corrections': list[dict], snap records (empty = none),
    }
    """
    import geopandas as gpd

    if rotor_diameter_m is None:
        rotor_diameter_m = float(wfm.windTurbines.diameter())

    crs = usable_land_gdf.crs
    usable_poly = _largest_polygon(usable_land_gdf)
    # Exclusion layers go through _explode_polygons, not _largest_polygon: a layer
    # of many setback buffers would otherwise become one polygon with most of the
    # exclusion area silently dropped, and "community setback buffers" is exactly
    # the typical multi-feature use.
    exclusion_polys = _explode_polygons(exclusion_gdfs)
    if boundary_tol_m is None:
        boundary_tol_m = 1e-3 * rotor_diameter_m

    # InclusionZone takes only the outer ring, so holes in the usable land (e.g.
    # punched by setbacks) would be lost and hole-centre points judged legal.
    # Holes are converted to ExclusionZones to put them back.
    boundary_zones = [InclusionZone(np.array(usable_poly.exterior.coords))]
    hole_polys = [sg.Polygon(ring) for ring in usable_poly.interiors]
    boundary_zones += [ExclusionZone(np.array(h.exterior.coords)) for h in hole_polys]
    boundary_zones += [ExclusionZone(np.array(p.exterior.coords)) for p in exclusion_polys]
    if hole_polys:
        print(f"  Usable land has {len(hole_polys)} inner ring(s) (holes); converted to exclusion zones")

    min_spacing_m = min_spacing_d * rotor_diameter_m
    rng = np.random.default_rng(seed)
    x0, y0, init_spacing_ok = _sample_initial_layout(
        usable_poly, exclusion_polys, n_wt, rng, min_spacing_m=min_spacing_m)

    cost_comp = PyWakeAEPCostModelComponent(wfm, n_wt=n_wt)  # default autograd gradients;
    # do not pass grad_method=None: the optimiser then converges less easily at the same maxiter

    tf = TopFarmProblem(
        design_vars={"x": x0, "y": y0},
        cost_comp=cost_comp,
        constraints=[
            XYBoundaryConstraint(boundary_zones, boundary_type="multi_polygon"),
            SpacingConstraint(min_spacing=min_spacing_m),
        ],
        driver=EasyScipyOptimizeDriver(optimizer="SLSQP", maxiter=maxiter),
    )

    init_method = "random"
    if use_smart_start:
        try:
            minx, miny, maxx, maxy = usable_poly.bounds
            xs_g = np.linspace(minx, maxx, smart_start_grid_n)
            ys_g = np.linspace(miny, maxy, smart_start_grid_n)
            XX, YY = np.meshgrid(xs_g, ys_g)
            ZZ = cost_comp.get_aep4smart_start(
                wd=np.arange(0, 360, smart_start_wd_step)
            )
            tf.smart_start(XX, YY, ZZ, min_space=min_spacing_m, seed=seed,
                           show_progress=False)
            init_method = "smart_start"
        except Exception as e:   # noqa: BLE001 - a poor start must not crash the pipeline
            print(f"  [WARN] smart_start failed ({type(e).__name__}: {e}); falling back to a random initial layout")

    cost_before, _ = tf.evaluate()
    cost_after, state, _ = tf.optimize()

    xs, ys = np.asarray(state["x"], dtype=float), np.asarray(state["y"], dtype=float)
    pts = [sg.Point(x, y) for x, y in zip(xs, ys)]

    # Tolerance-based feasibility verdict + push boundary-hugging points inward
    usable_tol = usable_poly.buffer(boundary_tol_m)
    excl_tol = [z.buffer(-boundary_tol_m) for z in exclusion_polys]
    excl_tol = [z for z in excl_tol if not z.is_empty]
    needs_snap = any(
        (not usable_tol.covers(p)) or any(z.covers(p) for z in excl_tol) for p in pts
    )
    corrections = []
    if needs_snap and snap_margin_m > 0:
        pts, corrections = _snap_to_safe_side(
            pts, usable_poly, exclusion_polys, snap_margin_m,
            max_penetration_m=snap_max_penetration_m)
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])

    inside_usable = all(usable_tol.covers(p) for p in pts)
    outside_exclusions = all(not any(z.covers(p) for z in excl_tol) for p in pts)
    pairwise = [
        pts[i].distance(pts[j]) for i in range(n_wt) for j in range(i + 1, n_wt)
    ]
    min_spacing_actual = min(pairwise) if pairwise else float("inf")
    feasible = inside_usable and outside_exclusions and min_spacing_actual >= min_spacing_m - 1e-6

    aep_before_gwh = -cost_before
    # If snapping moved coordinates, cost_after no longer matches the final
    # coordinates; recompute from the final coordinates in all cases
    energy = evaluate_layout(wfm, xs, ys)
    moved_any = any(c.get("snapped") for c in corrections)
    aep_after_gwh = energy["aep_net_gwh"] if moved_any else -cost_after

    # Reference frame of aep_before: when smart_start succeeds it is the
    # smart_start layout (feasible, comparable); when it falls back to a random
    # start whose spacing could not be satisfied, it is an infeasible layout and
    # the ratio is undefined.
    baseline_comparable = (init_method == "smart_start") or init_spacing_ok
    aep_vs_initial_pct = (aep_after_gwh - aep_before_gwh) / aep_before_gwh * 100
    if not baseline_comparable:
        print("  [WARN] The initial layout is neither smart_start nor spacing-feasible; "
              "aep_vs_initial_pct is not comparable, flagged baseline_comparable=False")
    if baseline_comparable and aep_vs_initial_pct < 0:
        print(f"  [WARN] Optimised AEP is below the initial layout ({aep_vs_initial_pct:+.2f}%): "
              "the optimiser may not have converged, or the initial layout itself was "
              "infeasible. Do not read this negative number as 'optimisation is useless'")

    points_gdf = gpd.GeoDataFrame(
        {"wtg_id": [f"{id_prefix}_{i}" for i in range(n_wt)]},
        geometry=pts,
        crs=crs,
    )
    return {
        "points_gdf": points_gdf,
        "aep_before_gwh": aep_before_gwh,
        "aep_after_gwh": aep_after_gwh,
        # It compares "initial layout vs optimised layout"; it is not any
        # industry-standard "improvement".
        "aep_vs_initial_pct": aep_vs_initial_pct,
        "aep_baseline_comparable": baseline_comparable,
        "aep_gross_gwh": energy["aep_gross_gwh"],
        "aep_net_gwh": energy["aep_net_gwh"],
        "wake_loss_pct": energy["wake_loss_pct"],
        "feasible": feasible,
        "min_spacing_actual_m": min_spacing_actual,
        "init_method": init_method,
        "boundary_snap_corrections": corrections,
    }
