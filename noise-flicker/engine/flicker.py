#!/usr/bin/env python3
"""Shadow-flicker engine - per receptor, minute-by-minute over a full year, two outputs.

## Thresholds come from the criteria registry, not from this file

Hours-per-year and minutes-per-day thresholds are always read through criteria.py.
Whether a threshold is binding is recorded in the registry (``enforceability``; see
criteria.py). The output of this engine is a **warning list and a record, not a hard
exclusion layer**.

## Geometry: exact ray / rotor-plane intersection, no small-angle approximation

A ray is cast from the receptor towards the sun and intersected with the rotor plane
(through the hub, normal = nacelle axis). If the intersection P satisfies
``|P - hub| <= rotor radius`` and lies on the sun side, the turbine shades the receptor
at that instant. This is shorter and more accurate than projecting an ellipse and testing
inside/outside (the rotor can subtend 15 degrees or more; a small-angle approximation is
off by about 2 %).

## Two outputs (always report both)

**Astronomical maximum** (worst case, the figure usually used for permitting)
  Assumes a cloudless year, rotor always turning and **the rotor plane always facing the
  receptor**. The rotor axis becomes the line of sight u and the test reduces to the cone
  test ``s.u >= cos(atan(R/L))``.
  Note: this is the industry worst-case convention, not a hard geometric upper bound.
  The supremum over all yaw angles is ``asin(R/L)``, which is strictly larger (a tilted
  rotor brings its near edge closer to the receptor and subtends a larger angle); the
  difference is largest close to the turbine and shrinks quickly with distance. Quote this
  convention whenever the worst case is presented externally.

**Expected** (for internal decisions)
  The wind rose gives a probability distribution of nacelle orientation; each wind-
  direction bin is intersected exactly and weighted by its frequency, then multiplied by
  the sunshine probability and the operating probability.
  Note: the sum runs over the ``asin(R/L)`` **superset** (it does not inherit the
  truncation of the astronomical convention), so "expected <= astronomical maximum" is
  **not guaranteed by construction** - any excess is a small numerical effect. Clamping the
  probability inputs to [0, 1] keeps it bounded, and the output verifies the relation per
  receptor (within a numerical tolerance).

## Distance cut-off: the "blade covers 20 % of the solar disc" convention

d_max = blade chord / (0.2 x solar angular diameter [rad]).
The blade chord must be supplied explicitly by the caller - see ``assess_flicker``.
The 20 % rule defines the **area to be examined**; it is not a per-time-step shading
criterion. Inverting it into a distance is the correct use. It is a modelling
convention, not a compliance threshold.

## Limits of validity

Screening level. **No terrain or building screening** (hills, trees or other buildings
blocking the sun all reduce real flicker below this result, so the result is
conservative). No actual turbine downtime records, no hourly cloud observations.
"""
from __future__ import annotations

import math

import numpy as np

import solar

# The jurisdiction-specific sentence of the disclaimer is assembled at run time from the
# registry by _disclaimer_for(); a hard-coded threshold here would ship with every output
# regardless of the jurisdiction and drift silently when the registry changes.

#: When a receptor is closer to its nearest turbine than this factor x rotor diameter, the
#: astronomical figure should not be quoted on its own. Tool assumption, configurable by
#: assigning flicker.NEAR_FIELD_WARN_FACTOR; not taken from any guideline.
NEAR_FIELD_WARN_FACTOR = 2.0

OUTPUT_DISCLAIMER = (
    "Screening-level shadow-flicker result, not an EIA/permitting-grade assessment. "
    "No terrain, building or vegetation screening (hence conservative); no hourly cloud "
    "observations and no actual turbine downtime records. The astronomical maximum uses "
    "the industry worst-case convention (rotor plane always facing the receptor); the "
    "expected value depends on the wind rose and sunshine probability supplied, and "
    "changes if those assumptions change."
)


def _disclaimer_for(crit: dict) -> str:
    """Append the nature of the jurisdiction's criterion to the disclaimer; the numbers
    come from the registry, never from this file."""
    h, m = crit.get("limit_hours_per_year"), crit.get("limit_minutes_per_day")
    if h is None and m is None:
        return (OUTPUT_DISCLAIMER + " This jurisdiction has **no numeric threshold** "
                "(study obligation only); this output is a record and internal warning, "
                "not a compliance determination.")
    veto = "a hard veto" if crit.get("is_hard_veto") else "not a hard veto"
    return (OUTPUT_DISCLAIMER + f" Jurisdiction threshold: {h} h/year, {m} min/day, {veto}"
            " - this output is a warning list, not an exclusion layer.")


class FlickerTurbine:
    """x/y must be in a projected CRS (metres), the same CRS as the receptors."""

    def __init__(self, x, y, hub_height_m, rotor_diameter_m, wtg_id="", crs=None):
        self.x, self.y = float(x), float(y)
        self.hub_height_m = float(hub_height_m)
        self.rotor_diameter_m = float(rotor_diameter_m)
        self.rotor_radius_m = self.rotor_diameter_m / 2.0
        self.id = wtg_id
        self.crs = crs


class FlickerReceptor:
    def __init__(self, x, y, rec_id="", height_m=1.5, kind="", crs=None):
        self.x, self.y = float(x), float(y)
        self.id, self.kind = rec_id, kind
        self.height_m = float(height_m)
        self.crs = crs


def max_distance_m(blade_chord_m: float, sun_cover_fraction: float = 0.20) -> float:
    """Maximum calculation distance from the German LAI "area to be examined" convention
    (WKA-Schattenwurfhinweise, 2020 update).

    Wording matters: the original text reads
    `Der zu pruefende Bereich ergibt sich aus dem Abstand zur WKA, in welchem die
    Sonnenflaeche gerade zu 20 % durch ein Rotorblatt verdeckt wird` -
    i.e. it **delimits the area to be examined**; it is not a per-time-step shading
    threshold. Inverting it into a distance is the correct use.
    Source: LAI WKA-Schattenwurfhinweise, Aktualisierung 2019, Stand 23.01.2020; ``sun_cover_fraction`` can be changed if
    the guidance applicable to you uses a different value.
    """
    sun_rad = math.radians(solar.SUN_ANGULAR_DIAMETER_DEG)
    return float(blade_chord_m) / (sun_cover_fraction * sun_rad)


def _time_grid(year: int, step_minutes: int, utc_offset_hours: float) -> np.ndarray:
    """UTC time grid covering the **local** calendar year.

    Slicing by the UTC year instead would truncate the first/last local day: the annual
    total is unaffected, but ``astro_max_minutes_per_day`` and ``days_affected`` would be
    biased.
    """
    off = np.timedelta64(int(utc_offset_hours * 3600), "s")
    return np.arange(np.datetime64(f"{year}-01-01T00:00:00") - off,
                     np.datetime64(f"{year+1}-01-01T00:00:00") - off,
                     np.timedelta64(step_minutes, "m"), dtype="datetime64[s]")


def _normalize_rose(wind_rose) -> list:
    """Wind rose -> [(azimuth deg, frequency)], frequencies normalised. Azimuth is the
    direction the wind blows **from** (meteorological convention)."""
    if wind_rose is None:
        return [(0.0, 1.0)]        # placeholder, never used by the caller
    items = list(wind_rose.items()) if isinstance(wind_rose, dict) else list(wind_rose)
    tot = sum(f for _, f in items)
    if tot <= 0:
        raise ValueError("wind rose frequencies sum to 0")
    return [(float(d) % 360.0, float(f) / tot) for d, f in items]


def assess_flicker(turbines, receptors, *, lat_deg: float, lon_deg: float,
                   utc_offset_hours: float, country: str, year: int = 2026,
                   step_minutes: int = 1, min_sun_elevation_deg: float = 3.0,
                   blade_chord_m: float | None = None,
                   blade_chord_source: str | None = None,
                   wind_rose=None, sunshine_probability: float | None = None,
                   operating_probability: float = 1.0) -> dict:
    """Main entry point. Returns per-receptor hours/year and minutes/day, exceedance list,
    provenance and disclaimer.

    ``country`` is the jurisdiction key in the criteria registry. Without wind_rose /
    sunshine_probability **only the astronomical maximum** is produced and the expected
    value is None - there is deliberately no default wind rose: an invented one would make
    the expected value look evidence-based.

    ``min_sun_elevation_deg`` defaults to 3 degrees, following the LAI WKA-Schattenwurfhinweise
    (2020), under which sun positions less than 3 degrees above the horizon are not counted
    (obstruction and atmospheric attenuation). Pass another value if your guidance differs.
    """
    import criteria
    import noise as _n

    # Three kinds of input that would silently yield an "everything complies" deliverable
    # are rejected before any computation.
    _n._check_geometry(turbines, receptors)
    for nm, v in (("sunshine_probability", sunshine_probability),
                  ("operating_probability", operating_probability)):
        if v is not None and not (0.0 <= float(v) <= 1.0):
            raise ValueError(
                f"{nm}={v} is outside [0,1]. Without this check a probability above 1 "
                f"makes the expected hours exceed the astronomical maximum, breaking the "
                f"'expected <= astronomical maximum' guarantee.")

    lim_h, lim_min, crit = criteria.flicker_limits(country)

    # Plausibility hint for the jurisdiction key vs the site latitude. A hint, not a gate:
    # the engine cannot judge how close is "close enough", and a hard latitude box would
    # wrongly block sites near a border. The band comes from the registry (lat_range_deg).
    lat_range = crit.get("lat_range_deg")
    geo_mismatch = bool(lat_range and not (lat_range[0] <= lat_deg <= lat_range[1]))

    if blade_chord_m is None:
        raise ValueError(
            "blade_chord_m (effective blade chord, m) must be given explicitly. There is "
            "deliberately no default:\nthrough the 20 % convention it directly sets d_max "
            "and therefore which receptors are reported as 0.00 h -\nan invented chord "
            "would make the delivered numbers look evidence-based. Use the blade chord "
            "from the manufacturer's data for the turbine being assessed.")
    # Record where the chord came from, so that a documented value and an unsourced one
    # cannot be confused in the output.
    chord_src = blade_chord_source or "not declared by caller"

    times = _time_grid(year, step_minutes, utc_offset_hours)
    elev, az = solar.position(times, lat_deg, lon_deg)
    up = elev > min_sun_elevation_deg
    t_up, s_up = times[up], solar.unit_vectors(elev[up], az[up])
    n_up = len(t_up)

    # local day index (for "maximum minutes on a single day")
    local_days = ((t_up.astype("datetime64[s]").astype(np.int64)
                   + int(utc_offset_hours * 3600)) // 86400).astype(np.int64)
    day_ids, day_idx = np.unique(local_days, return_inverse=True)
    # local hour (0-23): when in the day flicker occurs. The hour-of-day distribution
    # supports the common argument that flicker falls at dawn/dusk, when the sun is weak.
    local_hours = (((t_up.astype("datetime64[s]").astype(np.int64)
                     + int(utc_offset_hours * 3600)) % 86400) // 3600).astype(np.int64)

    d_max = max_distance_m(blade_chord_m)
    rose = _normalize_rose(wind_rose)
    do_expected = wind_rose is not None

    rows, n_ex_h, n_ex_d, n_invalid = [], 0, 0, 0
    for r in receptors:
        # A receptor under the swept area of a rotor is outside the validity of this model
        # (it would report an absurd number of hours and dominate worst_receptor in the
        # summary). Operation buildings or misregistered points in real building layers
        # can hit this. Flag it rather than report a meaningless number or abort the run.
        under = [t.id or "?" for t in turbines
                 if math.hypot(t.x - r.x, t.y - r.y) < t.rotor_radius_m]
        if under:
            n_invalid += 1
            rows.append({
                "receptor_id": r.id, "kind": r.kind, "x": r.x, "y": r.y,
                "astro_hours_per_year": None, "astro_max_minutes_per_day": None,
                "days_affected": None, "expected_hours_per_year": None,
                "exceeds_annual": None, "exceeds_daily": None,
                "status": "INVALID_UNDER_ROTOR",
                "note": f"inside the swept-area projection of turbine(s) {', '.join(under)}; "
                        f"the flicker model does not apply - check whether this point is a "
                        f"dwelling receptor or a misregistered point",
                "contributing_wtg_hours": {}, "hour_of_day_minutes": {},
            })
            continue
        astro = np.zeros(n_up, dtype=bool)          # astronomical max: any turbine counts
        prob = np.zeros(n_up, dtype=float)          # expected: probability of shading
        per_wtg = {}
        for t in turbines:
            dx, dy, dz = t.x - r.x, t.y - r.y, t.hub_height_m - r.height_m
            dist = math.hypot(dx, dy)
            if dist > d_max:
                continue
            L = math.sqrt(dx * dx + dy * dy + dz * dz)
            u = np.array([dx / L, dy / L, dz / L])   # receptor -> hub, ENU
            proj = s_up @ u
            # ---- astronomical max: industry worst-case convention (rotor plane always
            #      facing the receptor) => cone test. With the rotor facing the receptor the
            #      ray/disc intersection reduces to |P-hub| = L*tan(theta) <= R, so the
            #      threshold is atan(R/L).
            hit = proj >= math.cos(math.atan2(t.rotor_radius_m, L))
            # ---- candidate superset: the supremum over all yaw angles is asin(R/L) (a
            #      tilted rotor brings its near edge closer and subtends a larger angle),
            #      strictly larger than atan(R/L). Computing the expected value only on the
            #      astronomical mask would inherit the atan truncation and under-count most
            #      in the near field - exactly where 30 h is exceeded.
            cand = proj >= math.cos(math.asin(min(t.rotor_radius_m / L, 1.0)))
            if not cand.any():
                continue
            astro |= hit
            per_wtg[t.id or f"({t.x:.0f},{t.y:.0f})"] = \
                round(float(hit.sum()) * step_minutes / 60.0, 2)

            if do_expected:
                # ---- expected: exact ray / rotor-plane intersection per wind-direction bin
                idx = np.flatnonzero(cand)           # on the superset, no atan truncation
                s_sub = s_up[idx]
                hub_rel = np.array([dx, dy, dz])     # receptor -> hub
                p_sub = np.zeros(len(idx))
                for wdir, freq in rose:
                    w = math.radians(wdir)
                    n = np.array([math.sin(w), math.cos(w), 0.0])   # nacelle axis (horizontal)
                    sn = s_sub @ n
                    with np.errstate(divide="ignore", invalid="ignore"):
                        tt = (hub_rel @ n) / sn
                    # tt >= 0, not tt > 0: with a strict inequality a receptor lying exactly
                    # in the rotor plane (t=0) never registers a hit for any wind direction,
                    # so astro and expected disagree wildly.
                    ok = np.isfinite(tt) & (tt >= 0)
                    P = s_sub * tt[:, None] - hub_rel               # intersection rel. to hub
                    inside = ok & (np.einsum("ij,ij->i", P, P)
                                   <= t.rotor_radius_m ** 2)
                    p_sub += freq * inside
                # Approximation: with several turbines shading the same receptor the strict
                # semantics would OR the turbines within each wind-direction bin and then
                # weight by frequency; here each turbine is computed separately and the
                # maximum probability is taken. The error is always an underestimate and of
                # the order of a numerical tolerance, so the approximation is kept - but it
                # is not exact.
                np.maximum.at(prob, idx, np.minimum(p_sub, 1.0))

        # The INVALID guard radius equals the rotor radius - a cliff edge - and does not
        # cover the range where the astronomical figure is still wildly high. In the near
        # field the astronomical value is a supremum over an unrealisable pose (rotor always
        # facing the receptor) and can be more than twice the expected value; it should not
        # be quoted on its own. The threshold uses the diameter of the *nearest* turbine:
        # using the largest rotor in the whole layout would flag receptors next to a small
        # turbine because of a large turbine several km away.
        _t_near = min(turbines, key=lambda t: math.hypot(t.x - r.x, t.y - r.y))
        d_near = math.hypot(_t_near.x - r.x, _t_near.y - r.y)
        near_field = d_near < NEAR_FIELD_WARN_FACTOR * _t_near.rotor_diameter_m
        n_min = int(astro.sum()) * step_minutes
        hours = n_min / 60.0
        # maximum minutes on a single day
        per_day = np.bincount(day_idx[astro], minlength=len(day_ids)) * step_minutes
        max_day_min = int(per_day.max()) if len(per_day) else 0
        n_days = int((per_day > 0).sum())

        exp_hours = exp_raw = None
        if do_expected:
            f = 1.0
            if sunshine_probability is not None:
                f *= sunshine_probability
            f *= operating_probability
            exp_raw = float(prob.sum()) * step_minutes / 60.0 * f
            exp_hours = round(exp_raw, 2)

        # Since expected is summed over the asin superset, "<= astro" is not guaranteed by
        # construction, so it is verified per receptor and flagged rather than silently
        # clamped (clamping would hide a real error). The comparison uses the unrounded
        # value with a tolerance of half a time step: comparing the rounded exp_hours with
        # the unrounded hours raises false alarms whenever rounding goes up, and a self-check
        # that cries wolf on normal input will be ignored when it matters.
        exp_over = bool(exp_raw is not None
                        and exp_raw > hours + (step_minutes / 60.0) * 0.5)
        ex_h = lim_h is not None and hours > lim_h
        ex_d = lim_min is not None and max_day_min > lim_min
        n_ex_h += bool(ex_h)
        n_ex_d += bool(ex_d)
        rows.append({
            "receptor_id": r.id, "kind": r.kind, "x": r.x, "y": r.y,
            "astro_hours_per_year": round(hours, 2),
            "astro_max_minutes_per_day": max_day_min,
            "days_affected": n_days,
            "near_field_warning": near_field,
            "expected_hours_per_year": exp_hours,
            "expected_exceeds_astro": exp_over,
            "exceeds_annual": bool(ex_h), "exceeds_daily": bool(ex_d),
            "status": ("EXCEED" if (ex_h or ex_d) else "OK") if lim_h is not None
                      else "NO_THRESHOLD",
            "contributing_wtg_hours": dict(sorted(per_wtg.items(),
                                                  key=lambda kv: -kv[1])),
            "hour_of_day_minutes": {int(h): int(v) * step_minutes for h, v in
                                    enumerate(np.bincount(local_hours[astro],
                                                          minlength=24)) if v},
        })

    rows.sort(key=lambda d: (d["astro_hours_per_year"] is None,
                             -(d["astro_hours_per_year"] or 0.0)))
    valid = [r for r in rows if r["status"] != "INVALID_UNDER_ROTOR"]
    return {
        "_what": f"{country} per-receptor shadow flicker (astronomical maximum"
                 + (" + expected value" if do_expected
                    else "; no wind rose given, so no expected value") + ")",
        "_disclaimer": _disclaimer_for(crit),
        "criterion": crit,
        "model": {
            "solar_algorithm": "NOAA Solar Calculator (with atmospheric refraction correction)",
            "geometry": "exact ray / rotor-plane intersection, no small-angle approximation",
            "astronomical_max_assumption": "cloudless year + rotor always turning + rotor "
                                           "plane always facing the receptor",
            "min_sun_elevation_deg": min_sun_elevation_deg,
            "astro_convention": "industry worst case: rotor plane always facing the receptor "
                                "(threshold atan(R/L)). Candidates are screened with the "
                                "asin(R/L) superset, so the expected value is not truncated "
                                "by that convention.",
            "max_distance_m": round(d_max),
            "blade_chord_m": round(float(blade_chord_m), 2),
            "blade_chord_source": chord_src,
            "blade_chord_source_declared": blade_chord_source is not None,
            "max_distance_basis": f"blade chord {blade_chord_m:.2f} m / (20% x solar angular "
                                  f"diameter {solar.SUN_ANGULAR_DIAMETER_DEG} deg). "
                                  f"'Area to be examined' convention of the German LAI "
                                  f"WKA-Schattenwurfhinweise (2020) - a modelling "
                                  f"convention, not a compliance threshold.",
            "step_minutes": step_minutes, "year": year,
            "lat_deg": lat_deg, "lon_deg": lon_deg,
            "country_latitude_mismatch": geo_mismatch,
            "country_latitude_note": (
                f"latitude {lat_deg} deg is outside the latitude band {list(lat_range)} "
                f"declared for {country} in the criteria registry; check that the "
                f"jurisdiction key is correct." if geo_mismatch else None),
            "utc_offset_hours": utc_offset_hours,
            "terrain_and_building_screening": "not modelled (result is conservative)",
            "wind_rose_given": do_expected,
            "sunshine_probability": sunshine_probability,
            "operating_probability": operating_probability,
        },
        "n_turbines": len(turbines), "n_receptors": len(receptors),
        "summary": {
            "n_exceed_annual": n_ex_h, "n_exceed_daily": n_ex_d,
            "n_exceed_any": sum(1 for r in valid if r["status"] == "EXCEED"),
            "n_invalid_under_rotor": n_invalid,
            "n_near_field_warning": sum(1 for r in valid if r.get("near_field_warning")),
            "n_expected_exceeds_astro": sum(1 for r in valid
                                            if r.get("expected_exceeds_astro")),
            "worst_receptor": valid[0]["receptor_id"] if valid else None,
            "worst_hours_per_year": valid[0]["astro_hours_per_year"] if valid else None,
            "_how_to_read": "An exceedance does not by itself rule out a turbine position; "
                            "see criterion.limit_logic and the enforceability of the "
                            "criterion.",
        },
        "receptors": rows,
    }
