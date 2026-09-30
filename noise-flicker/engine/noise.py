#!/usr/bin/env python3
"""ISO 9613-2 noise propagation engine (screening level) - per receptor, configurable criteria.

Two design rules:
  1. Criteria are decoupled from the model: every limit comes from criteria.py;
     this file contains no limit values.
  2. Turbine data are inputs: sound power level, hub height and the octave-band
     spectral shape are always supplied by the caller. There are no built-in
     turbine parameters and no built-in spectrum.

## Model

    L_p(d) = 10*lg( sum_bands 10^[(L_W,band - A_div - A_atm - A_gr) / 10] )

  A_div = 20*lg(d_slant) + 11        point source, spherical spreading (ISO 9613-2 s.7.1)
  A_atm = alpha(f, T, RH, p) * d/1000 alpha from the ISO 9613-1 closed-form expression
  A_gr  = s.7.3.1 general method, hard ground G=0: for this geometry = -3 dB (a gain)
          Note: the s.7.3.2 alternative method with negative values clipped to zero
          silently drops that 3 dB and shortens compliance distances substantially;
          a regression test guards against it.
  A_bar = 0 (no terrain/building screening at screening level)
  D_c   = 0 (omnidirectional point source; ISO 9613-2 is a downwind model, so every
          direction is treated as downwind - no wind-direction reduction)

## Interface with increment-over-background criteria

ISO 9613-2 gives the LAeq of the source alone at the receptor (the "specific level").
Subtracting the background from a measured total level is a measurement step and is not
needed on the modelling path. Penalties, if any apply, belong in the configured limits;
the propagation model never adds them.

## Spectrum

``SPECTRA`` starts empty. Register octave-band shapes (A-weighted band level minus the
A-weighted total, dB, bands 63 Hz..8 kHz) with ``register_spectrum()`` or
``load_spectra(<json>)`` - ideally the manufacturer's spectrum for the turbine and mode
being assessed. See examples/spectrum.example.json.

## Limits of validity

Screening level, not an EIA-grade assessment: no measured background noise, no terrain
or building screening, spectral shape as supplied. The uncertainty from the spectral
shape, meteorology, ground model and hub height is quantified by sensitivity().
ISO 9613-2 is commonly quoted for source heights below about 30 m and distances up to
about 1 km; modern turbines at km-scale distances are an extrapolation - see
OUTPUT_DISCLAIMER.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

#: Registered octave-band spectral shapes: name -> {band_Hz: relative dB}. Only the shape
#: matters - band levels are always renormalised so that their energy sum equals L_WA.
#: Empty by default on purpose: supply the spectrum of the turbine being assessed.
SPECTRA: dict = {}

PA_STANDARD = 101.325e3

OUTPUT_DISCLAIMER = (
    "Screening-level ISO 9613-2 result, not an EIA/permitting-grade assessment. "
    "No measured background noise, no terrain or building screening, no meteorological "
    "statistics. L_WA and the octave-band spectral shape are inputs supplied by the user; "
    "see the sensitivity analysis for the effect of the spectral shape on distances. "
    "For increment-over-background criteria, conclusions hold only for the background "
    "LA90 used (measured or assumed, as labelled). ISO 9613-2 is commonly quoted for "
    "source heights below about 30 m and distances up to about 1 km; results for tall "
    "turbines at km-scale distances are an extrapolation beyond that range."
)


# ---------------------------------------------------------------- spectra
def register_spectrum(name: str, shape: dict) -> None:
    """Register a spectral shape {band_Hz: relative dB} covering all 8 octave bands."""
    try:
        clean = {int(float(k)): float(v) for k, v in shape.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(f"spectrum {name!r}: bands and values must be numeric ({e})") from None
    missing = [f for f in BANDS if f not in clean]
    extra = sorted(set(clean) - set(BANDS))
    if missing or extra:
        raise ValueError(f"spectrum {name!r}: needs exactly the bands {BANDS}; "
                         f"missing {missing}, unexpected {extra}")
    if not all(math.isfinite(v) for v in clean.values()):
        raise ValueError(f"spectrum {name!r} contains non-finite values")
    SPECTRA[name] = {f: clean[f] for f in BANDS}


def load_spectra(path) -> list:
    """Load ``{"spectra": {name: {band: dB, ...}, ...}}`` from JSON and register every entry.
    Keys starting with '_' are metadata and ignored. Returns the registered names."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    names = []
    for name, shape in (d.get("spectra") or {}).items():
        if name.startswith("_"):
            continue
        register_spectrum(name, {k: v for k, v in shape.items() if not str(k).startswith("_")})
        names.append(name)
    if not names:
        raise ValueError(f"{path}: no spectra found under the 'spectra' key")
    return names


def shift_spectrum(shape: dict, octaves: int) -> dict:
    """Shift a spectral shape by whole octaves (negative = towards low frequencies).

    Used by sensitivity() to bracket the uncertainty of the spectral shape without any
    extra data. Bands pushed past the ends are extrapolated linearly from the last two bands."""
    vals = [shape[f] for f in BANDS]
    n = len(vals)

    def at(i):
        if 0 <= i < n:
            return vals[i]
        if i < 0:
            return vals[0] + (vals[0] - vals[1]) * (-i)
        return vals[-1] + (vals[-1] - vals[-2]) * (i - n + 1)

    # band i takes the value from band i - octaves: a negative shift moves every level to a
    # lower band (e.g. -1: the 500 Hz value appears at 250 Hz)
    return {f: at(i - octaves) for i, f in enumerate(BANDS)}


def _derived_spectrum(base: str, octaves: int) -> str:
    name = f"{base}__shift{octaves:+d}oct"
    SPECTRA[name] = shift_spectrum(SPECTRA[base], octaves)
    return name


# ---------------------------------------------------------------- propagation
@functools.lru_cache(maxsize=1024)
def alpha_iso9613_1(f_hz: float, t_c: float, rh_pct: float, pa: float) -> float:
    """Atmospheric absorption coefficient, dB/km (ISO 9613-1 closed form).

    Checked against reference values in the self-test: a mistyped exponent does not raise,
    it just shifts every distance. Cached because alpha only depends on (band, meteorology)
    but is evaluated receptors x turbines x 8 times; bounded cache size."""
    T = t_c + 273.15
    T0, T01, pr = 293.15, 273.16, 101.325e3
    psat_over_pr = 10.0 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151)
    h = rh_pct * psat_over_pr / (pa / pr)
    frO = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa / pr) * (T / T0) ** -0.5 * (
        9.0 + 280.0 * h * math.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0)))
    f2 = f_hz * f_hz
    term_class = 1.84e-11 * (pa / pr) ** -1 * (T / T0) ** 0.5
    term_O = 0.01275 * math.exp(-2239.1 / T) / (frO + f2 / frO)
    term_N = 0.1068 * math.exp(-3352.0 / T) / (frN + f2 / frN)
    return 8.686 * f2 * (term_class + (T / T0) ** -2.5 * (term_O + term_N)) * 1000.0


@functools.lru_cache(maxsize=64)
def _spectrum_total(shape_items: tuple) -> float:
    """Energy sum (dB) of a spectral shape. Keyed on the shape *content*, not its name:
    SPECTRA entries are mutable dicts, and a name-keyed cache would keep a stale
    normalisation constant after an in-place edit (band energies would then no longer
    sum to L_WA, silently). Guarded by a regression test."""
    return 10.0 * math.log10(sum(10 ** (v / 10.0) for _, v in shape_items))


def band_levels(lwa: float, spectrum: str) -> dict:
    """Split the total A-weighted sound power level into octave bands using the named
    shape, normalised so that the band energies sum exactly to ``lwa``.

    Deliberately not cached as a whole: it returns a dict, and a shared cached dict could
    be mutated by one caller and silently corrupt every other result."""
    if spectrum not in SPECTRA:
        raise ValueError(f"unknown spectrum {spectrum!r}; registered: {sorted(SPECTRA)} "
                         f"(use register_spectrum() or load_spectra())")
    shape = SPECTRA[spectrum]
    tot = _spectrum_total(tuple(sorted(shape.items())))
    return {f: lwa + (shape[f] - tot) for f in BANDS}


def a_ground(d: float, h_src: float, h_rec: float, mode: str = "hard") -> float:
    """Ground effect A_gr, dB. The easiest place in this model to lose 3 dB.

    mode="hard" (default; s.7.3.1 general method, G=0):
        Table 3 with G=0 gives A_s = A_r = -1.5 dB and A_m = -3q, where
        q = 0 for d <= 30(h_s+h_r), else 1 - 30(h_s+h_r)/d.
        For turbine geometry 30(h_s+h_r) is several km, so screening distances fall inside
        it: q=0 and A_gr = -3 dB (a gain). Together with the +11 of free-field spherical
        spreading this equals hemispherical spreading (+8).

    mode="iso_alt" (s.7.3.2 alternative method, negative values clipped to zero):
        the zero-clipped range depends on h_m = (h_s + h_r)/2 (not only on the source
        height). The analytic break point solves 4.8 d^2 - 34 h_m d - 600 h_m = 0.
        Inside it the 3 dB hard-ground reflection gain is lost (optimistic); beyond it the
        term becomes an extra attenuation (more optimistic). Comparison only, never default.

    Why hard ground by default: G=1 (porous ground) yields a positive attenuation for this
    geometry, so G=0 is the conservative bound.
    """
    if mode == "iso_alt":
        h_m = (h_src + h_rec) / 2.0
        return max(0.0, 4.8 - (2.0 * h_m / d) * (17.0 + 300.0 / d))
    if mode != "hard":
        raise ValueError(f"unknown ground mode: {mode}")
    thresh = 30.0 * (h_src + h_rec)
    q = 0.0 if d <= thresh else 1.0 - thresh / d
    return -3.0 - 3.0 * q


class Atmosphere:
    """Site meteorology, spectrum and ground model. Temperature, humidity and spectrum have
    no defaults on purpose - the caller must state them for the site being assessed."""

    def __init__(self, t_c: float, rh_pct: float, pa: float = PA_STANDARD,
                 spectrum: str | None = None, ground: str = "hard",
                 receiver_height_m: float = 1.5):
        if spectrum is None:
            raise ValueError("spectrum is required: register a spectral shape "
                             "(noise.register_spectrum / noise.load_spectra) and pass its name")
        if spectrum not in SPECTRA:
            raise ValueError(f"unknown spectrum {spectrum!r}; registered: {sorted(SPECTRA)}")
        self.t_c, self.rh_pct, self.pa = t_c, rh_pct, pa
        self.spectrum, self.ground = spectrum, ground
        self.receiver_height_m = receiver_height_m

    def as_dict(self) -> dict:
        return {"temp_C": self.t_c, "rh_pct": self.rh_pct, "pressure_Pa": self.pa,
                "spectrum": self.spectrum,
                "spectrum_shape_dB": dict(SPECTRA[self.spectrum]),
                "ground": ("s.7.3.1 G=0 hard ground = -3 dB gain (conservative)"
                           if self.ground == "hard"
                           else "s.7.3.2 alternative method (optimistic, comparison only)"),
                "receiver_height_m": self.receiver_height_m}


def lp_single(lwa: float, d_horiz: float, *, h_src: float, atm: Atmosphere) -> float:
    """A-weighted sound pressure level dB(A) of one turbine at horizontal distance d_horiz."""
    h_rec = atm.receiver_height_m
    d = math.hypot(d_horiz, h_src - h_rec)          # slant distance
    lw = band_levels(lwa, atm.spectrum)
    agr = a_ground(d, h_src, h_rec, atm.ground)
    tot = 0.0
    for f in BANDS:
        a = (20.0 * math.log10(d) + 11.0
             + alpha_iso9613_1(f, atm.t_c, atm.rh_pct, atm.pa) * d / 1000.0 + agr)
        tot += 10 ** ((lw[f] - a) / 10.0)
    return 10.0 * math.log10(tot)


# ---------------------------------------------------------------- per receptor
class Turbine:
    """One turbine. x/y must be in a projected CRS (metres), never lon/lat."""

    def __init__(self, x: float, y: float, hub_height_m: float, lwa_dBA: float,
                 wtg_id: str = "", crs=None):
        self.x, self.y = float(x), float(y)
        self.hub_height_m, self.lwa_dBA = float(hub_height_m), float(lwa_dBA)
        self.id = wtg_id
        self.crs = crs          # carried so the engine can assert one CRS for the whole site


class Receptor:
    """One receptor (dwelling, school, ...). Same CRS as the turbines."""

    def __init__(self, x: float, y: float, rec_id: str = "",
                 background_la90_dBA: float | None = None, kind: str = "", crs=None):
        self.x, self.y = float(x), float(y)
        self.id, self.kind = rec_id, kind
        self.background_la90_dBA = background_la90_dBA
        self.crs = crs


# ------------------------------------------------------------- input guards
#: A nearest turbine-receptor distance above this is treated as an input error (wrong CRS
#: or units), not a real site: noise and flicker cut-offs are a few km at most.
#: Tool default for an input sanity check, not a physical criterion.
MAX_PLAUSIBLE_DISTANCE_M = 100_000.0


def assert_same_crs(*groups) -> object:
    """Assert that all objects come from the same projected CRS.

    Loading turbines and receptors separately with estimate_utm_crs() puts them in
    different UTM zones when a site straddles a zone boundary; Euclidean distances are then
    off by orders of magnitude and every receptor silently passes. Mixing tagged and
    untagged objects is also rejected: either every object carries a CRS (layers.py) or
    none does (hand-built test/what-if inputs)."""
    seen, untagged = {}, []
    for g in groups:
        for o in g:
            c = getattr(o, "crs", None)
            if c is None:
                untagged.append(getattr(o, "id", "?"))
            else:
                seen.setdefault(str(c), []).append(getattr(o, "id", "?"))
    if len(seen) > 1:
        detail = "; ".join(f"{k} (e.g. {v[0]})" for k, v in seen.items())
        raise ValueError(
            f"turbines and receptors are not in the same CRS: {detail}.\n"
            f"Across UTM zones distances are silently wrong by orders of magnitude. Load both "
            f"with layers.load_site(), or pass the same epsg= to both loaders.")
    if seen and untagged:
        raise ValueError(
            f"some objects carry a CRS ({next(iter(seen))}) and others do not (e.g. "
            f"{untagged[0]}, {len(untagged)} in total). CRS consistency cannot be checked when "
            f"layer-loaded and hand-built objects are mixed - pass crs= to the hand-built ones.")
    return next(iter(seen), None)


def _check_geometry(turbines, receptors) -> None:
    """Reject empty lists, non-finite coordinates and implausible scales - three inputs that
    would otherwise produce an 'everything complies' result."""
    if not turbines:
        raise ValueError("turbine list is empty. Returning an empty result is deliberately "
                         "avoided: an 'all receptors comply' deliverable would not show that "
                         "no turbine was ever included.")
    if not receptors:
        raise ValueError("receptor list is empty.")
    for tag, objs in (("turbine", turbines), ("receptor", receptors)):
        for o in objs:
            if not (math.isfinite(o.x) and math.isfinite(o.y)):
                raise ValueError(f"{tag} {getattr(o, 'id', '?')!r} has non-finite coordinates "
                                 f"({o.x}, {o.y}) - they would silently yield 0 or inf.")
    assert_same_crs(turbines, receptors)
    dmin = min(math.hypot(r.x - t.x, r.y - t.y) for r in receptors for t in turbines)
    if dmin > MAX_PLAUSIBLE_DISTANCE_M:
        raise ValueError(
            f"nearest turbine-receptor distance is {dmin/1000:.0f} km, above "
            f"{MAX_PLAUSIBLE_DISTANCE_M/1000:.0f} km.\nAlmost certainly a CRS or unit error "
            f"(degrees used as metres, layers in different UTM zones), not a real site.")


def lp_at_point(turbines, x: float, y: float, atm: Atmosphere) -> tuple:
    """Energy sum of all turbines at (x, y). Returns (L_p_total, dominant turbine,
    its L_p, nearest horizontal distance m).

    A single turbine complying does not mean the array complies: the array sum can push
    the compliance distance out considerably."""
    if not turbines:
        raise ValueError("turbine list is empty")
    tot, best, best_lp, dmin = 0.0, None, -math.inf, math.inf
    for t in turbines:
        d = math.hypot(x - t.x, y - t.y)
        d = max(d, 1.0)                              # receptor at the tower base
        lp = lp_single(t.lwa_dBA, d, h_src=t.hub_height_m, atm=atm)
        tot += 10 ** (lp / 10.0)
        if lp > best_lp:
            best, best_lp = t, lp
        dmin = min(dmin, d)
    return 10.0 * math.log10(tot), best, best_lp, dmin


def _criterion_block(country, period, ctype, global_info, rows) -> dict:
    """Top-level criterion provenance. With per-receptor limits there is no single limit:
    the block reports the range and how the limit was formed. Both branches return the
    same key set so downstream scripts can treat every jurisdiction alike."""
    import criteria
    c = criteria.get(country, "noise")
    if global_info is not None:
        lim = global_info.get("limit_dBA")
        return {**global_info, "period": period,
                "rule": f"{c.get('metric') or 'absolute limit at receptor'} <= {lim} dB(A)"
                        f" ({global_info.get('resolved')})",
                "limit_dBA_range": [lim, lim], "limit_dBA_distinct": 1,
                # absolute limits do not use a background level: None, not 0
                "n_receptors_with_measured_LA90": None,
                "n_receptors_with_assumed_LA90": None,
                "background_caveat": None, "boundary_caveat": None,
                "measurement_metric": c.get("measurement_metric"),
                "open_question": None,
                "_background_not_applicable": "this criterion does not depend on background noise"}
    lims = sorted({r["limit_dBA"] for r in rows})
    n_meas = sum(1 for r in rows if r["background_is_measured"])
    adj = c.get("background_adjustment")
    rule = (c.get("rule_text") or {}).get("night" if period == "night" else "day")
    if rule is None and adj is not None:
        rule = adj.get("rule")
    return {
        "mode": ctype, "period": period, "rule": rule,
        "limit_dBA": lims[0] if len(lims) == 1 else None,
        "limit_dBA_range": [lims[0], lims[-1]] if lims else None,
        "limit_dBA_distinct": len(lims),
        "n_receptors_with_measured_LA90": n_meas,
        "n_receptors_with_assumed_LA90": len(rows) - n_meas,
        "background_caveat": ((c.get("background_LA90") or {}).get("assumption_basis")
                              or (adj or {}).get("source")),
        "measurement_metric": c.get("measurement_metric"),
        "open_question": c.get("_open_question"),
        # For background-adjusted tables the table value is the core concept - pass it
        # through so a reader can tell the table value from table value + adjustment.
        "resolved": (rows[0].get("_resolved") if rows else None),
        "table_value_dBA": (rows[0].get("_table_value_dBA") if rows else None),
        "penalties_available": c.get("penalties_dB_A"),
        "boundary_caveat": (adj or {}).get("_boundary_caveat"),
        "_background_not_applicable": None,
        **c.provenance,
    }


def assess_receptors(turbines, receptors, atm: Atmosphere, *,
                     country: str, period: str = "night",
                     default_background_la90: float | None = None) -> dict:
    """Per-receptor assessment - the main entry point of this module.

    ``country`` is the jurisdiction key in the criteria registry. For criteria that need a
    background level each receptor may carry its own measured LA90; otherwise
    ``default_background_la90`` is used and flagged as an assumption. Absolute limits ignore
    background values.

    Returns a dict with per-receptor rows, summary, criterion provenance and disclaimer.
    """
    import criteria

    _check_geometry(turbines, receptors)
    ctype = criteria.get(country, "noise")["criterion_type"]
    per_receptor_bg = criteria.needs_background(country)
    rows, n_fail = [], 0
    global_limit, global_info = None, None
    if not per_receptor_bg:
        global_limit, global_info = criteria.noise_limit(country, period=period)

    for r in receptors:
        lp, worst, lp_worst, dmin = lp_at_point(turbines, r.x, r.y, atm)
        if per_receptor_bg:
            bg = r.background_la90_dBA if r.background_la90_dBA is not None \
                else default_background_la90
            limit, info = criteria.noise_limit(
                country, period=period, background_la90=bg,
                background_is_measured=r.background_la90_dBA is not None)
        else:
            limit, info = global_limit, global_info
        exceed = lp - limit
        if exceed > 0:
            n_fail += 1
        rows.append({
            "receptor_id": r.id, "kind": r.kind, "x": r.x, "y": r.y,
            "Lp_total_dBA": round(lp, 1),
            "limit_dBA": round(limit, 1),
            "exceedance_dB": round(exceed, 1),
            "status": "EXCEED" if exceed > 0 else "OK",
            "nearest_turbine_m": round(dmin),
            "dominant_wtg": worst.id or f"({worst.x:.0f},{worst.y:.0f})",
            "dominant_wtg_Lp_dBA": round(lp_worst, 1),
            "array_penalty_dB": round(lp - lp_worst, 1),
            "background_LA90_dBA": info.get("background_LA90_dBA"),
            "background_is_measured": info.get("background_is_measured"),
            "table_value_dBA": info.get("table_value_dBA"),
            "adjustment_dB": info.get("adjustment_dB"),
            "adjustment_branch": info.get("adjustment_branch"),
            "_resolved": info.get("resolved"),
            "_table_value_dBA": info.get("table_value_dBA"),
        })

    rows.sort(key=lambda d: -d["exceedance_dB"])
    worst_row = rows[0] if rows else None
    return {
        "_what": f"{country} per-receptor noise assessment (ISO 9613-2 screening, {period})",
        "_disclaimer": OUTPUT_DISCLAIMER,
        # Independent of receptor order: with per-receptor limits the top-level block
        # describes the criterion, not the limit of one arbitrary receptor.
        "criterion": _criterion_block(country, period, ctype, global_info, rows),
        "model": {"standard": "ISO 9613-2 downwind + ISO 9613-1 atmospheric absorption",
                  "A_div": "20lg(d_slant)+11", "A_bar": 0, "D_c": 0,
                  **atm.as_dict()},
        "n_turbines": len(turbines), "n_receptors": len(receptors),
        "summary": {
            "n_exceed": n_fail,
            "pct_exceed": round(100.0 * n_fail / len(receptors), 1) if receptors else 0.0,
            "worst_receptor": worst_row["receptor_id"] if worst_row else None,
            "worst_exceedance_dB": worst_row["exceedance_dB"] if worst_row else None,
            "max_array_penalty_dB": max((r["array_penalty_dB"] for r in rows), default=0.0),
        },
        "receptors": rows,
    }


# ---------------------------------------------------------------- distance / sensitivity
def required_distance(lwa: float, limit: float, *, h_src: float, atm: Atmosphere,
                      array_n: int = 1, spacing: float = 700.0,
                      lo: float = 50.0, hi: float = 20000.0) -> float:
    """Bisection for the compliance horizontal distance m (smallest d with L_p(d) <= limit).

    array_n > 1 uses an idealised row of n turbines at ``spacing`` with the receptor facing
    the nearest one - a rough figure for when no layout exists. With turbine coordinates,
    use assess_receptors instead. ``spacing`` (700 m) and the search bounds are tool
    defaults, not properties of any turbine or site.
    """
    def lp(d):
        if array_n <= 1:
            return lp_single(lwa, d, h_src=h_src, atm=atm)
        tot = 0.0
        for i in range(array_n):
            off = ((i + 1) // 2) * spacing * (1 if i % 2 else -1)
            tot += 10 ** (lp_single(lwa, math.hypot(d, off), h_src=h_src, atm=atm) / 10.0)
        return 10.0 * math.log10(tot)

    if lp(lo) <= limit:
        return lo
    # No solution raises instead of returning the upper bound: a silent 20 km would give
    # sensitivity() a fake percentage.
    if lp(hi) > limit:
        raise ValueError(
            f"no solution in {lo:.0f}-{hi:.0f} m: still {lp(hi):.1f} dB(A) at {hi:.0f} m > "
            f"limit {limit:.1f}.\nMost likely the limit is too strict or L_WA too high "
            f"(L_WA={lwa}, array_n={array_n}).")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if lp(mid) > limit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sensitivity(lwa: float, limit: float, *, h_src: float, atm: Atmosphere,
                array_n: int = 9, spacing: float = 700.0,
                meteo_variants=((15.0, 90.0), (30.0, 40.0)),
                hub_delta_m: float = 30.0) -> dict:
    """Quantify the distance uncertainty caused by the modelling assumptions. **Include
    this in any output** - a single distance without it presents assumptions as results.

    Spectral variants are the supplied shape shifted one octave down (low-frequency heavy)
    and one octave up (high-frequency heavy). ``array_n``, ``spacing``, ``meteo_variants``
    ((temperature C, relative humidity %) pairs) and ``hub_delta_m`` are tool defaults used
    only to demonstrate sensitivity; set them to the ranges relevant to your site."""
    base = required_distance(lwa, limit, h_src=h_src, atm=atm,
                             array_n=array_n, spacing=spacing)
    lf = _derived_spectrum(atm.spectrum, -1)
    hf = _derived_spectrum(atm.spectrum, +1)

    def A(**kw):
        d = dict(t_c=atm.t_c, rh_pct=atm.rh_pct, pa=atm.pa, spectrum=atm.spectrum,
                 ground=atm.ground, receiver_height_m=atm.receiver_height_m)
        d.update(kw)
        return Atmosphere(**d)

    out = []
    variants = [
        ("spectrum shifted 1 octave down (low-frequency heavy, conservative side)", A(spectrum=lf), h_src),
        ("spectrum shifted 1 octave up (high-frequency heavy)", A(spectrum=hf), h_src),
        *[(f"meteorology {t:g} C / {rh:g} % RH", A(t_c=float(t), rh_pct=float(rh)), h_src)
          for t, rh in meteo_variants],
        ("ground iso_alt (s.7.3.2 alternative method, optimistic)", A(ground="iso_alt"), h_src),
        (f"hub height -{hub_delta_m:g} m", atm, h_src - hub_delta_m),
        (f"hub height +{hub_delta_m:g} m", atm, h_src + hub_delta_m),
    ]
    for label, a, h in variants:
        d = required_distance(lwa, limit, h_src=h, atm=a, array_n=array_n, spacing=spacing)
        out.append({"variant": label, "d_m": round(d), "delta_m": round(d - base),
                    "delta_pct": round(100.0 * (d - base) / base, 1),
                    # iso_alt is a known-optimistic reference method, not a plausible bound
                    "is_assumption": "iso_alt" not in label})
    # Only genuine assumptions define the uncertainty span; the reference method is listed
    # separately so it cannot dominate the reported range.
    assumptions = [v for v in out if v["is_assumption"]]
    span = max(abs(v["delta_pct"]) for v in assumptions)
    top = max(assumptions, key=lambda v: abs(v["delta_pct"]))
    ref = [v for v in out if not v["is_assumption"]]
    return {"_disclaimer": OUTPUT_DISCLAIMER,
            "baseline_m": round(base), "variants": out,
            "max_abs_delta_pct": span,
            "dominant_assumption": top["variant"],
            "reference_only_variants": [v["variant"] for v in ref],
            "_read_this": f"Assumption uncertainty moves the compliance distance by up to "
                          f"{span:.0f}% (largest single item: {top['variant']}). Read any single "
                          f"distance as a range. Reference-only variants "
                          f"({', '.join(v['variant'] for v in ref) or 'none'}) are excluded from "
                          f"that range - they are known-optimistic methods, not plausible bounds."}
