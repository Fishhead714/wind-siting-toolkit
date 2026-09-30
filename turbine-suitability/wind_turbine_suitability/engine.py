"""Suitability engine: site conditions x turbine description -> verdict per constraint.

Design rules (the reason this module exists):

1. **Hard constraints are not weighted.** Any failing constraint fails the turbine; no other
   item can compensate. Mixing safety limits with yield or cost terms in one weighted score is
   the classic misuse of multi-criteria methods.
2. **Unknown is never pass.** A missing turbine limit, a missing or non-finite site value, or a
   basis mismatch gives *unknown*; unknowns on required constraints stop the overall result from
   being a pass.
3. **Limits are data.** Class limits come from a :class:`ClassTable`; turbine limits come from
   the :class:`TurbineSpec`. Only a few clearly labelled engineering defaults live in
   :class:`ScreeningConfig`.

This is screening-level logic. See ``DISCLAIMER``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .evidence import decide_tier, measurement_leaks, wording
from .iec import ClassTable, default_table
from .site import SiteConditions, is_finite_value
from .turbine import TurbineSpec

DISCLAIMER = (
    "Screening-level result. It does not replace a formal site suitability assessment, which "
    "is done by a qualified body from turbine load simulations (see IECRE OD-501). "
    "'Preliminary fit' means no known hard constraint was triggered; it does not mean "
    "'compliant with IEC 61400-1'. The bundled class table is an unverified example: check it "
    "against the standard you hold (see ClassTable.verified)."
)


@dataclass
class ScreeningConfig:
    """Engineering defaults. Not taken from a standard; every tolerance band and threshold below is an
    arbitrary example value. Change them to match your own practice."""
    min_tip_clearance_m: float = 25.0          # common rule of thumb
    band_v50_pct: float = 5.0
    band_vave_pct: float = 5.0
    band_ve50_pct: float = 5.0
    band_turbulence_pct: float = 10.0
    band_air_density_pct_of_range: float = 3.0
    band_altitude_pct: float = 5.0
    band_temp_c: float = 3.0
    band_temp_operating_c: float = 2.0
    band_shear_pct: float = 10.0
    band_flow_inclination_pct: float = 20.0
    band_extreme_density_pct: float = 2.0
    ntm_bin_range_ms: tuple = (4.0, 25.0)
    # V50/Vave plausibility window. ARBITRARY example default, deliberately wide; it only raises a
    # data-quality flag (never blocks). Low-wind and cyclone-prone sites can legitimately fall outside.
    v50_over_vave_ok_range: tuple = (2.5, 9.0)
    # verdict caps: a pass is downgraded to marginal when the named site flag is >= the threshold.
    # Flag names and thresholds are ARBITRARY example conventions; use any integer scale you like.
    cap_thresholds: tuple = (("vref", "v50_uncertainty_level", 2),
                             ("turbulence", "ti_uncertainty_level", 2),
                             ("shear", "shear_neutral_only", 1),
                             ("flow_inclination", "terrain_complexity_level", 2))
    required: tuple = ("tip_clearance", "vref", "vave", "turbulence", "air_density", "altitude",
                       "temp_high", "temp_low", "shear", "flow_inclination",
                       "air_density_extreme", "ve50_gust")


@dataclass
class ConstraintResult:
    id: str
    label: str
    verdict: str                 # pass | marginal | fail | unknown
    limit_type: str              # oem_hard | class_assumption | data_quality
    site_value: Any = None
    limit: Any = None
    margin: str = ""
    basis: str = ""
    reason: str = ""
    limit_derived: bool = False


def _R(cid, label, lt, **kw):
    return ConstraintResult(cid, label, kw.pop("verdict", "unknown"), lt, **kw)


def _site(site: SiteConditions, key: str):
    f = site.get(key)
    if f is None:
        return None, f"site field {key} missing"
    if f.evidence == "unavailable" or f.value is None:
        return None, f"site field {key} unavailable. {f.note}".strip()
    if not is_finite_value(f.value):
        return None, f"site field {key} is not finite ({f.value!r}); likely an upstream nodata"
    return f.value, ""


def _scalar_le(sv, limit, band):
    if sv > limit:
        return "fail", f"exceeds by {sv - limit:+.3g}"
    if sv > limit - band:
        return "marginal", f"only {limit - sv:.3g} of headroom (tolerance {band:.3g})"
    return "pass", f"headroom {limit - sv:.3g}"


def _interval_le(sv, limit):
    if not (isinstance(sv, (list, tuple)) and len(sv) == 2):
        return None, None
    lo, hi = float(sv[0]), float(sv[1])
    if hi <= limit:
        return "pass", f"interval [{lo}, {hi}] entirely within {limit}"
    if lo > limit:
        return "fail", f"interval [{lo}, {hi}] entirely above {limit}"
    return "marginal", f"interval [{lo}, {hi}] straddles {limit}; the estimate uncertainty spans the limit"


def _interval_ge(sv, limit):
    if not (isinstance(sv, (list, tuple)) and len(sv) == 2):
        return None, None
    lo, hi = float(sv[0]), float(sv[1])
    if lo >= limit:
        return "pass", f"interval [{lo}, {hi}] entirely above {limit}"
    if hi < limit:
        return "fail", f"interval [{lo}, {hi}] entirely below {limit}"
    return "marginal", f"interval [{lo}, {hi}] straddles {limit}"


class SuitabilityEngine:
    def __init__(self, table: Optional[ClassTable] = None, config: Optional[ScreeningConfig] = None):
        self.table = table or default_table()
        self.cfg = config or ScreeningConfig()

    # ---- generic scalar-vs-limit constraint --------------------------------------------
    def _scalar(self, cid, label, lt, site, skey, limit, band_pct, basis, derived=False,
                interval_op=None, band_abs=None, limit_field=""):
        sv, err = _site(site, skey)
        if sv is None:
            return _R(cid, label, lt, limit=limit, basis=basis, reason=err, limit_derived=derived)
        if limit is None:
            return _R(cid, label, lt, site_value=sv, basis=basis, limit_derived=derived,
                      reason=f"turbine gives no {limit_field}; a missing limit is not 'no limit'")
        if interval_op:
            fn = _interval_le if interval_op == "le" else _interval_ge
            v, m = fn(sv, limit)
            if v is None:
                return _R(cid, label, lt, site_value=sv, limit=limit, basis=basis, limit_derived=derived,
                          reason=f"site field {skey} must be a [low, high] interval, got {sv!r}")
            if band_abs and v == "pass":
                lo, hi = float(sv[0]), float(sv[1])
                edge = hi if interval_op == "le" else lo
                slack = (limit - edge) if interval_op == "le" else (edge - limit)
                if slack < band_abs:
                    v, m = "marginal", f"{m}; only {slack:.3g} of slack (tolerance {band_abs:.3g})"
            return ConstraintResult(cid, label, v, lt, sv, limit, m, basis, "", derived)
        if isinstance(sv, (list, tuple)):
            return _R(cid, label, lt, site_value=sv, limit=limit, basis=basis, limit_derived=derived,
                      reason=f"site field {skey} is an interval but this constraint expects a scalar")
        band = band_abs if band_abs is not None else abs(limit) * band_pct / 100.0
        v, m = _scalar_le(sv, limit, band)
        return ConstraintResult(cid, label, v, lt, sv, limit, m, basis, "", derived)

    # ---- individual constraints ----------------------------------------------------------
    def _tip_clearance(self, site, t):
        cid, label, basis = "tip_clearance", "Tip clearance (geometric feasibility)", \
            "hub height - rotor radius; the default minimum is an industry rule of thumb, not a standard value"
        hub, err = _site(site, "hub_height_m")
        need = t.min_tip_clearance_m if t.min_tip_clearance_m is not None else self.cfg.min_tip_clearance_m
        if hub is not None and isinstance(hub, (list, tuple)):
            return _R(cid, label, "oem_hard", site_value=hub, limit=need, basis=basis,
                      reason="hub_height_m must be a scalar")
        if hub is None:
            return _R(cid, label, "oem_hard", limit=need, basis=basis, reason=err)
        if t.rotor_diameter_m is None:
            return _R(cid, label, "oem_hard", site_value=hub, limit=need, basis=basis,
                      reason="turbine gives no rotor diameter")
        clr = hub - t.rotor_diameter_m / 2.0
        return ConstraintResult(cid, label, "pass" if clr >= need else "fail", "oem_hard",
                                round(clr, 2), need,
                                f"tip clearance {clr:.2f} m, required >= {need}", basis)

    def _vref(self, site, t):
        return self._scalar("vref", "50-yr wind V50 vs turbine Vref", "oem_hard", site, "v50_ms",
                            t.vref_max_ms, self.cfg.band_v50_pct,
                            "reference wind speed = 50-yr 10-min mean at hub height; site V50 must not exceed it",
                            limit_field="vref_max_ms")

    def _vave(self, site, t):
        limit, derived = t.vave_max_ms, False
        basis = "annual mean wind speed vs the class annual-mean limit"
        if limit is None and t.vref_max_ms is not None:
            limit = round(t.vref_max_ms * self.table.vave_over_vref, 4)
            derived = True
            basis += (f" (turbine gives no vave_max_ms; derived as {self.table.vave_over_vref} x Vref"
                      f"{'; NOTE special-class turbine, the ratio is not guaranteed' if str(t.iec_class).upper().startswith('S') else ''})")
        return self._scalar("vave", "Annual mean speed vs turbine Vave", "class_assumption", site, "vave_ms",
                            limit, self.cfg.band_vave_pct, basis, derived, limit_field="vave_max_ms")

    def _turbulence(self, site, t):
        cid, label, lt = "turbulence", "Turbulence intensity vs turbine I_ref", "class_assumption"
        basis = ("normal turbulence model: I_ref is the expected TI at 15 m/s while the site value is a 90th "
                 "percentile, so the limit is a curve I_ref*(0.75U+b)/U compared over the operating range; "
                 "wake-added turbulence is NOT included (real requirement can only be higher)")
        sv, err = _site(site, "ti_p90_range")
        if sv is None:
            return _R(cid, label, lt, basis=basis, reason=err)
        if t.ref_turbulence_iref is None:
            return _R(cid, label, lt, site_value=sv, basis=basis,
                      reason="turbine gives no ref_turbulence_iref; a missing limit is not 'no limit'")
        u_hi = self.cfg.ntm_bin_range_ms[1]
        if t.cutout_wind_speed_ms:
            u_hi = t.cutout_wind_speed_ms
        # the curve falls with U, so the tightest bin is the highest speed
        limit = round(self.table.ntm_limit_curve(t.ref_turbulence_iref, u_hi), 5)
        basis += f" [tightest bin U={u_hi:g} m/s: limit {limit}; note the 15 m/s helpers in turbulence.py are looser]"
        v, m = _interval_le(sv, limit)
        if v is None:
            return _R(cid, label, lt, site_value=sv, limit=limit, basis=basis,
                      reason="site ti_p90_range must be a [low, high] interval")
        if v == "pass":
            hi = float(sv[1])
            if limit - hi < limit * self.cfg.band_turbulence_pct / 100.0:
                v, m = "marginal", m + f"; slack below {self.cfg.band_turbulence_pct:g}% tolerance"
        return ConstraintResult(cid, label, v, lt, sv, limit, m, basis, "", True)

    def _air_density(self, site, t):
        cid, label, lt = "air_density", "Air density inside turbine range", "oem_hard"
        basis = "turbine datasheet mean air-density envelope"
        sv, err = _site(site, "air_density_kgm3")
        lo, hi = t.air_density_min_kgm3, t.air_density_max_kgm3
        if sv is None:
            return _R(cid, label, lt, limit=[lo, hi], basis=basis, reason=err)
        if isinstance(sv, (list, tuple)):
            return _R(cid, label, lt, site_value=sv, limit=[lo, hi], basis=basis,
                      reason="air_density_kgm3 must be a scalar")
        if lo is None or hi is None:
            return _R(cid, label, lt, site_value=sv, limit=[lo, hi], basis=basis,
                      reason="turbine gives no density range")
        band = (hi - lo) * self.cfg.band_air_density_pct_of_range / 100.0
        v = "fail" if (sv < lo or sv > hi) else ("marginal" if (sv < lo + band or sv > hi - band) else "pass")
        return ConstraintResult(cid, label, v, lt, sv, [lo, hi], f"range [{lo}, {hi}], site {sv}", basis)

    def _altitude(self, site, t):
        return self._scalar("altitude", "Altitude vs turbine maximum", "oem_hard", site, "elevation_max_m",
                            t.max_altitude_m, self.cfg.band_altitude_pct,
                            "site side uses the highest point of the area, not a representative point",
                            limit_field="max_altitude_m")

    def _temp_high(self, site, t):
        return self._scalar("temp_high", "High temperature vs operating upper limit", "oem_hard", site,
                            "temp_max_abs_c", t.operating_temp_max_c, 0,
                            "absolute maximum interval vs datasheet operating upper limit",
                            interval_op="le", band_abs=self.cfg.band_temp_c,
                            limit_field="operating_temp_max_c")

    def _temp_low(self, site, t):
        return self._scalar("temp_low", "Low temperature vs survival lower limit", "oem_hard", site,
                            "temp_min_abs_c", t.survival_temp_min_c, 0,
                            "absolute minimum interval vs datasheet survival lower limit",
                            interval_op="ge", band_abs=self.cfg.band_temp_c,
                            limit_field="survival_temp_min_c")

    def _temp_low_operating(self, site, t):
        r = self._scalar("temp_low_operating", "Low temperature vs operating lower limit (yield, not structure)",
                         "yield_impact", site, "temp_min_abs_c", t.operating_temp_min_c, 0,
                         "below the operating limit the turbine stops: lost energy, not damage",
                         interval_op="ge", band_abs=self.cfg.band_temp_operating_c,
                         limit_field="operating_temp_min_c")
        return r

    def _shear(self, site, t):
        limit, derived = t.max_shear_exponent, False
        basis = "turbine max shear exponent; falls back to the class default normal-profile exponent"
        if limit is None:
            limit, derived = self.table.default_shear_alpha, True
            basis += f" (turbine gives none; using default {limit})"
        return self._scalar("shear", "Wind shear exponent vs limit", "class_assumption", site, "shear_exponent",
                            limit, self.cfg.band_shear_pct, basis, derived)

    def _flow_inclination(self, site, t):
        limit, derived = t.max_flow_inclination_deg, False
        basis = "inflow angle vs turbine limit, falling back to the class default"
        if limit is None:
            limit, derived = self.table.default_flow_inclination_deg, True
            basis += f" (default {limit} deg)"
        return self._scalar("flow_inclination", "Inflow angle vs limit", "class_assumption", site,
                            "flow_inclination_deg", limit, self.cfg.band_flow_inclination_pct, basis, derived)

    def _air_density_extreme(self, site, t):
        cid, label, lt = "air_density_extreme", "Extreme air density (cold, dense end)", "oem_hard"
        basis = "extreme load cases use the cold, high-density end; the annual mean does not cover it"
        limit, derived = t.extreme_air_density_max_kgm3, False
        if limit is None and t.air_density_max_kgm3 is not None:
            limit, derived = t.air_density_max_kgm3, True
            basis += " (falls back to the mean-density envelope: a basis mismatch, so an exceedance is only 'marginal')"
        r = self._scalar(cid, label, lt, site, "air_density_cold_extreme_kgm3", limit,
                         self.cfg.band_extreme_density_pct, basis, derived,
                         limit_field="extreme_air_density_max_kgm3")
        if derived and r.verdict == "fail":
            r.verdict = "marginal"
            r.margin += " (proxy limit: confirm with the turbine maker)"
        return r

    def _ve50(self, site, t):
        limit, derived = t.ve50_max_ms, False
        basis = ("extreme gust vs turbine limit. NOTE: a fixed gust ratio only holds for standard classes; "
                 "in tropical-cyclone regions real gust factors can be far larger")
        if limit is None and t.vref_max_ms is not None:
            limit, derived = round(t.vref_max_ms * self.table.ve50_over_vref, 3), True
            basis += f" (derived as {self.table.ve50_over_vref} x Vref)"
        return self._scalar("ve50_gust", "Extreme gust Ve50 vs turbine limit", "oem_hard", site, "ve50_ms",
                            limit, self.cfg.band_ve50_pct, basis, derived, limit_field="ve50_max_ms")

    def _v50_consistency(self, site):
        cid, label = "v50_consistency", "V50 / Vave consistency (data quality)"
        basis = ("class tables have a fixed Vref/Vave ratio; a site estimate far from it means V50 or Vave "
                 "is unreliable. This judges the card, not the turbine.")
        a, e1 = _site(site, "v50_ms")
        b, e2 = _site(site, "vave_ms")
        if a is None or b is None or isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)) or b <= 0:
            return _R(cid, label, "data_quality", basis=basis, reason=e1 or e2 or "need scalar v50_ms and vave_ms")
        r = a / b
        lo, hi = self.cfg.v50_over_vave_ok_range
        ok = lo <= r <= hi
        return ConstraintResult(cid, label, "pass" if ok else "fail", "data_quality", round(r, 3), [lo, hi],
                                f"ratio {r:.2f} {'inside' if ok else 'OUTSIDE'} [{lo}, {hi}]", basis,
                                "" if ok else "flag only (does not change the verdict); which source to fix is a human call")

    # ---- caps ------------------------------------------------------------------------
    def _apply_caps(self, results, site):
        capped = []
        for r in results:
            if r.verdict != "pass":
                continue
            for cid, flag, thr in self.cfg.cap_thresholds:
                if cid != r.id:
                    continue
                v = site.value(flag)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= thr:
                    r.verdict = "marginal"
                    r.margin += f" | capped to marginal: site flag {flag}={v:g} (>= {thr:g})"
                    capped.append(r.id)
                    break
        return capped

    # ---- aggregation ---------------------------------------------------------------------
    ORDER = {"pass": 0, "marginal": 1, "exceeds_class_assumption": 2,
             "insufficient_data": 4, "fail": 5}

    def evaluate(self, site: SiteConditions, turbine: TurbineSpec, tier: Optional[str] = None) -> dict:
        tier = tier or decide_tier()
        results: List[ConstraintResult] = [
            self._tip_clearance(site, turbine), self._vref(site, turbine), self._vave(site, turbine),
            self._turbulence(site, turbine), self._air_density(site, turbine), self._altitude(site, turbine),
            self._temp_high(site, turbine), self._temp_low(site, turbine), self._shear(site, turbine),
            self._flow_inclination(site, turbine), self._air_density_extreme(site, turbine),
            self._ve50(site, turbine), self._temp_low_operating(site, turbine),
            self._v50_consistency(site),
        ]
        capped = self._apply_caps(results, site)
        required = set(self.cfg.required)
        fails = [r for r in results if r.verdict == "fail"]
        hard = [r for r in fails if r.limit_type == "oem_hard"]
        cls = [r for r in fails if r.limit_type == "class_assumption"]
        dq = [r for r in fails if r.limit_type == "data_quality"]
        yl = [r for r in fails if r.limit_type == "yield_impact"]
        unk = [r for r in results if r.verdict == "unknown" and r.id in required]
        marg = [r for r in results if r.verdict == "marginal"]
        if hard:
            overall = "fail"
        elif cls:
            overall = "exceeds_class_assumption"
        elif unk:
            overall = "insufficient_data"
        elif marg:
            overall = "marginal"
        else:
            overall = "pass"
        label = wording(tier, overall)
        if dq:
            label += f" | data-quality flag raised: {[r.id for r in dq]}"
        if yl:
            label += f" | yield-impact flag (not a structural limit): {[r.id for r in yl]}"
        leaked = measurement_leaks(tier, site)
        derate = None
        tmax = site.value("temp_max_abs_c")
        if turbine.derating_temp_c is not None and isinstance(tmax, (list, tuple)) and tmax \
                and is_finite_value(tmax) and float(tmax[0]) > turbine.derating_temp_c:
            derate = (f"site absolute-maximum interval {list(tmax)} C is entirely above the turbine's "
                      f"derating temperature {turbine.derating_temp_c} C: expect derated operation in hot "
                      "seasons. A yield issue, not a safety one; it does not change the verdict.")
        return {
            "site": site.name, "turbine": turbine.model,
            "hub_height_m": site.value("hub_height_m"),
            "turbine_source": turbine.source,
            "overall": overall, "overall_label": label, "tier": tier,
            "blocking_hard_limits": [r.id for r in hard],
            "exceeds_class_assumptions": [r.id for r in cls],
            "data_quality_flags": [r.id for r in dq],
            "yield_flags": [r.id for r in yl],
            "table_verified": self.table.verified,
            "verdict_capped": capped, "derating_flag": derate,
            "missing": [r.id for r in unk], "to_review": [r.id for r in marg],
            "derived_limits": [r.id for r in results if r.limit_derived and r.verdict != "unknown"],
            "tier_policy_violation": (f"measured fields in a model-only tier: {leaked}" if leaked else None),
            "constraints": [asdict(r) for r in results],
            "disclaimer": DISCLAIMER,
        }

    def screen(self, sites_by_hub: Dict[float, SiteConditions], catalog: Sequence[TurbineSpec],
               tier: Optional[str] = None) -> List[dict]:
        """Judgement unit = (turbine x hub-height variant).

        Hub height is a turbine attribute: TI falls and mean/extreme wind rise with height, so
        each turbine is evaluated on the card built for its own hub height. A turbine that
        declares no ``hub_height_options_m`` is *insufficient_data*; no site default is imposed.
        """
        tier = tier or decide_tier()
        if not sites_by_hub:
            raise ValueError("sites_by_hub is empty: supply at least one site card")
        out = []
        any_site = next(iter(sites_by_hub.values()))
        for t in catalog:
            hubs = t.hub_height_options_m
            if not hubs:
                r = self.evaluate(any_site, t, tier)
                r["overall"], r["hub_height_m"] = "insufficient_data", None
                r["overall_label"] = wording(tier, "insufficient_data")
                r["missing"] = sorted(set(r["missing"]) | {"hub_height_options_m"})
                out.append(r)
                continue
            for h in hubs:
                s = sites_by_hub.get(float(h))
                if s is None:
                    r = self.evaluate(any_site, t, tier)
                    r["overall"], r["hub_height_m"] = "insufficient_data", float(h)
                    r["overall_label"] = wording(tier, "insufficient_data")
                    r["missing"] = sorted(set(r["missing"]) | {f"site_card_for_hub_{h:g}m"})
                    out.append(r)
                    continue
                out.append(self.evaluate(s, t, tier))
        return sorted(out, key=lambda r: (self.ORDER[r["overall"]], len(r["missing"]),
                                          r["turbine"], r["hub_height_m"] or 0))
