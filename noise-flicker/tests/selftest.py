#!/usr/bin/env python3
"""Regression suite for the noise / shadow-flicker engine: ``python tests/selftest.py``.

Every test guards against a specific *silent* failure - a mistake that produces a
plausible-looking number instead of an error. When a check fails, read the docstring of
its test first: it states what the check prevents.

Inputs:
  * criteria  - tests/fixtures/criteria_fixture.json (fictional jurisdictions XI/XA/XB/XN);
                expected limits are computed from that file, never hard-coded here;
  * turbine   - examples/reference_turbine.json (an openly documented reference turbine);
                every turbine parameter in this file is taken from or derived from it;
  * spectrum  - the example octave-band spectrum named in reference_turbine.json.

Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import copy
import csv
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "engine"))

import numpy as np   # noqa: E402

import criteria      # noqa: E402
import flicker       # noqa: E402
import layers        # noqa: E402
import noise         # noqa: E402
import solar         # noqa: E402

# ---------------------------------------------------------------- shared inputs
FIXTURE = HERE / "fixtures" / "criteria_fixture.json"
FIX = json.loads(FIXTURE.read_text(encoding="utf-8"))["countries"]
REF = json.loads((ROOT / "examples" / "reference_turbine.json").read_text(encoding="utf-8"))

D = float(REF["rotor_diameter_m"])
HUB = float(REF["hub_height_m"])
LWA = float(REF["lwa_dBA"])
CHORD = float(REF["blade_chord_m"])
SPEC = REF["spectrum"]

# Registry via set_path() for this process and via the environment for worker processes
# (child processes do not inherit set_path()).
criteria.set_path(FIXTURE)
os.environ[criteria.ENV] = str(FIXTURE)
noise.load_spectra(ROOT / "examples" / REF["spectrum_file"])

XI_NOISE = FIX["XI"]["noise"]
ALLOW_NIGHT = float(XI_NOISE["allowance_dB"]["night"])
ALLOW_DAY = float(XI_NOISE["allowance_dB"]["day"])
XA_NIGHT = float(FIX["XA"]["noise"]["limits_dB_A"]["night"])
XB_NOISE = FIX["XB"]["noise"]
XB_TABLE_NIGHT = float(XB_NOISE["limits_dB_A"][XB_NOISE["binding"]]["night"])
XB_ADJ = XB_NOISE["background_adjustment"]
XB_MARGIN = float(XB_ADJ["margin_dB"])
FLICKER_LIMIT_H = float(FIX["XI"]["shadow_flicker"]["limit_hours_per_year"])

LOW_LAT = dict(lat_deg=20.0, lon_deg=75.0, utc_offset_hours=5)       # low latitude
EQ_S_LAT = dict(lat_deg=-10.0, lon_deg=-30.0, utc_offset_hours=-2)   # southern tropics, open ocean
HIGH_LAT = dict(lat_deg=54.0, lon_deg=9.0, utc_offset_hours=1)       # high latitude

CRS_A, CRS_B = "EPSG:32625", "EPSG:32626"
X0, Y0 = 280_000.0, 1_000_000.0        # synthetic projected origin (same as demo/make_demo_data.py)

PASSED, FAILS = [], []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    (PASSED if cond else FAILS).append(name)


def atm(t_c=22.0, rh=70.0, **kw):
    return noise.Atmosphere(t_c, rh, spectrum=kw.pop("spectrum", SPEC), **kw)


def fturb(x=0.0, y=0.0, *, hub=HUB, dia=D, wid="T1", crs=None):
    return flicker.FlickerTurbine(x, y, hub, dia, wid, crs=crs)


def nturb(x=0.0, y=0.0, *, hub=HUB, lwa=LWA, wid="T", crs=None):
    return noise.Turbine(x, y, hub, lwa, wid, crs=crs)


class _TempRegistry:
    """Point the criteria loader at a temporary copy of the fixture (so the fixture file is
    never modified) and restore the fixture afterwards, even on failure."""

    def __enter__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="wnf_crit_"))
        self.path = self.dir / "criteria.json"
        shutil.copyfile(FIXTURE, self.path)
        criteria.set_path(self.path)
        return self.path

    def __exit__(self, *exc):
        criteria.set_path(FIXTURE)
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


# ============================================================================ noise
def t1_alpha():
    """Atmospheric absorption. A mistyped exponent in the ISO 9613-1 closed form raises no
    error; it just shifts every distance. Reference values: ISO 9613-2:1996 Table 2."""
    print("\n[1] ISO 9613-1 absorption alpha vs ISO 9613-2:1996 Table 2")
    ref = {(10, 70): {63: 0.1, 125: 0.4, 250: 1.0, 500: 1.9, 1000: 3.7,
                      2000: 9.7, 4000: 32.8, 8000: 117.0},
           (20, 70): {63: 0.1, 125: 0.3, 250: 1.1, 500: 2.8, 1000: 5.0,
                      2000: 9.0, 4000: 22.9, 8000: 76.6},
           (15, 20): {63: 0.3, 125: 0.6, 250: 1.2, 500: 2.7, 1000: 8.2,
                      2000: 28.2, 4000: 88.8, 8000: 202.0}}
    bad = []
    for (t, rh), row in ref.items():
        for f, expect in row.items():
            got = noise.alpha_iso9613_1(f, t, rh, noise.PA_STANDARD)
            rel = abs(got - expect) / expect
            if not (rel < 0.15 or abs(got - expect) < 0.15):
                bad.append((t, rh, f, expect, round(got, 2)))
    check("all 24 reference values within 15 %", not bad, f"out of tolerance: {bad}" if bad else "")


def t2_ground_3db_trap():
    """Ground term. Using the s.7.3.2 alternative method with negative values clipped to
    zero silently drops the 3 dB hard-ground reflection gain and shortens compliance
    distances by tens of percent. Locks: hard ground is exactly -3 dB (a gain) and the
    alternative method is more optimistic."""
    print("\n[2] A_gr: the 3 dB trap")
    d_in = 0.8 * _analytic_root(HUB, 1.5)          # inside the iso_alt zero-clipped range
    hard = noise.a_ground(d_in, HUB, 1.5, "hard")
    alt = noise.a_ground(d_in, HUB, 1.5, "iso_alt")
    check("hard (G=0) = -3.0 dB gain", abs(hard + 3.0) < 1e-9, f"got {hard}")
    check("iso_alt inside its break point = 0 (reflection gain lost)", abs(alt) < 1e-9,
          f"got {alt} at {d_in:.0f} m")
    d1 = noise.required_distance(LWA, 35, h_src=HUB, atm=atm())
    d2 = noise.required_distance(LWA, 35, h_src=HUB, atm=atm(ground="iso_alt"))
    check("iso_alt gives a shorter (more optimistic) distance", d2 < d1,
          f"hard {d1:.0f} m vs alt {d2:.0f} m, {100*(d1-d2)/d1:.0f} % shorter")
    far_d = 3.0 * 30.0 * (HUB + 1.5)
    far = noise.a_ground(far_d, HUB, 1.5, "hard")
    check("beyond d = 30(hs+hr) q takes effect and A_gr moves below -3", far < -3.0,
          f"{far:.2f} dB at {far_d:.0f} m")


def t3_distance_self_consistency():
    """Compliance distance. A propagation change that breaks the array sum or the L_WA
    dependence would not raise; it would just move every setback. Self-consistency:
    array > single, monotonic in L_WA, +10 dB L_WA gives a clearly larger distance."""
    print("\n[3] required_distance self-consistency")
    a = atm()

    def dist(lwa, arr):
        return noise.required_distance(lwa, 35.0, h_src=HUB, atm=a,
                                       array_n=(9 if arr else 1), spacing=700.0)
    single, array = dist(LWA, False), dist(LWA, True)
    check("array distance > single-turbine distance", array > single,
          f"single {single:.0f} m, array {array:.0f} m")
    seq = [dist(LWA + dl, True) for dl in (-3.0, -1.5, 0.0, 1.5, 3.0)]
    check("distance increases monotonically with L_WA",
          all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)),
          " < ".join(f"{v:.0f}" for v in seq))
    plus10 = dist(LWA + 10.0, False)
    check("+10 dB L_WA gives a clearly larger distance (> 1.5x)", plus10 > 1.5 * single,
          f"{single:.0f} m -> {plus10:.0f} m")


def t4_superposition():
    """Energy summation must be 10 lg(N). Arithmetic addition or taking the maximum would
    silently give a wrong verdict - one turbine complying does not mean the array complies."""
    print("\n[4] multi-turbine energy sum")
    a = atm()
    for n in (2, 4, 9):
        ts = [nturb(wid=f"T{i}") for i in range(n)]              # co-located
        lp, _, lp1, _ = noise.lp_at_point(ts, 800.0, 0.0, a)
        expect = lp1 + 10 * math.log10(n)
        check(f"{n} co-located turbines = single + 10 lg({n})", abs(lp - expect) < 1e-9,
              f"{lp:.3f} vs {expect:.3f} dB(A)")


def t5_spectrum_order():
    """Low frequencies are absorbed less, so a spectrum shifted towards low frequencies must
    give a longer compliance distance. The wrong order means the spectral normalisation is
    broken."""
    print("\n[5] spectral shape: direction of the effect")
    noise.register_spectrum("_lf", noise.shift_spectrum(noise.SPECTRA[SPEC], -1))
    noise.register_spectrum("_hf", noise.shift_spectrum(noise.SPECTRA[SPEC], +1))
    ds = {sp: noise.required_distance(LWA, 35, h_src=HUB, atm=atm(spectrum=sp), array_n=9)
          for sp in ("_lf", SPEC, "_hf")}
    check("low-frequency shift > base > high-frequency shift",
          ds["_lf"] > ds[SPEC] > ds["_hf"], "  ".join(f"{k}={v:.0f}m" for k, v in ds.items()))


def t6_criteria_increment():
    """Increment-over-background criterion. Locks the night allowance and, above all, that a
    missing background raises instead of falling back to a default - a silent default would
    look like an absolute conclusion."""
    print("\n[6] increment criterion (XI)")
    bg = 30.0
    lim_n, info_n = criteria.noise_limit("XI", period="night", background_la90=bg)
    lim_d, _ = criteria.noise_limit("XI", period="daytime", background_la90=bg)
    check("night = LA90 + allowance[night]", abs(lim_n - (bg + ALLOW_NIGHT)) < 1e-9, f"got {lim_n}")
    check("day = LA90 + allowance[day]", abs(lim_d - (bg + ALLOW_DAY)) < 1e-9, f"got {lim_d}")
    check("night is stricter than day", lim_n < lim_d)
    check("output labels the background as not measured", info_n.get("background_is_measured") is False)
    try:
        criteria.noise_limit("XI", period="night")
        check("missing LA90 raises", False, "no error raised")
    except ValueError as e:
        rng = str(XI_NOISE["background_LA90"]["assumed_range_dB_A"])
        check("missing LA90 raises and quotes the assumed range", rng in str(e), rng)


def t7_criteria_guards():
    """Two registry guards: an unresearched entry must be refused, and a criterion without a
    numeric threshold must return None rather than an invented number."""
    print("\n[7] registry guards")
    for ctry, topic in (("XB", "flicker"), ("XN", "flicker"), ("XN", "noise")):
        try:
            (criteria.flicker_limits(ctry) if topic == "flicker"
             else criteria.noise_limit(ctry, period="night", background_la90=30.0))
            check(f"{ctry} {topic} NOT_RESEARCHED -> refused", False, "returned a value")
        except NotImplementedError:
            check(f"{ctry} {topic} NOT_RESEARCHED -> refused", True)
    h, m, _ = criteria.flicker_limits("XA")
    check("XA flicker without numeric threshold -> (None, None)", h is None and m is None)
    fx = FIX["XI"]["shadow_flicker"]
    h2, m2, i2 = criteria.flicker_limits("XI")
    check("XI flicker thresholds read from the registry and not a hard veto",
          h2 == fx["limit_hours_per_year"] and m2 == fx["limit_minutes_per_day"]
          and not i2["is_hard_veto"], f"{h2} h / {m2} min")
    lim_a, _ = criteria.noise_limit("XA", period="night")
    check("XA flat absolute night limit", abs(lim_a - XA_NIGHT) < 1e-9, f"got {lim_a}")
    _, info_b = criteria.noise_limit("XB", period="night", background_la90=XB_TABLE_NIGHT - 20)
    check("XB binding-zone night table value", abs(info_b["table_value_dBA"] - XB_TABLE_NIGHT) < 1e-9,
          f"got {info_b['table_value_dBA']}")
    try:
        criteria.jurisdiction("ZZ")
        check("unknown jurisdiction raises", False)
    except KeyError:
        check("unknown jurisdiction raises", True)


def t8_assess_end_to_end():
    """Per-receptor assessment: the array penalty must be reported, and a receptor that is
    geometrically closer must never come out quieter."""
    print("\n[8] per-receptor assessment end to end")
    a = atm()
    ts = [nturb(i * 700.0, 0.0, wid=f"WTG{i+1}") for i in range(5)]
    rs = [noise.Receptor(1400.0, d, f"R{d}") for d in (400, 800, 1600, 3000)]
    out = noise.assess_receptors(ts, rs, a, country="XI", period="night",
                                 default_background_la90=30.0)
    lps = [r["Lp_total_dBA"] for r in sorted(out["receptors"], key=lambda x: x["y"])]
    check("level falls monotonically with distance",
          all(lps[i] > lps[i + 1] for i in range(len(lps) - 1)), " > ".join(f"{v}" for v in lps))
    check("array penalty > 0 (summation active)", out["summary"]["max_array_penalty_dB"] > 0,
          f"max {out['summary']['max_array_penalty_dB']} dB")
    check("limit = 30 + allowance[night]",
          all(r["limit_dBA"] == round(30.0 + ALLOW_NIGHT, 1) for r in out["receptors"]))
    check("output carries the disclaimer", "Screening-level" in out["_disclaimer"])
    check("output carries criterion provenance", out["criterion"].get("registry", "").endswith(".json"))


# ============================================================================ solar / flicker
def t9_solar():
    """Solar position physics. Writing the afternoon branch of the NOAA azimuth as
    `360 - acos` is a common error that puts the noon sun on the wrong side of the sky."""
    print("\n[9] solar position (NOAA)")
    noon = np.array(["2026-03-20T12:00:00"], dtype="datetime64[s]")
    e_p, a_p = solar.position(noon, 30.0, 0.0)
    _, a_m = solar.position(noon, -30.0, 0.0)
    check("lat +30 equinox noon: sun azimuth ~180", abs(a_p[0] - 180.0) < 20.0, f"az={a_p[0]:.1f}")
    check("lat -30 equinox noon: sun azimuth ~0/360", a_m[0] < 20 or a_m[0] > 340, f"az={a_m[0]:.1f}")
    check("lat 30 equinox noon elevation ~60", abs(e_p[0] - 60.0) < 1.0, f"{e_p[0]:.2f} deg")
    for date in ("2026-06-21T12:00:00", "2026-12-21T12:00:00"):
        t = np.array([date], dtype="datetime64[s]")
        e, _ = solar.position(t, 0.0, 0.0, refraction=False)
        decl = 90.0 - e[0]
        check(f"{date[:10]} equator noon elevation gives |declination| ~23.44",
              abs(decl - 23.44) < 0.6, f"{decl:.2f} deg")
    day = np.arange(np.datetime64("2026-03-20T00:00:00"), np.datetime64("2026-03-21T00:00:00"),
                    np.timedelta64(1, "m"), dtype="datetime64[s]")
    e, a = solar.position(day, 0.0, 0.0)
    up = e > 0
    check("equinox sunrise azimuth ~90 (east)", abs(a[up][0] - 90.0) < 2.0, f"{a[up][0]:.1f} deg")
    check("equinox sunset azimuth ~270 (west)", abs(a[up][-1] - 270.0) < 2.0, f"{a[up][-1]:.1f} deg")
    check("equinox day length ~12 h", abs(up.sum() / 60.0 - 12.0) < 0.3, f"{up.sum()/60.0:.2f} h")


def t10_flicker_direction():
    """The shadow falls opposite the sun: a receptor west of the turbine is hit only in the
    morning, one to the east only in the afternoon. A mirrored azimuth flips this at once -
    far more sensitive than any annual total."""
    print("\n[10] flicker time of day vs direction")
    out = flicker.assess_flicker(
        [fturb()], [flicker.FlickerReceptor(-600, 0, "W"), flicker.FlickerReceptor(600, 0, "E")],
        country="XI", blade_chord_m=CHORD, **LOW_LAT)
    got = {r["receptor_id"]: set(r["hour_of_day_minutes"]) for r in out["receptors"]}
    check("west receptor only before noon", got["W"] and all(h < 12 for h in got["W"]),
          f"hours {sorted(got['W'])}")
    check("east receptor only after noon", got["E"] and all(h >= 12 for h in got["E"]),
          f"hours {sorted(got['E'])}")


def t11_flicker_distance():
    """Distance monotonicity, distance cut-off and rotor-size monotonicity. Any of them
    reversed means the ray/rotor intersection is wrong."""
    print("\n[11] flicker monotonicity and cut-off")
    dists = [300, 600, 1000, 1500]
    out = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-d, 0, str(d)) for d in dists],
                                 country="XI", blade_chord_m=CHORD, **LOW_LAT)
    h = {r["receptor_id"]: r["astro_hours_per_year"] for r in out["receptors"]}
    seq = [h[str(d)] for d in dists]
    check("hours fall monotonically with distance",
          all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)), " > ".join(f"{v:.1f}" for v in seq))
    dmax = flicker.max_distance_m(CHORD)
    far = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-(dmax + 500), 0, "far")],
                                 country="XI", blade_chord_m=CHORD, **LOW_LAT)
    check("beyond the distance cut-off -> 0 h", far["receptors"][0]["astro_hours_per_year"] == 0.0,
          f"d_max={dmax:.0f} m")
    big_d = D * 1.3
    big = flicker.assess_flicker([fturb(dia=big_d)], [flicker.FlickerReceptor(-600, 0, "r")],
                                 country="XI", blade_chord_m=CHORD,
                                 **LOW_LAT)["receptors"][0]["astro_hours_per_year"]
    check("larger rotor -> more flicker", big > h["600"],
          f"D x1.3 {big:.1f} h vs D {h['600']:.1f} h")


#: Receptor distance for the latitude test, in hub heights. Whether the poleward receptor
#: at high latitude is hit depends on the distance / hub-height ratio (shadow length), so the
#: distance scales with the turbine instead of being a fixed number of metres.
BEARING_DIST_HUBS = 4.0


def _four_bearings(site, ctry):
    r = BEARING_DIST_HUBS * HUB
    o = flicker.assess_flicker(
        [fturb()], [flicker.FlickerReceptor(-r, 0, "W"), flicker.FlickerReceptor(r, 0, "E"),
                    flicker.FlickerReceptor(0, r, "Yp"), flicker.FlickerReceptor(0, -r, "Ym")],
        country=ctry, blade_chord_m=CHORD, **site)
    return {r["receptor_id"]: r["astro_hours_per_year"] for r in o["receptors"]}


def t12_expected_and_latitude():
    """(a) The expected value must not exceed the astronomical maximum and must not vanish.
    (b) Latitude effect. The mechanism is *which bearings* are affected, not the duration at
    a single receptor: at high latitude the winter sun stays low, shadows are long and point
    poleward, so the poleward receptor is hit all winter; at low latitude the noon sun is
    always high and the receptors along the y axis stay at zero. All four bearings must be
    summed - omitting E (which equals W at both latitudes) mechanically lowers the ratio."""
    print("\n[12] expected value and latitude effect")
    ts, rs = [fturb()], [flicker.FlickerReceptor(-600, 0, "r")]
    kw = dict(country="XI", blade_chord_m=CHORD, **LOW_LAT)
    rose = {d: 1.0 for d in range(0, 360, 30)}
    a = flicker.assess_flicker(ts, rs, **kw)["receptors"][0]
    b = flicker.assess_flicker(ts, rs, wind_rose=rose, sunshine_probability=0.6, **kw)["receptors"][0]
    check("no wind rose -> expected is None", a["expected_hours_per_year"] is None)
    check("expected <= astronomical maximum", b["expected_hours_per_year"] <= b["astro_hours_per_year"],
          f"{b['expected_hours_per_year']} <= {b['astro_hours_per_year']}")
    check("expected > 0 (weighting did not zero the result)", b["expected_hours_per_year"] > 0)

    eq = _four_bearings(EQ_S_LAT, "XA")
    lo = _four_bearings(LOW_LAT, "XI")
    hi = _four_bearings(HIGH_LAT, "XI")
    check("lat -10: receptors along the y axis = 0", eq["Yp"] == 0.0 and eq["Ym"] == 0.0,
          f"Yp={eq['Yp']}, Ym={eq['Ym']}")
    check("lat 20: receptors along the y axis = 0", lo["Yp"] == 0.0 and lo["Ym"] == 0.0,
          f"Yp={lo['Yp']}, Ym={lo['Ym']}")
    check("lat 54: poleward receptor is the most affected", hi["Yp"] > hi["W"] > 0,
          f"poleward={hi['Yp']:.1f} > W={hi['W']:.1f} h")
    check("E ~ W at both latitudes (within 2 %)",
          100 * abs(eq["E"] - eq["W"]) / eq["W"] < 2.0 and 100 * abs(hi["E"] - hi["W"]) / hi["W"] < 2.0,
          f"low E/W={eq['E']:.1f}/{eq['W']:.1f}  high E/W={hi['E']:.1f}/{hi['W']:.1f}")
    tot_eq, tot_hi = sum(eq.values()), sum(hi.values())
    check("four-bearing total: low latitude < high latitude", tot_eq < tot_hi,
          f"{tot_eq:.1f} h vs {tot_hi:.1f} h = {100*tot_eq/tot_hi:.1f} %")
    # Numeric bands below depend on the reference turbine geometry (see module docstring).
    check("low-latitude W receptor between 50 % and 100 % of the high-latitude one",
          0.5 < eq["W"] / hi["W"] < 1.0, f"{100*eq['W']/hi['W']:.0f} %")
    check("four-bearing ratio low/high within 45-60 %", 0.45 < tot_eq / tot_hi < 0.60,
          f"{100*tot_eq/tot_hi:.1f} %")
    o = flicker.assess_flicker(ts, rs, country="XA", blade_chord_m=CHORD, **EQ_S_LAT)
    check("no numeric threshold -> status NO_THRESHOLD", o["receptors"][0]["status"] == "NO_THRESHOLD")
    check("no-threshold disclaimer says so", "no numeric threshold" in o["_disclaimer"])


# ============================================================================ input guards
def t13_crs_guard():
    """Turbine and receptor layers that each estimate their own UTM zone end up in different
    zones across a zone boundary; distances are then off by orders of magnitude and a
    receptor with a negative level still 'complies'. Guards: CRS label comparison and a
    magnitude check."""
    print("\n[13] cross-CRS and magnitude guards")
    a = atm(20.0, 70.0)
    ta = nturb(wid="W1", crs=CRS_A)
    try:
        noise.assess_receptors([ta], [noise.Receptor(600, 0, "H1", crs=CRS_B)], a,
                               country="XI", default_background_la90=30.0)
        check("different CRS raises", False, "passed silently")
    except ValueError as e:
        check("different CRS raises", "same CRS" in str(e))
    try:
        noise.assess_receptors([nturb(wid="W")], [noise.Receptor(500_000, 0, "H")], a,
                               country="XI", default_background_la90=30.0)
        check("implausible distance (500 km) raises", False, "passed silently")
    except ValueError as e:
        check("implausible distance (500 km) raises", "km" in str(e))
    o = noise.assess_receptors([ta], [noise.Receptor(600, 0, "H1", crs=CRS_A)], a,
                               country="XI", default_background_la90=30.0)
    check("same CRS passes", o["receptors"][0]["Lp_total_dBA"] > 0)


def t14_period_no_fallback():
    """An unrecognised period must raise. Silently falling back to the daytime allowance
    relaxes the night limit and shortens the setback."""
    print("\n[14] period has no silent fallback")
    bads = ("NIGHT", "evening", "nighttime", "", "Night")
    ok = 0
    for bad in bads:
        try:
            criteria.noise_limit("XI", period=bad, background_la90=30.0)
        except ValueError:
            ok += 1
    check(f"{len(bads)} invalid periods all raise", ok == len(bads), f"{ok}/{len(bads)}")
    check("night / daytime / day resolve as expected",
          criteria.noise_limit("XI", period="night", background_la90=30.0)[0] == 30.0 + ALLOW_NIGHT
          and criteria.noise_limit("XI", period="day", background_la90=30.0)[0] == 30.0 + ALLOW_DAY
          and criteria.noise_limit("XI", period="daytime", background_la90=30.0)[0] == 30.0 + ALLOW_DAY)
    try:
        criteria.noise_limit("XA", period="evening")
        check("absolute limit: invalid period raises", False)
    except (ValueError, KeyError):
        check("absolute limit: invalid period raises", True)


def t15_missing_allowance():
    """Allowances are numeric registry fields. A registry entry that lacks the allowance for
    the requested period must raise - never fall back to the other period's allowance or to
    a number parsed out of the rule text."""
    print("\n[15] missing allowance in the registry")
    check("fixture has numeric allowance_dB", isinstance(XI_NOISE.get("allowance_dB"), dict))
    with _TempRegistry() as p:
        d = json.loads(p.read_text(encoding="utf-8"))
        del d["countries"]["XI"]["noise"]["allowance_dB"]["night"]
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        try:
            criteria.noise_limit("XI", period="night", background_la90=30.0)
            check("registry without allowance_dB[night] raises", False, "passed silently")
        except ValueError as e:
            check("registry without allowance_dB[night] raises", "allowance_dB" in str(e))
        check("the day allowance is still served for period=day",
              criteria.noise_limit("XI", period="day", background_la90=30.0)[0] == 30.0 + ALLOW_DAY)
    check("after restore the limit is bg + allowance[night] again",
          criteria.noise_limit("XI", period="night", background_la90=30.0)[0] == 30.0 + ALLOW_NIGHT)


def _xb_expected(bg):
    if bg > XB_TABLE_NIGHT:
        return XB_TABLE_NIGHT + XB_ADJ["above_table_value"]
    if bg > XB_TABLE_NIGHT - XB_MARGIN:
        return XB_TABLE_NIGHT + XB_ADJ["within_margin_below"]
    return XB_TABLE_NIGHT + XB_ADJ["below_by_more_than_margin"]


def t16_background_adjustment():
    """Background-adjusted absolute limit: the table value moves with the background level.
    Handing out the fixed table value, or a limit without a background, is silently wrong."""
    print("\n[16] background-adjusted absolute limit (XB)")
    try:
        criteria.noise_limit("XB", period="night")
        check("missing background raises", False, "returned a number")
    except ValueError:
        check("missing background raises", True)
    cases = {"above table": XB_TABLE_NIGHT + 3,
             "at table (within margin)": XB_TABLE_NIGHT,
             "within margin": XB_TABLE_NIGHT - XB_MARGIN / 2,
             "exactly table - margin": XB_TABLE_NIGHT - XB_MARGIN,
             "below margin": XB_TABLE_NIGHT - XB_MARGIN - 5}
    got = {k: criteria.noise_limit("XB", period="night", background_la90=bg)[0] for k, bg in cases.items()}
    want = {k: _xb_expected(bg) for k, bg in cases.items()}
    check("bands follow margin/adjustments from the registry", got == want, str(got))
    check("the three adjustment bands give distinct limits",
          len({want["above table"], want["within margin"], want["below margin"]}) == 3)
    info = criteria.noise_limit("XB", period="night", background_la90=XB_TABLE_NIGHT - XB_MARGIN)[1]
    check("boundary value takes the lower band and says so",
          info["adjustment_dB"] == XB_ADJ["below_by_more_than_margin"] and info.get("boundary_caveat"))


def t17_empty_and_nan_inputs():
    """Empty turbine list, empty receptor list, NaN coordinates and NaN background would all
    silently produce 'everything complies'."""
    print("\n[17] empty and NaN inputs")
    a = atm(20.0, 70.0)
    T, R = [nturb()], [noise.Receptor(600, 0, "R")]
    fr = [flicker.FlickerReceptor(600, 0, "R")]
    kw = dict(country="XI", blade_chord_m=CHORD, **LOW_LAT)
    for name, fn in [
        ("noise: empty turbine list", lambda: noise.assess_receptors(
            [], R, a, country="XI", default_background_la90=30.0)),
        ("flicker: empty turbine list", lambda: flicker.assess_flicker([], fr, **kw)),
        ("noise: empty receptor list", lambda: noise.assess_receptors(
            T, [], a, country="XI", default_background_la90=30.0)),
        ("NaN coordinate", lambda: noise.assess_receptors(
            T, [noise.Receptor(float("nan"), 0, "R")], a, country="XI", default_background_la90=30.0)),
        ("NaN background LA90", lambda: noise.assess_receptors(
            T, [noise.Receptor(600, 0, "R", background_la90_dBA=float("nan"))], a,
            country="XI", default_background_la90=30.0)),
    ]:
        try:
            fn()
            check(name + " raises", False, "passed silently")
        except ValueError:
            check(name + " raises", True)


def t18_flicker_geometry():
    """(a) The expected value must be computed on the asin(R/L) superset, not inside the
    astronomical atan mask (that truncation under-counts in the near field).
    (b) A receptor under the rotor must be flagged INVALID_UNDER_ROTOR instead of reporting
    an absurd number of hours that dominates the summary."""
    print("\n[18] flicker near-field geometry")
    kw = dict(country="XI", blade_chord_m=CHORD, **LOW_LAT)
    rose = {d: 1.0 for d in range(0, 360, 30)}
    near = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-300, 0, "R")], wind_rose=rose,
                                  sunshine_probability=1.0, operating_probability=1.0,
                                  **kw)["receptors"][0]
    check("near-field expected > 0 and <= astronomical maximum",
          0 < near["expected_hours_per_year"] <= near["astro_hours_per_year"],
          f"exp {near['expected_hours_per_year']} / astro {near['astro_hours_per_year']}")
    o = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(0, 0, "under_rotor"),
                                           flicker.FlickerReceptor(-600, 0, "regular")],
                               wind_rose=rose, sunshine_probability=1.0, **kw)
    rows = {r["receptor_id"]: r for r in o["receptors"]}
    check("receptor under the rotor flagged INVALID_UNDER_ROTOR with no hours",
          rows["under_rotor"]["status"] == "INVALID_UNDER_ROTOR"
          and rows["under_rotor"]["astro_hours_per_year"] is None)
    check("summary worst receptor is not the invalid one",
          o["summary"]["worst_receptor"] == "regular" and o["summary"]["n_invalid_under_rotor"] == 1)
    check("output states the astronomical convention (worst case, not a hard bound)",
          "worst case" in o["model"]["astro_convention"])


def t19_probabilities_and_chord():
    """(a) Probabilities outside [0, 1] must raise; otherwise the expected value can exceed
    the astronomical maximum. (b) The blade chord is a required input that sets the
    distance cut-off: it must be echoed, required, and move d_max in the right direction."""
    print("\n[19] probability validation and blade chord")
    kw = dict(country="XI", blade_chord_m=CHORD, **LOW_LAT)
    bad = 0
    for p_ in (5.0, -1.0, 1.5):
        for field in ("operating_probability", "sunshine_probability"):
            try:
                flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-600, 0, "R")],
                                       wind_rose={0: 1}, **{field: p_}, **kw)
            except ValueError:
                bad += 1
    check("6 out-of-range probabilities all raise", bad == 6, f"{bad}/6")
    o = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-600, 0, "R")], **kw)
    check("model.blade_chord_m echoes the input chord",
          abs(o["model"]["blade_chord_m"] - round(CHORD, 2)) < 1e-9, f"{o['model']['blade_chord_m']} m")
    try:
        flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-600, 0, "R")],
                               country="XI", **LOW_LAT)
        check("missing chord raises", False, "a default was used")
    except ValueError as e:
        check("missing chord raises", "must be given explicitly" in str(e))
    dm = [flicker.max_distance_m(CHORD * f) for f in (0.5, 1.0, 1.5)]
    check("max_distance_m grows with chord", dm[0] < dm[1] < dm[2], " < ".join(f"{v:.0f}" for v in dm))
    dmax = dm[1]
    o2 = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(-(dmax * 1.05), 0, "out"),
                                            flicker.FlickerReceptor(-(dmax * 0.6), 0, "in")], **kw)
    h = {r["receptor_id"]: r["astro_hours_per_year"] for r in o2["receptors"]}
    check("receptor beyond d_max -> 0 h, receptor inside -> > 0 h", h["out"] == 0.0 and h["in"] > 0,
          f"d_max {dmax:.0f} m: out {h['out']} h, in {h['in']} h")


def t20_sensitivity_and_outputs():
    """Sensitivity output must not be dominated by the known-optimistic iso_alt reference
    variant; required_distance must raise when there is no solution instead of silently
    returning the upper bound; sensitivity output carries the disclaimer."""
    print("\n[20] sensitivity and output completeness")
    a = atm(20.0, 70.0)
    s_ = noise.sensitivity(LWA, 37.0, h_src=HUB, atm=a)
    check("iso_alt listed as reference-only", any("iso_alt" in v for v in s_["reference_only_variants"]))
    check("dominant assumption is not the iso_alt reference", "iso_alt" not in s_["dominant_assumption"],
          s_["dominant_assumption"])
    check("dominant assumption is a spectral shift", "spectrum" in s_["dominant_assumption"],
          s_["dominant_assumption"])
    iso = [v for v in s_["variants"] if "iso_alt" in v["variant"]][0]
    span_ok = s_["max_abs_delta_pct"] == max(abs(v["delta_pct"]) for v in s_["variants"] if v["is_assumption"])
    check("span > 0 and computed over assumptions only (iso_alt excluded)",
          s_["max_abs_delta_pct"] > 0 and span_ok and iso["is_assumption"] is False,
          f"span {s_['max_abs_delta_pct']} %, iso_alt {iso['delta_pct']} %")
    check("sensitivity carries _disclaimer", "_disclaimer" in s_)
    hd = round(HUB / 5.0)
    s2 = noise.sensitivity(LWA, 37.0, h_src=HUB, atm=a, meteo_variants=((12.0, 85.0),), hub_delta_m=hd)
    labels = [v["variant"] for v in s2["variants"]]
    check("meteo_variants / hub_delta_m drive the variant list",
          sum("meteorology" in v for v in labels) == 1 and any("12 C / 85 % RH" in v for v in labels)
          and any(f"+{hd:g} m" in v for v in labels) and any(f"-{hd:g} m" in v for v in labels), str(labels))
    try:
        noise.required_distance(LWA, -20.0, h_src=HUB, atm=a, array_n=9)
        check("no solution raises instead of silently returning the upper bound", False, "returned a number")
    except ValueError:
        check("no solution raises instead of silently returning the upper bound", True)
    check("bisection lower bound is returned when already compliant",
          noise.required_distance(LWA, LWA - 20.0, h_src=HUB, atm=a) == 50.0)


# ============================================================================ schema / provenance
def t21_schema_and_provenance():
    """Output schema and provenance: every criterion shape works end to end with one
    symmetric key set; mixed CRS tagging is rejected; period matching does not accept
    substrings; chord provenance; near-field warning; CSV stays machine-readable."""
    print("\n[21] schema, provenance and output format")
    a = atm(20.0, 70.0)
    T, R = [nturb()], [noise.Receptor(600, 0, "R")]
    KEYS = ("mode", "period", "rule", "limit_dBA", "limit_dBA_range", "limit_dBA_distinct",
            "n_receptors_with_measured_LA90", "n_receptors_with_assumed_LA90")
    xb_bg = XB_TABLE_NIGHT - XB_MARGIN / 2
    lims, miss = {}, []
    for ctry, bg in (("XI", 30.0), ("XA", None), ("XB", xb_bg)):
        o = noise.assess_receptors(T, R, a, country=ctry, period="night", default_background_la90=bg)
        lims[ctry] = o["receptors"][0]["limit_dBA"]
        miss += [f"{ctry}.{k}" for k in KEYS if k not in o["criterion"]]
    want = {"XI": 30.0 + ALLOW_NIGHT, "XA": XA_NIGHT, "XB": _xb_expected(xb_bg)}
    check("noise end to end for all three criterion shapes", lims == want, str(lims))
    check("criterion block has the core keys for every shape", not miss, str(miss))
    check("needs_background matches the criterion shape",
          all(criteria.needs_background(c) is v for c, v in (("XI", True), ("XA", False), ("XB", True))))

    try:
        noise.assess_receptors([nturb(crs=CRS_A)], [noise.Receptor(600, 0, "R")], a,
                               country="XI", default_background_la90=30.0)
        check("mixed CRS tagging (some objects without CRS) raises", False, "passed silently")
    except ValueError as e:
        check("mixed CRS tagging (some objects without CRS) raises", "carry a CRS" in str(e))
    o = noise.assess_receptors(T, R, a, country="XI", default_background_la90=30.0)
    check("no object tagged (hand-built inputs) still passes", o["n_receptors"] == 1)

    try:
        criteria._match_period({"midnight": 1}, "night")
        check("'midnight' is not matched as 'night'", False, "matched")
    except KeyError:
        check("'midnight' is not matched as 'night'", True)
    check("delimited keys still match", criteria._match_period({"x_night": 1, "x_day": 2}, "night") == "x_night")

    check("no built-in chord estimator (chord is an input)",
          not hasattr(flicker, "estimate_blade_chord_m")
          and not any("CHORD_OVER_DIAMETER" in n for n in dir(flicker)))

    near_x = -0.5 * flicker.NEAR_FIELD_WARN_FACTOR * D - 20.0
    far_x = -2.0 * flicker.NEAR_FIELD_WARN_FACTOR * D
    o = flicker.assess_flicker(
        [fturb()], [flicker.FlickerReceptor(near_x, 0, "near"), flicker.FlickerReceptor(far_x, 0, "far")],
        blade_chord_m=CHORD, wind_rose={0: 1, 90: 1}, sunshine_probability=0.5,
        country="XI", **HIGH_LAT)
    w = {r["receptor_id"]: r["near_field_warning"] for r in o["receptors"]}
    check("near-field receptor flagged (< factor x D), far one not", w == {"near": True, "far": False}, str(w))
    check("latitude inside the registry band -> no hint",
          o["model"]["country_latitude_mismatch"] is False and o["model"]["country_latitude_note"] is None)
    check("expected <= astro verified per receptor and not violated here",
          o["summary"]["n_expected_exceeds_astro"] == 0
          and all("expected_exceeds_astro" in r for r in o["receptors"]))

    band = [HIGH_LAT["lat_deg"] - 20.0, HIGH_LAT["lat_deg"] - 10.0]      # excludes the site
    with _TempRegistry() as p:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["countries"]["XI"]["lat_range_deg"] = band
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        om = flicker.assess_flicker([fturb()], [flicker.FlickerReceptor(far_x, 0, "far")],
                                    blade_chord_m=CHORD, country="XI", **HIGH_LAT)["model"]
    check("latitude outside the registry band -> hint (not a gate)",
          om["country_latitude_mismatch"] is True and str(band) in (om["country_latitude_note"] or ""),
          str(om["country_latitude_note"])[:90])

    out = Path(tempfile.mkdtemp(prefix="wnf_csv_")) / "f.csv"
    path, meta = layers.to_csv(o, out)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    check("csv.DictReader parses the CSV (no comment header)",
          len(rows) == 2 and "astro_hours_per_year" in rows[0], f"{len(rows)} rows")
    mt = meta.read_text(encoding="utf-8")
    check("limits of validity travel in a sidecar .meta.txt", meta.exists() and "Limits of validity" in mt)
    check("meta warns that INVALID rows are silently dropped by numeric filters", "silently drops" in mt)
    shutil.rmtree(out.parent, ignore_errors=True)

    check("flicker docstring does not claim the astro value cannot be exceeded",
          "cannot be exceeded" not in flicker.__doc__)
    check("flicker docstring names the larger asin supremum", "asin" in flicker.__doc__)
    check("noise disclaimer carries no hard-coded percentage that could drift from sensitivity()",
          re.search(r"\d+\s*%", noise.OUTPUT_DISCLAIMER) is None)


def t22_registry_cache_and_mixed_rotors():
    """Background value must reach the output; the criterion key set must be identical for
    every shape; the registry cache must not serve a stale snapshot; the near-field flag
    must use the nearest turbine's diameter; expected-vs-astro must compare unrounded
    values; chord provenance must be recorded."""
    print("\n[22] background in output, cache freshness, mixed rotors")
    a = atm(20.0, 70.0)
    T = [nturb()]
    got, want = {}, {}
    for bg in (XB_TABLE_NIGHT - XB_MARGIN - 5, XB_TABLE_NIGHT - 2, XB_TABLE_NIGHT + 5):
        r = noise.assess_receptors(T, [noise.Receptor(600, 0, "R")], a, country="XB", period="night",
                                   default_background_la90=bg)["receptors"][0]
        got[bg] = (r["limit_dBA"], r["background_LA90_dBA"])
        want[bg] = (_xb_expected(bg), bg)
    check("background value reaches the output and matches the limit", got == want, str(got))

    keys, per = set(), {}
    for ctry, bg in (("XI", 30.0), ("XA", None), ("XB", XB_TABLE_NIGHT)):
        c = noise.assess_receptors(T, [noise.Receptor(600, 0, "R")], a, country=ctry,
                                   default_background_la90=bg)["criterion"]
        per[ctry] = set(c)
        keys |= per[ctry]
    check("criterion block key set identical for every shape", all(per[k] == keys for k in per),
          str({k: sorted(keys - per[k]) for k in per}))
    xa = noise.assess_receptors(T, [noise.Receptor(600, 0, "R")], a, country="XA")["criterion"]
    check("absolute-limit rule is a rule, not a table-row label", "<=" in xa["rule"] and "dB(A)" in xa["rule"])
    check("absolute limit: background receptor counts are None, not 0",
          xa["n_receptors_with_measured_LA90"] is None and xa["n_receptors_with_assumed_LA90"] is None)
    xb = noise.assess_receptors(T, [noise.Receptor(600, 0, "R")], a, country="XB",
                                default_background_la90=XB_TABLE_NIGHT - XB_MARGIN)
    check("boundary caveat reaches the criterion block", xb["criterion"].get("boundary_caveat") is not None)
    check("background exactly table - margin takes the lower band",
          xb["receptors"][0]["limit_dBA"] == XB_TABLE_NIGHT + XB_ADJ["below_by_more_than_margin"])

    # Registry cache: one snapshot per run; an edit must take effect immediately.
    t0 = time.time()
    R1000 = [noise.Receptor(600 + i, 0, f"R{i}") for i in range(1000)]
    noise.assess_receptors(T, R1000, a, country="XI", default_background_la90=30.0)
    dt = time.time() - t0
    check("1000 receptors < 0.5 s (registry parsed once, not per receptor)", dt < 0.5, f"{dt:.2f} s")
    check("cache keeps a single snapshot", len(criteria._CACHE) == 1, f"{len(criteria._CACHE)}")
    needle = f'"night": {int(ALLOW_NIGHT)}'
    with _TempRegistry() as p:
        orig = p.read_text(encoding="utf-8")
        check("fixture edit target is unique", orig.count(needle) == 1, needle)
        criteria.noise_limit("XI", period="night", background_la90=30.0)       # warm the cache
        time.sleep(0.01)
        p.write_text(orig.replace(needle, f'"night": {int(ALLOW_NIGHT) - 1}'), encoding="utf-8")
        os.utime(p, None)
        got = criteria.noise_limit("XI", period="night", background_la90=30.0)[0]
        check("edited registry takes effect at once (no stale cache)",
              got == 30.0 + int(ALLOW_NIGHT) - 1, f"got {got}")
    check("after restore the fixture limit is back",
          criteria.noise_limit("XI", period="night", background_la90=30.0)[0] == 30.0 + ALLOW_NIGHT)

    # Near-field flag uses the nearest turbine, not the largest rotor of the layout.
    kw = dict(country="XI", blade_chord_m=CHORD, **LOW_LAT)
    small = fturb(0, 0, wid="small")
    big = fturb(7000, 0, hub=HUB + 30, dia=D * 2.0, wid="big")
    # between the small rotor's near-field radius (factor x D) and the big one's (factor x 2D)
    rx = -1.5 * flicker.NEAR_FIELD_WARN_FACTOR * D
    rec = [flicker.FlickerReceptor(rx, 0, "r")]
    mixed = flicker.assess_flicker([small, big], rec, **kw)["receptors"][0]
    alone = flicker.assess_flicker([small], rec, **kw)["receptors"][0]
    check("a large rotor 7 km away does not flag a receptor near a small one",
          mixed["near_field_warning"] == alone["near_field_warning"] is False, f"receptor at {abs(rx):.0f} m")
    near = flicker.assess_flicker([small], [flicker.FlickerReceptor(-0.5 * flicker.NEAR_FIELD_WARN_FACTOR * D
                                                                    - 10, 0, "r")], **kw)["receptors"][0]
    check("a genuine near-field receptor is still flagged", near["near_field_warning"] is True)

    src = inspect.getsource(flicker.assess_flicker)
    check("expected-vs-astro comparison uses the unrounded value",
          "exp_raw > hours" in src and "exp_hours > hours" not in src)

    und = flicker.assess_flicker([small], rec, **kw)["model"]
    dec = flicker.assess_flicker([small], rec, blade_chord_source="manufacturer data sheet", **kw)["model"]
    check("undeclared chord source is flagged as such",
          und["blade_chord_source_declared"] is False and "not declared" in und["blade_chord_source"])
    check("declared chord source is recorded verbatim",
          dec["blade_chord_source_declared"] is True and dec["blade_chord_source"] == "manufacturer data sheet")
    check("to_csv docstring describes the sidecar meta file",
          "meta.txt" in layers.to_csv.__doc__ and "first line" not in layers.to_csv.__doc__)


# ============================================================================ traceability / layers
def _zero_crossing(h_src, h_rec):
    """Smallest integer distance at which the iso_alt ground term leaves zero."""
    return next(d for d in range(20, 8000) if noise.a_ground(d, h_src, h_rec, "iso_alt") > 0)


def _analytic_root(h_src, h_rec):
    """Positive root of 4.8 d^2 - 34 h_m d - 600 h_m = 0, h_m = (h_s + h_r)/2."""
    hm = (h_src + h_rec) / 2.0
    return (34 * hm + math.sqrt((34 * hm) ** 2 + 4 * 4.8 * 600 * hm)) / (2 * 4.8)


def _make_synthetic_gpkg(out: Path):
    subprocess.run([sys.executable, str(ROOT / "demo" / "make_demo_data.py"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def t23_traceability_and_layers():
    """Adjusted limits must show table value + adjustment; the registry cache must notice a
    content change even when the modification time is restored; the iso_alt break point is
    pinned to its closed-form solution (it depends on the mean of source and receiver
    heights); and a synthetic layout is run end to end through the layers module, whose
    main purpose is to force both layers into one CRS."""
    print("\n[23] traceability, iso_alt break point, synthetic layers end to end")
    a = atm(20.0, 70.0)
    T = [nturb()]
    bg = XB_TABLE_NIGHT - 2.0
    o = noise.assess_receptors(T, [noise.Receptor(600, 0, "R")], a, country="XB", period="night",
                               default_background_la90=bg)
    r, c = o["receptors"][0], o["criterion"]
    adj = XB_ADJ["within_margin_below"]
    check("adjusted receptor row shows limit = table value + adjustment",
          (r["limit_dBA"], r["table_value_dBA"], r["adjustment_dB"]) == (XB_TABLE_NIGHT + adj, XB_TABLE_NIGHT, adj),
          f"{r['limit_dBA']} = {r['table_value_dBA']} + {r['adjustment_dB']}")
    check("criterion block carries the table value", c["table_value_dBA"] == XB_TABLE_NIGHT
          and c["resolved"] is not None)
    xa = noise.assess_receptors(T, [noise.Receptor(600, 0, "R")], a, country="XA")["receptors"][0]
    check("non-adjusted limit: no adjustment (None) and table value = limit",
          xa["adjustment_dB"] is None and xa["adjustment_branch"] is None
          and xa["table_value_dBA"] == xa["limit_dBA"] == XA_NIGHT)

    needle = f'"night": {int(ALLOW_NIGHT)}'
    with _TempRegistry() as p:
        orig = p.read_text(encoding="utf-8")
        criteria.noise_limit("XI", period="night", background_la90=30.0)       # warm the cache
        st = os.stat(p)
        p.write_text(orig.replace(needle, '"night": 99'), encoding="utf-8")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                        # restore mtime
        got = criteria.noise_limit("XI", period="night", background_la90=30.0)[0]
        check("content change detected although mtime was restored", got == 30.0 + 99, f"got {got}")
    check("after restore the fixture limit is back",
          criteria.noise_limit("XI", period="night", background_la90=30.0)[0] == 30.0 + ALLOW_NIGHT)

    hs_list = (30.0, HUB - 30.0, HUB, HUB + 30.0)
    ok = all(abs(_zero_crossing(hs, 1.5) - _analytic_root(hs, 1.5)) <= 1.0 for hs in hs_list)
    check("iso_alt zero crossing = closed-form root (four source heights)", ok,
          " ".join(f"h{hs:.0f}:{_zero_crossing(hs, 1.5)}~{_analytic_root(hs, 1.5):.1f}" for hs in hs_list))
    z15, z10 = _zero_crossing(HUB, 1.5), _zero_crossing(HUB, 10.0)
    check("break point depends on (h_s + h_r)/2, not only the source height",
          z10 > z15 and abs(z10 - _analytic_root(HUB, 10.0)) <= 1.0
          and "(h_s + h_r)/2" in noise.a_ground.__doc__,
          f"h_rec 1.5 -> {z15} m, h_rec 10 -> {z10} m")

    import contours
    check("contours CLI exposes --blade-chord-source", "--blade-chord-source" in inspect.getsource(contours.main))

    # ---- synthetic layers end to end
    import warnings
    warnings.filterwarnings("ignore")
    import geopandas as gpd
    d = Path(tempfile.mkdtemp(prefix="wnf_layers_"))
    try:
        src = _make_synthetic_gpkg(d / "site.gpkg")
        w = gpd.read_file(src, layer="wtg")
        rec = gpd.read_file(src, layer="receptors")
        gp = d / "mixed.gpkg"
        w.to_crs(4326).to_file(gp, layer="wtg", driver="GPKG")          # geographic
        rec.to_crs(CRS_B).to_file(gp, layer="rec", driver="GPKG")       # neighbouring zone
        tn, tf, rn, rf, site_crs = layers.load_site(
            gp, gp, turbine_layer="wtg", receptor_layer="rec", t_id_field="wtg_id", r_id_field="bid",
            t_hub_height_m=HUB, t_rotor_diameter_m=D, t_lwa_dBA=LWA)
        check("load_site puts both layers in one projected CRS (geographic + other zone in)",
              str(site_crs) == CRS_A and {str(o.crs) for o in tn} == {str(o.crs) for o in rn} == {CRS_A},
              f"{site_crs} / {len(tn)} turbines / {len(rn)} receptors")
        on = noise.assess_receptors(tn, rn, a, country="XI", period="night", default_background_la90=30.0)
        check("noise end to end through layers (not hand-built objects)",
              on["n_turbines"] == len(w) and on["n_receptors"] == len(rec),
              f"{on['n_turbines']} turbines / {on['n_receptors']} receptors / {on['summary']['n_exceed']} exceed")
        # The largest array penalty sits in the far field (where the level is low anyway),
        # not at the controlling nearest receptor: quoting the maximum penalty as support
        # for a setback conclusion is misleading.
        rows_ = on["receptors"]
        worst = min(rows_, key=lambda r: r["nearest_turbine_m"])
        far = max(rows_, key=lambda r: r["array_penalty_dB"])
        check("largest array penalty is farther out than the controlling receptor",
              far["nearest_turbine_m"] > worst["nearest_turbine_m"]
              and far["array_penalty_dB"] > worst["array_penalty_dB"],
              f"nearest {worst['nearest_turbine_m']} m penalty {worst['array_penalty_dB']} dB; "
              f"max penalty {far['array_penalty_dB']} dB at {far['nearest_turbine_m']} m")
        csv_p, _ = layers.to_csv(on, d / "site.csv")
        with open(csv_p, encoding="utf-8-sig") as f:
            cols = next(csv.reader(f))
        check("CSV exposes no internal keys", not [k for k in cols if k.startswith("_")], str(cols[-3:]))
        # contours.load_inputs: no implicit reprojection, CRS checked before loading
        for args, label in (((gp, "wtg"), "load_inputs: geographic turbine layer rejected"),
                            ((src, "wtg", gp, "rec"), "load_inputs: layers in different CRS rejected")):
            try:
                contours.load_inputs(*args, t_hub_height_m=HUB, t_rotor_diameter_m=D, t_lwa_dBA=LWA)
                check(label, False, "reprojected silently")
            except ValueError:
                check(label, True)
        tn2, _, rn2, _, crs2 = contours.load_inputs(src, "wtg", src, "receptors", t_id_field="wtg_id",
                                                    r_id_field="bid", t_hub_height_m=HUB,
                                                    t_rotor_diameter_m=D, t_lwa_dBA=LWA)
        check("load_inputs: same projected CRS loads", len(tn2) == len(w) and len(rn2) == len(rec)
              and str(crs2) == CRS_A)
        try:
            layers.turbines_from_gpkg(src, "wtg", rotor_diameter_m=D, lwa_dBA=LWA)
            check("missing turbine parameter raises (no default turbine)", False)
        except ValueError:
            check("missing turbine parameter raises (no default turbine)", True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t24_cache_freshness():
    """Caches must not freeze inputs. The spectral normalisation constant is cached; keyed on
    the spectrum *name* it would go stale after an in-place edit of SPECTRA and band energies
    would no longer sum to L_WA (error goes straight into distances). The absorption cache
    must not swallow meteorology differences."""
    print("\n[24] cache freshness (spectrum / meteorology edits must change results)")

    def band_sum(lw):
        return 10 * math.log10(sum(10 ** (v / 10.0) for v in lw.values()))
    base = band_sum(noise.band_levels(100.0, SPEC))
    check("baseline: band energies sum exactly to lwa", abs(base - 100.0) < 1e-9, f"{base:.4f} dB")
    a1 = atm()
    d_before = noise.required_distance(LWA, 35, h_src=HUB, atm=a1, array_n=9)
    orig = copy.deepcopy(noise.SPECTRA[SPEC])
    try:
        noise.SPECTRA[SPEC][8000] = -5.0                    # in-place edit
        s = band_sum(noise.band_levels(100.0, SPEC))
        check("after an in-place edit band energies still sum to lwa", abs(s - 100.0) < 1e-9, f"{s:.4f} dB")
        d_after = noise.required_distance(LWA, 35, h_src=HUB, atm=a1, array_n=9)
        check("after the edit the distance actually changes", abs(d_after - d_before) > 1.0,
              f"{d_before:.0f} m -> {d_after:.0f} m")
    finally:
        noise.SPECTRA[SPEC] = orig
    check("restored spectrum: back to baseline", abs(band_sum(noise.band_levels(100.0, SPEC)) - 100.0) < 1e-9)
    check("restored spectrum: distance back to original",
          abs(noise.required_distance(LWA, 35, h_src=HUB, atm=a1, array_n=9) - d_before) < 1e-6,
          f"{d_before:.1f} m")
    a_cold = noise.alpha_iso9613_1(4000, 10.0, 70.0, noise.PA_STANDARD)
    a_warm = noise.alpha_iso9613_1(4000, 30.0, 40.0, noise.PA_STANDARD)
    check("alpha cache keeps meteorology apart", abs(a_cold - a_warm) > 1e-6,
          f"{a_cold:.2f} vs {a_warm:.2f} dB/km")


# ============================================================================ contours
WORKERS = max(1, min(4, os.cpu_count() or 1))
#: Contour levels supplied by the caller (the engine has no default levels; the criterion
#: limit from the registry is always added).
NOISE_LEVELS = [30.0, 35.0, 40.0, 45.0]
FLICKER_LEVELS = [8.0]


def t25_contours_export():
    """Contour export. Guards three kinds of 'the map is drawn but wrong':
      (1) grid evaluation drifting from the engine (a table or kernel error shifts every
          line without raising) - noise grid nodes vs noise.lp_at_point, flicker kernel
          nodes vs assess_flicker;
      (2) single-turbine noise lines not circular, radii not decreasing with level, radii
          not matching required_distance;
      (3) schema drift (field names, types, layer names, CRS) that map templates rely on."""
    print("\n[25] contour export (contours.py)")
    import pandas as pd
    import pyogrio
    import contours
    a = atm(20.0, 70.0)
    fkw = dict(blade_chord_m=CHORD, blade_chord_source="reference turbine (examples)",
               wind_rose={0: 6, 30: 5, 60: 9, 90: 14, 120: 12, 150: 8,      # synthetic
                          180: 7, 210: 8, 240: 10, 270: 9, 300: 7, 330: 5},
               sunshine_probability=0.5, **LOW_LAT)

    # ---- (1) + (2) single-turbine noise
    tn1 = [nturb(X0, Y0, wid="T1")]
    xs, ys, Z, _ = contours.noise_grid(tn1, a, res=10.0, min_level=30.0)
    rng = np.random.default_rng(0)
    idx = [(int(rng.integers(len(ys))), int(rng.integers(len(xs)))) for _ in range(40)]
    dev = max(abs(Z[j, i] - noise.lp_at_point(tn1, xs[i], ys[j], a)[0]) for j, i in idx)
    check("noise grid nodes = noise.lp_at_point (< 0.005 dB)", dev < 0.005, f"max deviation {dev:.5f} dB")
    lines = contours.extract_lines(xs, ys, Z, [35.0, 40.0, 45.0])
    radii, circ, vs_req = {}, [], []
    for lv, g in lines:
        cc = np.asarray(g.coords)
        radii.setdefault(lv, []).append(np.hypot(cc[:, 0] - X0, cc[:, 1] - Y0))
    ok_round, ok_req = True, True
    for lv in (35.0, 40.0, 45.0):
        r = np.concatenate(radii.get(lv, [np.array([np.nan])]))
        spread = (r.max() - r.min()) / r.mean()
        req = noise.required_distance(LWA, lv, h_src=HUB, atm=a)
        circ.append(f"{lv:.0f}dB r={r.mean():.0f}m spread {100*spread:.2f}%")
        vs_req.append(f"{lv:.0f}dB {r.mean():.1f}~{req:.1f}")
        ok_round &= 100 * spread < 2.0
        ok_req &= abs(r.mean() - req) < 10.0
    check("single-turbine noise contour is a circle (radius spread < 2 %)", ok_round, "; ".join(circ))
    check("contour radius = required_distance (within one cell)", ok_req, "; ".join(vs_req))
    means = [np.concatenate(radii[lv]).mean() for lv in (35.0, 40.0, 45.0)]
    check("radius decreases with level", means[0] > means[1] > means[2], " > ".join(f"{m:.0f}" for m in means))

    # ---- single-turbine flicker: kernel nodes vs engine, 30 h line distance
    tf1 = [fturb(X0, Y0)]
    kw = dict(fkw, country="XI")
    fx, fy, FZ, finfo = contours.flicker_grid(tf1, kw, res=40.0, workers=WORKERS)
    probes = [(X0 + dx, Y0 + dy) for dx, dy in
              ((600, 0), (-800, 120), (1000, -200), (-400, 40), (1400, 280), (0, 800))]
    ex = flicker.assess_flicker(tf1, [flicker.FlickerReceptor(x, y, f"p{i}") for i, (x, y) in enumerate(probes)], **kw)
    exd = {r["receptor_id"]: r for r in ex["receptors"]}
    bad = []
    for i, (x, y) in enumerate(probes):
        j, k = int(round((y - fy[0]) / 40.0)), int(round((x - fx[0]) / 40.0))
        for case, key in (("astronomical_max", "astro_hours_per_year"), ("expected", "expected_hours_per_year")):
            if abs(FZ[case][j, k] - exd[f"p{i}"][key]) > 0.011:
                bad.append((i, case, FZ[case][j, k], exd[f"p{i}"][key]))
    check("flicker grid nodes = assess_flicker (both cases, single turbine)", not bad, str(bad[:3]))
    fl = contours.extract_lines(fx, fy, FZ["astronomical_max"], [FLICKER_LIMIT_H])
    fe = contours.extract_lines(fx, fy, FZ["expected"], [FLICKER_LIMIT_H])
    rmax = max(np.hypot(*(np.asarray(g.coords) - [X0, Y0]).T).max() for _, g in fl) if fl else 0
    rexp = max((np.hypot(*(np.asarray(g.coords) - [X0, Y0]).T).max() for _, g in fe), default=0.0)
    check("single-turbine criterion-limit astro line exists and lies between rotor diameter and d_max",
          bool(fl) and D < rmax < finfo["d_max"], f"farthest {rmax:.0f} m (d_max {finfo['d_max']:.0f} m)")
    fs = contours.extract_lines(fx, fy, FZ["astronomical_max"], [FLICKER_LIMIT_H], smooth_iter=2)
    hd = max(g0.hausdorff_distance(g1) for (_, g0), (_, g1) in zip(fl, fs))
    check("Chaikin smoothing: same line count, closed rings stay closed, shift <= 0.36 cell",
          len(fs) == len(fl) and hd <= 0.36 * 40.0
          and all(g1.is_closed == g0.is_closed for (_, g0), (_, g1) in zip(fl, fs)),
          f"{len(fl)} lines, Hausdorff {hd:.1f} m")
    check("criterion-limit expected line does not extend beyond the astro line", rexp <= rmax + 40.0,
          f"expected {rexp:.0f} m <= astro {rmax:.0f} m")

    # ---- (3) three turbines end to end + schema
    pos = [(X0, Y0), (X0 + 700, Y0 + 150), (X0 + 1400, Y0 - 100)]
    tn = [nturb(x, y, wid=f"T{i}", crs=CRS_A) for i, (x, y) in enumerate(pos)]
    tf = [fturb(x, y, wid=f"T{i}", crs=CRS_A) for i, (x, y) in enumerate(pos)]
    rp = [(X0 + 700, Y0 + 900), (X0 - 900, Y0), (X0 + 2600, Y0 + 50), (X0 + 0.5 * D / 2, Y0)]
    rn = [noise.Receptor(x, y, f"R{i}", crs=CRS_A) for i, (x, y) in enumerate(rp)]
    rf = [flicker.FlickerReceptor(x, y, f"R{i}", crs=CRS_A) for i, (x, y) in enumerate(rp)]
    lim_target = 37.0
    check("chosen criterion level is not one of the user levels", lim_target not in NOISE_LEVELS)
    bg_uniform = lim_target - ALLOW_NIGHT
    try:
        contours.build(tn, [], rn, [], CRS_A, country="XI", atm=a, background_la90=None,
                       do_flicker=False)
        check("build: no noise levels and no criterion limit raises", False, "built anyway")
    except ValueError as e:
        check("build: no noise levels and no criterion limit raises", "levels" in str(e))
    try:
        contours.build([], tf, [], rf, CRS_A, country="XA", do_noise=False, flicker_kw=fkw,
                       flicker_levels=None, workers=WORKERS)
        check("build: flicker without threshold and without levels raises", False, "built anyway")
    except ValueError as e:
        check("build: flicker without threshold and without levels raises", "levels" in str(e))
    out = contours.build(tn, tf, rn, rf, CRS_A, country="XI", atm=a, background_la90=bg_uniform,
                         noise_res=25.0, noise_levels=NOISE_LEVELS, flicker_kw=fkw, flicker_res=40.0,
                         flicker_levels=FLICKER_LEVELS, workers=WORKERS, wind_speed_ms=8.0)
    tmp = Path(tempfile.mkdtemp(prefix="wnf_contours_"))
    try:
        p = tmp / "c.gpkg"
        contours.write_gpkg(out, p)
        contours.write_gpkg(out, p)                    # idempotent: rewrite, not append
        names = {n for n, _ in pyogrio.list_layers(str(p))}
        check("GPKG has all five layers",
              names == {"noise_contour", "flicker_isoline", "flicker_band", "receptor", "params"}, str(sorted(names)))
        want = {"noise_contour": {"level_db": "float64", "metric": "object", "wind_speed_ms": "float64"},
                "flicker_isoline": {"hours_per_year": "float64", "case": "object"},
                "flicker_band": {"hours_min": "float64", "hours_max": "float64", "case": "object"},
                "receptor": {"receptor_id": "object", "noise_db": "float64",
                             "flicker_h": "float64", "exceeds": "int64"}}
        sch_ok, sch = True, []
        for lyr, fields in want.items():
            inf = pyogrio.read_info(str(p), layer=lyr)
            got = dict(zip(inf["fields"], [str(t) for t in inf["dtypes"]]))
            gt = {"receptor": "Point", "flicker_band": "Polygon"}.get(lyr, "LineString")
            sch_ok &= (got == fields and inf["crs"] == CRS_A and inf["geometry_type"] == gt)
            sch.append(f"{lyr}:{inf['geometry_type']}:{inf['features']}")
        check("schema field names / types / CRS as specified", sch_ok, " ".join(sch))
        nc = pyogrio.read_dataframe(str(p), layer="noise_contour")
        fi = pyogrio.read_dataframe(str(p), layer="flicker_isoline")
        rc = pyogrio.read_dataframe(str(p), layer="receptor")
        pr = pyogrio.read_dataframe(str(p), layer="params")
        fb = pyogrio.read_dataframe(str(p), layer="flicker_band")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # Band polygons vs exact receptor values: a receptor inside a band must have its exact
    # h/year inside [hours_min, hours_max) (receptors within max(2 h, 10 %) of a level count
    # as on the line - same tolerance as final_line_side_check).
    fb_a = fb[fb["case"] == "astronomical_max"]
    bad_band = 0
    for _, rr in rc.dropna(subset=["flicker_h"]).iterrows():
        h = float(rr["flicker_h"])
        hit = fb_a[fb_a.contains(rr.geometry)]
        lo = float(hit["hours_min"].max()) if len(hit) else None
        hi_s = hit["hours_max"].dropna()
        hi = float(hi_s.min()) if len(hi_s) else None
        near = lambda lv: abs(h - lv) <= max(2.0, 0.1 * lv)   # noqa: E731
        if lo is None:
            ok = h < min(fb_a["hours_min"]) or near(min(fb_a["hours_min"]))
        else:
            ok = (h >= lo or near(lo)) and (hi is None or h < hi or near(hi))
        bad_band += (not ok)
    check("flicker band polygons: receptor exact value inside its band (except on-line)",
          bad_band == 0 and len(fb_a) > 0, f"wrong band {bad_band}, polygons {len(fb_a)}")
    check("rewriting twice does not double features",
          len(nc) == len(out["noise_contour"]) and len(fi) == len(out["flicker_isoline"]))
    check("noise levels = user levels + criterion limit, metric LAeq, wind speed recorded",
          set(nc.level_db) == set(NOISE_LEVELS) | {lim_target}
          and set(nc.metric) == {"LAeq"} and set(nc.wind_speed_ms) == {8.0}, str(sorted(set(nc.level_db))))
    check("both flicker cases present, levels = user levels + criterion limit, limit line present",
          set(fi.case) == {"astronomical_max", "expected"}
          and set(fi.hours_per_year) <= set(FLICKER_LEVELS) | {FLICKER_LIMIT_H}
          and FLICKER_LIMIT_H in set(fi[fi.case == "astronomical_max"].hours_per_year),
          str(fi.groupby("case").hours_per_year.unique().to_dict()))
    r3 = rc.set_index("receptor_id")
    others = r3.drop(index="R3")
    check("receptor under the rotor: flicker_h/exceeds empty; others exceeds in {0,1}, noise_db set",
          pd.isna(r3.loc["R3", "flicker_h"]) and pd.isna(r3.loc["R3", "exceeds"])
          and set(others.exceeds.astype(int)) <= {0, 1} and others.noise_db.notna().all(),
          str(r3[["noise_db", "flicker_h", "exceeds"]].to_dict("index"))[:200])
    keys = set(pr.key)
    need = {"noise_model", "noise_lwa_dBA", "observer_height_m", "flicker_min_sun_elevation_deg",
            "noise_study_extent", "flicker_study_extent", "noise_grid_res_m", "flicker_grid_res_m",
            "flicker_cases", "noise_disclaimer", "flicker_disclaimer"}
    check("params table has every key the map legend needs", need <= keys, f"missing {sorted(need - keys)}")
    dv = json.loads(pr.set_index("key").loc["flicker_grid_vs_exact", "value"])
    fc = json.loads(pr.set_index("key").loc["flicker_final_line_side_check", "value"])
    check("written smoothed lines: receptor inside/outside agrees with exact value",
          fc["genuine"] == 0 and fc["beyond_margin"] == 0 and fc["n_checked"] > 0 and fc["n_open_lines"] == 0,
          f"checked {fc['n_checked']}, wrong side {fc['beyond_margin']}/{fc['all']}, open {fc['n_open_lines']}")
    fake = contours.final_line_side_check(
        [{"hours_per_year": lv, "case": cs, "geometry": g}
         for cs in ("astronomical_max",) for lv, g in
         contours.extract_lines(fx, fy, FZ["astronomical_max"], [FLICKER_LIMIT_H], smooth_iter=2)],
        {"far": (X0 + 0.9 * finfo["d_max"], Y0)}, {"astronomical_max": {"far": 250.0}}, [FLICKER_LIMIT_H])
    check("side check catches a mutant (far receptor claims 250 h)", fake["beyond_margin"] == 1, str(fake))
    check("flicker lines not on the wrong side of receptors", dv["side_mismatch_genuine"] == 0,
          str(dv["side_mismatch_examples"]))

    # Overlap refinement: summing single-turbine kernels double-counts simultaneous shading.
    # Refined nodes must equal assess_flicker (all turbines, per-minute OR), and at least
    # one node must drop below a level because of it - otherwise this check proves nothing.
    kw3 = dict(fkw, country="XI")
    lv3 = contours.resolve_levels(FLICKER_LEVELS, FLICKER_LIMIT_H, "flicker")
    gx, gy, Zr, ri = contours.flicker_grid(tf, kw3, res=40.0, workers=WORKERS, levels=lv3)
    _, _, Zs, _ = contours.flicker_grid(tf, kw3, res=40.0, workers=WORKERS, levels=None)
    diff = np.argwhere(np.abs(Zr["astronomical_max"] - Zs["astronomical_max"]) > 0.01)
    crossed = [tuple(q) for q in diff
               if any(Zs["astronomical_max"][tuple(q)] >= lv > Zr["astronomical_max"][tuple(q)] for lv in lv3)]
    pick = [q for q in crossed if ri["exact_mask"][q]][:6]
    ex3 = flicker.assess_flicker(tf, [flicker.FlickerReceptor(gx[i], gy[j], f"{j}_{i}", crs=CRS_A)
                                      for j, i in pick], **kw3) if pick else {"receptors": []}
    exm = {r["receptor_id"]: r["astro_hours_per_year"] for r in ex3["receptors"]}
    bad3 = [(j, i, Zr["astronomical_max"][j, i], exm[f"{j}_{i}"]) for j, i in pick
            if abs(Zr["astronomical_max"][j, i] - exm[f"{j}_{i}"]) > 0.011]
    check("overlap refinement: some nodes cross a level, refined value = assess_flicker",
          len(pick) > 0 and not bad3,
          f"refined {len(diff)}, crossing {len(crossed)}; sampled {len(pick)}, deviations {bad3[:2]}")

    # Duplicate receptor ids -> raise and list them, never merge silently
    rn_d = rn + [noise.Receptor(X0 + 3000, Y0, "R1", crs=CRS_A)]
    rf_d = rf + [flicker.FlickerReceptor(X0 + 3000, Y0, "R1", crs=CRS_A)]
    try:
        contours.build(tn, tf, rn_d, rf_d, CRS_A, country="XI", atm=a, background_la90=bg_uniform,
                       noise_levels=NOISE_LEVELS, do_flicker=False)
        check("duplicate receptor id raises and is listed", False, "passed silently")
    except ValueError as e:
        check("duplicate receptor id raises and is listed", "R1" in str(e), str(e)[:80])

    # Only per-receptor measured LA90, no site-wide background: still judge each receptor
    # against its own limit, but draw no criterion-limit contour. Backgrounds are chosen
    # relative to the modelled level so the verdicts do not depend on the turbine.
    at = (X0 + 700, Y0 + 900)
    lp_at = noise.lp_at_point(tn, *at, a)[0]
    bg_lo, bg_hi = lp_at - ALLOW_NIGHT - 5.0, lp_at - ALLOW_NIGHT + 5.0
    rn_m = [noise.Receptor(*at, "M0", background_la90_dBA=bg_lo, crs=CRS_A),
            noise.Receptor(*at, "M1", background_la90_dBA=bg_hi, crs=CRS_A),
            noise.Receptor(X0 - 900, Y0, "M2", crs=CRS_A)]
    om = contours.build(tn, [], rn_m, [], CRS_A, country="XI", atm=a, background_la90=None,
                        noise_levels=NOISE_LEVELS, do_flicker=False)
    rm = om["receptor"].set_index("receptor_id")
    pm = om["params"].set_index("key")
    check("per-receptor LA90: low background exceeds, high background complies, none -> not judged",
          rm.loc["M0", "exceeds"] == 1 and rm.loc["M1", "exceeds"] == 0
          and pd.isna(rm.loc["M2", "exceeds"]) and rm.notna().loc["M2", "noise_db"],
          str(rm[["noise_db", "exceeds"]].to_dict("index")))
    check("no site-wide background: no criterion-limit contour, params say why",
          set(om["noise_contour"].level_db) <= set(NOISE_LEVELS)
          and "no criterion-limit contour is drawn" in pm.loc["noise_limit_line_note", "value"])
    om2 = contours.build(tn, [], rn_m, [], CRS_A, country="XI", atm=a, background_la90=bg_uniform,
                         noise_levels=NOISE_LEVELS, do_flicker=False)
    check("both given: params say the contour uses the site-wide value, receptors their own",
          "site-wide background" in om2["params"].set_index("key").loc["noise_limit_line_note", "value"]
          and lim_target in set(om2["noise_contour"].level_db))

    # CRS guard: no implicit reprojection
    for bad_crs in ("EPSG:4326", None):
        try:
            contours.check_metric_crs(bad_crs)
            check(f"non-metric CRS {bad_crs} rejected", False)
        except ValueError:
            check(f"non-metric CRS {bad_crs} rejected", True)


# ============================================================================ main
TESTS = (t1_alpha, t2_ground_3db_trap, t3_distance_self_consistency, t4_superposition,
         t5_spectrum_order, t6_criteria_increment, t7_criteria_guards, t8_assess_end_to_end,
         t9_solar, t10_flicker_direction, t11_flicker_distance, t12_expected_and_latitude,
         t13_crs_guard, t14_period_no_fallback, t15_missing_allowance,
         t16_background_adjustment, t17_empty_and_nan_inputs, t18_flicker_geometry,
         t19_probabilities_and_chord, t20_sensitivity_and_outputs, t21_schema_and_provenance,
         t22_registry_cache_and_mixed_rotors, t23_traceability_and_layers, t24_cache_freshness,
         t25_contours_export)


def main():
    print("=" * 72)
    print("noise / shadow-flicker engine - regression suite")
    print("criteria registry:", criteria.criteria_path())
    print("reference turbine:", REF["name"])
    print("=" * 72)
    t_start = time.time()
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                       # a crashing test is a failure, not an abort
            check(f"{fn.__name__} ran without an exception", False, f"{type(e).__name__}: {e}")
    print("\n" + "=" * 72)
    print(f"{len(PASSED)} passed, {len(FAILS)} failed   ({time.time() - t_start:.0f} s)")
    if FAILS:
        print("failed: " + "; ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
