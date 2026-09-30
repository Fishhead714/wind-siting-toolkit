#!/usr/bin/env python3
"""Criteria registry loader - the single interface between the engine and any rules.

The engine contains **no criterion values**: noise limits, penalties, background
allowances and shadow-flicker thresholds are all read from a user-supplied JSON
registry. Keeping them in one file avoids the classic failure mode where the same
limit is copied into several places and the copies drift apart.

Locating the registry (first match wins):
  1. ``set_path(path)`` called by the program (e.g. ``contours.py --criteria``);
  2. the environment variable ``WTG_NOISE_FLICKER_CRITERIA``;
  3. otherwise ``FileNotFoundError``. There is deliberately **no built-in default**.

See ``examples/criteria.example.json`` for the schema. Supported noise criterion types:

  absolute_limit
      ``limits_dB_A`` is either flat ``{"day": x, "night": y}`` or zoned
      ``{"<zone>": {"day": x, "night": y}, ...}`` with ``"binding": "<zone>"``.
  absolute_limit_with_background_adjustment
      as above, plus ``background_adjustment`` = {``margin_dB``, ``above_table_value``,
      ``within_margin_below``, ``below_by_more_than_margin``, ``rule``, ``source``}:
      the table value is raised by an amount that depends on the background level.
  increment_over_background
      limit = background level + ``allowance_dB[period]`` - the generic "background + N dB"
      form found in published guidance such as ETSU-R-97 (UK) and NZS 6808 (New Zealand).
      The allowance values are supplied by the user; none are built in.

Shadow-flicker entries use ``limit_hours_per_year`` / ``limit_minutes_per_day`` (either may
be null, meaning "no numeric threshold").

``enforceability`` is free text recorded in the output provenance. One value has a meaning
in code: ``"statutory_hard"`` together with ``criterion_type: "annual_duration"`` marks the
flicker limit as a legally binding limit (``is_hard_veto`` = true in the output); any other
value is reported as guidance. A jurisdiction may declare ``lat_range_deg``
[min, max]; the flicker module then flags site latitudes outside that band.
An entry with ``"status": "NOT_RESEARCHED"`` raises instead of silently borrowing
another jurisdiction's numbers.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ENV = "WTG_NOISE_FLICKER_CRITERIA"
_EXPLICIT: Path | None = None


def set_path(path) -> None:
    """Select the registry file programmatically (takes precedence over the env var)."""
    global _EXPLICIT
    _EXPLICIT = None if path is None else Path(path).expanduser()


def criteria_path() -> Path:
    """Return the registry path: ``set_path()`` -> ``$WTG_NOISE_FLICKER_CRITERIA`` -> error."""
    for label, p in (("set_path()", _EXPLICIT),
                     (ENV, Path(os.environ[ENV]).expanduser() if os.environ.get(ENV) else None)):
        if p is not None:
            if not p.is_file():
                raise FileNotFoundError(f"criteria registry given via {label} does not exist: {p}")
            return p
    raise FileNotFoundError(
        "No criteria registry configured. Pass --criteria <file> (CLI), call "
        f"criteria.set_path(<file>), or set {ENV}=<file>.\n"
        "Start from examples/criteria.example.json and fill in the rules that apply to "
        "your site. Do not hard-code limits in the calling code to get around this error.")


#: (path, mtime_ns, size) -> parsed registry. Parsing once per run matters because
#: noise_limit() is called per receptor; keying on mtime *and* size makes an edited
#: file take effect immediately even if a copy tool preserved the modification time.
#: Known residual: a same-length rewrite that also restores the exact mtime is served
#: stale; hashing the content on every call would cost more than it protects.
_CACHE: dict = {}


def load() -> dict:
    p = criteria_path()
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _CACHE:
        _CACHE.clear()                 # keep a single snapshot only
        _CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
    return _CACHE[key]


# --------------------------------------------------------------------------
class Criterion(dict):
    """One criterion entry: a thin dict wrapper that also carries provenance."""

    def __init__(self, data: dict, *, country: str, topic: str, source: Path):
        super().__init__(data)
        self.country, self.topic, self.source = country, topic, source

    @property
    def provenance(self) -> dict:
        return {"country": self.country, "topic": self.topic,
                "criterion_type": self.get("criterion_type"),
                "enforceability": self.get("enforceability"),
                "verified_date": self.get("verified_date"),
                "source": self.get("source"),
                "registry": str(self.source)}


def jurisdiction(country: str) -> dict:
    """Raw registry node of one jurisdiction (``country`` is just the registry key)."""
    countries = load()["countries"]
    if country not in countries:
        raise KeyError(f"jurisdiction {country!r} is not in the criteria registry; "
                       f"available: {sorted(countries)}. Add it to the registry instead of "
                       f"hard-coding values at the call site.")
    return countries[country]


def get(country: str, topic: str) -> Criterion:
    """Criterion of one jurisdiction for ``topic`` in {'noise', 'shadow_flicker'}."""
    node = jurisdiction(country)
    if topic not in node:
        raise KeyError(f"{country} has no {topic!r} entry; available: "
                       f"{sorted(k for k in node if not k.startswith('_') and isinstance(node[k], dict))}")
    c = node[topic]
    if c.get("status") == "NOT_RESEARCHED":
        raise NotImplementedError(
            f"{country}.{topic} is marked NOT_RESEARCHED in the registry - the applicable "
            f"requirement has not been established. Do not substitute another jurisdiction's "
            f"criterion; research it first.\nRegistry note: {c.get('_note', '')}")
    return Criterion(c, country=country, topic=topic, source=criteria_path())


def _match_period(lims: dict, period: str) -> str:
    """Find the key for ``period`` in a limit table. Unknown periods raise (no silent fallback).

    Matching requires the key to *be* the word or to be delimited by '_'/'-', so that a key
    such as "midnight" can never be mistaken for "night"."""
    want = {"night": ("night",), "daytime": ("day", "daytime"), "day": ("day", "daytime")}
    if period not in want:
        raise ValueError(f"period={period!r} is not recognised; use 'night' or 'daytime' ('day' is an alias)")
    for suf in want[period]:
        for k in lims:
            if k == suf or k.endswith("_" + suf) or k.endswith("-" + suf) \
                    or k.startswith(suf + "_") or k.startswith(suf + "-"):
                return k
    raise KeyError(f"limit table has no {period} entry; keys: {sorted(lims)}")


def _alw_key(period: str) -> str:
    key = {"night": "night", "daytime": "day", "day": "day"}.get(period)
    if key is None:
        raise ValueError(
            f"period={period!r} is not recognised; use 'night' or 'daytime' ('day' is an alias). "
            f"There is deliberately no fallback: silently using the daytime allowance at night "
            f"would relax the limit.")
    return key


def _finite_background(country: str, value) -> float:
    bg = float(value)
    if not math.isfinite(bg):
        # NaN would propagate to limit=nan, and `exceed > 0` is False for NaN, so the
        # receptor would silently pass.
        raise ValueError(f"{country}: background LA90 is {value!r} (NaN/inf).")
    return bg


# --------------------------------------------------------------------------
def noise_limit(country: str, *, period: str = "night",
                background_la90: float | None = None,
                background_is_measured: bool = False) -> tuple[float, dict]:
    """Resolve the receptor sound-level limit dB(A) to compare with the propagation model.

    Returns ``(limit_dBA, info)``; ``info`` carries provenance that callers should copy
    into their output verbatim.

    For increment criteria the returned value is the ceiling for the modelled turbine level;
    ``allowance_dB`` per period is taken as given.
    """
    c = get(country, "noise")
    ctype = c["criterion_type"]

    if ctype in ("absolute_limit", "absolute_limit_with_background_adjustment"):
        lims = c["limits_dB_A"]
        if all(isinstance(v, dict) for v in lims.values()):          # zoned table
            zone = c.get("binding") or next(iter(lims))
            if zone not in lims:
                raise KeyError(f"{country}.noise binding zone {zone!r} not in limits_dB_A {sorted(lims)}")
            key = _match_period(lims[zone], period)
            base, note = float(lims[zone][key]), f"{country} zone {zone} {key}"
        else:
            key = _match_period(lims, period)
            base, note = float(lims[key]), f"{country} {key}"

        info = {"mode": ctype, "resolved": note, "table_value_dBA": base,
                "penalties_available": c.get("penalties_dB_A"),
                "measurement_metric": c.get("measurement_metric"),
                **c.provenance}

        adj = c.get("background_adjustment")
        if adj is None:
            return base, {**info, "limit_dBA": base}

        if background_la90 is None:
            raise ValueError(
                f"{country}: the limit depends on the background level ({adj.get('rule')}); "
                f"a background value is required.\nFor screening, pass an explicit assumed "
                f"value - it is labelled as an assumption in the output.")
        bg = _finite_background(country, background_la90)
        margin = float(adj["margin_dB"])
        # Boundary convention: bg exactly equal to (table - margin) takes the lower band.
        # Check the wording of your source ("less than" vs "at most") and adjust if needed.
        if bg > base:
            add, branch = adj["above_table_value"], "background > table value"
        elif bg > base - margin:
            add, branch = adj["within_margin_below"], "table - margin < background <= table"
        else:
            add, branch = adj["below_by_more_than_margin"], "background <= table - margin"
        add = float(add)
        return base + add, {**info, "limit_dBA": base + add,
                            "boundary_caveat": "background exactly equal to (table - margin) "
                                               "takes the lower band",
                            "background_LA90_dBA": bg,
                            "background_is_measured": bool(background_is_measured),
                            "adjustment_dB": add, "adjustment_branch": branch,
                            "adjustment_source": adj.get("source")}

    if ctype == "increment_over_background":
        ak = _alw_key(period)
        alw = c.get("allowance_dB")
        if not isinstance(alw, dict) or ak not in alw:
            raise ValueError(f"{country}.noise is missing the numeric field allowance_dB[{ak!r}]")
        allowance = float(alw[ak])

        if background_la90 is None:
            bgc = c.get("background_LA90") or {}
            raise ValueError(
                f"{country} uses an increment-over-background criterion; without a background "
                f"LA90 there is no absolute limit.\nRegistry status: {bgc.get('status')}\n"
                f"For screening, pass an explicit assumed value (suggested range: "
                f"{bgc.get('assumed_range_dB_A')}); it is labelled as an assumption in the output.")
        bg = _finite_background(country, background_la90)
        limit = bg + allowance
        return limit, {"mode": "increment_over_background",
                       "rule": (c.get("rule_text") or {}).get(ak),
                       "allowance_dB": allowance,
                       "background_LA90_dBA": bg,
                       "background_is_measured": bool(background_is_measured),
                       "background_caveat": (c.get("background_LA90") or {}).get("assumption_basis"),
                       "limit_dBA": limit,
                       "open_question": c.get("_open_question"),
                       **c.provenance}

    raise NotImplementedError(f"{country}.noise criterion_type={ctype!r} is not supported")


def flicker_limits(country: str) -> tuple[float | None, float | None, dict]:
    """Return (hours/year limit, minutes/day limit, info). **Either limit may be None**
    - e.g. a jurisdiction that only requires a study without a numeric threshold."""
    c = get(country, "shadow_flicker")
    h = c.get("limit_hours_per_year")
    m = c.get("limit_minutes_per_day")
    info = {"limit_hours_per_year": h, "limit_minutes_per_day": m,
            "limit_logic": c.get("limit_logic") or c.get("requirement"),
            "is_hard_veto": c.get("enforceability") == "statutory_hard"
                            and c.get("criterion_type") == "annual_duration",
            "lat_range_deg": jurisdiction(country).get("lat_range_deg"),
            **c.provenance}
    return h, m, info


def needs_background(country: str) -> bool:
    """Whether this jurisdiction's noise limit needs a background level.

    This is a property of the registry entry, not of one criterion type: both the
    increment criterion and the background-adjusted absolute limit need it."""
    c = get(country, "noise")
    return (c["criterion_type"] == "increment_over_background"
            or c.get("background_adjustment") is not None)


if __name__ == "__main__":
    print("registry:", criteria_path())
    for ct in load()["countries"]:
        try:
            lim, info = noise_limit(ct, period="night", background_la90=30.0)
            print(f"  {ct}.noise    night limit {lim:.1f} dB(A)  [{info['mode']}]")
        except Exception as e:
            print(f"  {ct}.noise    -> {type(e).__name__}: {str(e).splitlines()[0]}")
        try:
            h, m, i = flicker_limits(ct)
            print(f"  {ct}.flicker  {h} h/yr, {m} min/day  hard_veto={i['is_hard_veto']}")
        except Exception as e:
            print(f"  {ct}.flicker  -> {type(e).__name__}: {str(e).splitlines()[0]}")
