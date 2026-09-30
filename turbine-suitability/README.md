# wind-turbine-suitability

> **Scope and disclaimer.** This package does **screening-level** turbine-class suitability
> checks. It is **not** a formal site suitability assessment (which a qualified body
> issues from turbine load simulations, see IECRE OD-501) and it is **not** an energy-yield
> tool. "Preliminary fit" means no known hard constraint was triggered; it does **not** mean
> "compliant with IEC 61400-1". The text is embedded in every result as `DISCLAIMER`.

## What it decides

Given a description of the site (extreme wind, mean wind, turbulence, air density,
temperature extremes, shear, inflow angle, altitude, hub height) and a description of a
turbine's design envelope, it returns per-constraint verdicts (`pass`, `marginal`, `fail`,
`unknown`) and one overall result:

| Overall | Meaning |
|---|---|
| `pass` | no constraint triggered, none unknown |
| `marginal` | inside limits but within a tolerance band, or capped by a stated uncertainty flag |
| `exceeds_class_assumption` | a class *design assumption* (turbulence, shear, inflow angle, mean wind) is exceeded; a turbine maker's site-specific load check may still release it |
| `insufficient_data` | a required site value or turbine limit is missing |
| `fail` | a hard turbine limit is exceeded (extreme wind, altitude, temperature, density, tip clearance) |

Design rules the code follows (the tests check them):

1. **Hard constraints are not weighted.** One failing hard limit fails the turbine; nothing compensates.
2. **Unknown is never pass.** Missing limits, missing site values and non-finite values (NaN from an
   upstream no-data) all give `unknown`.
3. **Hard limits and design assumptions are worded differently.** Exceeding an assumption is not the
   same as "not suitable"; only the turbine maker can say that.
4. **Turbulence is an interval and a curve.** The reference intensity is an expected value while site
   estimates are 90th percentiles, so limits are converted first; the limit falls with wind speed, so
   the tightest bin is used. Intervals are compared whole, never by midpoint.
5. **Evidence tiers.** The tier is computed from facts (on-site months, long-term correction). Model-only
   tiers use qualified wording ("preliminary fit (model level)"), and every wording says "preliminary". The
   engine *reports* measured fields found in a model-only tier (`tier_policy_violation`); it does not block them.
6. **Hub height is a turbine attribute.** Each turbine is judged on the site card built for its own hub
   height; a turbine that declares none is `insufficient_data`.

## What it does not decide

Energy yield, wakes, layout, economics, foundations, noise, grid, permitting, or any load
calculation. Wake-added turbulence is **not** included, so the real turbulence requirement can only
be higher than the one checked here. Generic gust ratios do not hold in tropical-cyclone regions.

## The IEC 61400-1 relation, in words

The standard groups turbines into wind-speed classes by a reference extreme wind (50-year,
10-minute mean at hub height) and a linked annual-mean limit, and into turbulence categories by a
reference turbulence intensity at 15 m/s. Extreme gusts and annual-mean limits follow from the
reference wind by fixed ratios, extreme winds are extrapolated to hub height with a fixed exponent
(not the site's measured normal shear), and normal turbulence is a line in wind speed that
falls as speed rises. A tropical-cyclone class and a designer-defined special class exist.

**This package does not reproduce the standard.** The class parameters in
`wind_turbine_suitability/iec.py` are illustrative defaults, gathered from public secondary summaries
and **not verified against the standard text**. Check them against the edition you hold and, if
different, pass your own table with `ClassTable.from_dict(...)`. The screening logic contains no
hard-coded class limits.

## Install and run

```bash
pip install -e .          # no runtime dependencies
pip install pytest && pytest
python examples/run_synthetic.py
```

```python
from wind_turbine_suitability import SiteConditions, TurbineSpec, SuitabilityEngine, decide_tier

site = SiteConditions("my site").set("hub_height_m", 110.0).set("v50_ms", 38.0, "model")  # etc.
turbine = TurbineSpec("Generic 3 MW class", vref_max_ms=42.5, rotor_diameter_m=120.0)
result = SuitabilityEngine().evaluate(site, turbine, tier=decide_tier())
```

## Helpers for building site inputs

Pure Python, caller-supplied numbers only (no file reading, no downloads):

- `roughness`: Davenport-Wieringa z0 classes, area-weighted equivalent z0 averaged in turbulence space.
- `turbulence`: z0 interval to a 90th-percentile turbulence interval; the scatter-factor range rests on an
  assumed coefficient of variation and should be replaced by a measured one when a mast exists.
- `extremes`: Gumbel fit (returns `None` on short samples), 1-minute to 10-minute conversion, power-law
  extrapolation with the extreme-wind exponent.
- `temperature`: absolute-extreme interval (record extreme to return level) with a lapse-rate correction.

## Turbine data

No turbine library or power curves are shipped; this tool needs only envelope limits. Parameters in tests
and examples are synthetic ("Generic 3 MW class"). Supply your own turbine data in `TurbineSpec`.

## Arbitrary defaults you should review

All tolerance bands (5 % / 10 % / 3 % / 20 % ...), the 25 m tip clearance, the verdict-cap thresholds and the
V50/Vave plausibility window (2.5 to 9, flag only, never blocks) in `ScreeningConfig` are **arbitrary example
values**, not from any standard or study. Verdict-cap flags on the site card (`v50_uncertainty_level`,
`ti_uncertainty_level`, `shear_neutral_only`, `terrain_complexity_level`) use an integer scale of your choosing.

## Turbulence checks differ by design

`SuitabilityEngine` compares the site interval with the limit curve at the **tightest bin** (highest wind speed up to
cut-out). `verdict_against_iref` and `ClassTable.required_turbulence_category` use only the 15 m/s reference speed and
are looser; treat them as coarse classification helpers.

## Class table caveats

The default table mixes editions: turbulence category A+ exists only in the newer edition. Class T's annual-mean
limit is left undefined, and applying Vave = 0.2 x Vref to it is an assumption not confirmed from the standard.
Results carry `table_verified = False` and the report prints a warning until you set `verified=True` on a table you checked.

## Known limitations

- A cold-stop (operating lower temperature) exceedance is a yield flag, not a class or structural finding.
- Site turbulence estimated from roughness is a neutral-stability estimate and stacks three assumptions;
  measured shear at high wind can be several times the neutral estimate.
- Turbine envelope limits are read as given; whether a datasheet limit means what this code assumes
  (e.g. mean density envelope vs. extreme density) must be confirmed with the maker.

Licence: Apache-2.0. See `THIRD_PARTY_NOTICES.md`.
