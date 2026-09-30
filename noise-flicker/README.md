# noise_flicker - wind turbine noise and shadow-flicker screening engine

Screening-level calculations for wind turbine layouts:

- **Noise**: ISO 9613-2 downwind propagation with ISO 9613-1 atmospheric absorption,
  per receptor, energy-summed over the whole array; compliance distances and a
  sensitivity analysis of the modelling assumptions.
- **Shadow flicker**: minute-by-minute sun position over a full year (NOAA solar
  calculator equations), exact ray / rotor-disc intersection, astronomical worst case and
  (with a wind rose and sunshine probability) an expected case.
- **Contours**: gridded noise and flicker fields turned into contour lines / bands and
  written to a GeoPackage together with per-receptor results and a parameter table.

It is a screening tool, not an EIA-grade assessment: no terrain or building screening,
no measured background noise.

## Nothing is built in

The engine ships **no turbine data, no spectrum and no regulatory limits**. Everything
that depends on a specific turbine or jurisdiction is an input:

| Input | Where | Example file |
|---|---|---|
| Hub height, rotor diameter, L_WA, blade chord | CLI flags / layer fields / function arguments (required, no defaults) | `examples/reference_turbine.json` |
| Octave-band spectral shape | `--spectrum-file` / `noise.load_spectra()` | `examples/spectrum.example.json` |
| Noise and flicker criteria | `--criteria` / `criteria.set_path()` / `$WTG_NOISE_FLICKER_CRITERIA` | `examples/criteria.example.json` |

The example files hold **openly published values**, not defaults:

- `reference_turbine.json` - IEA Wind TCP Task 37 **IEA-3.4-130-RWT** land-based reference
  turbine (3.37 MW, D = 130 m, hub 110 m; Bortolotti et al. 2019, NREL/TP-5000-73492, Table 2).
  Blade chord 4.3 m = maximum chord in the turbine's public YAML definition (conservative for
  flicker). **L_WA = 106.3 dB(A) is an estimate** from the rated-power regression of
  van den Berg et al. (2008), not a published property of that turbine.
- `spectrum.example.json` - a generic MW-class octave shape derived as the mean of 10
  normalised spectra of >= 2 MW turbines (WINDFARMperception, 2008, Appendix B), renormalised;
  only the 8 derived values are included.
- `criteria.example.json` - international guidance for illustration only (IFC 2007 noise
  guideline values, WHO 2018 recommendation as information, LAI 2020 shadow-flicker values).
  **Example only - check the rules of your jurisdiction.**

The engine does not need any of them; they exist so the demo and the tests can run.
The flicker calculation distance depends on the blade chord, which is therefore a required
input (no estimate from the rotor diameter is offered).

### Criteria registry

`engine/criteria.py` documents the supported criterion shapes:

- `absolute_limit` - flat `{"day", "night"}` or zoned limit tables;
- `absolute_limit_with_background_adjustment` - table value raised depending on the
  background level (`margin_dB`, `above_table_value`, `within_margin_below`,
  `below_by_more_than_margin`);
- `increment_over_background` - limit = background level + `allowance_dB[period]`: the
  generic "background + N dB" form used in published guidance such as ETSU-R-97 and
  NZS 6808. It is a generic shape; the allowance values are always supplied by the user;
- shadow flicker: `limit_hours_per_year` / `limit_minutes_per_day` (null = no numeric threshold);
- `"status": "NOT_RESEARCHED"` raises instead of borrowing another jurisdiction's rule.
- `enforceability` is recorded as provenance. `"statutory_hard"` (with
  `criterion_type: "annual_duration"`) marks a flicker limit as legally binding and sets
  `is_hard_veto: true` in the output; every other value is reported as guidance.

Contour levels are not built in either: `contours.py` draws the criterion limit plus any
levels passed with `--noise-levels` / `--flicker-levels` / `--levels-file`
(`examples/contour_levels.example.json`: noise 35/40/45 dB(A), flicker 8/30 h/year).

## Requirements

Python >= 3.10, numpy, scipy, contourpy, shapely, pyproj, geopandas, pyogrio
(threadpoolctl optional). The point calculations in `noise.py`, `flicker.py` and
`solar.py` need numpy only.

## Quick start

```bash
PY=python3 demo/run_demo.sh            # synthetic layout -> demo/out/demo_contours.gpkg
python3 tests/selftest.py              # regression suite
python3 engine/contours.py --help
```

Python API:

```python
import sys; sys.path.insert(0, "engine")
import criteria, noise
criteria.set_path("my_criteria.json")
noise.load_spectra("my_spectrum.json")
atm = noise.Atmosphere(t_c=20, rh_pct=70, spectrum="my_turbine_mode")
res = noise.assess_receptors(turbines, receptors, atm, country="MYJURISDICTION", period="night")
```

## Limitations

- ISO 9613-2 is commonly quoted for source heights below about 30 m and distances up to
  about 1 km; applying it to modern turbines at km-scale distances is an extrapolation.
- Hard ground (G = 0) is the default because it is the conservative bound for this geometry.
- The flicker calculation distance follows the common convention that flicker is assessed
  where a blade covers at least 20 % of the solar disc; it therefore needs the blade chord.
- No terrain, vegetation or building screening (results are conservative).

## References

Data used in `examples/` (see also THIRD_PARTY_NOTICES.md):

- Bortolotti, P. et al. (2019), *IEA Wind TCP Task 37: Systems Engineering in Wind Energy -
  WP2.1 Reference Wind Turbines*, NREL/TP-5000-73492. https://www.nrel.gov/docs/fy19osti/73492.pdf
- IEA Wind Task 37, IEA-3.4-130-RWT repository (Apache-2.0).
  https://github.com/IEAWindTask37/IEA-3.4-130-RWT
- van den Berg, F., Pedersen, E., Bouma, J., Bakker, R. (2008), *WINDFARMperception: Visual and
  acoustic impact of wind turbine farms on residents*, Final Report. https://pure.rug.nl/ws/files/14620621/WFp-final.pdf
- Møller, H., Pedersen, C. S. (2011), Low-frequency noise from large wind turbines,
  J. Acoust. Soc. Am. 129(6), 3727-3744.
- IFC (2007), *Environmental, Health, and Safety (EHS) General Guidelines*, section 1.7 Noise.
- WHO Regional Office for Europe (2018), *Environmental Noise Guidelines for the European Region*.
- LAI (2020), *WKA-Schattenwurfhinweise*, Stand 23.01.2020.

Methods:

- ISO 9613-1:1993, Acoustics - Attenuation of sound during propagation outdoors - Part 1.
- ISO 9613-2:1996, Acoustics - Attenuation of sound during propagation outdoors - Part 2.
- NOAA ESRL Solar Calculator (equations after Meeus, *Astronomical Algorithms*).
- Chaikin, G. M. (1974), An algorithm for high-speed curve generation.

## License

Apache License 2.0 (see `LICENSE`). Third-party sources of example values are listed in
`THIRD_PARTY_NOTICES.md`; no third-party files are redistributed.
