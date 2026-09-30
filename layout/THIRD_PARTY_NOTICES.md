# Third-party notices

No third-party source code is copied into this package. It imports the libraries below at
run time. Licences were read from the installed package metadata (`License-Expression`,
`License`, or trove classifiers) in a clean virtual environment; none of the runtime dependencies is GPL or AGPL.

| Library | Version checked | Licence (from metadata) |
|---|---|---|
| TOPFARM | 2.6.2 | MIT |
| PyWake | 2.6.20 | MIT |
| rasterio | 1.5.1 | BSD-3-Clause |
| geopandas | 1.2.0 | BSD-3-Clause |
| shapely | 2.1.2 | BSD-3-Clause |
| scipy | 1.18.1 | BSD (classifier; licence file is BSD-3-Clause style) |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pandas | 3.0.6 | BSD-3-Clause |
| pyproj | 3.8.0 | MIT |

Transitive dependencies were not audited individually. Re-check before redistribution
that bundles binaries.

## Literature and documentation behind constants

Status "not verified" means the author has not re-read the source for this release.

- **RIX (Ruggedness Index) and the critical slope tan = 0.3.** Bowen, A. J. and Mortensen,
  N. G. (1996), *Exploring the limits of WAsP*, Risoe-R-995, Risoe National Laboratory.
  The 5 % and 1 % routing thresholds are **not** from this source; they are example
  defaults. Not verified against the original text for this release.
- **Topographic Position Index.** Weiss, A. D. (2001), *Topographic Position and
  Landforms Analysis*, poster, ESRI User Conference. The default radius and threshold
  used here are example values. Not verified.
- **Jensen wake decay constant k = 0.075 (land) / 0.04 (offshore).** WAsP documentation.
  Page not re-checked; not verified.
- **TurbOPark / TurboGaussian deficit.** Implemented in PyWake (MIT). The offshore
  constant A = 0.04 is the PyWake default, confirmed in its source. The onshore value
  A = 0.064 was reported in a conference poster (WindEurope 2025, poster PO110, a
  single-case value); **not verified** and used only as an example default.
- **IEC 61400-1 turbulence constants** (reference intensities 0.16 / 0.14 / 0.12 for
  classes A / B / C, b = 5.6 m/s, 8 degree inflow inclination, C_CT = 1.15 complex-terrain
  factor). Quoted as Edition 3 (2005); the class A+ value 0.18 and the applicability of
  C_CT are **not verified**; Edition 4 (2019) was not checked. The standard is a paid
  document and none of its text is reproduced.
- **Capacity density of onshore wind plants.** Denholm, P. et al. (2009), *Land-Use
  Requirements of Modern Wind Power Plants in the United States*, NREL/TP-6A2-45834.
  Cited as a context reference only; numbers not re-verified.
- **Air density.** International Standard Atmosphere lapse-rate formulas; IEC 61400-12-1
  for the humidity-term form. Not re-verified against the standard text.

## Design credit: VelantisWind (GPL-3.0)

The directional-ellipse spacing check in `wind_layout/spacingcheck.py` follows the design of the
QGIS plugin [VelantisWind](https://github.com/Velantis-Wind/VelantisWind) (GPL-3.0),
`spacing_core/geometry.py`: an elliptical envelope per turbine approximated by a 64-vertex polygon,
candidate pairs found with a spatial index, and a three-level result (conflict / near / ok).

No code was copied. The module here is a separate implementation in pure shapely and STRtree with
no QGIS dependency, plus a closed-form ellipse test that the upstream does not have. A comparison of the two files
(Python `difflib.SequenceMatcher` over non-blank, non-comment lines with whitespace normalised, against
upstream `spacing_core/geometry.py` at its default branch on 2026-09-30) gave a similarity ratio of 0.04
and no identical line longer than 25 characters. This is an independent implementation, so the package
is Apache-2.0; the credit is given because the design is not
the author's own. If you need the upstream's GUI workflow, use VelantisWind itself.
