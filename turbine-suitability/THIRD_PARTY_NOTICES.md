# Third-party notices

No third-party source code, data set, turbine model library or power curve is included in this
package, and it has no runtime dependencies (Python standard library only). Tests need `pytest`
(MIT).

## Public-data readers deliberately left out

This package does not include readers for external wind, land-cover or temperature data sets: they
need raster or dataframe libraries, and this release stays
pure Python. If you add such readers, check and state each data set's licence yourself.

## Literature behind constants

The author has not re-read these sources for this release. All values are marked example defaults.

- **Surface roughness classes.** Wieringa, J. (1992), "Updating the Davenport roughness
  classification", J. Wind Eng. Ind. Aerodyn. 41-44, 357-368; based on Davenport (1960).
  Representative z0 per class in `roughness.py` are the commonly quoted values.
- **IEC 61400-1 (wind turbines - design requirements).** Class parameters in `iec.py` (reference
  wind speeds, reference turbulence intensities, the ratios for annual mean and extreme gust, the
  0.11 extreme-wind exponent, the additive constant of the normal-turbulence line, default inflow angle
  and normal-profile exponent) are widely quoted headline values from public secondary summaries.
  **Not reproduced from the standard and not verified against it; the edition (Ed. 3 vs Ed. 4) of each
  value is uncertain.** No table, text or figure of the standard is copied. Verify before relying on them.
- **IECRE OD-501** (site suitability assessment rules): referenced only to state that this tool does not replace it.
- **1-minute to 10-minute wind conversion (0.88)** and **standard-atmosphere lapse rate (6.5 C/km)**:
  commonly used engineering constants; not re-verified.
- **Gumbel extreme-value fit** by the method of moments: textbook statistics.
- **Coefficient of variation range 0.15 to 0.30** for the turbulence scatter factor: an assumption for onshore
  sites without a documented authoritative source.
