# Third-party notices

No third-party source code or data files are redistributed in this repository. The files in
`examples/` quote individual numeric values from the publications below; they are credited here.

## IEA-3.4-130-RWT (IEA Wind TCP Task 37)

- Values used: rated power 3.37 MW, rotor diameter 130 m, hub height 110 m (report Table 2);
  maximum blade chord 4.298 m (read from `yaml/IEA-3.4-130-RWT.yaml`,
  commit aacff396f46b35f7104d83be095f859684013405).
- Report: Bortolotti, P. et al. (2019), NREL/TP-5000-73492, https://www.nrel.gov/docs/fy19osti/73492.pdf.
- Repository: https://github.com/IEAWindTask37/IEA-3.4-130-RWT, licensed under the
  Apache License, Version 2.0. The repository does not ship a NOTICE file. If the YAML file is
  ever redistributed with this project, include a copy of the Apache-2.0 license and keep the
  repository's attribution.

## WINDFARMperception Final Report (2008)

- van den Berg, F., Pedersen, E., Bouma, J., Bakker, R. (2008), https://pure.rug.nl/ws/files/14620621/WFp-final.pdf.
- Used: the sound power regression of section 3.2 (to estimate L_WA from rated power) and the
  mean of the normalised octave spectra of Appendix B (8 derived values only; the source table
  is not reproduced).

## Møller & Pedersen (2011)

- Møller, H., Pedersen, C. S. (2011), Low-frequency noise from large wind turbines,
  J. Acoust. Soc. Am. 129(6), 3727-3744.
- Used only as a visual cross-check of the derived spectral shape (Fig. 14); no values are
  taken from it.

## Guidance documents (values quoted as examples in `criteria.example.json`)

- IFC (2007), Environmental, Health, and Safety General Guidelines, Table 1.7.1.
- WHO Regional Office for Europe (2018), Environmental Noise Guidelines for the European Region, section 3.4.
- LAI (2020), WKA-Schattenwurfhinweise, Stand 23.01.2020 (also the source of the 20 % solar-disc coverage rule and the 3 degree
  minimum solar elevation default and the 8 h / 30 h contour levels in the examples).
