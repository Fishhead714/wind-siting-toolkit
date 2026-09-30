# Third-party notices

No third-party source code is copied into this package. It imports the libraries below at
run time. Licences were read from the installed package metadata in a clean virtual
environment; none of the runtime dependencies is GPL or AGPL.

| Library | Version checked | Licence (from metadata) |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| shapely | 2.1.2 | BSD-3-Clause (classifier: BSD License) |
| geopandas | 1.2.0 | BSD-3-Clause |
| pyproj | 3.8.0 | MIT |
| pandas (pulled in by geopandas) | 3.0.6 | BSD-3-Clause |

Transitive dependencies were not audited individually.

## Ideas and conventions

The method is a plain regular grid rotated to a chosen axis, with spacing expressed as
multiples of the rotor diameter D (for example "5D x 3D"). That is a textbook layout
convention, not the work of a single author, and no source or text was taken from any
project. Numeric defaults in `EXAMPLE_DEFAULTS` are illustrative and have no cited source.
