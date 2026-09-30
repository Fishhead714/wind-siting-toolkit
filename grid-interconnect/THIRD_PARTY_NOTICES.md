# Third-party notices

No third-party source code is copied into this package. It imports the libraries below at
run time. Licences were read from the installed package metadata (`License-Expression`,
`License`, or trove classifiers) in a clean virtual environment; none of the runtime
dependencies is GPL or AGPL.

| Library | Version checked | Licence (from metadata) |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| scipy | 1.18.1 | BSD (classifier; licence file is BSD-3-Clause style) |
| shapely | 2.1.2 | BSD-3-Clause |
| geopandas | 1.2.0 | BSD-3-Clause |
| networkx | 3.7 | BSD-3-Clause |
| scikit-image | 0.26.0 | BSD (classifier; BSD-3-Clause style) |
| pyproj | 3.8.0 | MIT |

Transitive dependencies were not audited individually.

## Algorithms

Minimum spanning tree (scipy), Dijkstra shortest paths (networkx) and minimum-cost path
through an array (scikit-image `route_through_array`) are standard textbook methods used via
those libraries. The hybrid "direct link versus road link" pair cost and the sensitivity
tooling are the author's own formulation; no upstream project is adapted.
