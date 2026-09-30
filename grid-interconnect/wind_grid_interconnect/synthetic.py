"""Synthetic scenario generator (invented geometry; no real place or data).

Coordinates are metres in a local azimuthal-equidistant CRS centred on (0, 0)
of a made-up datum (``SYNTHETIC_CRS``). It does not correspond to any site.

Layout (x to the east, y to the north, ~12 km x 10 km):

* one long east-west road and a north-south side road, plus a short spur;
* a north-south river (a water barrier) crossed by the main road at a bridge;
* ``n_turbines`` turbines in two clusters at varying distances from the roads;
* a collection point (substation) at the east end of the main road.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, box

#: Local metric CRS with no relation to any real location.
SYNTHETIC_CRS = "+proj=aeqd +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=6371000 +units=m +no_defs"
BOUNDS = (0.0, 0.0, 12000.0, 10000.0)


@dataclass
class Scenario:
    roads: list
    river_lines: list
    lakes: list
    turbines: np.ndarray       # (n, 2)
    substation: np.ndarray     # (2,)
    transmission_line: object  # a line a tap could connect to
    bounds: tuple
    terrain_mult: np.ndarray   # multiplier grid for ``cell`` metres (see make_scenario)
    cell: float

    @property
    def points(self) -> np.ndarray:
        """Turbines followed by the substation (the last row)."""
        return np.vstack([self.turbines, self.substation])


def make_scenario(n_turbines: int = 16, seed: int = 0, cell: float = 100.0) -> Scenario:
    rng = np.random.default_rng(seed)
    main = LineString([(0, 5000), (2500, 5200), (3000, 5100), (6000, 5000), (9000, 4800), (11500, 5000)])
    side = LineString([(3000, 0), (3100, 3000), (3000, 5100), (3200, 8000), (3000, 10000)])
    spur = LineString([(9000, 4800), (9300, 2600), (9600, 1500)])
    # the main road and the river cross near (6000, 5000)
    river = LineString([(6100, 0), (5900, 3000), (6100, 5000), (6000, 7500), (6200, 10000)])
    lake = box(9800, 7800, 11200, 8800)
    n_a = n_turbines // 2
    n_b = n_turbines - n_a
    # cluster A: north-west of the river, 1.2-3 km north of the main road, on both sides of the side road
    A = np.column_stack([rng.uniform(1200, 5000, n_a), rng.uniform(6300, 8300, n_a)])
    # cluster B: east of the river, south of the main road, 0.5-3 km from the spur / main road
    B = np.column_stack([rng.uniform(7000, 10800, n_b), rng.uniform(1200, 3800, n_b)])
    turbines = np.vstack([A, B]).round(0)
    substation = np.array([11500.0, 5000.0])
    tl = LineString([(12000, 9500), (11400, 6500), (11000, 3000), (10500, 0)])
    rows = int(np.ceil((BOUNDS[3] - BOUNDS[1]) / cell))
    cols = int(np.ceil((BOUNDS[2] - BOUNDS[0]) / cell))
    field = gaussian_filter(rng.normal(size=(rows, cols)), sigma=8)
    field = (field - field.min()) / (field.max() - field.min())
    terrain_mult = 1.0 + 1.5 * field        # 1.0 (flat) .. 2.5 (rough): invented
    return Scenario([main, side, spur], [river], [lake], turbines, substation, tl,
                    BOUNDS, terrain_mult, cell)
