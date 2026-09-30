"""Turbine spacing conflict check using elliptical envelopes (pure shapely).

Each turbine gets an ellipse envelope (semi-major axis along the wind direction,
semi-minor axis across it), discretised to a 64-vertex polygon. A spatial index
does the coarse search, then candidate pairs are tested with real polygon
intersection / distance and classified CONFLICT / NEAR / OK.

Design credit: the envelope design (64-vertex ellipse, spatial-index coarse search,
conflict / near / ok levels) follows the QGIS plugin VelantisWind (GPL-3.0); this is a separate
pure-shapely implementation and no code was copied. See THIRD_PARTY_NOTICES.md.
"""
import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

ELLIPSE_VERTICES = 64


def ellipse_conflict(dx, dy, a, b, azimuth_deg):
    """Whether two co-oriented congruent elliptical envelopes overlap (closed form).

    The Minkowski sum of two congruent, co-oriented ellipses (semi-axis ``a``
    along the wind, ``b`` across it) is the same ellipse scaled by 2, so they
    intersect iff the centre offset, expressed in the ellipse frame, satisfies

        (du/a)^2 + (dv/b)^2 < 4

    where du is the component along the major (downwind) axis and dv the
    perpendicular component.

    Why this function exists: if candidate selection uses an isotropic circular
    exclusion radius equal to the *larger* spacing (``downwind_d * D``, e.g. 5D),
    the maximum centre distance the ellipse criterion can flag is exactly
    ``2a = downwind_d * D``. A circle of that radius is therefore stricter than the
    ellipse test and a CONFLICT is mathematically impossible, so a spacing check
    run afterwards is a tautology. Using the ellipse criterion during selection
    also allows denser packing than the circle, by a factor up to
    ``a/b = downwind_d/crosswind_d``.

    :param dx, dy: centre offset (metres, projected CRS); may be ndarrays
    :param a: downwind semi-axis = downwind_d * D / 2
    :param b: crosswind semi-axis = crosswind_d * D / 2
    :param azimuth_deg: major-axis azimuth (compass bearing, clockwise from north),
        same convention as ``_ellipse_polygon``
    :return: bool (or bool ndarray); True means conflict
    """
    theta = math.radians(azimuth_deg)
    ux, uy = math.sin(theta), math.cos(theta)      # major axis (downwind) unit vector
    vx, vy = math.cos(theta), -math.sin(theta)     # perpendicular (crosswind)
    du = np.asarray(dx) * ux + np.asarray(dy) * uy
    dv = np.asarray(dx) * vx + np.asarray(dy) * vy
    return (du / a) ** 2 + (dv / b) ** 2 < 4.0


def ellipse_required_distance(a, b, azimuth_deg, bearing_deg):
    """Minimum allowed centre distance between two turbines along a bearing.

    For diagnostics and reports, not used for the verdict. Returns ``2a`` when
    bearing equals azimuth (strictest) and ``2b`` when perpendicular to it.
    """
    phi = math.radians(bearing_deg - azimuth_deg)
    return 2.0 / math.sqrt((math.cos(phi) / a) ** 2 + (math.sin(phi) / b) ** 2)


def sector_energy_weights(windrose):
    """Sector weights by wind *energy*, not by occurrence frequency.

    Wind power density scales with the cube of wind speed. For a Weibull
    distribution E[V^3] = A^3 * Gamma(1 + 3/k), so a sector's energy share is
    proportional to freq * A^3 * Gamma(1 + 3/k).

    The most frequent direction and the most energetic direction are often
    different: a frequent but weak sector matters less for wakes and production
    than an occasional but strong one. Spacing rules guard against wakes in the
    energy-carrying directions, so orientation should follow energy.

    :param windrose: dict with sector_freq_pct / sector_weibull_A / sector_weibull_k
    :return: (sector_center_deg, weights); weights are normalised to sum to 1
    """
    freq = np.asarray(windrose["sector_freq_pct"], dtype=float)
    A = np.asarray(windrose["sector_weibull_A"], dtype=float)
    k = np.asarray(windrose["sector_weibull_k"], dtype=float)
    n = len(freq)
    centers = np.arange(n) * (360.0 / n)
    energy = freq * A ** 3 * np.array([math.gamma(1 + 3 / ki) for ki in k])
    return centers, energy / energy.sum()


def dominant_energy_azimuth(windrose):
    """Centre azimuth (deg) of the highest-energy sector.

    Use this instead of the highest-frequency sector as the main direction for
    the spacing check.
    """
    centers, w = sector_energy_weights(windrose)
    return float(centers[int(np.argmax(w))])


def high_energy_azimuths(windrose, coverage=0.8):
    """Azimuths of the sectors that together carry ``coverage`` of the energy.

    Sectors are taken in descending energy order. Passing the result as
    ``azimuth_deg`` to ``check_spacing`` draws one ellipse per direction and
    takes their union; a single dominant direction only protects that direction
    and can miss under-spaced rows for bimodal or multimodal wind roses.
    """
    centers, w = sector_energy_weights(windrose)
    order = np.argsort(-w)
    cum = np.cumsum(w[order])
    keep = order[: int(np.searchsorted(cum, coverage) + 1)]
    return [float(centers[i]) for i in sorted(keep)]


def _ellipse_polygon(cx, cy, a, b, azimuth_deg):
    """Ellipse envelope polygon at (cx, cy) in a local projected CRS (metres).

    :param a: semi-major axis, downwind (downwind_d * D / 2)
    :param b: semi-minor axis, crosswind (crosswind_d * D / 2)
    :param azimuth_deg: main wind direction as a compass bearing (north = 0,
        clockwise). This is not a mathematical angle, so it differs from the
        output of atan2.
    """
    theta = math.radians(azimuth_deg)
    ux, uy = math.sin(theta), math.cos(theta)   # major axis (downwind) unit vector
    vx, vy = math.cos(theta), -math.sin(theta)  # perpendicular (crosswind) unit vector
    t = np.linspace(0, 2 * math.pi, ELLIPSE_VERTICES, endpoint=False)
    ca, sb = a * np.cos(t), b * np.sin(t)
    xs = cx + ca * ux + sb * vx
    ys = cy + ca * uy + sb * vy
    return Polygon(zip(xs, ys))


def check_spacing(
    points_gdf,
    rotor_diameter_m,
    azimuth_deg,
    downwind_d=5.0,
    crosswind_d=3.0,
    near_margin=0.10,
    id_col="wtg_id",
):
    """Turbine spacing conflict check.

    :param points_gdf: geopandas.GeoDataFrame of candidate / final turbine
        points. Must be in a projected (metre) CRS and contain the ``id_col`` column.
    :param rotor_diameter_m: rotor diameter D, a single global value
    :param azimuth_deg: main wind direction (compass bearing, clockwise from
        north). Either a **scalar** or a **sequence** of azimuths; for a sequence
        one ellipse is drawn per direction and the union is used (bimodal /
        multimodal wind roses). Ways to choose the direction(s):
          - ``dominant_energy_azimuth(windrose)``: the single highest-energy sector
          - ``high_energy_azimuths(windrose, 0.8)``: all sectors covering 80% of energy
    :param downwind_d: downwind spacing multiple. Example default 5D; the
        appropriate value depends on your wake model, terrain and turbine.
        Set your own.
    :param crosswind_d: crosswind spacing multiple. Example default 3D; set your own.
    :param near_margin: "near" threshold as a fraction of the characteristic
        radius (= max(a, b)). Example default 10%; set your own.
    :param id_col: name of the turbine-ID column
    :return: {
        'status': {wtg_id: 'CONFLICT'|'NEAR'|'OK'},
        'conflicts': [(id1, id2, centre_distance_m), ...],
        'near': [(id1, id2, centre_distance_m), ...],
        'ellipses_gdf': GeoDataFrame of each point's ellipse envelope (union
        of the per-direction ellipses when several azimuths are given)
    }
    """
    if points_gdf.crs is None or not points_gdf.crs.is_projected:
        raise ValueError(
            "points_gdf must be in a projected (metre) CRS, not lon/lat; "
            "reproject to a local UTM zone first"
        )

    a = downwind_d * rotor_diameter_m / 2
    b = crosswind_d * rotor_diameter_m / 2
    near_tol = near_margin * max(a, b)

    azimuths = np.atleast_1d(np.asarray(azimuth_deg, dtype=float)).tolist()

    ids = points_gdf[id_col].astype(str).tolist()
    geoms = list(points_gdf.geometry)
    if len(azimuths) == 1:
        ellipses = [_ellipse_polygon(pt.x, pt.y, a, b, azimuths[0]) for pt in geoms]
    else:
        # Multi-direction: one ellipse per direction, union. A single-direction
        # ellipse only guards downwind spacing in that one direction, so
        # bimodal / multimodal roses could miss an under-spaced row in another
        # energy-carrying direction.
        ellipses = [
            unary_union([_ellipse_polygon(pt.x, pt.y, a, b, az) for az in azimuths])
            for pt in geoms
        ]

    tree = STRtree(ellipses)
    status = {i: "OK" for i in ids}
    conflicts = []
    near = []
    seen_pairs = set()

    for i, poly in enumerate(ellipses):
        # Buffer by near_tol before querying: STRtree.query only prefilters by
        # bounding box, so two ellipses whose boundaries are closer than near_tol
        # but whose bounding boxes do not overlap would otherwise miss a NEAR.
        for j in tree.query(poly.buffer(near_tol)):
            j = int(j)
            if j <= i or (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            other = ellipses[j]
            center_dist = geoms[i].distance(geoms[j])
            if poly.intersects(other):
                conflicts.append((ids[i], ids[j], center_dist))
                status[ids[i]] = "CONFLICT"
                status[ids[j]] = "CONFLICT"
            elif poly.distance(other) < near_tol:
                near.append((ids[i], ids[j], center_dist))
                if status[ids[i]] == "OK":
                    status[ids[i]] = "NEAR"
                if status[ids[j]] == "OK":
                    status[ids[j]] = "NEAR"

    import geopandas as gpd

    ellipses_gdf = gpd.GeoDataFrame(
        {id_col: ids, "status": [status[i] for i in ids]},
        geometry=ellipses,
        crs=points_gdf.crs,
    )
    return {
        "status": status,
        "conflicts": conflicts,
        "near": near,
        "ellipses_gdf": ellipses_gdf,
    }
