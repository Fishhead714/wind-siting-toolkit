"""Oriented rectangular turbine grid."""
from __future__ import annotations

import numpy as np


def oriented_grid(bounds, dx: float, dy: float, azimuth_deg: float,
                  offset: tuple[float, float] = (0.0, 0.0)):
    """Grid nodes covering ``bounds`` with rows spaced ``dx`` along the wind axis.

    ``azimuth_deg`` is the orientation of the downwind axis, measured clockwise
    from grid north. Only the axis matters, so azimuth and azimuth+180 give the
    same node set. The downwind axis has spacing ``dx`` and the perpendicular
    (crosswind) axis spacing ``dy``. ``offset`` shifts the
    lattice as a fraction of (dx, dy), each in [0, 1).

    Returns two 1-D arrays (x, y) in the CRS of ``bounds``.
    """
    if dx <= 0 or dy <= 0:
        raise ValueError("spacings must be positive")
    az = np.radians(azimuth_deg)
    d = np.array([np.sin(az), np.cos(az)])      # downwind axis
    p = np.array([np.cos(az), -np.sin(az)])     # crosswind axis (perpendicular)
    minx, miny, maxx, maxy = bounds
    corners = np.array([[minx, miny], [maxx, miny], [minx, maxy], [maxx, maxy]])
    u, v = corners @ d, corners @ p
    iu = (np.arange(np.floor(u.min() / dx), np.ceil(u.max() / dx) + 1) + offset[0]) * dx
    iv = (np.arange(np.floor(v.min() / dy), np.ceil(v.max() / dy) + 1) + offset[1]) * dy
    U, V = np.meshgrid(iu, iv)
    X = U * d[0] + V * p[0]
    Y = U * d[1] + V * p[1]
    return X.ravel(), Y.ravel()
