"""Minimal in-memory raster helpers (numpy only, metric CRS).

A raster here is a 2-D array plus its upper-left corner and a square pixel
size in metres, in the same CRS as the developable area. Reading files is left
to the caller (rasterio, GDAL, ...), keeping this package dependency-light.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Raster:
    array: np.ndarray      # shape (rows, cols), row 0 is the northern edge
    x0: float              # x of the upper-left corner
    y0: float              # y of the upper-left corner
    res: float             # pixel size in metres (square pixels)

    def rowcol(self, x, y):
        cols = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        rows = np.floor((self.y0 - np.asarray(y)) / self.res).astype(int)
        return rows, cols

    def sample(self, x, y, outside=np.nan):
        """Nearest-pixel values; ``outside`` where a point is off the raster."""
        rows, cols = self.rowcol(x, y)
        h, w = self.array.shape
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.full(rows.shape, outside, dtype="float64")
        out[ok] = self.array[rows[ok], cols[ok]]
        return out

    def disk_fraction(self, x, y, radius_m: float):
        """Mean of the raster over a disk around each point.

        Intended for 0/1 masks (e.g. "usable land"). Points whose disk leaves
        the raster get 0.0: the caller cannot know what lies outside, so the
        conservative answer is used.
        """
        x = np.atleast_1d(np.asarray(x, dtype="float64"))
        y = np.atleast_1d(np.asarray(y, dtype="float64"))
        rpx = radius_m / self.res
        r = int(np.ceil(rpx))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= rpx * rpx
        n_disk = disk.sum()
        h, w = self.array.shape
        rows, cols = self.rowcol(x, y)
        out = np.zeros(x.shape, dtype="float64")
        arr = np.nan_to_num(self.array.astype("float64"), nan=0.0)
        for k in range(x.size):
            r0, r1, c0, c1 = rows[k] - r, rows[k] + r + 1, cols[k] - r, cols[k] + r + 1
            if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
                continue
            out[k] = (arr[r0:r1, c0:c1] * disk).sum() / n_disk
        return out


def slope_degrees(dem: Raster) -> Raster:
    """Slope in degrees from a DEM on a metric grid (central differences)."""
    gy, gx = np.gradient(dem.array.astype("float64"), dem.res)
    return Raster(np.degrees(np.arctan(np.hypot(gx, gy))), dem.x0, dem.y0, dem.res)
