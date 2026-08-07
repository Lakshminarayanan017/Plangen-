"""
geometry.py — dependency-free polygon geometry for the data pipeline.

No shapely / PIL / matplotlib on this machine, and a training data
pipeline should not depend on heavy optional libraries anyway. Everything
here is pure numpy: shoelace area/centroid and an even-odd ray-casting
rasterizer. Coordinates are SVG pixels (x right, y down) throughout.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = Sequence[Point]


def _as_xy(poly: Polygon) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(poly, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
        raise ValueError(f"polygon needs >=3 (x,y) points, got shape {a.shape}")
    # drop a duplicated closing vertex if present
    if np.allclose(a[0], a[-1]):
        a = a[:-1]
    return a[:, 0], a[:, 1]


def signed_area(poly: Polygon) -> float:
    """Shoelace signed area (px^2). Positive = counter-clockwise in a
    y-up frame; sign is irrelevant to callers that take abs()."""
    x, y = _as_xy(poly)
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def area(poly: Polygon) -> float:
    return abs(signed_area(poly))


def centroid(poly: Polygon) -> Point:
    """Area-weighted polygon centroid (px). Falls back to the vertex mean
    for degenerate (zero-area) polygons."""
    x, y = _as_xy(poly)
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * float(cross.sum())
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    cx = float(((x + x1) * cross).sum() / (6.0 * a))
    cy = float(((y + y1) * cross).sum() / (6.0 * a))
    return cx, cy


def bbox_of(polys: List[Polygon]) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) over several polygons."""
    xs, ys = [], []
    for p in polys:
        a = np.asarray(p, dtype=np.float64)
        xs.append(a[:, 0]); ys.append(a[:, 1])
    X = np.concatenate(xs); Y = np.concatenate(ys)
    return float(X.min()), float(Y.min()), float(X.max()), float(Y.max())


def rasterize_union(polys: List[Polygon], grid: int,
                    origin: Tuple[float, float],
                    extent: Tuple[float, float]) -> np.ndarray:
    """Rasterize the UNION of polygons onto a `grid`×`grid` uint8 mask.

    Cell (r, c) samples the point at the cell CENTER in world coords:
        x = origin_x + (c + 0.5)/grid * extent_w
        y = origin_y + (r + 0.5)/grid * extent_h
    Even-odd (ray-cast) test per polygon, OR-combined. Row 0 = top (min y),
    matching SVG's y-down convention → north is row 0.
    """
    ox, oy = origin
    ew, eh = extent
    cs = (np.arange(grid) + 0.5) / grid
    px = ox + cs * ew                     # (grid,) x per column
    py = oy + cs * eh                     # (grid,) y per row
    gx, gy = np.meshgrid(px, py)          # (grid, grid)
    qx = gx.ravel(); qy = gy.ravel()
    inside = np.zeros(qx.shape, dtype=bool)
    for poly in polys:
        inside |= _points_in_polygon(qx, qy, poly)
    return inside.reshape(grid, grid).astype(np.uint8)


def _points_in_polygon(qx: np.ndarray, qy: np.ndarray,
                       poly: Polygon) -> np.ndarray:
    """Vectorized even-odd ray casting for many query points vs one polygon."""
    x, y = _as_xy(poly)
    n = len(x)
    inside = np.zeros(qx.shape, dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = x[i], y[i], x[j], y[j]
        # does a horizontal ray from the point cross edge (i, j)?
        cond = ((yi > qy) != (yj > qy))
        # x-coordinate of the edge at the query's y
        denom = (yj - yi)
        denom = np.where(denom == 0.0, 1e-12, denom)
        xcross = xi + (qy - yi) / denom * (xj - xi)
        inside ^= cond & (qx < xcross)
        j = i
    return inside
