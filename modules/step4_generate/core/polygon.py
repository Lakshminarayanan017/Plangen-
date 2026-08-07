"""
polygon.py — irregular plots on the same integer lattice.

Real plots are not rectangles. They are surveyed polygons with a splayed
corner, a road chamfer, or a boundary that runs at 7 degrees to everything
else. The lattice handles this without any change to the ownership model:
cells outside the polygon are simply OUTSIDE, exactly like the cells beyond
a rectangular plot's wall.

What DOES need care is where the building goes. The carver splits
rectangles, so the buildable core is the largest rectangle that fits inside
the setback-adjusted polygon; everything inside the plot but outside that
core is open space (which is what a setback strip IS). This module supplies
the two primitives that needs — rasterization and the largest inscribed
rectangle — and nothing else. No floating-point geometry survives past this
boundary: the outputs are cell indices.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from modules.step4_generate.core import units

Point = Tuple[float, float]          # (x, y) in FEET
Rect = Tuple[int, int, int, int]     # (x0, y0, x1, y1) in cells, half-open


class PolygonError(ValueError):
    """The polygon cannot be used as a plot boundary."""


def signed_area_sqft(points: Sequence[Point]) -> float:
    """Shoelace area; positive when the ring winds counter-clockwise."""
    n = len(points)
    total = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def normalize(points: Sequence[Point]) -> List[Point]:
    """Validate a plot ring and return it closed-free and CCW-consistent."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise PolygonError(f"a plot needs at least 3 corners, got {len(pts)}")
    area = signed_area_sqft(pts)
    if abs(area) < 1.0:
        raise PolygonError(f"plot area {abs(area):.2f} sqft is degenerate")
    return pts if area > 0 else list(reversed(pts))


def bbox_cells(points: Sequence[Point]) -> Tuple[int, int, float, float]:
    """(width_cells, height_cells, x_min_ft, y_min_ft) of the ring's bbox."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys)
    w = int(np.ceil((max(xs) - x0) * units.CELLS_PER_FOOT))
    h = int(np.ceil((max(ys) - y0) * units.CELLS_PER_FOOT))
    return w, h, x0, y0


def rasterize(points: Sequence[Point], width_cells: int, height_cells: int,
              origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """(h, w) bool mask of cells whose CENTRE lies inside the polygon.

    Centre sampling, not corner sampling: a cell is in the plot when its
    middle is, which makes the rasterized area converge on the true area
    from both sides instead of systematically over-reporting it."""
    ox, oy = origin
    cols = (np.arange(width_cells) + 0.5) / units.CELLS_PER_FOOT + ox
    rows = (np.arange(height_cells) + 0.5) / units.CELLS_PER_FOOT + oy
    gx, gy = np.meshgrid(cols, rows)

    inside = np.zeros((height_cells, width_cells), dtype=bool)
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        if y0 == y1:
            continue
        # ray casting to +x: does the horizontal line through this cell
        # centre cross this edge, and is the crossing to the right?
        straddles = ((y0 > gy) != (y1 > gy))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (gy - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddles & (gx < x_cross)
    return inside


def erode(mask: np.ndarray, cells: int) -> np.ndarray:
    """Shrink a mask by `cells` in the 4-connected sense — the wall ring of
    an irregular plot, or a uniform setback."""
    out = mask.copy()
    for _ in range(max(0, cells)):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        shrunk[0, :] = False
        shrunk[-1, :] = False
        shrunk[:, 0] = False
        shrunk[:, -1] = False
        out = shrunk
    return out


def largest_inscribed_rectangle(mask: np.ndarray) -> Optional[Rect]:
    """The biggest axis-aligned all-True rectangle in `mask`.

    Standard histogram-and-stack maximal rectangle, O(rows * cols): for each
    row, the height of the True run ending there per column, then the
    largest rectangle under that histogram. Exact — this is the buildable
    core, and guessing it would put walls outside the plot.
    """
    if mask.size == 0 or not mask.any():
        return None
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int64)
    best = (0, None)

    for y in range(h):
        heights = np.where(mask[y], heights + 1, 0)
        stack: List[int] = []                 # columns of increasing height
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            while stack and int(heights[stack[-1]]) > cur:
                idx = stack.pop()
                height = int(heights[idx])
                left = stack[-1] + 1 if stack else 0
                area = height * (x - left)
                if area > best[0]:
                    best = (area, (left, y - height + 1, x, y + 1))
            if x < w:
                stack.append(x)
    return best[1]


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """4-connected True regions, largest first. Used to give every leftover
    piece of open space its own face — a room must be connected, so two
    disjoint setback strips cannot be one face."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    out: List[np.ndarray] = []
    current = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            stack = [(sy, sx)]
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                            and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
            out.append(labels == current)
    out.sort(key=lambda m: -int(m.sum()))
    return out


def rect_area_sqft(rect: Rect) -> float:
    x0, y0, x1, y1 = rect
    return units.area_sqft((x1 - x0) * (y1 - y0))
