"""
state.py — R1: the spatial state the decoder is allowed to look at.

The v2 post-mortem in one sentence: the decoder conditioned on embeddings of
the PREVIOUS room's (cell, size) and nothing else, so it packed a plot the
way you would pack a suitcase from a written list instead of by looking at
it. The legality mask in `masked_decode` then corrected the output at
inference — which is why the model's own confidence sat at 0.355 against a
tau of 0.35 and it fell back on 41.8% of proposals. The model was never
trained to anticipate the constraint that was being applied to it.

This module builds the thing it should have been looking at: a small
multi-channel raster, on the SAME 32x32 grid the cell head predicts over,
describing the world immediately BEFORE each room is placed.

Pure NumPy on purpose. The identical builder feeds torch training and the
NumPy production path, so train-time state and inference-time state cannot
drift apart — the class of bug that cost v2 two silent failures.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ml.placer_v3.config import (
    STATE_CH_CLAIMED, STATE_CH_ENTRANCE, STATE_CH_FOOTPRINT,
    STATE_CH_PRIVATE, STATE_CH_PUBLIC, STATE_CHANNELS, STATE_GRID,
    ZONE_TO_ID,
)

_ZONE_PUBLIC = ZONE_TO_ID["public"]
_ZONE_PRIVATE = ZONE_TO_ID["private"]

_SIDE_BAND = {
    "N": (slice(0, 3), slice(None)),
    "S": (slice(STATE_GRID - 3, STATE_GRID), slice(None)),
    "W": (slice(None), slice(0, 3)),
    "E": (slice(None), slice(STATE_GRID - 3, STATE_GRID)),
}


def claim_radius(size_class: int) -> int:
    """Exclusion radius in cells that a placed room casts on the seed grid.

    Byte-identical to `tier2_placer.masked_decode._claim_radius`. It is
    redefined here rather than imported so v3 does not depend on the v2
    package it is meant to replace; `tests/ml/test_placer_v3.py` asserts the
    two agree for every size class, so the parity is enforced, not assumed.
    """
    return int(np.clip(round((size_class ** 0.5) / 2.0), 0, 3))


def footprint_from_boundary(boundary64: np.ndarray) -> np.ndarray:
    """(64,64) footprint -> (32,32) float mask on the seed grid.

    A seed cell is buildable when ANY part of its 2x2 block in the 64-grid is
    inside the footprint — the same "any fill" rule
    `masked_decode.boundary_legal_cells` uses to decide legality, so the
    state the model sees and the mask applied to its logits agree by
    construction.
    """
    g = STATE_GRID
    scale = max(1, boundary64.shape[0] // g)
    blocks = boundary64[:g * scale, :g * scale].reshape(g, scale, g, scale)
    out = (blocks.max(axis=(1, 3)) > 0).astype(np.float32)
    if not out.any():                       # degenerate footprint
        out[:] = 1.0
    return out


def static_planes(boundary64: np.ndarray, entrance_side: str
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The two channels that do not change as rooms are placed."""
    footprint = footprint_from_boundary(boundary64)
    entrance = np.zeros((STATE_GRID, STATE_GRID), dtype=np.float32)
    band = _SIDE_BAND.get(entrance_side)
    if band is not None:
        entrance[band] = 1.0
    entrance *= footprint                   # only where the plot actually is
    return footprint, entrance


def _stamp(plane: np.ndarray, row: int, col: int, radius: int) -> None:
    r0, r1 = max(0, row - radius), min(STATE_GRID, row + radius + 1)
    c0, c1 = max(0, col - radius), min(STATE_GRID, col + radius + 1)
    plane[r0:r1, c0:c1] = 1.0


class PlacementState:
    """Incremental occupancy, mutated one placed room at a time.

    Used directly by the sampler (which genuinely does not know the future)
    and by `build_state_stack` (which does, and simply replays it). One
    implementation, so the two can never disagree.
    """

    def __init__(self, boundary64: np.ndarray, entrance_side: str):
        self.footprint, self.entrance = static_planes(boundary64,
                                                      entrance_side)
        self.claimed = np.zeros((STATE_GRID, STATE_GRID), dtype=np.float32)
        self.public = np.zeros((STATE_GRID, STATE_GRID), dtype=np.float32)
        self.private = np.zeros((STATE_GRID, STATE_GRID), dtype=np.float32)

    def snapshot(self) -> np.ndarray:
        """(C, 32, 32) float32 view of the world right now."""
        out = np.empty((STATE_CHANNELS, STATE_GRID, STATE_GRID),
                       dtype=np.float32)
        out[STATE_CH_FOOTPRINT] = self.footprint
        out[STATE_CH_ENTRANCE] = self.entrance
        out[STATE_CH_CLAIMED] = self.claimed
        out[STATE_CH_PUBLIC] = self.public
        out[STATE_CH_PRIVATE] = self.private
        return out

    def place(self, row: int, col: int, size_class: int, zone_id: int
              ) -> None:
        radius = claim_radius(size_class)
        _stamp(self.claimed, row, col, radius)
        if zone_id == _ZONE_PUBLIC:
            _stamp(self.public, row, col, radius)
        elif zone_id == _ZONE_PRIVATE:
            _stamp(self.private, row, col, radius)

    def legal_cells(self) -> np.ndarray:
        """(1024,) bool — inside the footprint and not already claimed.

        Never returns all-False: a program that claims every cell would
        otherwise dead-end the sampler, so an exhausted board falls back to
        the bare footprint. That is the same guard v2 used, kept because it
        is the difference between a cramped plan and no plan.
        """
        legal = (self.footprint > 0) & (self.claimed == 0)
        if not legal.any():
            legal = self.footprint > 0
        return legal.reshape(-1)


def build_state_stack(boundary64: np.ndarray, entrance_side: str,
                      cells: Sequence[int], sizes: Sequence[int],
                      zone_ids: Sequence[int]) -> np.ndarray:
    """(N, C, 32, 32): the state BEFORE each of N rooms is placed.

    Teacher forcing means the whole trajectory is known up front, so the
    entire stack is built in one replay instead of N forward passes. Step i
    sees rooms 0..i-1 and never room i — the causality the decoder relies on
    is enforced here, in the data, not left to a mask downstream.

    `sizes` are 1-based size CLASSES (1..40), matching the engine contract
    and `claim_radius`, not the 0-based head indices.
    """
    n = len(cells)
    state = PlacementState(boundary64, entrance_side)
    stack = np.empty((max(n, 1), STATE_CHANNELS, STATE_GRID, STATE_GRID),
                     dtype=np.float32)
    if n == 0:
        stack[0] = state.snapshot()
        return stack[:0]
    for i in range(n):
        stack[i] = state.snapshot()          # BEFORE placing room i
        row, col = divmod(int(cells[i]), STATE_GRID)
        state.place(row, col, int(sizes[i]), int(zone_ids[i]))
    return stack


def transform_state_planes(planes: np.ndarray, flip: bool, k: int
                           ) -> np.ndarray:
    """Apply the dataset's dihedral augmentation to a state stack.

    Must mirror `dataset._augment` exactly: flip columns first, then rotate
    90 degrees clockwise k times. An augmentation that moved the seed cells
    without moving the state would teach the model that the two are
    unrelated, which is the whole signal.
    """
    out = planes
    if flip:
        out = out[..., ::-1]
    for _ in range(k % 4):
        out = np.rot90(out, k=-1, axes=(-2, -1))
    return np.ascontiguousarray(out)
