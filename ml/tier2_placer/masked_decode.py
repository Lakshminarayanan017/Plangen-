"""
masked_decode.py — constrained autoregressive sampling (the v1 fix).

Rooms are placed one at a time in generation order. Before sampling each
room's seed cell, illegal cells have their logit set to −∞, so a
structurally invalid layout is UNSAMPLABLE (v1 discouraged bad output; this
forbids it). Active legality masks:

  (a) outside the boundary footprint            — always
  (b) within the claimed radius of a placed room — always (radius from size)

Hooks (no-op until the request carries the constraint):
  (c) zone-forbidden cells, (d) hard must-adjacency.

Sampling with temperature over the legal remainder yields diverse proposals.
Below the confidence threshold τ the caller falls back to the statistical
proposer.

CONFIDENCE (calibrated against how the model is trained — see the note in
`neighborhood_confidence`): the mean post-mask probability mass inside the
3x3 cell neighborhood of the argmax, NOT the raw top-1 probability. Training
uses Gaussian label smoothing at sigma=1 cell, so a PERFECTLY fitted model
puts only ~0.159 on its single best cell — a top-1 threshold above that is
unreachable by construction and would mute the model forever. The same
perfect fit puts ~0.779 in the 3x3 block, which is what this measures.

Works on both the torch model and the NumPy inference model — both expose
`encode(arrays)` and `decode_step(memory, arrays, prev_cell, prev_size)`
returning (cell_logits, size_logits) as numpy arrays for the current prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ml.tier2_placer.config import SEED_GRID

NEG_INF = -1e9


@dataclass
class Placement:
    room_index: int          # index into the generation-ordered room list
    row: int
    col: int
    size_class: int          # 1..40


@dataclass
class DecodeResult:
    placements: List[Placement]
    confidence: float        # mean 3x3-neighborhood mass (the gate metric)
    top1_confidence: float = 0.0   # mean top-1 prob (diagnostic only)


# Probability mass a perfectly fitted model (sigma=1 Gaussian labels) puts on
# its best cell / its best 3x3 block. Used to express tau as a fraction of
# what is actually achievable, and asserted by test_confidence_ceiling.
PERFECT_TOP1 = 0.15915
PERFECT_NEIGHBORHOOD = 0.77948


def neighborhood_confidence(probs: np.ndarray, cell: int,
                            grid: int = SEED_GRID) -> float:
    """Probability mass in the 3x3 block centered on `cell`.

    Why not top-1: the training loss spreads the target over a sigma=1
    Gaussian (train.py::gaussian_soft_targets), so the best attainable top-1
    probability is ~0.159 regardless of how well the model learns. Scoring
    confidence on a quantity the model is trained NOT to maximize made the
    default tau (0.35) unreachable — every proposal fell back, silently. The
    3x3 mass is the quantity the loss actually optimizes.
    """
    g = probs.reshape(grid, grid)
    r, c = divmod(int(cell), grid)
    return float(g[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum())


def boundary_legal_cells(boundary_mask64: np.ndarray) -> np.ndarray:
    """(1024,) bool — cell c=(r,c) on the 32-grid is legal iff its 2×2 block
    in the 64-grid footprint has any fill."""
    g = SEED_GRID
    scale = boundary_mask64.shape[0] // g               # 2
    legal = np.zeros(g * g, dtype=bool)
    for r in range(g):
        for c in range(g):
            blk = boundary_mask64[r*scale:(r+1)*scale, c*scale:(c+1)*scale]
            legal[r * g + c] = bool(blk.any())
    if not legal.any():                                 # degenerate footprint
        legal[:] = True
    return legal


def _claim_radius(size_class: int) -> int:
    """Exclusion radius (cells) a placed room casts, growing with its size."""
    return int(np.clip(round((size_class ** 0.5) / 2.0), 0, 3))


def _apply_claim(legal: np.ndarray, row: int, col: int, radius: int) -> None:
    g = SEED_GRID
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            if 0 <= r < g and 0 <= c < g:
                legal[r * g + c] = False


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample(model, arrays: dict, *, temperature: float = 1.0,
           rng: Optional[np.random.Generator] = None,
           greedy: bool = False) -> DecodeResult:
    """One constrained proposal. `model` exposes encode()/decode_step()
    returning numpy logits. `arrays` are the numpy input arrays
    (tier2_placer.features)."""
    rng = rng or np.random.default_rng(0)
    n = int(arrays["n_rooms"])
    base_legal = boundary_legal_cells(arrays["boundary"][0])
    memory = model.encode(arrays)

    placements: List[Placement] = []
    sampled_cells: List[int] = []
    sampled_sizes: List[int] = []
    claimed = base_legal.copy()
    top1_probs: List[float] = []
    nbhd_probs: List[float] = []

    for i in range(n):
        prev_cell = np.array(
            [model.cell_count if j == 0 else sampled_cells[j - 1]
             for j in range(i + 1)], dtype=np.int64)
        prev_size = np.array(
            [model.size_count if j == 0 else sampled_sizes[j - 1]
             for j in range(i + 1)], dtype=np.int64)
        cell_logits, size_logits = model.decode_step(
            memory, arrays, prev_cell, prev_size)
        cl = cell_logits[i].astype(np.float64).copy()

        legal = claimed.copy()
        if not legal.any():
            legal = base_legal.copy()          # never dead-end
        cl[~legal] = NEG_INF

        probs = _softmax(cl / max(temperature, 1e-3))
        if greedy:
            cell = int(np.argmax(cl))
        else:
            cell = int(rng.choice(len(cl), p=probs))
        # confidence is always read off the UNTEMPERED posterior, so a
        # high-temperature variant is not scored as a less certain model
        posterior = _softmax(cl)
        top1_probs.append(float(posterior.max()))
        nbhd_probs.append(neighborhood_confidence(posterior,
                                                  int(np.argmax(cl))))

        # size head (unconstrained)
        size = int(np.argmax(size_logits[i])) + 1       # 0..39 -> 1..40

        row, col = divmod(cell, SEED_GRID)
        placements.append(Placement(i, row, col, size))
        sampled_cells.append(cell)
        sampled_sizes.append(size - 1)
        _apply_claim(claimed, row, col, _claim_radius(size))

    return DecodeResult(
        placements=placements,
        confidence=float(np.mean(nbhd_probs)) if nbhd_probs else 0.0,
        top1_confidence=float(np.mean(top1_probs)) if top1_probs else 0.0)


def sample_k(model, arrays: dict, k: int, *, temperature: float = 1.0,
             seed: int = 0) -> List[DecodeResult]:
    """K diverse proposals (variant 0 is greedy — the model's best guess)."""
    out = [sample(model, arrays, greedy=True,
                  rng=np.random.default_rng(seed))]
    for v in range(1, k):
        out.append(sample(model, arrays, temperature=temperature,
                          rng=np.random.default_rng(seed * 1000 + v)))
    return out
