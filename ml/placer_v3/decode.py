"""
decode.py — constrained autoregressive sampling.

Rooms are placed one at a time in generation order. Before each cell is
sampled, illegal cells have their logit set to -inf, so an invalid layout is
UNSAMPLABLE rather than merely discouraged. v3 changes what "the model knows"
at that moment: the same occupancy the mask is computed from is also fed to
the network as state, so it can anticipate the constraint instead of being
corrected by it.

CONFIDENCE. Training spreads the cell target over a sigma=1 Gaussian, so a
PERFECTLY fitted model puts only ~0.159 on its single best cell. A tau above
that is unreachable by construction — v2 shipped with exactly that bug and
fell back on every proposal until it was found. Confidence here is therefore
the 3x3 neighbourhood mass (perfect ceiling ~0.779), which is the quantity
the loss actually optimises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from ml.placer_v3.config import STATE_GRID
from ml.placer_v3.state import PlacementState

NEG_INF = -1e9

# Mass a perfectly fitted (sigma=1) model puts on its best cell / best 3x3.
PERFECT_TOP1 = 0.15915
PERFECT_NEIGHBORHOOD = 0.77948

# Fall back to the statistical proposer below this 3x3 mass. 0.35 is ~45% of
# what a perfect fit achieves — compare it ONLY against `confidence`, never
# against `top1_confidence`, whose ceiling is 0.159.
DEFAULT_TAU = 0.35


@dataclass
class Placement:
    room_index: int
    row: int
    col: int
    size_class: int          # 1..40
    aspect_class: int
    orientation_class: int
    band_index: int


@dataclass
class DecodeResult:
    placements: List[Placement]
    confidence: float
    top1_confidence: float = 0.0


def neighborhood_confidence(probs: np.ndarray, cell: int,
                            grid: int = STATE_GRID) -> float:
    g = probs.reshape(grid, grid)
    r, c = divmod(int(cell), grid)
    return float(g[max(0, r - 1):r + 2, max(0, c - 1):c + 2].sum())


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


@torch.no_grad()
def sample(net, arrays: Dict, *, temperature: float = 1.0,
           rng: Optional[np.random.Generator] = None,
           greedy: bool = False, device: str = "cpu") -> DecodeResult:
    """One constrained proposal from a torch PlacerNetV3.

    The prefix is re-run each step (O(n^2) forwards). At n <= 24 rooms that
    is microseconds and it keeps the sampler using the exact same modules as
    training — a KV cache here would be a second code path to keep in sync,
    which is what went wrong in v2.
    """
    rng = rng or np.random.default_rng(0)
    net.eval()
    n = int(arrays["n_rooms"])
    if n == 0:
        return DecodeResult([], 0.0, 0.0)

    zone_ids = np.asarray(arrays["zone_ids"])
    state = PlacementState(arrays["boundary"][0], arrays["entrance_side"])

    def t(key, dtype=torch.long):
        return torch.as_tensor(np.asarray(arrays[key]), dtype=dtype,
                               device=device)

    tensors = {
        "type_ids": t("type_ids"), "zone_ids": t("zone_ids"),
        "floor_ids": t("floor_ids"), "vastu_dir_ids": t("vastu_dir_ids"),
        "vastu_strength_ids": t("vastu_strength_ids"),
        "edge_index": t("edge_index"),
        "boundary": t("boundary", torch.float32),
        "global": t("global", torch.float32),
    }
    memory = net.encode(tensors)

    placements: List[Placement] = []
    cells: List[int] = []
    sizes: List[int] = []            # 0-based head indices
    state_rows: List[np.ndarray] = []
    top1: List[float] = []
    nbhd: List[float] = []

    cell_count = net.cfg.cell_count
    size_count = net.cfg.size_count

    for i in range(n):
        state_rows.append(state.snapshot())
        stack = torch.as_tensor(np.stack(state_rows), dtype=torch.float32,
                                device=device)
        prev_cell = torch.as_tensor(
            [cell_count if j == 0 else cells[j - 1] for j in range(i + 1)],
            dtype=torch.long, device=device)
        prev_size = torch.as_tensor(
            [size_count if j == 0 else sizes[j - 1] for j in range(i + 1)],
            dtype=torch.long, device=device)

        out = net.decode_prefix(memory, tensors, prev_cell, prev_size, stack)
        logits = out["cell"][i].float().cpu().numpy().astype(np.float64)

        legal = state.legal_cells()
        logits = np.where(legal, logits, NEG_INF)

        if greedy:
            cell = int(np.argmax(logits))
        else:
            cell = int(rng.choice(len(logits),
                                  p=_softmax(logits / max(temperature, 1e-3))))

        # confidence is read off the UNTEMPERED posterior, so a hot variant
        # is never scored as a less certain model
        posterior = _softmax(logits)
        top1.append(float(posterior.max()))
        nbhd.append(neighborhood_confidence(posterior, int(np.argmax(logits))))

        size = int(out["size"][i].argmax()) + 1
        row, col = divmod(cell, STATE_GRID)
        placements.append(Placement(
            room_index=i, row=row, col=col, size_class=size,
            aspect_class=int(out["aspect"][i].argmax()),
            orientation_class=int(out["orientation"][i].argmax()),
            band_index=int(out["band"][i].argmax())))

        cells.append(cell)
        sizes.append(size - 1)
        state.place(row, col, size, int(zone_ids[i]))

    return DecodeResult(
        placements=placements,
        confidence=float(np.mean(nbhd)),
        top1_confidence=float(np.mean(top1)))


def sample_k(net, arrays: Dict, k: int, *, temperature: float = 1.0,
             seed: int = 0, device: str = "cpu") -> List[DecodeResult]:
    """K proposals; variant 0 is greedy (the model's honest best guess)."""
    out = [sample(net, arrays, greedy=True, device=device,
                  rng=np.random.default_rng(seed))]
    for v in range(1, k):
        out.append(sample(net, arrays, temperature=temperature, device=device,
                          rng=np.random.default_rng(seed * 1000 + v)))
    return out
