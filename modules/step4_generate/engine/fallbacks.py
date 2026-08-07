"""
fallbacks.py — statistical-prior twins of the ML tiers.

The engine must run end-to-end BEFORE any model is trained, and must
degrade gracefully when a trained tier is unavailable or low-confidence.
This is the all-algorithmic Tier-2: an affinity-aware zone-prior proposer.

Beyond raw zone depths it encodes real architectural priors:
  • hub rooms (dining/hallway) seed at the plot's across-center — the
    free-space connector position
  • wet rooms seed BESIDE their affinity target (bathroom next to a
    bedroom, utility/store next to the kitchen) so the carver lands them
    adjacent and the connector can door them correctly
  • anchors are placed center-out by size (big rooms get prime positions)

The trained Tier-2 placer replaces this class behind the same contract;
its merge gate is beating this proposer on the harness (engine doc §7).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from modules.step4_generate.engine.contracts import (
    SEED_GRID, EngineRequest, LayoutProposal, Placement, RoomSpec,
)
from modules.step4_generate.engine.priors import ZonePriors

_ZONE_ORDER = {"public": 0, "service": 1, "private": 2}

# Depth of each zone measured from the entrance (0 = far wall, 1 = entrance).
_ZONE_DEPTH = {"public": 0.85, "service": 0.50, "private": 0.18}

_HUB_TYPES = {"dining_room", "hallway", "foyer", "passage"}
_BED_TYPES = {"master_bedroom", "bedroom"}
_WET_TYPES = {"bathroom", "toilet"}
_KITCHEN_HELPERS = {"utility", "store", "storage", "laundry"}


def _to_grid(depth: float, across: float, side: str) -> Tuple[int, int]:
    """Map (depth-from-entrance, across) ∈ [0,1]² to a (row, col) seed cell
    for the given entrance side."""
    if side == "S":
        y, x = depth, across
    elif side == "N":
        y, x = 1.0 - depth, across
    elif side == "E":
        y, x = across, depth
    elif side == "W":
        y, x = across, 1.0 - depth
    else:
        raise ValueError(f"bad entrance side {side!r}")
    n = SEED_GRID - 1
    return (max(0, min(n, round(y * n))), max(0, min(n, round(x * n))))


def _center_out_slots(n: int) -> List[float]:
    """n across-positions ordered center-first: 0.5, then alternating
    outward — prime positions go to the rooms placed first."""
    if n <= 1:
        # n == 0 (a zone with only attached rooms, e.g. a ground-floor common
        # bath and no bedroom) must not crash min(range(0)); return no slots.
        return [0.5] if n == 1 else []
    slots = sorted((i + 1) / (n + 1) for i in range(n))
    center = min(range(n), key=lambda i: abs(slots[i] - 0.5))
    order = [center]
    step = 1
    while len(order) < n:
        for cand in (center - step, center + step):
            if 0 <= cand < n and cand not in order:
                order.append(cand)
        step += 1
    return [slots[i] for i in order]


class PriorProposer:
    """Affinity-aware proposer. Depth priors come from the 5K-plan
    extraction (engine/priors.py) when available — measured reality —
    blended toward the room's zone constant so user zoning always wins
    the argument; hand-tuned constants are only the fallback. Variant
    index + request seed drive an rng so each of K candidates is
    distinct and reproducible."""

    def __init__(self, priors: Optional[ZonePriors] = None):
        self.priors = priors if priors is not None else ZonePriors()

    def _depth_base(self, rtype: str, zone: str) -> float:
        # a hub room living in the private zone (the bedroom passage) must
        # sit at the zone BOUNDARY where it can touch the service hub —
        # buried deep among bedrooms it fragments the free space (FSP-001)
        if rtype in _HUB_TYPES and zone == "private":
            return 0.38
        zone_depth = _ZONE_DEPTH[zone]
        learned = self.priors.depth_for(rtype) if self.priors.loaded else None
        if learned is None:
            return zone_depth
        return 0.5 * zone_depth + 0.5 * learned

    def propose(self, request: EngineRequest, variant: int = 0
                ) -> LayoutProposal:
        rng = random.Random(request.seed * 1_000_003 + variant)
        jd = rng.uniform  # jitter draw

        by_zone: Dict[str, List[RoomSpec]] = {}
        for spec in request.rooms:
            by_zone.setdefault(spec.zone, []).append(spec)

        placements: List[Placement] = []
        for zone in sorted(by_zone, key=_ZONE_ORDER.__getitem__):
            zone_specs = by_zone[zone]
            anchors = [s for s in zone_specs
                       if s.rtype not in _WET_TYPES | _KITCHEN_HELPERS]
            attached = [s for s in zone_specs
                        if s.rtype in _WET_TYPES | _KITCHEN_HELPERS]

            # hub first, then remaining anchors by size (center-out slots)
            anchors.sort(key=lambda s: (s.rtype not in _HUB_TYPES,
                                        -s.target_sqft))
            if variant > 0 and len(anchors) > 1:
                # variant 0 is the canonical prior layout; other variants
                # permute anchors so K candidates explore genuinely
                # different band compositions
                tail = anchors[1:]
                rng.shuffle(tail)
                anchors = anchors[:1] + tail
                if variant % 3 == 2:
                    anchors = list(reversed(anchors))

            slots = _center_out_slots(len(anchors))
            anchor_pos: Dict[str, Tuple[float, float]] = {}
            for spec, across in zip(anchors, slots):
                depth = _clamp(self._depth_base(spec.rtype, zone)
                               + jd(-0.08, 0.08))
                across = _clamp(across + jd(-0.05, 0.05))
                anchor_pos[spec.name] = (depth, across)
                placements.append(Placement(
                    room=spec.name,
                    seed_cell=_to_grid(depth, across, request.entrance_side),
                    size_class=spec.size_class,
                ))

            # attached rooms seed beside their affinity target
            beds = sorted((s for s in anchors if s.rtype in _BED_TYPES),
                          key=lambda s: -s.target_sqft)
            kitchens = [s for s in request.rooms if s.rtype == "kitchen"]
            bed_cycle = 0
            for spec in attached:
                target = self._affinity_target(
                    spec, beds, kitchens, bed_cycle, anchors)
                if spec.rtype in _WET_TYPES and beds:
                    bed_cycle += 1
                depth, across = self._beside(
                    target, anchor_pos, zone, rng)
                placements.append(Placement(
                    room=spec.name,
                    seed_cell=_to_grid(depth, across, request.entrance_side),
                    size_class=spec.size_class,
                ))

        return LayoutProposal(placements=placements, source="fallback-prior")

    @staticmethod
    def _affinity_target(spec: RoomSpec, beds: List[RoomSpec],
                         kitchens: List[RoomSpec], bed_cycle: int,
                         anchors: List[RoomSpec]) -> Optional[RoomSpec]:
        if spec.rtype in _WET_TYPES and beds:
            return beds[bed_cycle % len(beds)]
        if spec.rtype in _KITCHEN_HELPERS and kitchens:
            return kitchens[0]
        return anchors[0] if anchors else None

    @staticmethod
    def _beside(target: Optional[RoomSpec],
                anchor_pos: Dict[str, Tuple[float, float]],
                zone: str, rng: random.Random) -> Tuple[float, float]:
        if target is not None and target.name in anchor_pos:
            t_depth, t_across = anchor_pos[target.name]
            side = 1 if t_across <= 0.5 else -1
            return (_clamp(t_depth + rng.uniform(-0.04, 0.04)),
                    _clamp(t_across + side * 0.16))
        return (_clamp(_ZONE_DEPTH[zone] + rng.uniform(-0.08, 0.08)),
                _clamp(rng.uniform(0.15, 0.85)))


def _clamp(v: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, v))
