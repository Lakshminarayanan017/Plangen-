"""
mandala.py — `data/vastuRules1.json` as queryable geometry.

That file describes a Paramasayika Mandala (9x9 = 81 padas) and, until now,
nothing read it except one helper nobody called. It carries four things a
floor-plan engine can genuinely check:

  32 outer gates      where the main door may sit on each side, scored, with
                      10 padas marked hard_block
  13 energy fields    the Brahmasthan among them, with the usages that must
                      not land inside it
  6 marma diagonals   lines no wall junction or door centre may fall on
  global modifiers    plot aspect cap, staircase turn, fire-water separation

DATA DEFECTS, resolved rather than papered over. Two marma endpoints name a
pada that disagrees with the gate of that id:

    W3_Mukhya   W3 is 'Sugriva'; 'Mukhya' is N3
    W7_Shosha   W7 is 'Shok'

Both IDs resolve, and the id is the unambiguous half of the reference, so
endpoints are resolved BY ID and the name mismatch is reported in
`load_report()` rather than silently accepted or silently guessed.

Degrades honestly: a missing or malformed file yields `available == False`
and every rule that depends on it stands down, exactly like the rest of the
engine's knowledge sources.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from modules.step4_generate.engine.vastu.compass import (
    BRAHMA_HI, BRAHMA_LO, MANDALA, CompassFrame,
)

log = logging.getLogger("PlanGen.Vastu")

_DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "data",
    "vastuRules1.json")


@dataclass(frozen=True)
class Gate:
    """One of the 32 perimeter padas a main door can occupy."""
    id: str
    name: str
    pada: Tuple[int, int]        # (x east, y north) on the 9x9
    status: str                  # ideal | permitted | soft_block | hard_block
    score: float                 # 0.0 .. 1.0

    @property
    def side(self) -> str:
        return self.id[0]

    @property
    def is_blocked(self) -> bool:
        return self.status == "hard_block"


@dataclass(frozen=True)
class EnergyField:
    """One of the 13 inner fields, as a pada rectangle."""
    name: str
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]
    ideal_usage: Tuple[str, ...]
    hard_blocks: Tuple[str, ...]

    def contains(self, px: int, py: int) -> bool:
        return (self.x_range[0] <= px <= self.x_range[1]
                and self.y_range[0] <= py <= self.y_range[1])


@dataclass(frozen=True)
class MarmaLine:
    """A diagonal between two perimeter padas that must stay clear."""
    from_id: str
    to_id: str
    from_pada: Tuple[int, int]
    to_pada: Tuple[int, int]
    rule: str


@dataclass
class Mandala:
    available: bool = False
    gates: Dict[str, Gate] = field(default_factory=dict)
    fields: Dict[str, EnergyField] = field(default_factory=dict)
    marma: List[MarmaLine] = field(default_factory=list)
    structural: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source: str = ""

    # ── gates ───────────────────────────────────────────────────────────
    def gate_at(self, px: int, py: int) -> Optional[Gate]:
        for gate in self.gates.values():
            if gate.pada == (px, py):
                return gate
        return None

    def gates_on_side(self, compass_side: str) -> List[Gate]:
        return sorted((g for g in self.gates.values()
                       if g.side == compass_side),
                      key=lambda g: g.id)

    def best_gates(self, compass_side: str) -> List[Gate]:
        """Ideal-scoring gates on a side, for "you could move the door here"
        advice rather than a bare penalty."""
        return [g for g in self.gates_on_side(compass_side)
                if g.status == "ideal"]

    # ── the Brahmasthan ─────────────────────────────────────────────────
    @property
    def brahma(self) -> Optional[EnergyField]:
        return self.fields.get("Brahma")

    def brahma_blocked_usages(self) -> Tuple[str, ...]:
        b = self.brahma
        return b.hard_blocks if b else ()

    # ── marma, in grid space ────────────────────────────────────────────
    def marma_segments(self, frame: CompassFrame, h: int, w: int
                       ) -> List[Tuple[MarmaLine, Tuple[float, float],
                                       Tuple[float, float]]]:
        """Each diagonal as a grid-space segment ((x0,y0), (x1,y1))."""
        out = []
        for line in self.marma:
            a = _pada_center(frame, line.from_pada, h, w)
            b = _pada_center(frame, line.to_pada, h, w)
            out.append((line, a, b))
        return out

    def load_report(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "source": self.source,
            "gates": len(self.gates),
            "hard_block_gates": sum(1 for g in self.gates.values()
                                    if g.is_blocked),
            "ideal_gates": sum(1 for g in self.gates.values()
                               if g.status == "ideal"),
            "energy_fields": len(self.fields),
            "marma_lines": len(self.marma),
            "brahmasthan": bool(self.brahma),
            "warnings": list(self.warnings),
        }


def _pada_center(frame: CompassFrame, pada: Tuple[int, int], h: int, w: int
                 ) -> Tuple[float, float]:
    x0, y0, x1, y1 = frame.pada_rect(pada[0], pada[1], h, w)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def point_segment_distance(px: float, py: float,
                           a: Tuple[float, float],
                           b: Tuple[float, float]) -> float:
    """Distance from a point to a segment, in cells."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _load_raw(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("vastu mandala unavailable (%s); "
                    "pada-level rules stand down", exc)
        return None


def load(path: Optional[str] = None) -> Mandala:
    """Read the mandala. Never raises — an absent file is a degraded mode."""
    path = os.path.normpath(path or _DEFAULT_PATH)
    raw = _load_raw(path)
    if raw is None:
        return Mandala(available=False, source=path,
                       warnings=[f"{path} could not be read"])

    warnings: List[str] = []

    gates: Dict[str, Gate] = {}
    for entry in raw.get("outer_perimeter_gates_32", []):
        try:
            coords = entry["coords"]
            gates[entry["id"]] = Gate(
                id=entry["id"], name=entry.get("name", entry["id"]),
                pada=(int(coords[0]), int(coords[1])),
                status=entry.get("status", "permitted"),
                score=float(entry.get("score", 0.5)))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            warnings.append(f"gate {entry!r} skipped ({exc})")
    if len(gates) != 32:
        warnings.append(f"expected 32 perimeter gates, loaded {len(gates)}")

    fields: Dict[str, EnergyField] = {}
    for name, spec in (raw.get("inner_energy_fields_13") or {}).items():
        try:
            fields[name] = EnergyField(
                name=name,
                x_range=(int(spec["x_range"][0]), int(spec["x_range"][1])),
                y_range=(int(spec["y_range"][0]), int(spec["y_range"][1])),
                ideal_usage=tuple(spec.get("ideal_usage", ())),
                hard_blocks=tuple(spec.get("hard_blocks", ())))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            warnings.append(f"energy field {name!r} skipped ({exc})")

    brahma = fields.get("Brahma")
    if brahma and (brahma.x_range != (BRAHMA_LO, BRAHMA_HI)
                   or brahma.y_range != (BRAHMA_LO, BRAHMA_HI)):
        warnings.append(
            f"Brahmasthan is x{brahma.x_range} y{brahma.y_range}, not the "
            f"central 3x3 the compass frame assumes "
            f"([{BRAHMA_LO},{BRAHMA_HI}] on both axes)")

    marma: List[MarmaLine] = []
    for entry in raw.get("marma_vulnerable_diagonals", []):
        try:
            from_ref, to_ref = entry["from_pada"], entry["to_pada"]
            from_id, to_id = from_ref.split("_")[0], to_ref.split("_")[0]
            a, b = gates.get(from_id), gates.get(to_id)
            if a is None or b is None:
                warnings.append(f"marma {from_ref}->{to_ref}: unknown gate id")
                continue
            # resolve by ID; report the name disagreement rather than guess
            for ref, gate in ((from_ref, a), (to_ref, b)):
                named = ref.split("_", 1)[1] if "_" in ref else ""
                if named and not gate.name.lower().startswith(
                        named.lower()[:4]):
                    warnings.append(
                        f"marma endpoint {ref!r} names a pada that disagrees "
                        f"with gate {gate.id} ({gate.name!r}); resolved by id")
            marma.append(MarmaLine(from_id=from_id, to_id=to_id,
                                   from_pada=a.pada, to_pada=b.pada,
                                   rule=entry.get("rule", "")))
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"marma {entry!r} skipped ({exc})")

    mandala = Mandala(
        available=bool(gates),
        gates=gates, fields=fields, marma=marma,
        structural=dict(raw.get("global_structural_modifiers") or {}),
        warnings=warnings, source=path)
    if warnings:
        for note in warnings:
            log.info("vastu mandala: %s", note)
    log.info("vastu mandala loaded: %d gates, %d fields, %d marma lines",
             len(gates), len(fields), len(marma))
    return mandala


_CACHE: Dict[str, Mandala] = {}


def shared(path: Optional[str] = None) -> Mandala:
    """Process-wide cached load — the reviewer builds a context per candidate
    and re-reading a 30 KB JSON per rule per candidate is pure waste."""
    key = os.path.normpath(path or _DEFAULT_PATH)
    if key not in _CACHE:
        _CACHE[key] = load(key)
    return _CACHE[key]
