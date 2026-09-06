"""
plan_edit.py — "make the kitchen bigger", turned into something the engine
can act on.

WHY EDITS RE-DERIVE RATHER THAN MUTATE. The obvious implementation is to
change the finished plan: grow the kitchen face, push the neighbours over.
It is also wrong. Room sizes, floor assignment, bathroom attachment, Vastu
directions and the adjacency graph are all DERIVED by step 3 from each
other — change the bedroom count by hand and the bath links, the floor split
and the program capacity all quietly refer to a house that no longer exists.

So an edit changes the BRIEF and the pipeline runs again. One source of
truth, and every derivation stays consistent by construction.

Two layers, because not every edit is a requirement:

  requirement edits   add/remove a room, floor count, entrance, Vastu on/off
                      -> BuildingRequirements, re-enriched from scratch
  override edits      resize, re-orient, force an adjacency
                      -> applied AFTER enrichment, because the enricher
                         would otherwise re-derive them from its own rules
                         and silently discard what the user just asked for

CONTINUITY. An edit re-runs with the SAME seed. A user who says "make the
kitchen bigger" wants their plan with a bigger kitchen, not a different
house — and the engine is seeded from the run id, so holding it fixed keeps
the band structure recognisable.

NOTHING IS SILENTLY IGNORED. An instruction this module cannot parse comes
back in `unparsed` and the caller is expected to say so. A plan that quietly
differs from what was asked is the failure mode this whole file exists to
avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modules.step4_generate.engine.vastu.compass import normalize_direction

# ── how much "bigger" is ────────────────────────────────────────────────────
# Deliberately modest. A user saying "bigger" once expects a noticeable
# change, not a different program; they can say it twice.
_SCALE = {
    "much bigger": 1.50, "far bigger": 1.50, "way bigger": 1.50,
    "much larger": 1.50, "far larger": 1.50,
    "bigger": 1.25, "larger": 1.25, "wider": 1.25, "spacious": 1.25,
    "slightly bigger": 1.12, "a bit bigger": 1.12,
    "much smaller": 0.65, "far smaller": 0.65,
    "smaller": 0.80, "tighter": 0.80, "compact": 0.80,
    "slightly smaller": 0.90, "a bit smaller": 0.90,
}
# longest first, so "much bigger" is not matched as "bigger"
_SCALE_WORDS = sorted(_SCALE, key=len, reverse=True)

_ADD = re.compile(r"\b(add|include|i want|i need|put in|give me)\b")
_REMOVE = re.compile(r"\b(remove|drop|delete|get rid of|no need for|"
                     r"don'?t want|without)\b")
_MOVE = re.compile(r"\b(move|shift|put|place|relocate)\b")
_FLOORS = re.compile(r"\b(g\s*\+\s*(\d)|(\d)\s*floors?|"
                     r"(single|one|two|three)\s*floors?|duplex)\b")
_ENTRANCE = re.compile(r"\b(entrance|entry|main door|door)\b")
_ADJACENT = re.compile(r"\b(next to|beside|adjacent to|near|close to|"
                       r"connected to|opens? (?:in)?to)\b")

ACTIONS = ("add", "remove", "resize", "move", "adjacency", "floors",
           "entrance", "vastu")


@dataclass
class EditIntent:
    """One instruction, in terms the pipeline can act on."""
    action: str                       # see ACTIONS
    target: Optional[str] = None      # canonical room type, when relevant
    value: Any = None                 # scale factor / sector / count / bool
    other: Optional[str] = None       # second room, for adjacency
    raw: str = ""

    @property
    def is_requirement(self) -> bool:
        """Requirement edits re-run the whole pipeline; override edits are
        applied after enrichment."""
        return self.action in ("add", "remove", "floors", "entrance", "vastu")

    def describe(self) -> str:
        if self.action == "resize":
            if isinstance(self.value, tuple) and self.value[0] == "abs":
                return f"{_pretty(self.target)} set to {self.value[1]:.0f} sqft"
            factor = float(self.value or 1.0)
            word = "larger" if factor > 1 else "smaller"
            return (f"{_pretty(self.target)} "
                    f"{abs(round((factor - 1) * 100))}% {word}")
        if self.action == "move":
            return f"{_pretty(self.target)} moved to the {self.value}"
        if self.action == "add":
            return f"added {_pretty(self.target)}"
        if self.action == "remove":
            return f"removed {_pretty(self.target)}"
        if self.action == "adjacency":
            return (f"{_pretty(self.target)} placed next to "
                    f"{_pretty(self.other)}")
        if self.action == "floors":
            return f"{self.value} floor(s)"
        if self.action == "entrance":
            return f"entrance from the {self.value}"
        if self.action == "vastu":
            return ("Vastu compliance on" if self.value
                    else "Vastu compliance off")
        return self.raw

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "target": self.target,
                "value": self.value, "other": self.other, "raw": self.raw,
                "describes": self.describe()}


@dataclass
class EditPlan:
    intents: List[EditIntent] = field(default_factory=list)
    unparsed: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.intents)

    def summary(self) -> str:
        if not self.intents:
            return "Nothing in that could be turned into a change."
        return "; ".join(i.describe() for i in self.intents)

    def to_dict(self) -> Dict[str, Any]:
        return {"intents": [i.to_dict() for i in self.intents],
                "unparsed": list(self.unparsed),
                "summary": self.summary()}


def _pretty(rtype: Optional[str]) -> str:
    return (rtype or "room").replace("_", " ")


# ── room vocabulary ─────────────────────────────────────────────────────────

def _room_vocabulary() -> Dict[str, str]:
    """phrase -> canonical room type.

    Built from the SAME alias table `room_resolver` uses
    (sources/enricher_rules.json), so anything the parser understands in a
    brief it also understands in an edit — a user should not have to learn a
    second vocabulary to change their own plan.
    """
    from modules.step3_enrich.room_resolver import (
        DISPLAY_TO_NORM, NORM_TO_DISPLAY,
    )
    vocab: Dict[str, str] = {}
    for norm, display in NORM_TO_DISPLAY.items():
        vocab[display.lower()] = norm
        vocab[norm.replace("_", " ")] = norm
    for alias, norm in DISPLAY_TO_NORM.items():
        vocab[alias.lower()] = norm
    # a few phrasings people actually type
    vocab.update({
        "master": "master_bedroom", "master bed": "master_bedroom",
        "bed room": "bedroom", "bath": "bathroom", "loo": "toilet",
        "wc": "toilet", "puja": "pooja_room", "puja room": "pooja_room",
        "prayer room": "pooja_room", "hall": "living_room",
        "drawing": "drawing_room", "car park": "car_parking",
        "parking": "car_parking", "stairs": "staircase",
        "store": "store_room", "utility": "utility_room",
        "study": "study_room", "kids room": "bedroom_kids",
        "guest room": "bedroom_guest",
    })
    return vocab


_VOCAB: Optional[Dict[str, str]] = None


def _vocab() -> Dict[str, str]:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = _room_vocabulary()
    return _VOCAB


def find_room(text: str, exclude: Optional[str] = None
              ) -> Optional[Tuple[str, int]]:
    """Longest room phrase in `text` -> (canonical type, position).

    Longest-first matters: "master bedroom" must not match as "bedroom", or
    "make the master bedroom bigger" would resize the wrong room.
    """
    low = text.lower()
    best: Optional[Tuple[str, int]] = None
    best_len = 0
    for phrase, norm in _vocab().items():
        if norm == exclude or len(phrase) < 3:
            continue
        pos = low.find(phrase)
        if pos < 0:
            continue
        # whole-word-ish: avoid "bath" inside "bathroom mat"
        after = low[pos + len(phrase):pos + len(phrase) + 1]
        if after and after.isalpha():
            continue
        if len(phrase) > best_len:
            best, best_len = (norm, pos), len(phrase)
    return best


# ── parsing ─────────────────────────────────────────────────────────────────

def _split_clauses(text: str) -> List[str]:
    parts = re.split(r"[.;\n]|,\s*(?=and\b)|\band also\b|\balso\b|\band\b",
                     text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p and p.strip()]


def parse(text: str) -> EditPlan:
    """Free text -> structured intents. Deterministic, no LLM.

    Every clause that cannot be turned into an intent is returned in
    `unparsed` rather than dropped, so the caller can tell the user which
    part of their instruction did not land.
    """
    plan = EditPlan()
    if not text or not text.strip():
        return plan

    for clause in _split_clauses(text):
        intent = _parse_clause(clause)
        if intent is not None:
            plan.intents.append(intent)
        elif len(clause.split()) >= 2:
            plan.unparsed.append(clause)
    return plan


def _parse_clause(clause: str) -> Optional[EditIntent]:
    low = clause.lower().strip()
    if not low:
        return None

    # ── Vastu on/off ────────────────────────────────────────────────────
    if "vastu" in low:
        off = bool(re.search(r"\b(no|not|without|off|disable|remove|drop)\b",
                             low))
        return EditIntent("vastu", value=not off, raw=clause)

    # ── floors ──────────────────────────────────────────────────────────
    m = _FLOORS.search(low)
    if m and re.search(r"\bfloors?\b|\bg\s*\+|\bduplex\b|\bstorey", low):
        words = {"single": 1, "one": 1, "two": 2, "three": 3}
        n = None
        if m.group(2):
            n = int(m.group(2)) + 1          # G+1 -> 2 floors
        elif m.group(3):
            n = int(m.group(3))
        elif m.group(4):
            n = words.get(m.group(4))
        elif "duplex" in low:
            n = 2
        if n and 1 <= n <= 3:
            return EditIntent("floors", value=n, raw=clause)

    # ── entrance ────────────────────────────────────────────────────────
    if _ENTRANCE.search(low):
        direction = normalize_direction(_first_direction(low))
        if direction:
            return EditIntent("entrance", value=direction, raw=clause)

    # ── adjacency first: "put X next to Y" is positional, not longest ───
    # `find_room` returns the LONGEST phrase in a string, so on "put the
    # kitchen next to the dining room" it answered "dining room" and the
    # second lookup then found nothing. Split at the marker and read one
    # room from each side instead.
    marker = _ADJACENT.search(low)
    if marker:
        left = find_room(low[:marker.start()])
        right = find_room(low[marker.end():])
        if left and right and left[0] != right[0]:
            return EditIntent("adjacency", target=left[0], other=right[0],
                              raw=clause)

    room = find_room(low)

    # ── remove ──────────────────────────────────────────────────────────
    if _REMOVE.search(low) and room:
        return EditIntent("remove", target=room[0], raw=clause)

    # ── move to a compass sector ────────────────────────────────────────
    if room and (_MOVE.search(low) or _first_direction(low)):
        direction = normalize_direction(_first_direction(low))
        if direction:
            return EditIntent("move", target=room[0], value=direction,
                              raw=clause)

    # ── resize ──────────────────────────────────────────────────────────
    if room:
        absolute = re.search(r"(\d{2,4})\s*(?:sq\s*\.?\s*ft|sqft|square feet)",
                             low)
        if absolute:
            return EditIntent("resize", target=room[0],
                              value=("abs", float(absolute.group(1))),
                              raw=clause)
        for word in _SCALE_WORDS:
            if word in low:
                return EditIntent("resize", target=room[0],
                                  value=_SCALE[word], raw=clause)

    # ── add (last: "add a study" has no other signal) ───────────────────
    if room and _ADD.search(low):
        return EditIntent("add", target=room[0], raw=clause)

    return None


def _first_direction(text: str) -> Optional[str]:
    m = re.search(r"\b(north[\s\-]?east|north[\s\-]?west|south[\s\-]?east|"
                  r"south[\s\-]?west|north|south|east|west|"
                  r"ne|nw|se|sw|centre|center)\b", text)
    return m.group(1) if m else None


# ── applying ────────────────────────────────────────────────────────────────

def apply_to_requirements(requirements: Dict[str, Any],
                          intents: Sequence[EditIntent]
                          ) -> Tuple[Dict[str, Any], List[str]]:
    """Requirement-level edits, on a COPY of the brief dict.

    Returns (new requirements, notes). Anything that changes the program,
    the floor count or the orientation belongs here, because step 3 has to
    re-derive sizes, floors and attachments from it.
    """
    import copy

    reqs = copy.deepcopy(requirements or {})
    notes: List[str] = []
    rooms = list(reqs.get("rooms") or [])

    from modules.step3_enrich.room_resolver import NORM_TO_DISPLAY

    for intent in intents:
        if intent.action == "add" and intent.target:
            display = NORM_TO_DISPLAY.get(intent.target,
                                          _pretty(intent.target).title())
            existing = next((r for r in rooms
                             if _norm_of(r.get("room_type")) == intent.target),
                            None)
            if existing:
                existing["quantity"] = int(existing.get("quantity", 1)) + 1
                notes.append(f"a second {_pretty(intent.target)} was added")
            else:
                rooms.append({"room_type": display, "quantity": 1,
                              "specific_requirements": None,
                              "preferred_floor": None})
                notes.append(f"{_pretty(intent.target)} added to the brief")

        elif intent.action == "remove" and intent.target:
            before = len(rooms)
            kept = []
            for r in rooms:
                if _norm_of(r.get("room_type")) != intent.target:
                    kept.append(r)
                    continue
                qty = int(r.get("quantity", 1))
                if qty > 1:
                    r["quantity"] = qty - 1
                    kept.append(r)
            rooms = kept
            notes.append(f"{_pretty(intent.target)} removed"
                         if len(rooms) != before or True else "")

        elif intent.action == "floors":
            reqs["number_of_floors"] = int(intent.value)
            notes.append(f"floor count set to {intent.value}")

        elif intent.action == "entrance":
            ctx = dict(reqs.get("plot_context") or {})
            ctx["entrance_side"] = _to_long_direction(intent.value)
            if not ctx.get("road_facing_sides"):
                ctx["road_facing_sides"] = [ctx["entrance_side"]]
            ctx.setdefault("shape", "rectangular")
            reqs["plot_context"] = ctx
            notes.append(f"entrance set to the {intent.value}")

        elif intent.action == "vastu":
            reqs["vastu_compliant"] = bool(intent.value)
            notes.append("Vastu compliance "
                         + ("enabled" if intent.value else "disabled"))

    reqs["rooms"] = rooms
    return reqs, [n for n in notes if n]


def overrides_from(intents: Sequence[EditIntent]) -> Dict[str, Any]:
    """The edits that must survive enrichment.

    Step 3 derives a room's size from matched-plan statistics and its Vastu
    direction from the rule book. Both would overwrite what the user just
    asked for, so these are recorded and re-applied afterwards.
    """
    scale: Dict[str, float] = {}
    absolute: Dict[str, float] = {}
    sector: Dict[str, str] = {}
    adjacency: List[Tuple[str, str]] = []
    for intent in intents:
        if intent.action == "resize" and intent.target:
            if isinstance(intent.value, tuple) and intent.value[0] == "abs":
                absolute[intent.target] = float(intent.value[1])
            elif isinstance(intent.value, (int, float)):
                scale[intent.target] = scale.get(intent.target, 1.0) * \
                    float(intent.value)
        elif intent.action == "move" and intent.target and intent.value:
            sector[intent.target] = intent.value
        elif intent.action == "adjacency" and intent.target and intent.other:
            adjacency.append((intent.target, intent.other))
    return {"scale": scale, "absolute": absolute, "sector": sector,
            "adjacency": adjacency}


def apply_overrides(enriched, overrides: Dict[str, Any]) -> List[str]:
    """Apply the post-enrichment overrides IN PLACE. Returns notes.

    Sizes are clamped to the room's NBC minimum and its phase-02 ceiling: an
    edit may stretch a room, it may not make it illegal or turn a bathroom
    into a hall.
    """
    from modules.step3_enrich.program_synth import max_sqft_for

    notes: List[str] = []
    scale = overrides.get("scale") or {}
    absolute = overrides.get("absolute") or {}
    sector = overrides.get("sector") or {}

    for room in enriched.rooms:
        rtype = room.room_type
        want: Optional[float] = None
        if rtype in absolute:
            want = absolute[rtype]
        elif rtype in scale:
            want = room.target_area_sqft * scale[rtype]
        if want is not None:
            ceiling = max(room.target_area_sqft,
                          max_sqft_for(rtype, room.target_area_sqft)) * 1.6
            new = max(room.min_area_sqft, min(want, ceiling))
            if abs(new - room.target_area_sqft) > 1.0:
                notes.append(
                    f"{room.display_name}: {room.target_area_sqft:.0f} -> "
                    f"{new:.0f} sqft"
                    + (" (clamped to what the plot and NBC allow)"
                       if abs(new - want) > 1.0 else ""))
                width = max(room.min_width_ft,
                            room.target_width_ft * (new / room.target_area_sqft) ** 0.5)
                room.target_area_sqft = round(new, 2)
                room.target_width_ft = round(width, 2)
                room.target_length_ft = round(new / max(width, 0.1), 2)
                room.max_area_sqft = round(max(room.max_area_sqft, new), 2)

        if rtype in sector:
            room.preferred_direction = sector[rtype]
            if room.vastu is not None:
                room.vastu.preferred_directions = [sector[rtype]]
                room.vastu.prohibited_directions = [
                    d for d in room.vastu.prohibited_directions
                    if d != sector[rtype]]
            notes.append(f"{room.display_name}: placed toward "
                         f"{sector[rtype]}")
    return notes


def _norm_of(room_type: Optional[str]) -> str:
    if not room_type:
        return ""
    found = find_room(str(room_type).lower())
    return found[0] if found else str(room_type).lower().replace(" ", "_")


def _to_long_direction(short: str) -> str:
    return {"N": "north", "S": "south", "E": "east", "W": "west",
            "NE": "north_east", "NW": "north_west",
            "SE": "south_east", "SW": "south_west"}.get(short, "north")
