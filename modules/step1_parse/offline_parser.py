"""
offline_parser.py — deterministic rule-based requirement extraction.

The LLM parser is the primary path, but API quota/keys can and do fail.
This module keeps the chat functional with zero LLM calls:

  extract(text)                → partial BuildingRequirements-shaped dict
                                 containing ONLY fields found in the text
  merge_partial(existing, new) → shallow field-level merge (new wins)
  apply_answer(data, field, a) → deterministically apply an interactive
                                 answer to the specific field asked about;
                                 returns None if the answer can't be mapped

It is also the FIRST choice for interactive widget answers (direction /
floors / vastu / plot size): applying "North-East" to the facing field is
string mapping, not language understanding — no LLM call needed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ── directions ────────────────────────────────────────────────────────
_DIRECTIONS = [  # longest patterns first so "north-east" wins over "north"
    (r"north[\s\-_]*east|ne\b", "north_east"),
    (r"north[\s\-_]*west|nw\b", "north_west"),
    (r"south[\s\-_]*east|se\b", "south_east"),
    (r"south[\s\-_]*west|sw\b", "south_west"),
    (r"north|n\b", "north"),
    (r"south|s\b", "south"),
    (r"east|e\b", "east"),
    (r"west|w\b", "west"),
]

_ROOM_KEYWORDS = [
    (r"pooja", "Pooja Room"),
    (r"study", "Study Room"),
    (r"store\s*room|storage", "Store Room"),
    (r"balcon(?:y|ies)", "Balcony"),
    (r"dining", "Dining Room"),
    (r"car\s*parking|parking|garage", "Car Parking"),
    (r"servant", "Servant Room"),
    (r"utilit", "Utility"),
    (r"staircase|stairs", "Staircase"),
]


def parse_direction(text: str) -> Optional[str]:
    """'North-East', 'ne', 'facing north east please' → 'north_east'."""
    low = text.lower()
    for pattern, value in _DIRECTIONS:
        if re.search(pattern, low):
            return value
    return None


def parse_floors(text: str) -> Optional[int]:
    low = text.lower()
    if re.search(r"g\s*\+\s*2", low):
        return 3
    if re.search(r"g\s*\+\s*1|duplex", low):
        return 2
    if re.search(r"single|ground\s*(floor|only)|one\s*floor|1\s*floor", low):
        return 1
    m = re.search(r"\b([123])\b\s*floors?", low)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(one|two|three)\b", low)
    if m:
        return {"one": 1, "two": 2, "three": 3}[m.group(1)]
    m = re.fullmatch(r"\s*([123])\s*", low)
    if m:
        return int(m.group(1))
    return None


def parse_yes_no(text: str) -> Optional[bool]:
    low = text.lower().strip()
    if re.search(r"\b(no|not|without|skip|don'?t|nah)\b", low):
        return False
    if re.search(r"\b(yes|yeah|ok(?:ay)?|sure|please|definitely|enable|want)\b", low) \
            or low in ("y", "haan", "ha"):
        return True
    return None


def parse_plot_dims(text: str) -> Optional[Dict[str, Any]]:
    """'30x40', '30 x 40 ft', '30 by 40 feet plot' → dims dict (feet)."""
    m = re.search(r"(\d{2,3})\s*(?:x|×|\*|by)\s*(\d{2,3})", text.lower())
    if not m:
        return None
    w, l = float(m.group(1)), float(m.group(2))
    unit = "m" if re.search(r"\bmet(?:er|re)s?\b|\bm\b(?!i)", text.lower()) else "ft"
    if unit == "m":                     # store in feet, like the LLM parser
        w, l = round(w * 3.28084, 1), round(l * 3.28084, 1)
    return {"width": w, "length": l, "unit": "ft",
            "total_area_sqft": round(w * l, 2)}


def _extract_rooms(low: str) -> List[Dict[str, Any]]:
    rooms: List[Dict[str, Any]] = []

    def add(rtype: str, qty: int = 1):
        for r in rooms:
            if r["room_type"] == rtype:
                r["quantity"] = max(r["quantity"], qty)
                return
        rooms.append({"room_type": rtype, "quantity": qty,
                      "specific_requirements": None, "preferred_floor": None})

    bhk = re.search(r"(\d)\s*bhk", low)
    beds = re.search(r"(\d+)\s*bed\s*rooms?", low)
    if bhk or beds:
        n = int((bhk or beds).group(1))
        if n > 0:
            master = 1 if re.search(r"master", low) or n >= 3 else 0
            if master:
                add("Master Bedroom", 1)
            if n - master > 0:
                add("Bedroom", n - master)
            add("Living Room", 1)
            add("Kitchen", 1)

    baths = re.search(r"(\d+)\s*(?:bath\s*rooms?|toilets?|washrooms?)", low)
    if baths:
        add("Bathroom", int(baths.group(1)))
    elif re.search(r"bath\s*room|toilet|washroom", low):
        add("Bathroom", 1)

    for pattern, rtype in _ROOM_KEYWORDS:
        m = re.search(r"(\d+)?\s*" + pattern, low)
        if m:
            add(rtype, int(m.group(1)) if m.group(1) else 1)
    return rooms


def extract(text: str) -> Dict[str, Any]:
    """Rule-based extraction. Returns ONLY the fields present in the text."""
    low = text.lower()
    out: Dict[str, Any] = {}

    dims = parse_plot_dims(text)
    if dims:
        out["plot_dimensions"] = dims

    facing = None
    m = re.search(
        r"(north[\s\-]*east|north[\s\-]*west|south[\s\-]*east|"
        r"south[\s\-]*west|north|south|east|west)[\s\-]*facing", low)
    if m:
        facing = parse_direction(m.group(1))
    elif re.search(r"facing|road\s+(?:on|is|side)", low):
        facing = parse_direction(low)
    if facing:
        out["plot_context"] = {
            "shape": "rectangular", "road_facing_sides": [facing],
            "north_direction": None, "entrance_side": None,
            "image_source_notes": None,
        }

    floors = parse_floors(low)
    if floors is not None and re.search(r"floor|g\s*\+|duplex|single|ground", low):
        out["number_of_floors"] = floors

    if "vastu" in low:
        out["vastu_compliant"] = not re.search(
            r"(no|not|without|skip)\s+(?:\w+\s+)?vastu|vastu\s+(?:no|not needed)", low)

    rooms = _extract_rooms(low)
    if rooms:
        out["rooms"] = rooms

    if re.search(r"stilt", low):
        out["parking_type"] = "stilt"
    elif re.search(r"garage", low):
        out["parking_type"] = "garage"

    for word in ("villa", "bungalow", "duplex"):
        if word in low:
            out["building_type"] = word
            break
    return out


def merge_partial(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow field-level merge: new values win; rooms are unioned by type."""
    merged = dict(existing or {})
    for key, value in new.items():
        if key == "rooms" and merged.get("rooms"):
            rooms = {r["room_type"]: dict(r) for r in merged["rooms"]}
            for r in value:
                rooms[r["room_type"]] = dict(r)
            merged["rooms"] = list(rooms.values())
        elif key == "plot_context" and merged.get("plot_context"):
            ctx = dict(merged["plot_context"])
            for k, v in value.items():
                if v not in (None, [], ""):
                    ctx[k] = v
            merged["plot_context"] = ctx
        else:
            merged[key] = value
    return merged


def apply_answer(data: Dict[str, Any], field_key: str,
                 answer: str) -> Optional[Dict[str, Any]]:
    """Deterministically apply an interactive answer to the field that was
    asked about. Returns the updated dict, or None if unmappable (caller
    should then fall back to the LLM merge)."""
    updated = dict(data or {})
    ctx = dict(updated.get("plot_context") or {
        "shape": "rectangular", "road_facing_sides": [],
        "north_direction": None, "entrance_side": None,
        "image_source_notes": None,
    })

    if field_key.startswith("plot_context.road_facing_sides"):
        d = parse_direction(answer)
        if not d:
            return None
        ctx["road_facing_sides"] = [d]
        updated["plot_context"] = ctx
        return updated

    if field_key.startswith("plot_context.entrance_side"):
        d = parse_direction(answer)
        if d is None and parse_yes_no(answer):
            d = (ctx.get("road_facing_sides") or [None])[0]  # "road side is fine"
        if not d:
            return None
        ctx["entrance_side"] = d
        updated["plot_context"] = ctx
        return updated

    if field_key.startswith("plot_context.north_direction"):
        d = parse_direction(answer)
        if not d:
            return None
        ctx["north_direction"] = d
        updated["plot_context"] = ctx
        return updated

    if field_key == "number_of_floors":
        n = parse_floors(answer)
        if n is None:
            return None
        updated["number_of_floors"] = n
        return updated

    if field_key == "vastu_compliant":
        v = parse_yes_no(answer)
        if v is None:
            return None
        updated["vastu_compliant"] = v
        return updated

    if field_key.startswith("plot_dimensions"):
        dims = parse_plot_dims(answer)
        if not dims:
            return None
        updated["plot_dimensions"] = dims
        return updated

    if field_key == "rooms":
        rooms = _extract_rooms(answer.lower())
        if not rooms:
            return None
        updated["rooms"] = rooms
        return updated

    if field_key == "parking_type":
        low = answer.lower()
        for word in ("stilt", "garage", "none"):
            if word in low or (word == "none" and parse_yes_no(answer) is False):
                updated["parking_type"] = word if word != "none" else "none"
                return updated
        return None

    return None   # building_type / style / free-text → LLM path
