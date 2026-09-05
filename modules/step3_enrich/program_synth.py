"""
program_synth.py — decide WHAT to build, before deciding where it goes.

The gap this fills. Until now the program came from BHK statistics and was
then only ever SCALED: on a 500 sqft plot seven rooms were shrunk toward
their NBC minimums until the request was refused, and on a 4,200 sqft plot
the same seven were inflated by one uniform factor until a 45 sqft bathroom
came out at 266. Nothing ever asked whether the program suited the plot.

    measured, 60x70 plot, 675 sqft program:
        Bath 1     45 sqft target  ->  266 sqft   (5.9x)
        Living    200 sqft target  -> 1176 sqft   (5.9x)

WHY NOT JUST PICK THE HIGHEST-SCORING PROGRAM. Because the reviewer's soft
score is a measure of how well a program was REALIZED, not whether it suits
the plot — fewer rooms means less area drift, fewer walls and fewer chances
to violate anything, so a 1BHK scores 100 on a 4,200 sqft plot. Selecting by
score would recommend a studio for a mansion. Measured, across 14 plot sizes:
the best-scoring program was `1BHK` at every size above 500 sqft.

So capacity is read as FEASIBILITY instead — the largest program that still
lands above a quality floor — and the table below is that measurement, not a
guess. The user's own request always outranks it; synthesis adjusts at the
margins and says, out loud, what it changed and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ── the measured capacity curve ─────────────────────────────────────────────
# Each row: (max net-buildable sqft for this tier, tier name, bedrooms).
# Derived by running the engine over 14 plot sizes x 7 program shapes and
# taking the largest program that still scored >= 60. The scores behind it:
#
#     net sqft   studio  1BHK  2BHK  2BHK+  3BHK  3BHK+  4BHK
#         252       75      -     -      -     -      -     -
#         300       75     66     -      -     -      -     -
#         432       96     78    30      -     -      -     -
#         500       97     99    69     36     -      -     -
#         616       97    100    87     75    -9    -39     -
#         952       97    100    88     75    45     -1  -112
#        1200       97     98    98     75    70     58    17
#        1575       97    100    98     75    72     72    24
#        2000       97    100    98     76    69     69    54
#
# "-" is a request the engine refused outright. Note 4BHK never clears 60 at
# any size — nine-plus-room programs are a real weakness of the current
# carver, recorded here rather than hidden.
_CAPACITY: Tuple[Tuple[float, str, int], ...] = (
    (300.0, "studio", 0),
    (480.0, "1BHK", 1),
    (600.0, "2BHK", 2),
    (1100.0, "2BHK+", 2),
    (1500.0, "3BHK", 3),
    (2400.0, "3BHK+", 3),
    (float("inf"), "4BHK", 4),
)

# Scale regimes. The boundaries are the same measurement read differently:
# below ~600 sqft the engine starts refusing ordinary programs, and above
# ~1800 a normal program stops filling the plot and starts inflating.
REGIME_COMPACT_MAX = 600.0
REGIME_SPACIOUS_MIN = 1800.0

# Rooms worth ADDING when a plot has surplus, in Indian-practice priority
# order: (rtype, display, min net sqft before it is worth adding, target).
# Deliberately conservative — the alternative to adding a room is inflating
# the ones already there, and a 266 sqft bathroom helps nobody.
_SURPLUS_LADDER: Tuple[Tuple[str, str, float, float], ...] = (
    ("dining_room", "Dining Room", 800.0, 110.0),
    ("pooja_room", "Pooja Room", 1000.0, 40.0),
    ("utility_room", "Utility Room", 1150.0, 45.0),
    ("store_room", "Store Room", 1400.0, 45.0),
    ("study_room", "Study Room", 1700.0, 90.0),
)

# Rooms that may be dropped to fit a compact plot, LEAST useful first. Only
# implicit rooms are ever dropped: a room the user asked for by name is never
# removed silently, and the caller is told when one has to go.
_DROP_ORDER: Tuple[str, ...] = (
    "store_room", "utility_room", "study_room", "passage",
    "pooja_room", "dining_room",
)

# Per-room upper bounds, as a multiple of the room's target. Without these
# the engine's settler spreads every surplus square foot proportionally, and
# proportional is exactly wrong: a living room can absorb 60% more, a
# bathroom cannot absorb 500%.
_MAX_MULTIPLE: Dict[str, float] = {
    "living_room": 1.9, "drawing_room": 1.9, "dining_room": 1.7,
    "master_bedroom": 1.6, "bedroom": 1.5, "study_room": 1.5,
    "kitchen": 1.5, "foyer": 1.5, "passage": 1.6,
    "bathroom": 1.35, "toilet": 1.3, "pooja_room": 1.4,
    "store_room": 1.5, "utility_room": 1.5, "staircase": 1.25,
    "car_parking": 1.4, "servant_room": 1.4,
}
DEFAULT_MAX_MULTIPLE = 1.6

# Absolute ceilings, sqft. A bedroom above this stops being a bedroom.
_ABSOLUTE_MAX: Dict[str, float] = {
    "bathroom": 70.0, "toilet": 45.0, "pooja_room": 70.0,
    "store_room": 90.0, "utility_room": 90.0, "kitchen": 220.0,
    "staircase": 140.0, "passage": 130.0,
}


def regime_for(net_sqft: float) -> str:
    if net_sqft <= REGIME_COMPACT_MAX:
        return "compact"
    if net_sqft >= REGIME_SPACIOUS_MIN:
        return "spacious"
    return "normal"


def capacity_for(net_sqft: float) -> Tuple[str, int]:
    """(tier name, bedrooms the plot comfortably supports)."""
    for limit, name, beds in _CAPACITY:
        if net_sqft <= limit:
            return name, beds
    return _CAPACITY[-1][1], _CAPACITY[-1][2]


def max_sqft_for(rtype: str, target_sqft: float) -> float:
    """The most this room should ever be allowed to grow to."""
    grown = target_sqft * _MAX_MULTIPLE.get(rtype, DEFAULT_MAX_MULTIPLE)
    ceiling = _ABSOLUTE_MAX.get(rtype)
    return round(min(grown, ceiling) if ceiling else grown, 1)


@dataclass
class Decision:
    """One change synthesis made, in terms a user can act on."""
    action: str          # add | drop | keep | cap | warn
    room: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"action": self.action, "room": self.room,
                "reason": self.reason}


@dataclass
class ProgramPlan:
    regime: str
    net_sqft: float
    capacity: str
    requested_bedrooms: int
    decisions: List[Decision] = field(default_factory=list)
    headline: str = ""
    added: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "net_sqft": round(self.net_sqft, 1),
            "capacity": self.capacity,
            "requested_bedrooms": self.requested_bedrooms,
            "headline": self.headline,
            "added": list(self.added),
            "dropped": list(self.dropped),
            "decisions": [d.to_dict() for d in self.decisions],
        }


def _is_bedroom(rtype: str) -> bool:
    return "bedroom" in rtype


def _explicit(room) -> bool:
    """A room the user actually asked for, as opposed to one an earlier
    stage inferred. Only inferred rooms are ever dropped."""
    return not getattr(room, "implicit_room", False)


def plan_program(rooms: Sequence, net_sqft_per_floor: float, floors: int = 1,
                 *, allow_add: bool = True, allow_drop: bool = True
                 ) -> ProgramPlan:
    """Decide what belongs on this plot. Returns the DECISIONS; the caller
    applies them, so nothing here mutates its input.

    `net_sqft_per_floor` is the buildable area of ONE floor; the whole
    program is judged against the total across floors, because a 3BHK on two
    small floors is a different question from a 3BHK on one small floor.
    """
    total_net = max(1.0, net_sqft_per_floor * max(1, floors))
    regime = regime_for(net_sqft_per_floor)
    tier, tier_beds = capacity_for(total_net)

    have = list(rooms)
    have_types = [getattr(r, "room_type", "") for r in have]
    bedrooms = sum(1 for t in have_types if _is_bedroom(t))
    program_sqft = sum(float(getattr(r, "target_area_sqft", 0.0))
                       for r in have)

    plan = ProgramPlan(regime=regime, net_sqft=total_net, capacity=tier,
                       requested_bedrooms=bedrooms)

    density = program_sqft / total_net

    # ── over capacity: say so, and shed inferred extras only ────────────
    if bedrooms > tier_beds and tier_beds > 0:
        plan.decisions.append(Decision(
            action="warn", room=f"{bedrooms} bedrooms",
            reason=(f"{total_net:.0f} sqft of buildable area comfortably "
                    f"carries a {tier} ({tier_beds} bedroom"
                    f"{'s' if tier_beds != 1 else ''}). A {bedrooms}-bedroom "
                    f"plan here means rooms at or near their NBC minimums.")))

    if allow_drop and density > 0.95:
        # shed inferred conveniences before compressing everything: a plan
        # without a store room beats a plan where every bedroom is 9x10
        for rtype in _DROP_ORDER:
            if density <= 0.85:
                break
            for room in have:
                if getattr(room, "room_type", "") != rtype or _explicit(room):
                    continue
                have.remove(room)
                area = float(getattr(room, "target_area_sqft", 0.0))
                program_sqft -= area
                density = program_sqft / total_net
                name = getattr(room, "display_name", rtype)
                plan.dropped.append(name)
                plan.decisions.append(Decision(
                    action="drop", room=name,
                    reason=(f"the program needed {area:.0f} sqft it did not "
                            f"have; this room was inferred, not requested")))
                break

    # ── surplus: add rooms rather than inflate the ones there ───────────
    if allow_add and density < 0.72:
        present = set(have_types)
        for rtype, display, min_net, target in _SURPLUS_LADDER:
            if total_net < min_net or rtype in present:
                continue
            if (program_sqft + target) / total_net > 0.80:
                break
            program_sqft += target
            present.add(rtype)
            plan.added.append(display)
            plan.decisions.append(Decision(
                action="add", room=display,
                reason=(f"{total_net - program_sqft:.0f} sqft would otherwise "
                        f"have been spread across the existing rooms")))
        density = program_sqft / total_net

    # ── the headline ────────────────────────────────────────────────────
    plan.headline = _headline(plan, bedrooms, tier_beds, density, total_net)
    return plan


def _headline(plan: ProgramPlan, bedrooms: int, tier_beds: int,
              density: float, total_net: float) -> str:
    bits: List[str] = [
        f"{total_net:.0f} sqft buildable — a {plan.capacity} plot "
        f"({plan.regime})."
    ]
    if plan.added:
        bits.append(f"Added {', '.join(plan.added)} rather than inflating "
                    f"the rooms you asked for.")
    if plan.dropped:
        bits.append(f"Dropped {', '.join(plan.dropped)} to keep the rooms "
                    f"you asked for above their minimums.")
    if bedrooms > tier_beds > 0:
        bits.append(f"{bedrooms} bedrooms on this plot will be tight; "
                    f"{tier_beds} is the comfortable number.")
    if not plan.added and not plan.dropped and bedrooms <= tier_beds:
        bits.append("The program fits the plot as requested.")
    return " ".join(bits)


def surplus_after_caps(rooms: Sequence, net_sqft: float) -> float:
    """Buildable area left over once every room is at its maximum.

    Above zero, the plot cannot be filled with rooms without bloating them —
    which is the signal to open a courtyard rather than keep growing
    bedrooms. Vastu wants exactly that at the Brahmasthan, so the two
    requirements point the same way.
    """
    ceiling = sum(max_sqft_for(getattr(r, "room_type", ""),
                               float(getattr(r, "target_area_sqft", 0.0)))
                  for r in rooms)
    return max(0.0, net_sqft - ceiling)
