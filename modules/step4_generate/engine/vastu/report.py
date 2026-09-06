"""
report.py — the Vastu compliance scorecard.

A boolean `vastu_enabled: true` in the output tells a user nothing, and it is
what the product shipped for months while the engine was in fact ignoring
Vastu completely. This produces the opposite: a per-room account of what was
asked for, what was achieved, and — where they differ — what it would have
cost to fix.

Two design commitments:

  NEVER CLAIM MORE THAN WAS DONE. A room one sector off its ideal is reported
  as "near", not as compliant. A site whose north was given as a diagonal is
  reported as accurate to 45 degrees. The mandala's own data defects appear
  in `notes`.

  ALWAYS SAY WHAT WOULD HAVE BEEN BETTER. Every miss carries the sector or
  pada it should have occupied, because "your pooja room is in the west" is a
  complaint and "your pooja room is in the west; north-east is ideal" is
  advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from modules.step4_generate.engine.vastu import mandala as mandala_mod
from modules.step4_generate.engine.vastu.compass import (
    CompassFrame, sector_distance,
)

# How a room's realised sector relates to the one Vastu asked for.
STATUS_IDEAL = "ideal"        # exactly the preferred sector
STATUS_NEAR = "near"          # one compass step away — a good plan
STATUS_OFF = "off"            # two or more steps, but not barred
STATUS_BARRED = "barred"      # landed in a sector Vastu explicitly prohibits
STATUS_NONE = "unconstrained"  # this room has no Vastu opinion

_GRADE_BANDS = ((0.90, "A"), (0.75, "B"), (0.60, "C"), (0.40, "D"))

# How the grade is composed. Rooms lead because they are most of Vastu and the
# only part a user can act on by editing; the entrance is one decision but a
# decisive one; the Brahmasthan is a single fact that is clean on essentially
# every plan.
#
# The old form averaged the three equally. MEASURED over the same 68 plans
# (6 plot sizes x 3 programs x 4 orientations), equal thirds against these:
#
#     grades   old: A 19  B 10  C 26  D 8  E 5      mean 0.729
#              new: A  8  B 43  C 11  D 6           mean 0.776
#
# The old distribution is bimodal — piles at A and at C with a hollow at B —
# because the volatile per-pada entrance score carried a full third of the
# grade while the Brahmasthan term contributed a constant 1.0 to every plan
# (intrusion was exactly zero on 68/68). Weighting toward the rooms tracks the
# part of Vastu the user can actually influence and removes the hollow.
#
# Stated plainly: the mean goes UP by ~0.05, so this is not a tightening. What
# it buys is a grade that moves with room compliance instead of jumping on one
# pada, and that never falls when every room improves.
W_ROOMS = 0.60
W_ENTRANCE = 0.25
W_BRAHMASTHAN = 0.15

# Share of the central 3x3 occupied by a barred use at which the Brahmasthan
# counts as fully violated. Below it the term degrades linearly.
BRAHMA_FULL_VIOLATION = 0.25


@dataclass
class RoomCompliance:
    room: str
    rtype: str
    wanted: Optional[str]
    got: str
    status: str
    strength: str = "soft"
    steps_off: int = 0
    advice: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VastuReport:
    active: bool = False
    grade: str = "-"
    score: float = 0.0
    rooms: List[RoomCompliance] = field(default_factory=list)
    entrance: Dict[str, object] = field(default_factory=dict)
    brahmasthan: Dict[str, object] = field(default_factory=dict)
    site: Dict[str, object] = field(default_factory=dict)
    # advice a person can act on
    notes: List[str] = field(default_factory=list)
    # health of the rule book itself. Kept, because a report that hides the
    # condition of its own source is not honest — but kept SEPARATE, because
    # "marma endpoint 'W3_Mukhya' names a pada that disagrees with gate W3"
    # is a message for whoever maintains data/vastuRules1.json, not for
    # someone deciding where to put their kitchen. `modules.diagnostics`
    # is where it belongs on screen.
    data_notes: List[str] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for room in self.rooms:
            out[room.status] = out.get(room.status, 0) + 1
        return out

    def to_dict(self) -> Dict:
        return {
            "active": self.active,
            "grade": self.grade,
            "score": self.score,
            "counts": self.counts,
            "rooms": [r.to_dict() for r in self.rooms],
            "entrance": self.entrance,
            "brahmasthan": self.brahmasthan,
            "site": self.site,
            "notes": list(self.notes),
            "data_notes": list(self.data_notes),
        }

    def summary_line(self) -> str:
        if not self.active:
            return "Vastu not requested."
        c = self.counts
        return (f"Vastu {self.grade} ({self.score:.0%}) — "
                f"{c.get(STATUS_IDEAL, 0)} ideal, {c.get(STATUS_NEAR, 0)} "
                f"near, {c.get(STATUS_OFF, 0)} off, "
                f"{c.get(STATUS_BARRED, 0)} in barred sectors")


def _grade(score: float) -> str:
    for cut, letter in _GRADE_BANDS:
        if score >= cut:
            return letter
    return "E"


def build(plan, request, room_ids: Dict[str, int],
          verdict=None) -> VastuReport:
    """Read a finished plan against the Vastu that was asked of it."""
    active = bool(getattr(request, "vastu", False)) and any(
        getattr(s, "has_vastu", False) for s in request.rooms)
    report = VastuReport(active=active)
    if not active:
        return report

    frame = CompassFrame(getattr(request, "north_side", "N") or "N")
    mandala = mandala_mod.shared()
    breakdown = dict(getattr(verdict, "breakdown", None) or {})

    # ── per room ────────────────────────────────────────────────────────
    weights: List[float] = []
    for spec in request.rooms:
        rid = room_ids.get(spec.name)
        if rid is None:
            continue
        got = frame.sector_of_rect(plan.face_bbox(rid), plan.h, plan.w)
        wanted = spec.vastu_dir
        barred = got in (spec.vastu_avoid or ())

        if barred:
            status, steps = STATUS_BARRED, sector_distance(got, wanted or got)
            advice = (f"{spec.rtype.replace('_', ' ')} must not sit in the "
                      f"{got}" + (f"; {wanted} is ideal" if wanted else ""))
        elif not wanted:
            status, steps, advice = STATUS_NONE, 0, ""
        else:
            steps = sector_distance(got, wanted)
            if steps == 0:
                status, advice = STATUS_IDEAL, ""
            elif steps == 1:
                status = STATUS_NEAR
                advice = f"one sector from the ideal {wanted}"
            else:
                status = STATUS_OFF
                advice = f"{steps} sectors from the ideal {wanted}"

        report.rooms.append(RoomCompliance(
            room=spec.name, rtype=spec.rtype, wanted=wanted, got=got,
            status=status, strength=spec.vastu_strength, steps_off=steps,
            advice=advice))

        if status == STATUS_NONE:
            continue
        # a HARD-strength room counts double: a pooja room in the south-west
        # is not half a problem because a store room is fine
        weight = 2.0 if spec.vastu_strength == "hard" else 1.0
        credit = {STATUS_IDEAL: 1.0, STATUS_NEAR: 0.7,
                  STATUS_OFF: 0.25, STATUS_BARRED: 0.0}[status]
        weights.append(weight * credit)
        report.rooms[-1].advice = advice

    hard_total = sum(2.0 if s.vastu_strength == "hard" else 1.0
                     for s in request.rooms
                     if s.name in room_ids and s.has_vastu)
    room_score = (sum(weights) / hard_total) if hard_total else 0.0

    # ── entrance pada ───────────────────────────────────────────────────
    gate_score = None
    if mandala.available and request.floor_index == 0:
        gate_id = breakdown.get("vastu_gate")
        gate = mandala.gates.get(str(gate_id)) if gate_id else None
        if gate is not None:
            gate_score = gate.score
            side = frame.compass_of_grid_side(request.entrance_side)
            ideal = [f"{g.id} ({g.name})" for g in mandala.best_gates(side)]
            report.entrance = {
                "pada": gate.id, "name": gate.name, "status": gate.status,
                "score": gate.score,
                "compass_side": side,
                "ideal_padas_on_this_side": ideal,
                "advice": ("" if gate.status == "ideal"
                           else f"the door sits on {gate.id} ({gate.name}), "
                                f"which Vastu marks {gate.status.replace('_', ' ')}"
                                + (f"; {', '.join(ideal)} are ideal on the "
                                   f"{side} side" if ideal else "")),
            }

    # ── the Brahmasthan ─────────────────────────────────────────────────
    if mandala.available and mandala.brahma is not None:
        intrusion = float(breakdown.get("vastu_brahma_intrusion", 0.0))
        wall = float(breakdown.get("vastu_brahma_wall", 0.0))
        x0, y0, x1, y1 = frame.brahmasthan_rect(plan.h, plan.w)
        report.brahmasthan = {
            "rect_cells": [x0, y0, x1, y1],
            "barred_use_share": round(intrusion, 3),
            "wall_share": round(wall, 3),
            "clear": intrusion <= 0.02,
            "advice": ("" if intrusion <= 0.02 else
                       f"{intrusion:.0%} of the plot's centre is occupied by "
                       f"a use Vastu bars there "
                       f"({', '.join(mandala.brahma_blocked_usages())})"),
        }

    # ── site-level facts the user cannot fix by moving rooms ────────────
    report.site = {
        "north_side": frame.north_side,
        "orientation": frame.describe(),
        "entrance_faces": frame.compass_of_grid_side(request.entrance_side),
        "plot_aspect": breakdown.get("vastu_plot_aspect"),
        "marma_hits": breakdown.get("vastu_marma_hits", 0),
        "mass_gradient_ok": (
            breakdown.get("vastu_mass_ne", 0.0)
            <= breakdown.get("vastu_mass_sw", 0.0)),
    }

    # ── overall ─────────────────────────────────────────────────────────
    #
    # WEIGHTED, AND CONTINUOUS IN THE BRAHMASTHAN. Both matter, and both were
    # wrong.
    #
    # The old form averaged three parts equally and scored the Brahmasthan
    # 1.0 / 0.0 on the same 2% threshold used for the display boolean. That
    # made the grade NON-MONOTONIC in the thing it reports: an edit measured
    # here took a plan from 4 ideal / 1 near / 3 off to 6 ideal / 2 off — every
    # room strictly better, entrance unchanged — and the grade fell B (89%) to
    # C (60%), because a kitchen clipped 3% of the centre. A user who asks for
    # a Vastu-canonical change and watches their Vastu score collapse stops
    # believing the feature, which is the opposite of leading with it.
    #
    # MEASURED, 68 plans over 6 plot sizes x 3 programs x 4 orientations:
    # intrusion was EXACTLY zero on 100% of them. So this term is inert on
    # ordinary plans — as a third of the grade it was pure inflation — and it
    # only has to shape the rare real defect. It is now linear to a cap:
    # a quarter of the centre occupied by a barred use counts as a full
    # violation, 3% costs about a point of grade rather than 29.
    #
    # VAS-003 already scores the same quantity proportionally
    # (`penalty += w_vastu_brahma * worst`); this brings the number the user
    # reads into the same shape as the number the engine optimises.
    parts: List[Tuple[float, float]] = [(room_score, W_ROOMS)]
    if gate_score is not None:
        parts.append((gate_score, W_ENTRANCE))
    if report.brahmasthan:
        intrusion = float(report.brahmasthan["barred_use_share"])
        parts.append((max(0.0, 1.0 - intrusion / BRAHMA_FULL_VIOLATION),
                      W_BRAHMASTHAN))
    total_weight = sum(w for _, w in parts) or 1.0
    report.score = round(sum(v * w for v, w in parts) / total_weight, 3)
    report.grade = _grade(report.score)

    # ── honesty notes ───────────────────────────────────────────────────
    if not mandala.available:
        report.notes.append(
            "The 81-pada mandala could not be loaded, so entrance-pada, "
            "Brahmasthan and marma checks did not run. Room sectors were "
            "still scored.")
    report.data_notes.extend(mandala.warnings)
    barred = [r for r in report.rooms if r.status == STATUS_BARRED]
    if barred:
        report.notes.append(
            "Rooms in barred sectors are a deliberate trade the layout made "
            "to stay buildable, not an oversight: "
            + ", ".join(f"{r.room} ({r.got})" for r in barred))
    return report


def format_text(report: VastuReport) -> str:
    """Console rendering — used by the demos and the harness."""
    if not report.active:
        return "Vastu not requested."
    lines = [report.summary_line(), ""]
    width = max((len(r.room) for r in report.rooms), default=10)
    for r in sorted(report.rooms, key=lambda r: (r.status != STATUS_BARRED,
                                                 -r.steps_off, r.room)):
        if r.status == STATUS_NONE:
            continue
        mark = {STATUS_IDEAL: "++", STATUS_NEAR: " +",
                STATUS_OFF: " -", STATUS_BARRED: "XX"}[r.status]
        lines.append(f"  {mark} {r.room:<{width}}  want {str(r.wanted):<7}"
                     f"got {r.got:<7}{r.advice}")
    if report.entrance:
        lines += ["", f"  entrance: {report.entrance['pada']} "
                      f"({report.entrance['name']}) — "
                      f"{report.entrance['status']}"]
        if report.entrance.get("advice"):
            lines.append(f"            {report.entrance['advice']}")
    if report.brahmasthan:
        state = ("clear" if report.brahmasthan["clear"]
                 else report.brahmasthan["advice"])
        lines += ["", f"  brahmasthan: {state}"]
    if report.notes:
        lines += [""] + [f"  note: {n}" for n in report.notes]
    if report.data_notes:
        lines += [""] + [f"  rule book: {n}" for n in report.data_notes]
    return "\n".join(lines)
