"""
Flaw-injection tests: every reviewer rule must catch a synthetically
planted defect (implementation plan §4 / M1 gate). Plans here are built
by hand on the GridPlan substrate — small, exact, and each contains
precisely one flaw.
"""

import unittest

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest, RoomSpec
from modules.step4_generate.engine.validator import BasicValidator


def check(plan, rooms, wide_pairs=(), door_pairs=(), entrance=None,
          gate=None, config=None):
    """Run the reviewer over a hand-built plan. rooms: [(name, rtype)...]
    matching plan room names; opening pairs by room name. `gate`: room
    name to receive an exterior "wide" opening (vehicle gate), matching
    carve/connections.py::place_vehicle_gate's semantics."""
    ids = {name: plan.find_room(name).id for name, _ in rooms}
    for a, b in wide_pairs:
        plan.add_opening(ids[a], ids[b], "wide")
    for a, b, swing in door_pairs:
        plan.add_opening(ids[a], ids[b], "door", swing=swing)
    if entrance:
        plan.add_exterior_opening(ids[entrance], "S")
    if gate:
        plan.add_exterior_opening(ids[gate], "S", kind="wide")
    request = EngineRequest(
        plot_w_ft=30, plot_h_ft=40, entrance_side="S",
        rooms=[RoomSpec(n, t, 100, zone="public") for n, t in rooms],
    )
    return BasicValidator(config or EngineConfig()).check(plan, request, ids)


def three_room_plan():
    """Living (left), Kitchen (right-top), Bed (right-bottom)."""
    plan = GridPlan.from_feet(30, 40)
    right = plan.split(1, "v", units.cells(15), name="right")
    plan.rename(1, "Living", "living_room")
    bed = plan.split(right, "h", units.cells(20), name="Bed")
    plan.rename(right, "Kitchen", "kitchen")
    plan.rename(bed, "Bed", "bedroom")
    return plan


ROOMS3 = [("Living", "living_room"), ("Kitchen", "kitchen"),
          ("Bed", "bedroom")]


class TestFreespaceRules(unittest.TestCase):
    def test_fsp001_detects_room_hanging_off_private(self):
        # Bed2 reachable only through Bed1 — not opening onto free space
        plan = GridPlan.from_feet(30, 40)
        right = plan.split(1, "v", units.cells(15), name="right")
        plan.rename(1, "Living", "living_room")
        bed2 = plan.split(right, "h", units.cells(20), name="Bed2")
        plan.rename(right, "Bed1", "bedroom")
        plan.rename(bed2, "Bed2", "bedroom")
        rooms = [("Living", "living_room"), ("Bed1", "bedroom"),
                 ("Bed2", "bedroom")]
        verdict = check(plan, rooms,
                        door_pairs=[("Living", "Bed1", "b"),
                                    ("Bed1", "Bed2", "b")],
                        entrance="Living",
                        config=EngineConfig(fsp001_hard=True))
        self.assertTrue(any("FSP-001" in h and "Bed2" in h
                            for h in verdict.hard), verdict.hard)

    def test_fsp001_soft_mode_records_violations(self):
        plan = GridPlan.from_feet(30, 40)
        right = plan.split(1, "v", units.cells(15), name="right")
        plan.rename(1, "Living", "living_room")
        bed2 = plan.split(right, "h", units.cells(20), name="Bed2")
        plan.rename(right, "Bed1", "bedroom")
        plan.rename(bed2, "Bed2", "bedroom")
        rooms = [("Living", "living_room"), ("Bed1", "bedroom"),
                 ("Bed2", "bedroom")]
        verdict = check(plan, rooms,
                        door_pairs=[("Living", "Bed1", "b"),
                                    ("Bed1", "Bed2", "b")],
                        entrance="Living")     # default: soft severity
        self.assertFalse(any("FSP-001" in h for h in verdict.hard))
        self.assertGreaterEqual(
            verdict.breakdown["freespace_violations"], 1)

    def test_fsp001_accepts_attached_bath(self):
        plan = GridPlan.from_feet(30, 40)
        right = plan.split(1, "v", units.cells(15), name="right")
        plan.rename(1, "Living", "living_room")
        bath = plan.split(right, "h", units.cells(28), name="Bath")
        plan.rename(right, "Bed", "bedroom")
        plan.rename(bath, "Bath", "bathroom")
        rooms = [("Living", "living_room"), ("Bed", "bedroom"),
                 ("Bath", "bathroom")]
        verdict = check(plan, rooms,
                        door_pairs=[("Living", "Bed", "b"),
                                    ("Bed", "Bath", "b")],
                        entrance="Living")
        self.assertFalse(any("FSP-001" in h for h in verdict.hard),
                         verdict.hard)

    def test_fsp002_flags_corridor_waste(self):
        # a passage eating ~45% of the floor
        plan = GridPlan.from_feet(30, 40)
        hall = plan.split(1, "h", units.cells(20), name="Passage")
        plan.rename(1, "Living", "living_room")
        plan.rename(hall, "Passage", "passage")
        rooms = [("Living", "living_room"), ("Passage", "passage")]
        verdict = check(plan, rooms, wide_pairs=[("Living", "Passage")],
                        entrance="Passage")
        self.assertGreater(verdict.breakdown["circulation_fraction"], 0.14)

    def test_fsp003_flags_narrow_wide_opening(self):
        plan = three_room_plan()
        ids = {n: plan.find_room(n).id for n, _ in ROOMS3}
        plan.add_opening(ids["Living"], ids["Kitchen"], "wide",
                         width=20)                    # 30" — sub-walkable
        request = EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S",
            rooms=[RoomSpec(n, t, 100) for n, t in ROOMS3])
        plan.add_opening(ids["Living"], ids["Bed"], "door")
        plan.add_exterior_opening(ids["Living"], "S")
        verdict = BasicValidator().check(plan, request, ids)
        self.assertGreaterEqual(verdict.breakdown["narrow_passages"], 1)

    def test_fsp004_reports_hub_distance(self):
        plan = three_room_plan()
        verdict = check(plan, ROOMS3,
                        wide_pairs=[("Living", "Kitchen")],
                        door_pairs=[("Living", "Bed", "b")],
                        entrance="Living")
        self.assertIn("hub_mean_dist", verdict.breakdown)
        self.assertEqual(verdict.breakdown["hub_mean_dist"], 1.0)


def parking_plan():
    """Living (left), Parking (right) — both span the full depth and so
    both touch the S exterior wall (the request's entrance_side)."""
    plan = GridPlan.from_feet(30, 40)
    right = plan.split(1, "v", units.cells(15), name="right")
    plan.rename(1, "Living", "living_room")
    plan.rename(right, "Parking", "parking")
    return plan


PARKING_ROOMS = [("Living", "living_room"), ("Parking", "parking")]


class TestParkingVehicleGate(unittest.TestCase):
    def test_str003_detects_missing_vehicle_gate(self):
        plan = parking_plan()
        verdict = check(plan, PARKING_ROOMS, entrance="Living")
        self.assertTrue(any("STR-003" in h for h in verdict.hard),
                        verdict.hard)

    def test_str003_passes_with_vehicle_gate(self):
        plan = parking_plan()
        verdict = check(plan, PARKING_ROOMS, entrance="Living",
                        gate="Parking")
        self.assertFalse(any("STR-003" in h for h in verdict.hard),
                         verdict.hard)

    def test_fsp001_does_not_require_parking_interior_opening(self):
        # parking has ONLY its exterior vehicle gate — no interior opening
        # onto the free space at all — and must not be flagged for it,
        # even under fsp001_hard=True (the strictest routing).
        plan = parking_plan()
        verdict = check(plan, PARKING_ROOMS, entrance="Living",
                        gate="Parking", config=EngineConfig(fsp001_hard=True))
        self.assertFalse(any("FSP-001" in h for h in verdict.hard),
                         verdict.hard)


class TestWallRules(unittest.TestCase):
    def _four_room_plan(self, right_split_offset_cells):
        """A(living) | B(dining) on top, C/D bedrooms below; the right
        band's horizontal wall is offset by the given amount."""
        plan = GridPlan.from_feet(30, 40)
        right = plan.split(1, "v", units.cells(15), name="R")
        low_l = plan.split(1, "h", units.cells(20), name="LL")
        low_r = plan.split(right, "h",
                           units.cells(20) + right_split_offset_cells,
                           name="LR")
        for rid, (n, t) in zip((1, right, low_l, low_r),
                               [("A", "living_room"), ("B", "dining_room"),
                                ("C", "bedroom"), ("D", "bedroom")]):
            plan.rename(rid, n, t)
        rooms = [("A", "living_room"), ("B", "dining_room"),
                 ("C", "bedroom"), ("D", "bedroom")]
        return plan, rooms

    def test_wal002_detects_jog(self):
        plan, rooms = self._four_room_plan(4)      # 6" offset = jog
        verdict = check(plan, rooms, wide_pairs=[("A", "B")],
                        door_pairs=[("A", "C", "b"), ("B", "D", "b")],
                        entrance="C")
        self.assertFalse(verdict.hard, verdict.hard)
        self.assertGreaterEqual(verdict.breakdown["wall_jogs"], 1)

    def test_wal003_detects_toothpick_run(self):
        plan, rooms = self._four_room_plan(units.cells(1))  # 1' offset
        verdict = check(plan, rooms, wide_pairs=[("A", "B")],
                        door_pairs=[("A", "C", "b"), ("B", "D", "b")],
                        entrance="C")
        self.assertFalse(verdict.hard, verdict.hard)
        self.assertGreaterEqual(verdict.breakdown["toothpick_walls"], 1)

    def test_wal001_efficiency_reported(self):
        plan = three_room_plan()
        verdict = check(plan, ROOMS3, wide_pairs=[("Living", "Kitchen")],
                        door_pairs=[("Living", "Bed", "b")],
                        entrance="Living")
        self.assertIn("wall_efficiency", verdict.breakdown)
        self.assertGreater(verdict.breakdown["wall_efficiency"], 0.5)


class TestDoorRules(unittest.TestCase):
    def test_dor001_detects_swing_collision(self):
        # two doors into Bed's top-left corner from PERPENDICULAR walls —
        # their leaf sweeps overlap inside the bedroom
        plan = three_room_plan()
        ids = {n: plan.find_room(n).id for n, _ in ROOMS3}
        plan.rename(ids["Kitchen"], "Kitchen", "dining_room")
        walls = plan.shared_walls()
        lb = next(sw for sw in walls
                  if {sw.room_a, sw.room_b} == {ids["Living"], ids["Bed"]})
        kb = next(sw for sw in walls
                  if {sw.room_a, sw.room_b} == {ids["Kitchen"], ids["Bed"]})
        swing_lb = "a" if lb.room_a == ids["Bed"] else "b"
        swing_kb = "a" if kb.room_a == ids["Bed"] else "b"
        plan.add_opening(lb.room_a, lb.room_b, "door", swing=swing_lb,
                         at=lb.along_lo + 6)     # vertical wall, top end
        plan.add_opening(kb.room_a, kb.room_b, "door", swing=swing_kb,
                         at=kb.along_lo + 6)     # horizontal wall, left end
        plan.add_opening(ids["Living"], ids["Kitchen"], "wide")
        plan.add_exterior_opening(ids["Living"], "S")
        retyped = [("Living", "living_room"), ("Kitchen", "dining_room"),
                   ("Bed", "bedroom")]
        request = EngineRequest(plot_w_ft=30, plot_h_ft=40,
                                entrance_side="S",
                                rooms=[RoomSpec(n, t, 100)
                                       for n, t in retyped])
        verdict = BasicValidator().check(plan, request, ids)
        self.assertTrue(any("DOR-001" in h for h in verdict.hard),
                        verdict.hard)

    def test_dor002_detects_cornered_hinge(self):
        plan = three_room_plan()
        ids = {n: plan.find_room(n).id for n, _ in ROOMS3}
        walls = plan.shared_walls()
        lb = next(sw for sw in walls
                  if {sw.room_a, sw.room_b} == {ids["Living"], ids["Bed"]})
        plan.add_opening(lb.room_a, lb.room_b, "door", stub=2,
                         at=lb.along_lo + 2)         # 3" from the corner
        verdict = check(plan, ROOMS3, wide_pairs=[("Living", "Kitchen")],
                        entrance="Living")
        self.assertGreaterEqual(
            verdict.breakdown["hinge_stub_violations"], 1)

    def test_dor003_detects_wrong_swing(self):
        plan = three_room_plan()
        ids = {n: plan.find_room(n).id for n, _ in ROOMS3}
        walls = plan.shared_walls()
        lb = next(sw for sw in walls
                  if {sw.room_a, sw.room_b} == {ids["Living"], ids["Bed"]})
        wrong = "a" if plan.rooms[lb.room_a].rtype == "living_room" else "b"
        plan.add_opening(lb.room_a, lb.room_b, "door", swing=wrong)
        verdict = check(plan, ROOMS3, wide_pairs=[("Living", "Kitchen")],
                        entrance="Living")
        self.assertGreaterEqual(verdict.breakdown["wrong_swing_doors"], 1)

    def test_dor004_rewards_cross_ventilation(self):
        plan = three_room_plan()
        ids = {n: plan.find_room(n).id for n, _ in ROOMS3}
        verdict_before = check(plan, ROOMS3,
                               wide_pairs=[("Living", "Kitchen")],
                               door_pairs=[("Living", "Bed", "b")],
                               entrance="Living")
        # bed has a door on the h-wall; give it a window on the v exterior
        plan.add_exterior_opening(ids["Bed"], "E", kind="window",
                                  width=units.cells(4))
        request = EngineRequest(plot_w_ft=30, plot_h_ft=40,
                                entrance_side="S",
                                rooms=[RoomSpec(n, t, 100)
                                       for n, t in ROOMS3])
        verdict = BasicValidator().check(plan, request, ids)
        self.assertGreater(verdict.breakdown["cross_vent_rooms"],
                           verdict_before.breakdown["cross_vent_rooms"])


class TestExistingRulesStillFire(unittest.TestCase):
    def test_opn001_kitchen_door(self):
        plan = three_room_plan()
        verdict = check(plan, ROOMS3,
                        door_pairs=[("Living", "Kitchen", "b"),
                                    ("Living", "Bed", "b")],
                        entrance="Living")
        self.assertTrue(any("OPN-001" in h for h in verdict.hard))

    def test_cir002_unreachable(self):
        plan = three_room_plan()
        verdict = check(plan, ROOMS3, wide_pairs=[("Living", "Kitchen")],
                        entrance="Living")     # Bed has no opening at all
        self.assertTrue(any("CIR-002" in h for h in verdict.hard))


if __name__ == "__main__":
    unittest.main()
