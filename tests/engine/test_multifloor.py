"""
Multi-floor tests (M7): the vertical rules, the reservation that makes stair
continuity possible, and the end-to-end G+1 gate.

M7's gate: G+1 briefs produce a building where stacking and continuity rules
pass. The load-bearing assertion is stair overlap — VRT-001 is hard, and it
is only satisfiable because the carver RESERVES the footprint and the
settler is forbidden from optimizing it away.
"""

import logging
import unittest

from modules.step4_generate.carve.reserved import carve_with_reservation, isolate, snap_reservation
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import CarveError, GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest, RoomSpec
from modules.step4_generate.engine.multifloor import (
    BuildingResult, FloorContext, floor_request, generate_building,
)
from modules.step4_generate.engine.vertical import STAIR_OVERLAP_MIN, check_stacking

logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)


GROUND = [
    RoomSpec("Living Room", "living_room", 260, zone="public"),
    RoomSpec("Kitchen", "kitchen", 110, zone="service"),
    RoomSpec("Dining Room", "dining_room", 130, zone="service"),
    RoomSpec("Bedroom 2", "bedroom", 150),
    RoomSpec("Bath 1", "bathroom", 50),
]
FIRST = [
    RoomSpec("Master Bedroom", "master_bedroom", 190, zone="private"),
    RoomSpec("Bedroom 3", "bedroom", 150),
    RoomSpec("Bath 2", "bathroom", 50),
    RoomSpec("Passage", "hallway", 60, zone="private"),
]


def _base(w=30, h=50, side="S", seed=7):
    return EngineRequest(plot_w_ft=w, plot_h_ft=h, entrance_side=side,
                         rooms=[], n_floors=2, k=6, seed=seed,
                         name=f"{w}x{h}_{side}")


class TestReservation(unittest.TestCase):
    def test_isolate_produces_the_exact_rectangle(self):
        plan = GridPlan.from_feet(30, 50)
        rect = (units.cells(10), units.cells(14),
                units.cells(18), units.cells(26))
        face, regions = isolate(plan, rect, "S")
        self.assertEqual(plan.face_bbox(face), rect)
        self.assertTrue(regions)
        plan.verify()

    def test_isolate_handles_a_reservation_against_the_edge(self):
        plan = GridPlan.from_feet(30, 50)
        ext = plan.ext_wall
        rect = (plan.w - ext - units.cells(8), units.cells(14),
                plan.w - ext, units.cells(24))
        face, _ = isolate(plan, rect, "S")
        self.assertEqual(plan.face_bbox(face), rect)
        plan.verify()

    def test_isolate_refuses_a_reservation_outside_the_plot(self):
        plan = GridPlan.from_feet(30, 50)
        with self.assertRaises(CarveError):
            isolate(plan, (0, 0, units.cells(8), units.cells(10)), "S")

    def test_snap_absorbs_an_unusable_sliver(self):
        plan = GridPlan.from_feet(30, 50)
        ext = plan.ext_wall
        # 1'6" from the east wall — no room could live in the leftover
        rect = (plan.w - ext - units.cells(8) - units.cells(1, 6),
                units.cells(14),
                plan.w - ext - units.cells(1, 6), units.cells(24))
        snapped = snap_reservation(plan, rect)
        self.assertEqual(snapped[2], plan.w - ext)

    def test_carve_with_reservation_places_every_room(self):
        from modules.step4_generate.engine.contracts import LayoutProposal, Placement
        plan = GridPlan.from_feet(32, 50)
        rooms = [RoomSpec("Staircase", "staircase", 75, zone="service"),
                 RoomSpec("Master Bedroom", "master_bedroom", 190),
                 RoomSpec("Bedroom 3", "bedroom", 150),
                 RoomSpec("Bath 2", "bathroom", 50)]
        request = EngineRequest(plot_w_ft=32, plot_h_ft=50,
                                entrance_side="S", rooms=rooms, n_floors=2)
        proposal = LayoutProposal(placements=[
            Placement("Staircase", (16, 24), 3),
            Placement("Master Bedroom", (6, 8), 8),
            Placement("Bedroom 3", (22, 8), 6),
            Placement("Bath 2", (26, 24), 2)], source="test")
        rect = (units.cells(20), units.cells(18),
                units.cells(28), units.cells(30))
        ids = carve_with_reservation(plan, request, proposal,
                                     reserved_room="Staircase", rect=rect,
                                     config=EngineConfig())
        self.assertEqual(set(ids), {s.name for s in rooms})
        self.assertEqual(plan.face_bbox(ids["Staircase"]),
                         snap_reservation(plan, rect))
        plan.verify()


class TestVerticalRules(unittest.TestCase):
    """One planted defect per rule, on hand-built floors."""

    @staticmethod
    def _floor(stair_x0_ft, rtypes=("living_room", "staircase")):
        plan = GridPlan.from_feet(30, 40)
        stair = plan.split(1, "v", units.cells(stair_x0_ft),
                           name="Staircase", rtype="staircase")
        plan.rename(1, "Living Room", rtypes[0])
        ids = {"Living Room": 1, "Staircase": stair}
        plan.add_exterior_opening(1, "S")
        plan.add_opening(1, stair, "wide")
        return plan, ids

    def _request(self, floor=1):
        return EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S", n_floors=2,
            floor_index=floor,
            rooms=[RoomSpec("Living Room", "living_room", 300, zone="public"),
                   RoomSpec("Staircase", "staircase", 80, zone="service")])

    def test_identical_floors_pass_vrt001(self):
        low, low_ids = self._floor(18)
        up, up_ids = self._floor(18)
        verdict = check_stacking(low, low_ids, up, up_ids)
        self.assertTrue(verdict.ok, verdict.hard)
        self.assertAlmostEqual(verdict.breakdown["stair_overlap"], 1.0)

    def test_a_moved_stair_fails_vrt001(self):
        low, low_ids = self._floor(18)
        up, up_ids = self._floor(10)
        verdict = check_stacking(low, low_ids, up, up_ids)
        self.assertFalse(verdict.ok)
        self.assertTrue(any("VRT-001" in h for h in verdict.hard))
        self.assertLess(verdict.breakdown["stair_overlap"], STAIR_OVERLAP_MIN)

    def test_a_missing_upper_stair_fails_vrt001(self):
        low, low_ids = self._floor(18)
        up = GridPlan.from_feet(30, 40)
        up.rename(1, "Hall", "living_room")
        up.add_exterior_opening(1, "S")
        verdict = check_stacking(low, low_ids, up, {"Hall": 1})
        self.assertTrue(any("VRT-001" in h for h in verdict.hard))

    def test_a_different_footprint_fails_vrt002(self):
        low, low_ids = self._floor(18)
        up, up_ids = self._floor(18)
        wide = GridPlan.from_feet(34, 40)
        verdict = check_stacking(low, low_ids, wide, {"x": 1})
        self.assertTrue(any("VRT-002" in h for h in verdict.hard))

    def test_wet_stacking_is_measured(self):
        low, low_ids = self._floor(18)
        up, up_ids = self._floor(18)
        low.rooms[low_ids["Living Room"]].rtype = "bathroom"
        up.rooms[up_ids["Living Room"]].rtype = "bathroom"
        verdict = check_stacking(low, low_ids, up, up_ids)
        self.assertAlmostEqual(verdict.breakdown["wet_stack_fraction"], 1.0)

    def test_unstacked_wet_rooms_cost_score(self):
        low, low_ids = self._floor(18)
        up, up_ids = self._floor(18)
        low.rooms[low_ids["Living Room"]].rtype = "bathroom"
        up.rooms[up_ids["Staircase"]].rtype = "bathroom"
        verdict = check_stacking(low, low_ids, up, up_ids)
        self.assertLess(verdict.breakdown["wet_stack_fraction"], 0.5)
        self.assertLess(verdict.soft_score, 100.0)

    def test_wall_alignment_is_measured(self):
        low, low_ids = self._floor(18)
        up, up_ids = self._floor(18)
        verdict = check_stacking(low, low_ids, up, up_ids)
        self.assertGreater(verdict.breakdown["wall_alignment"], 0.9)


class TestUpperFloorEntrance(unittest.TestCase):
    def test_an_upper_floor_is_entered_from_the_stair_not_the_street(self):
        request = floor_request(_base(), 1, FIRST)
        from modules.step4_generate.engine.orchestrator import Orchestrator
        result = Orchestrator().generate(request)
        self.assertTrue(result.ranked)
        plan = result.best.plan
        exterior_doors = [op for op in plan.openings
                          if op.is_exterior and op.kind == "door"]
        self.assertEqual(exterior_doors, [],
                         "an upper floor must not have a front door")

    def test_the_ground_floor_still_gets_a_front_door(self):
        from modules.step4_generate.engine.orchestrator import Orchestrator
        result = Orchestrator().generate(floor_request(_base(), 0, GROUND))
        doors = [op for op in result.best.plan.openings
                 if op.is_exterior and op.kind == "door"]
        self.assertEqual(len(doors), 1)


class TestBuildingGate(unittest.TestCase):
    """M7's gate, end to end."""

    @classmethod
    def setUpClass(cls):
        cls.result: BuildingResult = generate_building(
            _base(), [GROUND, FIRST])

    def test_every_floor_is_planned(self):
        self.assertTrue(self.result.ok, self.result.warnings)
        self.assertEqual(len(self.result.floors), 2)

    def test_no_vertical_hard_violation_survives(self):
        for floor in self.result.floors:
            if floor.vertical is not None:
                self.assertEqual(floor.vertical.hard, [])

    def test_the_stair_lands_on_the_stair_below(self):
        upper = self.result.floors[1]
        self.assertGreaterEqual(upper.vertical.breakdown["stair_overlap"],
                                STAIR_OVERLAP_MIN)

    def test_the_reserved_stair_survives_settle(self):
        """Settle chases area targets; without freezing the reservation it
        moved the stair and overlap collapsed to 23-59%."""
        lower, upper = (f.candidate for f in self.result.floors)
        low_rect = next(lower.plan.face_bbox(rid)
                        for name, rid in lower.room_ids.items()
                        if lower.plan.rooms[rid].rtype == "staircase")
        up_rect = next(upper.plan.face_bbox(rid)
                       for name, rid in upper.room_ids.items()
                       if upper.plan.rooms[rid].rtype == "staircase")
        self.assertEqual(low_rect, up_rect)

    def test_the_score_reflects_vertical_agreement(self):
        self.assertGreater(self.result.score, 0.0)
        floors = [f.candidate.verdict.soft_score for f in self.result.floors]
        self.assertLessEqual(self.result.score,
                             sum(floors) / len(floors) + 1e-6)

    def test_summary_is_serializable(self):
        import json
        json.dumps(self.result.summary())

    def test_the_orchestrator_is_left_as_found(self):
        from modules.step4_generate.engine.fallbacks import PriorProposer
        from modules.step4_generate.engine.orchestrator import Orchestrator
        orch = Orchestrator()
        proposer, realizer, settler = (orch.proposer, orch.realizer,
                                       orch.settler)
        generate_building(_base(seed=11), [GROUND, FIRST], orchestrator=orch)
        self.assertIs(orch.proposer, proposer)
        self.assertIs(orch.realizer, realizer)
        self.assertIsInstance(proposer, PriorProposer)

    def test_floor_context_reports_the_stair(self):
        ground = self.result.floors[0].candidate
        ctx = FloorContext.from_plan(ground.plan, ground.room_ids)
        self.assertIsNotNone(ctx.stair_rect)
        self.assertEqual(ctx.stair_name, "Staircase")


if __name__ == "__main__":
    unittest.main()
