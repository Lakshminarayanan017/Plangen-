"""
test_program_synth.py — the layer that decides WHAT to build.

The safety property here mirrors the Vastu one: a program that declares no
size ceilings must behave exactly as it did before phase 02
(`TestNoCapsUnchanged`). Everything else is the new behaviour — sizing the
footprint to the program instead of the program to the plot.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from modules.step3_enrich.program_synth import (
    DEFAULT_MAX_MULTIPLE, REGIME_COMPACT_MAX, REGIME_SPACIOUS_MIN,
    capacity_for, max_sqft_for, plan_program, regime_for, surplus_after_caps,
)
from modules.step4_generate.carve.standards import clamp_to_range


def _room(rtype, name, area, implicit=False):
    return NS(room_type=rtype, display_name=name, target_area_sqft=area,
              implicit_room=implicit)


def _program(beds=3):
    rooms = [_room("living_room", "Living Room", 150),
             _room("kitchen", "Kitchen", 75),
             _room("master_bedroom", "Master Bedroom", 120)]
    for i in range(2, beds + 1):
        rooms.append(_room("bedroom", f"Bedroom {i}", 105))
    rooms += [_room("bathroom", "Bath 1", 35),
              _room("bathroom", "Bath 2", 35, implicit=True),
              _room("utility_room", "Utility Room", 45, implicit=True),
              _room("store_room", "Store Room", 45, implicit=True)]
    return rooms


class TestCapacityCurve(unittest.TestCase):
    """The curve is a MEASUREMENT (14 plot sizes x 7 programs, largest that
    scored >= 60), so these assert the shape of that measurement."""

    def test_capacity_is_monotone(self):
        seen = []
        for sqft in (200, 350, 500, 700, 1000, 1300, 1800, 2500, 4000):
            _, beds = capacity_for(sqft)
            seen.append(beds)
        self.assertEqual(seen, sorted(seen), "capacity must never shrink")

    def test_known_points(self):
        self.assertEqual(capacity_for(250)[0], "studio")
        self.assertEqual(capacity_for(400)[0], "1BHK")
        self.assertEqual(capacity_for(550)[0], "2BHK")
        self.assertEqual(capacity_for(1300)[0], "3BHK")
        self.assertEqual(capacity_for(9999)[0], "4BHK")

    def test_regimes_partition(self):
        self.assertEqual(regime_for(REGIME_COMPACT_MAX - 1), "compact")
        self.assertEqual(regime_for(REGIME_COMPACT_MAX + 1), "normal")
        self.assertEqual(regime_for(REGIME_SPACIOUS_MIN - 1), "normal")
        self.assertEqual(regime_for(REGIME_SPACIOUS_MIN + 1), "spacious")


class TestSizeCeilings(unittest.TestCase):
    def test_a_bathroom_cannot_become_a_bedroom(self):
        """The bug this exists for: on a 60x70 plot a 45 sqft bath was
        realised at 266."""
        self.assertLessEqual(max_sqft_for("bathroom", 45), 70.0)

    def test_living_rooms_have_more_headroom_than_bathrooms(self):
        living = max_sqft_for("living_room", 200) / 200
        bath = max_sqft_for("bathroom", 45) / 45
        self.assertGreater(living, bath)

    def test_absolute_ceiling_beats_the_multiple(self):
        # 200 sqft bath x 1.35 would be 270; the absolute ceiling is 70
        self.assertLessEqual(max_sqft_for("bathroom", 200), 70.0)

    def test_unknown_type_gets_the_default(self):
        self.assertAlmostEqual(max_sqft_for("gym_room", 100),
                               100 * DEFAULT_MAX_MULTIPLE, places=1)

    def test_ceiling_never_below_target(self):
        for rtype in ("bathroom", "living_room", "kitchen", "staircase"):
            for area in (10, 50, 200, 800):
                self.assertGreaterEqual(
                    max(area, max_sqft_for(rtype, area)), area)


class TestPlanProgram(unittest.TestCase):
    def test_surplus_adds_rooms_instead_of_inflating(self):
        plan = plan_program(_program(3), 1700, 1)
        self.assertTrue(plan.added)
        self.assertIn("Dining Room", plan.added)
        self.assertIn("rather than inflating", plan.headline)

    def test_compact_drops_inferred_rooms_only(self):
        plan = plan_program(_program(3), 420, 1)
        self.assertTrue(plan.dropped)
        # every dropped room was implicit; none was user-requested
        explicit = {"Living Room", "Kitchen", "Master Bedroom",
                    "Bedroom 2", "Bedroom 3", "Bath 1"}
        self.assertFalse(set(plan.dropped) & explicit,
                         f"dropped a requested room: {plan.dropped}")

    def test_over_capacity_warns_with_numbers(self):
        plan = plan_program(_program(3), 420, 1)
        warns = [d for d in plan.decisions if d.action == "warn"]
        self.assertTrue(warns)
        self.assertIn("bedroom", warns[0].reason)

    def test_a_fitting_program_is_left_alone(self):
        plan = plan_program(_program(2), 900, 1)
        self.assertEqual(plan.dropped, [])

    def test_multi_floor_judges_total_area(self):
        """A 3BHK across two small floors is a different question from a
        3BHK on one small floor."""
        one = plan_program(_program(3), 500, 1)
        two = plan_program(_program(3), 500, 2)
        self.assertEqual(one.net_sqft, 500)
        self.assertEqual(two.net_sqft, 1000)
        self.assertGreaterEqual(capacity_for(two.net_sqft)[1],
                                capacity_for(one.net_sqft)[1])

    def test_flags_disable_each_half(self):
        self.assertEqual(plan_program(_program(3), 1700, 1,
                                      allow_add=False).added, [])
        self.assertEqual(plan_program(_program(3), 420, 1,
                                      allow_drop=False).dropped, [])

    def test_does_not_mutate_its_input(self):
        rooms = _program(3)
        before = [r.display_name for r in rooms]
        plan_program(rooms, 420, 1)
        self.assertEqual([r.display_name for r in rooms], before)

    def test_serialises(self):
        import json
        blob = json.dumps(plan_program(_program(3), 1700, 1).to_dict())
        self.assertIn("capacity", blob)
        self.assertIn("headline", blob)

    def test_surplus_after_caps(self):
        rooms = _program(2)
        self.assertEqual(surplus_after_caps(rooms, 300.0), 0.0)
        self.assertGreater(surplus_after_caps(rooms, 5000.0), 0.0)


class TestClampToRange(unittest.TestCase):
    def test_total_is_preserved(self):
        got, _ = clamp_to_range({"a": 500.0, "b": 500.0},
                                {"a": 40.0, "b": 100.0},
                                {"a": 100.0, "b": 900.0})
        self.assertEqual(sum(got.values()), 1000)

    def test_ceiling_is_respected_and_surplus_moves(self):
        got, over = clamp_to_range({"bath": 500.0, "living": 500.0},
                                   {"bath": 40.0, "living": 100.0},
                                   {"bath": 100.0, "living": 900.0})
        self.assertEqual(got["bath"], 100)
        self.assertEqual(got["living"], 900)
        self.assertEqual(over, 0.0)

    def test_unplaceable_surplus_is_reported(self):
        got, over = clamp_to_range({"bath": 500.0, "living": 500.0},
                                   {"bath": 40.0, "living": 100.0},
                                   {"bath": 100.0, "living": 300.0})
        self.assertGreater(over, 0.0)
        self.assertEqual(got["bath"], 100)
        self.assertEqual(got["living"], 300)

    def test_minimums_still_win(self):
        got, _ = clamp_to_range({"a": 10.0, "b": 990.0},
                                {"a": 200.0, "b": 100.0},
                                {"a": 1000.0, "b": 1000.0})
        self.assertGreaterEqual(got["a"], 200)


class TestNoCapsUnchanged(unittest.TestCase):
    """A request declaring no ceilings must behave exactly as it did before
    phase 02 — the settler's original proportional distribution."""

    def test_settle_targets_identical_without_caps(self):
        from modules.step4_generate.carve.standards import (
            clamp_to_minimums, nbc_min_area_cells,
        )
        from modules.step4_generate.core.grid_plan import GridPlan
        from modules.step4_generate.engine.contracts import (
            EngineRequest, RoomSpec,
        )
        from modules.step4_generate.engine.settle import (
            scaled_targets_with_overflow,
        )

        plan = GridPlan.from_feet(40, 50)
        rid = plan.split(1, "h", plan.h // 2, name="Bath", rtype="bathroom")
        plan.rename(1, "Living", "living_room")
        ids = {"Living": 1, "Bath": rid}
        request = EngineRequest(
            plot_w_ft=40, plot_h_ft=50, entrance_side="S",
            rooms=[RoomSpec("Living", "living_room", 200, "public"),
                   RoomSpec("Bath", "bathroom", 45, "private")])

        got, overflow = scaled_targets_with_overflow(plan, request, ids)
        self.assertEqual(overflow, 0.0)

        total = sum(plan.face_area_cells(r) for r in ids.values())
        raw = {ids[s.name]: total * s.target_sqft / 245.0
               for s in request.rooms}
        floors = {ids[s.name]: float(nbc_min_area_cells(s.rtype) or 0)
                  for s in request.rooms}
        expected = {r: max(1, v)
                    for r, v in clamp_to_minimums(raw, floors).items()}
        self.assertEqual(got, expected)

    def test_caps_actually_bind(self):
        from modules.step4_generate.core import units
        from modules.step4_generate.core.grid_plan import GridPlan
        from modules.step4_generate.engine.contracts import (
            EngineRequest, RoomSpec,
        )
        from modules.step4_generate.engine.settle import (
            scaled_targets_with_overflow,
        )

        plan = GridPlan.from_feet(40, 50)
        rid = plan.split(1, "h", plan.h // 2, name="Bath", rtype="bathroom")
        plan.rename(1, "Living", "living_room")
        ids = {"Living": 1, "Bath": rid}
        request = EngineRequest(
            plot_w_ft=40, plot_h_ft=50, entrance_side="S",
            rooms=[RoomSpec("Living", "living_room", 200, "public",
                            max_sqft=400.0),
                   RoomSpec("Bath", "bathroom", 45, "private",
                            max_sqft=60.0)])
        got, overflow = scaled_targets_with_overflow(plan, request, ids)
        self.assertLessEqual(units.area_sqft(got[rid]), 61.0)
        self.assertGreater(overflow, 0.0,
                           "a capped program on a big plot must report "
                           "the area it cannot absorb")


class TestFootprintFit(unittest.TestCase):
    """Sizing the building to the program — a 2BHK on a large plot is a
    2BHK house in a garden, not a 2BHK stretched over the whole plot."""

    def _plan(self, pw, pl, beds=2):
        from models import (
            BuildingRequirements, EnrichedPlan, EnrichedRoom, FloorPlan,
            Setbacks,
        )
        from modules.step2_match.indian_standards import get_room_minimums

        def mk(rid, rtype, name, area, width):
            n = get_room_minimums(rtype)
            area = max(area, n["min_area_sqft"])
            width = max(width, n["min_width_ft"])
            return EnrichedRoom(
                room_id=rid, room_type=rtype, display_name=name,
                target_width_ft=width,
                target_length_ft=round(area / width, 2),
                target_area_sqft=area, min_width_ft=n["min_width_ft"],
                min_length_ft=n["min_width_ft"],
                min_area_sqft=n["min_area_sqft"],
                max_area_sqft=max_sqft_for(rtype, area),
                ceiling_height_ft=9.0, preferred_floor=0)

        rooms = [mk("living_1", "living_room", "Living Room", 200, 14),
                 mk("kitchen_1", "kitchen", "Kitchen", 100, 9),
                 mk("master_bedroom_1", "master_bedroom",
                    "Master Bedroom", 160, 12)]
        for i in range(2, beds + 1):
            rooms.append(mk(f"bedroom_{i}", "bedroom", f"Bedroom {i}",
                            130, 11))
        rooms.append(mk("bathroom_1", "bathroom", "Bath 1", 45, 5))
        nw, nl = pw - 6, pl - 8
        return EnrichedPlan(
            original_requirements=BuildingRequirements(),
            plot_width_ft=pw, plot_length_ft=pl, plot_area_sqft=pw * pl,
            setbacks=Setbacks(front=5, rear=3, left=3, right=3),
            net_buildable_width_ft=nw, net_buildable_length_ft=nl,
            net_buildable_area_sqft=nw * nl, total_floors=1,
            floors=[FloorPlan(floor_number=0, floor_label="Ground Floor",
                              room_ids=[r.room_id for r in rooms])],
            rooms=rooms, max_ground_coverage_sqft=pw * pl * 0.6,
            max_far_total_sqft=pw * pl * 1.5)

    def test_big_plot_small_program_shrinks_the_building(self):
        from api.engine_bridge import _footprint_for_program
        ep = self._plan(60, 70)
        w, h, notes = _footprint_for_program(ep, 54, 62, multi=False)
        self.assertLess(w * h, 54 * 62,
                        "a 2BHK must not be spread over 3,348 sqft")
        self.assertTrue(notes)
        self.assertIn("open ground", notes[0])

    def test_matched_plot_is_left_alone(self):
        from api.engine_bridge import _footprint_for_program
        ep = self._plan(30, 40)
        w, h, notes = _footprint_for_program(ep, 24, 32, multi=False)
        self.assertEqual((w, h), (24, 32))
        self.assertEqual(notes, [])

    def test_aspect_is_preserved(self):
        from api.engine_bridge import _footprint_for_program
        ep = self._plan(60, 70)
        w, h, _ = _footprint_for_program(ep, 54, 62, multi=False)
        self.assertAlmostEqual(w / h, 54 / 62, delta=0.12)

    def test_never_shrinks_below_the_engine_minimum(self):
        from api.engine_bridge import MIN_FOOTPRINT_FT, _footprint_for_program
        ep = self._plan(200, 200, beds=1)
        w, h, _ = _footprint_for_program(ep, 190, 190, multi=False)
        self.assertGreaterEqual(w, MIN_FOOTPRINT_FT)
        self.assertGreaterEqual(h, MIN_FOOTPRINT_FT)


class TestCapacityAdvice(unittest.TestCase):
    def test_failure_message_is_actionable(self):
        """"engine produced no valid plan" tells a builder nothing."""
        from api.engine_bridge import _capacity_advice
        ep = TestFootprintFit()._plan(20, 26, beds=2)
        advice = _capacity_advice(ep, 0, 16, 20)
        self.assertIn("sqft of rooms", advice)
        self.assertIn("carries a", advice)
        self.assertIn("Reduce the room count", advice)


if __name__ == "__main__":
    unittest.main()
