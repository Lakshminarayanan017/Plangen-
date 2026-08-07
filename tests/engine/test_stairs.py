"""
Staircase tests (M2): lattice-exact geometry, program injection, fitting,
the STR-S flaw-injection suite, and the end-to-end G+1 gate.

The gate this file enforces (implementation_plan_v2.md M2): every multi-floor
brief produces a plan with a fitted staircase and zero STR-S hard violations.
"""

import logging
import unittest

from modules.step4_generate.carve import stairs as stair_geom
from modules.step4_generate.carve.standards import type_min_long, type_min_side
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest, RoomSpec
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.engine.program import inject_implicit_rooms, stair_area_sqft
from modules.step4_generate.engine.stairs import StairFitter
from modules.step4_generate.engine.validator import BasicValidator
from ml.harness.briefs import stair_briefs


class TestStairGeometry(unittest.TestCase):
    def test_riser_count_inside_the_code_band(self):
        for height in (8.5, 9.0, 10.0, 11.0, 12.0, 13.5):
            n, riser = stair_geom.riser_count(height)
            self.assertGreaterEqual(riser, stair_geom.MIN_RISER_IN, height)
            self.assertLessEqual(riser, stair_geom.MAX_RISER_IN, height)
            self.assertAlmostEqual(n * riser, height * 12, places=6)

    def test_standard_floor_height_matches_the_plan(self):
        n, riser = stair_geom.riser_count(10.0)
        self.assertEqual(n, 17)
        self.assertAlmostEqual(riser, 120 / 17, places=4)

    def test_every_dimension_is_on_the_lattice(self):
        for v in stair_geom.variants(10.0):
            for value in (*v.min_footprint, *v.comfort_footprint,
                          v.tread_cells, v.width_cells, v.landing_cells):
                self.assertEqual(value, int(value))
                self.assertGreater(value, 0)

    def test_tread_is_at_least_the_code_minimum(self):
        # NBC minimum going is 250 mm (9.84"); ours must be >= that AND on
        # the 1.5" lattice, which is why it is 10.5" and not a nominal 10".
        self.assertGreaterEqual(units.to_inches(stair_geom.TREAD_CELLS), 9.84)

    def test_treads_are_one_fewer_than_risers_per_flight(self):
        for v in stair_geom.variants(10.0):
            self.assertEqual(sum(v.treads_per_flight),
                             v.risers - v.flights)

    def test_comfort_footprint_adds_exactly_one_landing(self):
        for v in stair_geom.variants(10.0):
            self.assertEqual(v.comfort_footprint[1] - v.min_footprint[1],
                             v.landing_cells)

    def test_carvable_variant_is_not_a_corridor(self):
        v = stair_geom.carvable_variant(10.0)
        w, l = v.min_footprint
        self.assertLessEqual(max(w, l), 2 * min(w, l))
        self.assertEqual(v.kind, "dogleg")

    def test_riser_count_refuses_an_impossible_height(self):
        with self.assertRaises(stair_geom.StairError):
            stair_geom.riser_count(0.0)

    def test_narrow_flight_is_refused(self):
        with self.assertRaises(stair_geom.StairError):
            stair_geom.variants(10.0, width_cells=units.cells(2))


class TestStandardsStayInSync(unittest.TestCase):
    """The carver's minimums are DERIVED from the geometry — never typed."""

    def test_min_side_and_long_match_the_carvable_variant(self):
        v = stair_geom.carvable_variant(10.0)
        width, length = v.min_footprint
        self.assertEqual(type_min_side("staircase"), width)
        self.assertEqual(type_min_long("staircase"), length)

    def test_types_without_a_run_requirement_report_zero(self):
        self.assertEqual(type_min_long("bedroom"), 0)


class TestStairFitting(unittest.TestCase):
    @staticmethod
    def _face(w_ft, h_ft):
        plan = GridPlan.from_feet(w_ft + 2, h_ft + 2)
        plan.rename(1, "Staircase", "staircase")
        return plan

    def test_fits_a_dogleg_in_a_dogleg_shaped_face(self):
        plan = self._face(8, 12)
        flight = stair_geom.fit_face(plan, 1)
        self.assertIsNotNone(flight)
        self.assertEqual(flight.variant.kind, "dogleg")

    def test_fits_a_straight_run_in_a_corridor_face(self):
        plan = self._face(4, 20)
        flight = stair_geom.fit_face(plan, 1)
        self.assertIsNotNone(flight)
        self.assertEqual(flight.variant.kind, "straight")

    def test_rejects_a_face_that_is_merely_big_enough_in_area(self):
        # 9'x9' = 81 sqft, more than the dog-leg's 65 — and unusable,
        # because no flight's RUN fits. Area is not shape.
        plan = self._face(9, 9)
        self.assertIsNone(stair_geom.fit_face(plan, 1))

    def test_orientation_is_detected_both_ways(self):
        self.assertEqual(stair_geom.fit_face(self._face(8, 14), 1).run_axis,
                         "h")
        self.assertEqual(stair_geom.fit_face(self._face(14, 8), 1).run_axis,
                         "v")

    def test_comfort_fit_is_preferred_when_the_face_allows_it(self):
        self.assertTrue(stair_geom.fit_face(self._face(8, 16), 1).both_landings)
        self.assertFalse(stair_geom.fit_face(self._face(8, 11), 1).both_landings)

    def test_fit_plan_attaches_and_unfitted_reports(self):
        plan = self._face(8, 12)
        flights = stair_geom.fit_plan(plan)
        self.assertEqual(len(flights), 1)
        self.assertIs(plan.stairs, flights)
        self.assertEqual(stair_geom.unfitted_faces(plan), [])

        bad = self._face(9, 9)
        stair_geom.fit_plan(bad)
        self.assertEqual(stair_geom.unfitted_faces(bad), [1])

    def test_tread_lines_stay_inside_the_face(self):
        plan = self._face(8, 14)
        flight = stair_geom.fit_face(plan, 1)
        x0, y0, x1, y1 = flight.rect
        lines = stair_geom.tread_lines(flight)
        self.assertGreater(len(lines), 8)
        for lx0, ly0, lx1, ly1 in lines:
            self.assertTrue(x0 <= lx0 <= x1 and x0 <= lx1 <= x1)
            self.assertTrue(y0 <= ly0 <= y1 and y0 <= ly1 <= y1)

    def test_higher_floor_needs_a_longer_run(self):
        short = stair_geom.carvable_variant(9.0).min_footprint[1]
        tall = stair_geom.carvable_variant(13.0).min_footprint[1]
        self.assertGreater(tall, short)


class TestProgramInjection(unittest.TestCase):
    @staticmethod
    def _request(n_floors=2, rooms=None):
        return EngineRequest(
            plot_w_ft=30, plot_h_ft=50, entrance_side="S", n_floors=n_floors,
            rooms=rooms or [
                RoomSpec("Living Room", "living_room", 220, zone="public"),
                RoomSpec("Kitchen", "kitchen", 90, zone="service"),
                RoomSpec("Master Bedroom", "master_bedroom", 160),
                RoomSpec("Bath 1", "bathroom", 45)])

    def test_single_floor_gets_no_staircase(self):
        req, notes = inject_implicit_rooms(self._request(n_floors=1))
        self.assertEqual([s.rtype for s in req.rooms].count("staircase"), 0)
        self.assertEqual(notes, [])

    def test_multi_floor_gets_exactly_one_staircase(self):
        req, notes = inject_implicit_rooms(self._request())
        stairs = [s for s in req.rooms if s.rtype == "staircase"]
        self.assertEqual(len(stairs), 1)
        self.assertGreaterEqual(stairs[0].target_sqft, stair_area_sqft())
        self.assertTrue(any("staircase injected" in n for n in notes))

    def test_injection_does_not_mutate_the_request(self):
        req = self._request()
        before = len(req.rooms)
        inject_implicit_rooms(req)
        self.assertEqual(len(req.rooms), before)

    def test_injection_is_idempotent(self):
        once, _ = inject_implicit_rooms(self._request())
        twice, notes = inject_implicit_rooms(once)
        self.assertEqual(len(once.rooms), len(twice.rooms))
        self.assertEqual(notes, [])

    def test_user_supplied_undersized_stair_is_raised_not_duplicated(self):
        rooms = [RoomSpec("Living Room", "living_room", 220, zone="public"),
                 RoomSpec("Stair", "stairs", 20.0, zone="service")]
        req, notes = inject_implicit_rooms(self._request(rooms=rooms))
        stairs = [s for s in req.rooms if s.rtype == "staircase"]
        self.assertEqual(len(stairs), 1)
        self.assertEqual(stairs[0].name, "Stair")
        self.assertGreaterEqual(stairs[0].target_sqft, stair_area_sqft())
        self.assertTrue(any("area raised" in n for n in notes))


def _stair_plan(face_w_ft, face_h_ft, connected=True, n_floors=2):
    """Living room (left) + a staircase face of EXACTLY face_w x face_h
    (right), entrance in the living room. `connected=False` leaves the
    stair sealed, which is STR-S03's planted defect."""
    ext = units.EXT_WALL_CELLS
    wall = units.INT_WALL_CELLS
    stair_w = units.cells(face_w_ft)
    plot_w = units.cells(face_w_ft + 20) + 2 * ext + wall
    plot_h = units.cells(face_h_ft) + 2 * ext

    plan = GridPlan(plot_w, plot_h)
    stair = plan.split(1, "v", plot_w - ext - wall - stair_w,
                       name="Staircase", rtype="staircase")
    plan.rename(1, "Living Room", "living_room")
    assert plan.clear_dims(stair) == (stair_w, plot_h - 2 * ext)

    rooms = [("Living Room", "living_room"), ("Staircase", "staircase")]
    ids = {name: plan.find_room(name).id for name, _ in rooms}
    plan.add_exterior_opening(ids["Living Room"], "S")
    if connected:
        plan.add_opening(ids["Living Room"], stair, "wide")
    request = EngineRequest(
        plot_w_ft=round(plot_w / units.CELLS_PER_FOOT),
        plot_h_ft=round(plot_h / units.CELLS_PER_FOOT),
        entrance_side="S", n_floors=n_floors,
        rooms=[RoomSpec(n, t, 120, zone="public") for n, t in rooms])
    return plan, request, ids


class TestStairRules(unittest.TestCase):
    """One planted defect per rule (the M1/M2 coverage law)."""

    def _check(self, plan, request, ids):
        StairFitter(EngineConfig()).fit(plan, request, ids)
        return BasicValidator(EngineConfig()).check(plan, request, ids)

    def test_str_s01_catches_a_missing_staircase(self):
        plan = GridPlan.from_feet(30, 40)
        plan.rename(1, "Living Room", "living_room")
        ids = {"Living Room": 1}
        request = EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S", n_floors=2,
            rooms=[RoomSpec("Living Room", "living_room", 300, zone="public")])
        plan.add_exterior_opening(1, "S")
        verdict = self._check(plan, request, ids)
        self.assertTrue(any("STR-S01" in h for h in verdict.hard))

    def test_str_s01_silent_on_single_floor(self):
        plan, request, ids = _stair_plan(8, 12, n_floors=1)
        verdict = self._check(plan, request, ids)
        self.assertFalse(any("STR-S01" in h for h in verdict.hard))

    def test_str_s02_catches_an_unbuildable_face(self):
        plan, request, ids = _stair_plan(9, 9)
        verdict = self._check(plan, request, ids)
        self.assertTrue(any("STR-S02" in h for h in verdict.hard))

    def test_str_s02_passes_a_buildable_face(self):
        plan, request, ids = _stair_plan(8, 12)
        verdict = self._check(plan, request, ids)
        self.assertFalse(any("STR-S02" in h for h in verdict.hard))

    def test_str_s03_catches_an_unreachable_stair(self):
        plan, request, ids = _stair_plan(8, 12, connected=False)
        verdict = self._check(plan, request, ids)
        self.assertTrue(any("STR-S03" in h for h in verdict.hard))

    def test_str_s04_penalizes_a_single_landing(self):
        cramped, req_c, ids_c = _stair_plan(8, 11)
        roomy, req_r, ids_r = _stair_plan(8, 16)
        v_cramped = self._check(cramped, req_c, ids_c)
        v_roomy = self._check(roomy, req_r, ids_r)
        self.assertEqual(v_cramped.breakdown.get("stair_single_landing"), 1)
        self.assertEqual(v_roomy.breakdown.get("stair_single_landing"), 0)

    def test_str_s05_penalizes_stealing_the_entrance_frontage(self):
        plan, request, ids = _stair_plan(8, 12)
        verdict = self._check(plan, request, ids)
        # the stair spans the full plot depth here, so it does reach the
        # S frontage — the rule must have measured it
        self.assertIn("stair_frontage_ft", verdict.breakdown)


class TestMultiFloorGate(unittest.TestCase):
    """M2's gate: every G+1 brief plans, with a fitted stair and no STR-S."""

    @classmethod
    def setUpClass(cls):
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)
        orch = Orchestrator()
        cls.results = [(b, orch.generate(b)) for b in stair_briefs(k=4)]

    def test_every_multi_floor_brief_produces_a_plan(self):
        failed = [b.name for b, r in self.results if not r.best]
        self.assertEqual(failed, [])

    def test_every_plan_has_a_fitted_staircase(self):
        for brief, result in self.results:
            plan = result.best.plan
            self.assertEqual(len(plan.stairs), 1, brief.name)
            self.assertEqual(stair_geom.unfitted_faces(plan), [], brief.name)

    def test_no_str_s_hard_violation_survives(self):
        for brief, result in self.results:
            for cand in result.ranked:
                bad = [h for h in cand.verdict.hard if h.startswith("STR-S")]
                self.assertEqual(bad, [], brief.name)

    def test_fitted_flight_fits_inside_its_face(self):
        for brief, result in self.results:
            for flight in result.best.plan.stairs:
                x0, y0, x1, y1 = flight.rect
                fw, fl = flight.variant.min_footprint
                face = sorted((x1 - x0, y1 - y0))
                need = sorted((fw, fl))
                self.assertGreaterEqual(face[0], need[0], brief.name)
                self.assertGreaterEqual(face[1], need[1], brief.name)

    def test_renderer_draws_the_fitted_flight(self):
        from modules.step4_generate.render.svg_render import render_svg
        svg = render_svg(self.results[0][1].best.plan)
        self.assertIn("stairup", svg)


if __name__ == "__main__":
    unittest.main()
