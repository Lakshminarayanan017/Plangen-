"""End-to-end tests for the E2+E3 production pipeline."""

import unittest

import numpy as np

from modules.step4_generate.carve.settle import settle
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineRequest, RoomSpec
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.engine.realizer import HubRealizer, StripRealizer
from modules.step4_generate.engine.fallbacks import PriorProposer
from modules.step4_generate.engine.settle import scaled_targets_cells


def request_20x45(k=4, seed=7):
    return EngineRequest(
        plot_w_ft=20, plot_h_ft=45, entrance_side="S",
        rooms=[
            RoomSpec("Parking", "parking", 118, zone="public"),
            RoomSpec("Drawing Room", "drawing_room", 80, zone="public"),
            RoomSpec("Dining Room", "dining_room", 82, zone="service"),
            RoomSpec("Kitchen", "kitchen", 62, zone="service"),
            RoomSpec("Bed Room 3", "bedroom", 90, zone="private"),
            RoomSpec("C. Bath 2", "bathroom", 20, zone="private"),
            RoomSpec("Bed Room 1", "bedroom", 82, zone="private"),
            RoomSpec("C. Bath 1", "bathroom", 26, zone="private"),
            RoomSpec("Bed Room 2", "bedroom", 90, zone="private"),
        ],
        k=k, seed=seed,
    )


class TestWallSliding(unittest.TestCase):
    def test_pair_slide_conserves_partition(self):
        plan = GridPlan.from_feet(30, 40)
        right = plan.split(1, "v", units.cells(15), name="Right")
        total_before = plan.face_area_cells(1) + plan.face_area_cells(right)
        w_before, _ = plan.clear_dims(1)
        sw = plan.shared_walls()[0]
        self.assertTrue(plan.slide_pair_wall(sw, 8, 16, 16))
        self.assertEqual(
            plan.face_area_cells(1) + plan.face_area_cells(right),
            total_before)
        w_left, _ = plan.clear_dims(1)
        self.assertEqual(w_left, w_before + 8)          # grew by 12"
        plan.verify()

    def test_pair_slide_respects_minimum(self):
        plan = GridPlan.from_feet(30, 40)
        plan.split(1, "v", units.cells(15), name="Right")
        sw = plan.shared_walls()[0]
        w_right = units.cells(15) - 6  # right side width in cells (approx)
        self.assertFalse(
            plan.slide_pair_wall(sw, w_right, 8, units.cells(4)))
        plan.verify()

    def test_line_slide_moves_whole_band(self):
        plan = GridPlan.from_feet(20, 45)
        bottom = plan.split(1, "h", units.cells(20), name="Bottom")
        plan.split(bottom, "v", units.cells(10), name="Bottom-Right")
        lines = plan.interior_lines()
        h_lines = [l for l in lines if l[0] == "h"]
        self.assertEqual(len(h_lines), 1)
        axis, pos = h_lines[0]
        heights_before = {
            rid: plan.face_bbox(rid)[3] - plan.face_bbox(rid)[1]
            for rid in plan.rooms
        }
        self.assertTrue(plan.slide_line(axis, pos, 4,
                                        {rid: 16 for rid in plan.rooms}))
        for rid in plan.rooms:
            h_after = plan.face_bbox(rid)[3] - plan.face_bbox(rid)[1]
            self.assertNotEqual(h_after, 0)
        plan.verify()
        # top grew by 4, bottoms shrank by 4
        self.assertEqual(
            plan.face_bbox(1)[3] - plan.face_bbox(1)[1],
            heights_before[1] + 4)

    def test_settle_reduces_area_error(self):
        plan = GridPlan.from_feet(30, 40)
        right = plan.split(1, "v", units.cells(10), name="Right")
        plan.rename(1, "Left")
        # want them equal-ish: targets 50/50 of total
        total = plan.face_area_cells(1) + plan.face_area_cells(right)
        targets = {1: total // 2, right: total // 2}
        err_before = abs(plan.face_area_cells(1) - targets[1]) / targets[1]
        err_after = settle(plan, targets, {1: 24, right: 24})
        self.assertLess(err_after, err_before)
        self.assertLess(err_after, 0.05)
        plan.verify()


class TestHubPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Orchestrator().generate(request_20x45(k=6))

    def test_produces_valid_candidates(self):
        self.assertGreaterEqual(len(self.result.ranked), 1)
        for cand in self.result.ranked:
            cand.plan.verify()
            self.assertFalse(cand.verdict.hard)

    def test_kitchen_is_doorless(self):
        for cand in self.result.ranked:
            plan = cand.plan
            kid = plan.find_room("Kitchen").id
            interior_ops = [op for op in plan.openings
                            if not op.is_exterior
                            and kid in (op.room_a, op.room_b)]
            self.assertTrue(interior_ops, "kitchen must connect somewhere")
            for op in interior_ops:
                self.assertEqual(op.kind, "wide",
                                 "kitchen must be wide-open, never doored")

    def test_main_entrance_exists_on_entrance_side(self):
        for cand in self.result.ranked:
            doors = [op for op in cand.plan.openings
                     if op.is_exterior and op.kind == "door"]
            self.assertEqual(len(doors), 1)
            # south side → wall band at the bottom of the grid
            self.assertEqual(doors[0].wall_hi, cand.plan.h)

    def test_parking_has_vehicle_gate(self):
        # request_20x45 is the narrowest realistic plot in the golden set
        # (20' wide) — proves the vehicle gate is achievable in practice,
        # not just defined as a rule.
        for cand in self.result.ranked:
            plan = cand.plan
            pid = plan.find_room("Parking").id
            gates = [op for op in plan.openings
                     if op.is_exterior and op.kind == "wide"
                     and op.room_a == pid]
            self.assertTrue(gates, "parking must have a vehicle gate")

    def test_main_entrance_not_hosted_by_parking(self):
        for cand in self.result.ranked:
            plan = cand.plan
            pid = plan.find_room("Parking").id
            doors = [op for op in plan.openings
                     if op.is_exterior and op.kind == "door"]
            self.assertTrue(all(op.room_a != pid for op in doors),
                            "pedestrian entrance must not be through "
                            "parking")

    def test_settle_calibrates_areas(self):
        best = self.result.best
        self.assertLess(best.verdict.breakdown["area_drift"], 0.15)

    def test_fidelity_beats_strip_baseline(self):
        req = request_20x45(k=4)
        proposer = PriorProposer()
        hub_fid, strip_fid = [], []
        for v in range(req.k):
            proposal = proposer.propose(req, variant=v)
            _, _, f_hub, _ = HubRealizer().realize(req, proposal)
            hub_fid.append(f_hub)
            try:
                _, _, f_strip, _ = StripRealizer().realize(req, proposal)
                strip_fid.append(f_strip)
            except Exception:
                strip_fid.append(0.0)
        self.assertGreater(sum(hub_fid) / len(hub_fid),
                           sum(strip_fid) / len(strip_fid))
        self.assertGreater(sum(hub_fid) / len(hub_fid), 0.70)

    def test_deterministic(self):
        again = Orchestrator().generate(request_20x45(k=6))
        self.assertTrue(np.array_equal(self.result.best.plan.grid,
                                       again.best.plan.grid))


class TestScaledTargets(unittest.TestCase):
    def test_targets_sum_to_actual_room_area(self):
        req = request_20x45()
        proposal = PriorProposer().propose(req)
        plan, room_ids, _, _ = HubRealizer().realize(req, proposal)
        targets = scaled_targets_cells(plan, req, room_ids)
        total_rooms = sum(plan.face_area_cells(r) for r in room_ids.values())
        self.assertAlmostEqual(sum(targets.values()) / total_rooms, 1.0,
                               delta=0.01)


class TestExpandedEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = Orchestrator().generate(request_20x45(k=4))

    def test_windows_on_habitable_rooms(self):
        plan = self.result.best.plan
        windows = [op for op in plan.openings
                   if op.is_exterior and op.kind == "window"]
        self.assertGreaterEqual(len(windows), 3)
        # front/rear rooms of a row house must be windowed
        breakdown = self.result.best.verdict.breakdown
        self.assertLessEqual(breakdown["windowless"], 3)

    def test_new_rule_breakdown_keys(self):
        b = self.result.best.verdict.breakdown
        for key in ("nbc_violations", "daylightless", "windowless",
                    "circ_excess_hops", "wet_social_doors"):
            self.assertIn(key, b)

    def test_priors_loaded_from_extraction(self):
        from modules.step4_generate.engine.priors import ZonePriors
        priors = ZonePriors()
        if priors.loaded:                       # extraction file present
            d = priors.depth_for("bedroom")
            self.assertIsNotNone(d)
            self.assertTrue(0.0 <= d <= 1.0)
        missing = ZonePriors(path="does/not/exist.json")
        self.assertFalse(missing.loaded)
        self.assertIsNone(missing.depth_for("bedroom"))

    def test_retry_and_topup_fill_k(self):
        # the 20x30 1bhk brief used to keep only 3/6 — retry + top-up
        # must deliver the requested k (or very nearly)
        from modules.step4_generate.engine.contracts import RoomSpec
        req = EngineRequest(
            plot_w_ft=20, plot_h_ft=30, entrance_side="S",
            rooms=[
                RoomSpec("Living Room", "living_room", 96, zone="public"),
                RoomSpec("Kitchen", "kitchen", 43, zone="service"),
                RoomSpec("Master Bedroom", "master_bedroom", 77,
                         zone="private"),
                RoomSpec("Bath 1", "bathroom", 22, zone="private"),
            ],
            k=6, seed=20260716,
        )
        result = Orchestrator().generate(req)
        # a 20x30 4-room plot has few DISTINCT layouts; dedup may cap the
        # count below k — but retry + top-up must deliver at least 4 real,
        # distinct, valid plans where the naive loop kept 3
        self.assertGreaterEqual(len(result.ranked), 4)
        grids = {c.plan.grid.tobytes() for c in result.ranked}
        self.assertEqual(len(grids), len(result.ranked))

    def test_request_validation_rejects_garbage(self):
        from modules.step4_generate.engine.contracts import EngineRequestError, RoomSpec
        bad = EngineRequest(
            plot_w_ft=5, plot_h_ft=40, entrance_side="Q",
            rooms=[RoomSpec("A", "bedroom", -10, zone="nowhere")],
        )
        with self.assertRaises(EngineRequestError):
            Orchestrator().generate(bad)


if __name__ == "__main__":
    unittest.main()
