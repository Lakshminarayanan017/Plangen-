import unittest

import numpy as np

from modules.step4_generate.engine.contracts import EngineRequest, OpeningWish, RoomSpec
from modules.step4_generate.engine.fallbacks import PriorProposer
from modules.step4_generate.engine.orchestrator import Orchestrator


def make_request(k=4, seed=1):
    return EngineRequest(
        plot_w_ft=30, plot_h_ft=40, entrance_side="S",
        rooms=[
            RoomSpec("Living Room", "living_room", 190, zone="public"),
            RoomSpec("Dining", "dining_room", 130, zone="service"),
            RoomSpec("Kitchen", "kitchen", 110, zone="service"),
            RoomSpec("Master Bedroom", "master_bedroom", 165, zone="private"),
            RoomSpec("Bedroom 2", "bedroom", 140, zone="private"),
            RoomSpec("Bath 1", "bathroom", 45, zone="private"),
        ],
        wishes=[OpeningWish("Dining", "Kitchen", "wide")],
        k=k, seed=seed,
    )


class TestPriorProposer(unittest.TestCase):
    def test_all_rooms_placed_once(self):
        req = make_request()
        prop = PriorProposer().propose(req)
        names = [p.room for p in prop.placements]
        self.assertEqual(sorted(names), sorted(s.name for s in req.rooms))

    def test_public_zone_leads_generation_order(self):
        prop = PriorProposer().propose(make_request())
        self.assertEqual(prop.placements[0].room, "Living Room")

    def test_entrance_side_moves_public_zone(self):
        req_s = make_request()
        req_n = make_request()
        req_n.entrance_side = "N"
        seed_s = PriorProposer().propose(req_s).placements[0].seed_cell
        seed_n = PriorProposer().propose(req_n).placements[0].seed_cell
        self.assertGreater(seed_s[0], 16)   # public near south (high row)
        self.assertLess(seed_n[0], 16)      # public near north (low row)

    def test_deterministic(self):
        a = PriorProposer().propose(make_request(), variant=2)
        b = PriorProposer().propose(make_request(), variant=2)
        self.assertEqual(a.placements, b.placements)


class TestOrchestrator(unittest.TestCase):
    def test_end_to_end_produces_ranked_candidates(self):
        result = Orchestrator().generate(make_request(k=4))
        self.assertGreaterEqual(len(result.ranked), 1)
        for cand in result.ranked:
            self.assertFalse(cand.verdict.hard)
            self.assertIsNotNone(cand.fidelity)
            self.assertTrue(0.0 <= cand.fidelity <= 1.0)
            cand.plan.verify()

    def test_ranking_is_sorted(self):
        result = Orchestrator().generate(make_request(k=5))
        scores = [c.verdict.soft_score for c in result.ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deterministic_given_seed(self):
        r1 = Orchestrator().generate(make_request(k=3, seed=9))
        r2 = Orchestrator().generate(make_request(k=3, seed=9))
        self.assertTrue(np.array_equal(r1.best.plan.grid, r2.best.plan.grid))

    def test_variants_produce_diversity(self):
        result = Orchestrator().generate(make_request(k=6))
        grids = [c.plan.grid.tobytes() for c in result.ranked]
        self.assertGreater(len(set(grids)), 1)

    def test_telemetry_present(self):
        result = Orchestrator().generate(make_request())
        self.assertIn("mean_fidelity", result.tier_notes)
        self.assertIn("kept/discarded/dup", result.tier_notes)
        self.assertIn("total_ms", result.tier_notes)
        for cand in result.ranked:
            self.assertIn("settle", cand.timings_ms)


if __name__ == "__main__":
    unittest.main()
