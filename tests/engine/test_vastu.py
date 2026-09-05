"""
test_vastu.py — the Vastu layer's contracts.

The load-bearing test in this file is `TestNonVastuUnaffected`: a request
that did not ask for Vastu must be scored EXACTLY as it was before any of
this existed. A feature that silently changes every other plan is not a
feature, it is a regression with a marketing name.
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import (
    EngineConfig, EngineRequest, RoomSpec,
)
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.engine.vastu import mandala as mandala_mod
from modules.step4_generate.engine.vastu import report as report_mod
from modules.step4_generate.engine.vastu.compass import (
    CompassFrame, normalize_direction, sector_distance,
)


def _rooms(vastu: bool):
    def V(d=None, avoid=(), s="soft"):
        return (dict(vastu_dir=d, vastu_avoid=avoid, vastu_strength=s)
                if vastu else {})
    return [
        RoomSpec("Living Room", "living_room", 200, "public", **V("NE", ("SW",))),
        RoomSpec("Kitchen", "kitchen", 110, "service", **V("SE", ("NE",))),
        RoomSpec("Master Bedroom", "master_bedroom", 170, "private",
                 **V("SW", ("NE", "SE"))),
        RoomSpec("Pooja Room", "pooja_room", 40, "public",
                 **V("NE", ("S", "SW", "SE"), "hard")),
        RoomSpec("Bath 1", "bathroom", 45, "private", **V("NW", ("NE", "SW"))),
    ]


def _request(vastu: bool, **kw) -> EngineRequest:
    base = dict(plot_w_ft=40, plot_h_ft=50, entrance_side="N",
                north_side="N", k=4, seed=99, rooms=_rooms(vastu),
                vastu=vastu)
    base.update(kw)
    return EngineRequest(**base)


class TestCompassFrame(unittest.TestCase):
    H, W = 90, 60

    def test_every_sector_round_trips_in_every_orientation(self):
        for north in ("N", "E", "S", "W"):
            frame = CompassFrame(north)
            for sector in ("N", "NE", "E", "SE", "S", "SW", "W", "NW",
                           "center"):
                row, col = frame.sector_center(sector, self.H, self.W)
                self.assertEqual(frame.sector_of(row, col, self.H, self.W),
                                 sector, f"north={north} sector={sector}")

    def test_north_up_frame_is_the_intuitive_one(self):
        frame = CompassFrame("N")
        self.assertEqual(frame.sector_of(2, 2, self.H, self.W), "NW")
        self.assertEqual(frame.sector_of(self.H - 3, self.W - 3,
                                         self.H, self.W), "SE")
        self.assertEqual(frame.sector_of(self.H // 2, self.W // 2,
                                         self.H, self.W), "center")

    def test_rotation_actually_rotates(self):
        """With north on the EAST edge, the grid's right-middle IS north."""
        frame = CompassFrame("E")
        self.assertEqual(
            frame.sector_of(self.H // 2, self.W - 3, self.H, self.W), "N")
        self.assertEqual(frame.compass_of_grid_side("E"), "N")
        self.assertEqual(frame.grid_side_of_compass("N"), "E")

    def test_side_mappings_are_inverse(self):
        for north in ("N", "E", "S", "W"):
            frame = CompassFrame(north)
            for side in ("N", "E", "S", "W"):
                self.assertEqual(
                    frame.grid_side_of_compass(
                        frame.compass_of_grid_side(side)), side)

    def test_brahmasthan_is_one_ninth_and_orientation_invariant(self):
        rects = set()
        for north in ("N", "E", "S", "W"):
            rect = CompassFrame(north).brahmasthan_rect(self.H, self.W)
            rects.add(rect)
            x0, y0, x1, y1 = rect
            share = (x1 - x0) * (y1 - y0) / (self.H * self.W)
            self.assertAlmostEqual(share, 1 / 9, delta=0.02)
        self.assertEqual(len(rects), 1, "the centre is the centre, rotated")

    def test_pada_corners(self):
        frame = CompassFrame("N")
        self.assertEqual(frame.pada_of(0, 0, self.H, self.W), (0, 8))
        self.assertEqual(frame.pada_of(self.H - 1, self.W - 1,
                                       self.H, self.W), (8, 0))

    def test_sector_distance(self):
        self.assertEqual(sector_distance("NE", "NE"), 0)
        self.assertEqual(sector_distance("NE", "N"), 1)
        self.assertEqual(sector_distance("NE", "SW"), 4)
        self.assertEqual(sector_distance("N", "NW"), 1)   # wraps
        self.assertEqual(sector_distance("NE", "center"), 2)

    def test_normalize_direction(self):
        self.assertEqual(normalize_direction("north_east"), "NE")
        self.assertEqual(normalize_direction(" ne "), "NE")
        self.assertEqual(normalize_direction("Brahmasthan"), "center")
        self.assertIsNone(normalize_direction("upwards"))
        self.assertIsNone(normalize_direction(None))

    def test_bad_north_rejected(self):
        with self.assertRaises(ValueError):
            CompassFrame("up")


class TestMandala(unittest.TestCase):
    def setUp(self):
        self.m = mandala_mod.shared()

    def test_loads_the_full_system(self):
        self.assertTrue(self.m.available)
        self.assertEqual(len(self.m.gates), 32)
        self.assertEqual(len(self.m.marma), 6)
        self.assertIsNotNone(self.m.brahma)

    def test_gate_lookup_by_pada(self):
        gate = self.m.gate_at(0, 8)           # NW corner
        self.assertIsNotNone(gate)
        self.assertEqual(gate.id, "N1")
        self.assertTrue(gate.is_blocked)

    def test_every_side_offers_an_ideal_pada(self):
        """Advice depends on this: there must always be somewhere better to
        put the door than a blocked pada."""
        for side in ("N", "E", "S", "W"):
            self.assertTrue(self.m.best_gates(side),
                            f"no ideal gate on the {side} side")

    def test_known_data_defects_are_reported_not_swallowed(self):
        """W3 is 'Sugriva' but a marma line calls it 'Mukhya'; W7 is 'Shok'
        but a marma line calls it 'Shosha'. Both resolve by id — the point is
        that the disagreement is surfaced."""
        joined = " ".join(self.m.warnings)
        self.assertIn("W3", joined)
        self.assertIn("W7", joined)
        self.assertIn("resolved by id", joined)

    def test_marma_endpoints_all_resolved(self):
        for line in self.m.marma:
            self.assertIn(line.from_id, self.m.gates)
            self.assertIn(line.to_id, self.m.gates)

    def test_missing_file_degrades_quietly(self):
        absent = mandala_mod.load("/nonexistent/vastu.json")
        self.assertFalse(absent.available)
        self.assertEqual(absent.gates, {})
        self.assertTrue(absent.warnings)


class TestNonVastuUnaffected(unittest.TestCase):
    """The safety property: adding Vastu must not move a non-Vastu plan."""

    def test_no_vas_breakdown_when_vastu_off(self):
        result = Orchestrator(config=EngineConfig()).generate(_request(False))
        self.assertTrue(result.best)
        vas = [k for k in result.best.verdict.breakdown if k.startswith("vastu")]
        self.assertEqual(vas, [], f"VAS rules fired on a non-Vastu request: {vas}")

    def test_vastu_flag_without_directions_is_inactive(self):
        """A Vastu request whose rooms carry no directions is a DATA failure
        (the enricher warns about it). It must not be scored as compliance."""
        request = _request(False)
        request = EngineRequest(**{**request.to_dict(),
                                   "rooms": _rooms(False), "vastu": True})
        result = Orchestrator(config=EngineConfig()).generate(request)
        self.assertTrue(result.best)
        vas = [k for k in result.best.verdict.breakdown if k.startswith("vastu")]
        self.assertEqual(vas, [])

    def test_default_north_is_the_old_implicit_assumption(self):
        self.assertEqual(EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S",
            rooms=[RoomSpec("A", "bedroom", 100)]).north_side, "N")


class TestVastuRules(unittest.TestCase):
    def _best(self, **kw):
        result = Orchestrator(config=EngineConfig(**kw)).generate(
            _request(True))
        self.assertTrue(result.best, "vastu request produced no plan")
        return result.best

    def test_rules_fire_and_record_evidence(self):
        breakdown = self._best().verdict.breakdown
        for key in ("vastu_sector_steps", "vastu_prohibited",
                    "vastu_brahma_intrusion", "vastu_marma_hits",
                    "vastu_mass_sw", "vastu_mass_ne"):
            self.assertIn(key, breakdown)

    def test_bias_reaches_the_proposer(self):
        """Regression: Orchestrator built PriorProposer() with no config, so
        `vastu_bias` silently did nothing at every value and a sweep over it
        returned five identical rows. Assert the knob CHANGES something —
        a `<=` here passes vacuously when the knob is dead."""
        off = Orchestrator(config=EngineConfig(vastu_bias=0.0)).generate(
            _request(True)).best
        on = Orchestrator(config=EngineConfig(vastu_bias=0.7)).generate(
            _request(True)).best
        self.assertNotEqual(
            on.proposal.placements, off.proposal.placements,
            "vastu_bias produced identical proposals — the config is not "
            "reaching the proposer")

    def test_bias_improves_placement(self):
        """With the bias on, rooms land closer to their Vastu sector."""
        off = Orchestrator(config=EngineConfig(vastu_bias=0.0)).generate(
            _request(True)).best
        on = Orchestrator(config=EngineConfig(vastu_bias=0.7)).generate(
            _request(True)).best
        self.assertLess(
            on.verdict.breakdown["vastu_sector_steps"],
            off.verdict.breakdown["vastu_sector_steps"],
            "the vastu bias did not improve sector placement")

    def test_hard_mode_can_disqualify(self):
        """`vastu_hard` promotes a HARD-strength breach to a hard violation.
        Verified on a synthetic plan rather than by hoping the carver
        produces one."""
        from modules.step4_generate.engine.rules.base import ReviewContext
        from modules.step4_generate.engine.rules.vastu import prohibited_sector

        plan = GridPlan.from_feet(40, 40)
        rid = plan.split(1, "h", plan.h // 2, name="Pooja Room",
                         rtype="pooja_room")
        plan.rename(1, "Living Room", "living_room")
        # the new face is the southern half -> a barred sector for pooja
        spec = RoomSpec("Pooja Room", "pooja_room", 40, "public",
                        vastu_dir="NE", vastu_avoid=("S", "SW", "SE"),
                        vastu_strength="hard")
        request = EngineRequest(
            plot_w_ft=40, plot_h_ft=40, entrance_side="N", north_side="N",
            vastu=True, rooms=[spec])
        ctx = ReviewContext(plan=plan, request=request,
                            room_ids={"Pooja Room": rid},
                            config=EngineConfig(vastu_hard=True))
        prohibited_sector(ctx)
        self.assertTrue(ctx.hard, "hard mode did not disqualify a barred room")
        self.assertIn("VAS-002", ctx.hard[0])

    def test_soft_mode_penalises_but_does_not_disqualify(self):
        from modules.step4_generate.engine.rules.base import ReviewContext
        from modules.step4_generate.engine.rules.vastu import prohibited_sector

        plan = GridPlan.from_feet(40, 40)
        rid = plan.split(1, "h", plan.h // 2, name="Pooja Room",
                         rtype="pooja_room")
        spec = RoomSpec("Pooja Room", "pooja_room", 40, "public",
                        vastu_dir="NE", vastu_avoid=("S", "SW", "SE"),
                        vastu_strength="hard")
        request = EngineRequest(
            plot_w_ft=40, plot_h_ft=40, entrance_side="N", north_side="N",
            vastu=True, rooms=[spec])
        ctx = ReviewContext(plan=plan, request=request,
                            room_ids={"Pooja Room": rid},
                            config=EngineConfig(vastu_hard=False))
        prohibited_sector(ctx)
        self.assertEqual(ctx.hard, [])
        self.assertGreater(ctx.penalty, 0.0)


class TestVastuReport(unittest.TestCase):
    def setUp(self):
        request = _request(True)
        self.best = Orchestrator(config=EngineConfig()).generate(request).best
        self.report = report_mod.build(self.best.plan,
                                       self.best.request or request,
                                       self.best.room_ids, self.best.verdict)

    def test_active_and_graded(self):
        self.assertTrue(self.report.active)
        self.assertIn(self.report.grade, list("ABCDE"))
        self.assertGreaterEqual(self.report.score, 0.0)
        self.assertLessEqual(self.report.score, 1.0)

    def test_every_room_accounted_for(self):
        named = {r.room for r in self.report.rooms}
        for spec in self.best.request.rooms:
            self.assertIn(spec.name, named)

    def test_misses_carry_advice(self):
        for room in self.report.rooms:
            if room.status in ("near", "off", "barred"):
                self.assertTrue(room.advice,
                                f"{room.room} missed with no advice")

    def test_serialises(self):
        import json
        blob = json.dumps(self.report.to_dict())
        self.assertIn("brahmasthan", blob)
        self.assertIn("grade", blob)

    def test_inactive_report_says_so(self):
        request = _request(False)
        best = Orchestrator(config=EngineConfig()).generate(request).best
        rep = report_mod.build(best.plan, request, best.room_ids, best.verdict)
        self.assertFalse(rep.active)
        self.assertIn("not requested", report_mod.format_text(rep))


class TestBridgeMapping(unittest.TestCase):
    def test_diagonal_north_resolves_and_warns(self):
        from api.engine_bridge import _north_side
        warnings = []
        self.assertEqual(_north_side("north_east", warnings), "N")
        self.assertTrue(warnings)
        self.assertIn("45", warnings[0])

    def test_cardinal_north_is_silent(self):
        from api.engine_bridge import _north_side
        warnings = []
        self.assertEqual(_north_side("east", warnings), "E")
        self.assertEqual(warnings, [])

    def test_missing_north_defaults_to_N(self):
        from api.engine_bridge import _north_side
        self.assertEqual(_north_side(None, []), "N")

    def test_room_vastu_extraction(self):
        from api.engine_bridge import _room_vastu
        from models import VastuConstraint

        class Room:
            vastu = VastuConstraint(preferred_directions=["NE"],
                                    prohibited_directions=["S", "SW"],
                                    constraint_type="hard")
        got = _room_vastu(Room())
        self.assertEqual(got["vastu_dir"], "NE")
        self.assertEqual(got["vastu_avoid"], ("S", "SW"))
        self.assertEqual(got["vastu_strength"], "hard")

    def test_room_without_vastu_yields_nothing(self):
        from api.engine_bridge import _room_vastu

        class Room:
            vastu = None
        self.assertEqual(_room_vastu(Room()), {})


if __name__ == "__main__":
    unittest.main()
