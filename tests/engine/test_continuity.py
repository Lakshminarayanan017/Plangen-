"""
test_continuity.py — "the same house, with the change I asked for".

The property under test: an edit re-runs the pipeline, and re-running must
not hand the user a different house. Seed-alone made resemblance likely;
ranking candidates by resemblance makes it a property of the selection.

Measured before these were written (30x45 brief, four edits):

    edit                                   w=0    w=0.35
    living room much bigger                94%      94%
    add a study room                       66%      66%
    move master bedroom to the north-east  42%     100%
    kitchen 200 sqft + dining smaller      40%      92%

which is why the module exists at all, and why it is honest about doing
nothing for a simple resize.
"""

from __future__ import annotations

import unittest

from modules.step4_generate.engine import continuity as cont
from modules.step4_generate.engine.contracts import Candidate, Verdict


def _sig(**rooms) -> cont.LayoutSignature:
    """rooms: name=(cx, cy, area_share)."""
    return cont.LayoutSignature(rooms={
        name: cont.RoomFootprint(cx, cy, 0.2, 0.2, share)
        for name, (cx, cy, share) in rooms.items()})


class _Plan:
    """Minimal GridPlan stand-in: room id -> (x0, y0)."""
    w = h = 100

    def __init__(self, positions):
        self.positions = positions

    def face_bbox(self, rid):
        x, y = self.positions[rid]
        return (x, y, x + 20, y + 20)


def _cand(score, positions, room_ids=None):
    return Candidate(plan=_Plan(positions), proposal=None,
                     verdict=Verdict(hard=[], soft_score=score),
                     room_ids=room_ids or {"a": 0, "b": 1})


class TestCompare(unittest.TestCase):
    def test_identical_layouts_score_one(self):
        a = _sig(kitchen=(0.2, 0.2, 0.5), living=(0.8, 0.8, 0.5))
        self.assertAlmostEqual(cont.compare(a, a).similarity, 1.0, places=3)

    def test_swapped_rooms_score_low(self):
        a = _sig(kitchen=(0.1, 0.1, 0.5), living=(0.9, 0.9, 0.5))
        b = _sig(kitchen=(0.9, 0.9, 0.5), living=(0.1, 0.1, 0.5))
        self.assertLess(cont.compare(a, b).similarity, 0.2)

    def test_displacement_is_area_weighted(self):
        """Moving the living room is a different plan; nudging a bathroom is
        the same plan. An unweighted mean would score those alike."""
        base = _sig(living=(0.2, 0.2, 0.8), bath=(0.8, 0.8, 0.2))
        moved_big = _sig(living=(0.8, 0.8, 0.8), bath=(0.8, 0.8, 0.2))
        moved_small = _sig(living=(0.2, 0.2, 0.8), bath=(0.2, 0.2, 0.2))
        self.assertLess(cont.compare(base, moved_big).similarity,
                        cont.compare(base, moved_small).similarity)

    def test_added_rooms_do_not_count_as_displacement(self):
        """An edit that adds a study SHOULD change the room set."""
        before = _sig(kitchen=(0.2, 0.2, 0.5), living=(0.8, 0.8, 0.5))
        after = _sig(kitchen=(0.2, 0.2, 0.4), living=(0.8, 0.8, 0.4),
                     study=(0.5, 0.5, 0.2))
        score = cont.compare(before, after)
        self.assertGreater(score.similarity, 0.9)
        self.assertEqual(score.gained, ["study"])
        self.assertEqual(score.lost, [])

    def test_removed_rooms_are_reported(self):
        before = _sig(kitchen=(0.2, 0.2, 0.5), store=(0.8, 0.8, 0.5))
        after = _sig(kitchen=(0.2, 0.2, 1.0))
        score = cont.compare(before, after)
        self.assertEqual(score.lost, ["store"])
        self.assertAlmostEqual(score.coverage, 0.5, places=3)

    def test_moved_list_names_what_shifted(self):
        before = _sig(kitchen=(0.1, 0.1, 0.5), living=(0.5, 0.5, 0.5))
        after = _sig(kitchen=(0.7, 0.7, 0.5), living=(0.5, 0.5, 0.5))
        score = cont.compare(before, after)
        self.assertEqual(score.moved, ["kitchen"])

    def test_empty_signatures_are_survivable(self):
        empty = cont.LayoutSignature()
        self.assertEqual(cont.compare(empty, _sig(a=(0, 0, 1))).similarity, 0)
        self.assertEqual(cont.compare(_sig(a=(0, 0, 1)), empty).similarity, 0)

    def test_no_shared_rooms(self):
        score = cont.compare(_sig(a=(0.1, 0.1, 1.0)), _sig(b=(0.1, 0.1, 1.0)))
        self.assertEqual(score.similarity, 0.0)
        self.assertEqual(score.coverage, 0.0)

    def test_names_match_case_insensitively(self):
        a = cont.LayoutSignature(rooms={
            "Master Bedroom": cont.RoomFootprint(0.2, 0.2, 0.2, 0.2, 1.0)})
        b = cont.LayoutSignature(rooms={
            "master bedroom": cont.RoomFootprint(0.2, 0.2, 0.2, 0.2, 1.0)})
        # signature builders normalise, so compare directly on normalised keys
        self.assertEqual(cont._norm_name("Master Bedroom"),
                         cont._norm_name("  master   bedroom "))
        self.assertNotEqual(list(a.names), list(b.names))


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.prev = _sig(a=(0.1, 0.1, 0.5), b=(0.8, 0.8, 0.5))
        # identical layout, mediocre score
        self.familiar = _cand(88.0, {0: (10, 10), 1: (80, 80)})
        # rearranged, marginally better score
        self.rearranged = _cand(92.0, {0: (70, 70), 1: (10, 10)})

    def test_without_a_reference_the_engine_order_is_untouched(self):
        ranked = cont.rank_candidates([self.rearranged, self.familiar],
                                      cont.LayoutSignature())
        self.assertIs(ranked[0][0], self.rearranged)

    def test_weight_zero_is_the_engine_order(self):
        ranked = cont.rank_candidates([self.rearranged, self.familiar],
                                      self.prev, weight=0.0)
        self.assertIs(ranked[0][0], self.rearranged)

    def test_a_close_second_that_preserves_the_plan_wins(self):
        ranked = cont.rank_candidates([self.rearranged, self.familiar],
                                      self.prev, weight=0.35)
        self.assertIs(ranked[0][0], self.familiar)

    def test_familiarity_cannot_buy_a_bad_plan(self):
        """The floor is the whole safety property: an edit must not be able
        to trade quality away for looking the same."""
        awful = _cand(50.0, {0: (10, 10), 1: (80, 80)})    # identical, bad
        good = _cand(95.0, {0: (70, 70), 1: (10, 10)})     # rearranged, good
        ranked = cont.rank_candidates([good, awful], self.prev, weight=0.35)
        self.assertIs(ranked[0][0], good)

    def test_floor_is_relative_to_the_best_available(self):
        """On a brief the engine finds hard, every candidate scores low and
        an absolute floor would reject them all."""
        a = _cand(30.0, {0: (10, 10), 1: (80, 80)})
        b = _cand(34.0, {0: (70, 70), 1: (10, 10)})
        ranked = cont.rank_candidates([b, a], self.prev, weight=0.35)
        self.assertIs(ranked[0][0], a, "the familiar one is within tolerance")

    def test_never_returns_empty_when_given_candidates(self):
        only = _cand(10.0, {0: (10, 10), 1: (80, 80)})
        ranked = cont.rank_candidates([only], self.prev, weight=0.35,
                                      quality_floor=99.0)
        self.assertEqual(len(ranked), 1)

    def test_ordering_is_by_blended_rank(self):
        ranked = cont.rank_candidates([self.rearranged, self.familiar],
                                      self.prev, weight=0.35)
        self.assertGreaterEqual(ranked[0][2], ranked[1][2])


class TestMultiFloorGroundPick(unittest.TestCase):
    """A G+1 edit has to keep the house too.

    The single-floor path ranked candidates for resemblance from the start;
    the multi-floor path did not look at `previous` at all, so "make the
    kitchen bigger" on a G+1 returned a different building. G+1 is the common
    case for the users this is built for.

    Floor 0 is the only floor with a free choice — every floor above is
    planned against the one below — so that is where continuity belongs, and
    the vertical rules must stay untouched above it.
    """

    def setUp(self):
        from modules.step4_generate.engine.contracts import EngineConfig
        self.config = EngineConfig()
        self.prev = _sig(a=(0.1, 0.1, 0.5), b=(0.8, 0.8, 0.5))
        self.familiar = _cand(88.0, {0: (10, 10), 1: (80, 80)})
        self.rearranged = _cand(92.0, {0: (70, 70), 1: (10, 10)})
        self.result = type("R", (), {
            "ranked": [self.rearranged, self.familiar], "discarded": []})()

    def _pick(self, **kw):
        from modules.step4_generate.engine.multifloor import _pick_floor
        return _pick_floor(self.result, None, self.config, 0, **kw)

    def test_ground_floor_keeps_the_layout_on_an_edit(self):
        chosen, vertical, notes = self._pick(previous=self.prev)
        self.assertIs(chosen, self.familiar)
        self.assertIsNone(vertical, "floor 0 has nothing below to check")
        self.assertTrue(any("kept the layout" in n for n in notes))

    def test_a_first_run_is_untouched(self):
        """No previous plan means the engine's own order, exactly as before."""
        chosen, _, notes = self._pick()
        self.assertIs(chosen, self.rearranged)
        self.assertEqual(notes, [])

    def test_weight_zero_is_the_engine_order(self):
        chosen, _, _ = self._pick(previous=self.prev, continuity_weight=0.0)
        self.assertIs(chosen, self.rearranged)

    def test_no_note_when_the_top_candidate_already_wins(self):
        """Saying "kept the layout you had" when nothing was traded away
        would be a claim about a decision that was never made."""
        self.result.ranked = [self.familiar, self.rearranged]
        chosen, _, notes = self._pick(previous=self.prev)
        self.assertIs(chosen, self.familiar)
        self.assertEqual(notes, [])

    def test_signature_reaches_the_building_result(self):
        from modules.step4_generate.engine.multifloor import BuildingResult
        self.assertIsNone(BuildingResult(floors=[]).continuity)


class TestSignatureBuilders(unittest.TestCase):
    def test_from_a_grid_plan(self):
        plan = _Plan({0: (10, 10), 1: (60, 60)})
        sig = cont.signature_of_plan(plan, {"Kitchen": 0, "Living Room": 1})
        self.assertEqual(set(sig.names), {"kitchen", "living room"})
        self.assertAlmostEqual(sum(r.area_share for r in sig.rooms.values()),
                               1.0, places=5)

    def test_from_a_layout_floor(self):
        from models import LayoutFloor, PlacedRoom
        floor = LayoutFloor(
            floor_number=0, net_width_ft=30.0, net_length_ft=40.0,
            rooms=[PlacedRoom(room_id="k", room_type="kitchen",
                              display_name="Kitchen", floor=0, x_ft=0.0,
                              y_ft=0.0, width_ft=10.0, length_ft=10.0,
                              area_sqft=100.0),
                   PlacedRoom(room_id="l", room_type="living_room",
                              display_name="Living Room", floor=0,
                              x_ft=10.0, y_ft=10.0, width_ft=20.0,
                              length_ft=20.0, area_sqft=400.0)])
        sig = cont.signature_of_floor(floor)
        self.assertEqual(set(sig.names), {"kitchen", "living room"})
        self.assertAlmostEqual(sig.rooms["living room"].area_share, 0.8,
                               places=3)

    def test_degenerate_floor_is_empty_not_a_crash(self):
        from models import LayoutFloor
        self.assertFalse(cont.signature_of_floor(
            LayoutFloor(floor_number=0, net_width_ft=0.0,
                        net_length_ft=0.0, rooms=[])))

    def test_the_two_builders_agree(self):
        """THE REGRESSION. `compare` puts two signatures in ONE coordinate
        space, so a signature built from a GridPlan and one built from the
        LayoutFloor that GridPlan became must land in the same frame.

        They did not: `_floor_from_plan` writes y_ft from the SW corner while
        the grid counts from the NW, so every room was mirrored about the
        horizontal centre line. The API edit path therefore ranked candidates
        by how UNLIKE the previous plan they were — the exact inverse of the
        property this module exists to provide, and invisible in the engine's
        own tests because both sides there are GridPlans."""
        from api.engine_bridge import _floor_from_plan
        from modules.step4_generate.core import units

        cpf = units.CELLS_PER_FOOT

        class _Room:
            def __init__(self, name, rtype):
                self.name, self.rtype = name, rtype

        class _P:
            """A 30x40ft plan with rooms deliberately OFF the centre line —
            a symmetric layout would pass under the mirrored frame too."""
            w, h = int(30 * cpf), int(40 * cpf)
            boxes = {0: (0, 0, int(10 * cpf), int(8 * cpf)),        # NW
                     1: (int(10 * cpf), int(30 * cpf),
                         int(30 * cpf), int(40 * cpf))}             # SE
            rooms = {0: _Room("Kitchen", "kitchen"),
                     1: _Room("Living Room", "living_room")}

            def face_bbox(self, rid):
                return self.boxes[rid]

            def area_sqft(self, rid):
                x0, y0, x1, y1 = self.boxes[rid]
                return (x1 - x0) * (y1 - y0) / (cpf * cpf)

        plan = _P()
        from_grid = cont.signature_of_plan(
            plan, {"Kitchen": 0, "Living Room": 1})
        from_floor = cont.signature_of_floor(_floor_from_plan(plan, 0, 30, 40))

        score = cont.compare(from_grid, from_floor)
        self.assertGreater(
            score.similarity, 0.99,
            "a plan compared with itself through the other builder must be "
            f"identical, got {score.describe()}")
        self.assertEqual(score.moved, [])

    def test_signature_round_trips_through_json(self):
        sig = _sig(kitchen=(0.2, 0.3, 0.6), living=(0.7, 0.8, 0.4))
        back = cont.signature_from_dict(cont.signature_to_dict(sig))
        self.assertAlmostEqual(cont.compare(sig, back).similarity, 1.0,
                               places=6)
        self.assertEqual(set(back.names), set(sig.names))

    def test_signature_from_junk_is_empty_not_a_crash(self):
        for blob in (None, {}, {"kitchen": "nonsense"}, {"a": [1, 2]}):
            self.assertFalse(cont.signature_from_dict(blob))

    def test_signature_is_scale_free(self):
        """Phase 02 can resize the FOOTPRINT when a room is added, so a
        signature in absolute cells would report a redesign when nothing
        actually moved."""
        small = _Plan({0: (10, 10), 1: (60, 60)})
        big = type("P", (), {"w": 200, "h": 200,
                             "face_bbox": lambda s, rid: (
                                 {0: (20, 20), 1: (120, 120)}[rid][0],
                                 {0: (20, 20), 1: (120, 120)}[rid][1],
                                 {0: (20, 20), 1: (120, 120)}[rid][0] + 40,
                                 {0: (20, 20), 1: (120, 120)}[rid][1] + 40)})()
        ids = {"Kitchen": 0, "Living Room": 1}
        a = cont.signature_of_plan(small, ids)
        b = cont.signature_of_plan(big, ids)
        self.assertGreater(cont.compare(a, b).similarity, 0.99)


if __name__ == "__main__":
    unittest.main()
