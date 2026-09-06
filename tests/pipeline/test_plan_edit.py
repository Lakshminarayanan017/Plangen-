"""
test_plan_edit.py — turning "make the kitchen bigger" into a plan change.

Two bugs found while building this are pinned below, because both were
plausible-looking and silently wrong:

  * `describe()` crashed on an absolute size, because the value is a tuple
    for "150 sqft" and a float for "bigger".
  * "put the kitchen next to the dining room" resolved the WRONG room:
    `find_room` returns the longest phrase in a string, so it answered
    "dining room" and the second lookup then found nothing. Adjacency is
    positional and is now parsed by splitting at the marker.

The governing rule for the whole module: nothing is silently ignored. An
instruction that cannot be parsed comes back in `unparsed`.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api.server import app, sessions
from modules.step3_enrich import plan_edit


class TestParsing(unittest.TestCase):
    def _one(self, text):
        plan = plan_edit.parse(text)
        self.assertEqual(len(plan.intents), 1,
                         f"expected one intent from {text!r}, got "
                         f"{[i.action for i in plan.intents]}")
        return plan.intents[0]

    def test_relative_resize(self):
        i = self._one("make the kitchen bigger")
        self.assertEqual((i.action, i.target), ("resize", "kitchen"))
        self.assertGreater(i.value, 1.0)

    def test_much_bigger_beats_bigger(self):
        """Longest-phrase-first: 'much bigger' must not match as 'bigger'."""
        small = self._one("make the kitchen bigger").value
        large = self._one("make the kitchen much bigger").value
        self.assertGreater(large, small)

    def test_absolute_resize_and_its_description(self):
        i = self._one("make the kitchen 150 sqft")
        self.assertEqual(i.value, ("abs", 150.0))
        self.assertIn("150", i.describe())      # regression: used to crash

    def test_every_action_describes_without_crashing(self):
        for text in ("make the kitchen bigger", "make the kitchen 150 sqft",
                     "move the pooja room to the north east",
                     "add a study room", "remove the store room",
                     "put the kitchen next to the dining room",
                     "make it two floors", "entrance from the east",
                     "no vastu"):
            for intent in plan_edit.parse(text).intents:
                self.assertTrue(intent.describe())

    def test_move_to_a_sector(self):
        i = self._one("move the pooja room to the north east")
        self.assertEqual((i.action, i.target, i.value),
                         ("move", "pooja_room", "NE"))

    def test_adjacency_reads_rooms_positionally(self):
        """Regression: 'dining room' is the longest phrase in the clause, so
        a longest-match parser named it as the room being moved."""
        i = self._one("put the kitchen next to the dining room")
        self.assertEqual(i.action, "adjacency")
        self.assertEqual(i.target, "kitchen")
        self.assertEqual(i.other, "dining_room")

    def test_adjacency_synonyms(self):
        for text in ("the pooja room should be near the living room",
                     "kitchen beside the dining room",
                     "put the study adjacent to the master bedroom"):
            intents = plan_edit.parse(text).intents
            self.assertTrue(any(i.action == "adjacency" for i in intents),
                            text)

    def test_add_and_remove(self):
        self.assertEqual(self._one("add a study room").action, "add")
        self.assertEqual(self._one("remove the store room").action, "remove")

    def test_floors(self):
        for text, want in (("make it two floors", 2), ("G+1", 2),
                           ("make it a single floor", 1),
                           ("3 floors please", 3)):
            intents = [i for i in plan_edit.parse(text).intents
                       if i.action == "floors"]
            self.assertTrue(intents, text)
            self.assertEqual(intents[0].value, want, text)

    def test_vastu_on_and_off(self):
        self.assertIs(self._one("make it vastu compliant").value, True)
        self.assertIs(self._one("no vastu please").value, False)

    def test_multiple_clauses(self):
        plan = plan_edit.parse(
            "make the living room bigger. move the master bedroom to the "
            "south west and add a pooja room")
        self.assertEqual({i.action for i in plan.intents},
                         {"resize", "move", "add"})

    def test_unparseable_is_reported_not_dropped(self):
        plan = plan_edit.parse("paint the walls blue and add a jacuzzi")
        self.assertFalse(plan.ok)
        self.assertTrue(plan.unparsed)

    def test_partially_parseable_keeps_both_halves(self):
        plan = plan_edit.parse("make the kitchen bigger and paint it blue")
        self.assertTrue(any(i.action == "resize" for i in plan.intents))
        self.assertTrue(plan.unparsed)

    def test_empty_input(self):
        for text in ("", "   ", None):
            self.assertFalse(plan_edit.parse(text).ok)

    def test_master_bedroom_is_not_matched_as_bedroom(self):
        i = self._one("make the master bedroom bigger")
        self.assertEqual(i.target, "master_bedroom")

    def test_requirement_vs_override_split(self):
        """Add/remove/floors/entrance/vastu re-run the pipeline; resize and
        move are applied after enrichment, because step 3 would otherwise
        re-derive them from its own rules and discard the user's request."""
        for text, is_req in (("add a study room", True),
                             ("remove the store room", True),
                             ("make it two floors", True),
                             ("entrance from the east", True),
                             ("no vastu", True),
                             ("make the kitchen bigger", False),
                             ("move the kitchen to the south east", False)):
            for intent in plan_edit.parse(text).intents:
                self.assertEqual(intent.is_requirement, is_req, text)


class TestApplyToRequirements(unittest.TestCase):
    def setUp(self):
        self.brief = {
            "rooms": [{"room_type": "Kitchen", "quantity": 1},
                      {"room_type": "Bedroom", "quantity": 2}],
            "number_of_floors": 1,
        }

    def _apply(self, text):
        intents = [i for i in plan_edit.parse(text).intents
                   if i.is_requirement]
        return plan_edit.apply_to_requirements(self.brief, intents)

    def test_does_not_mutate_the_original(self):
        before = len(self.brief["rooms"])
        self._apply("add a study room")
        self.assertEqual(len(self.brief["rooms"]), before)

    def test_add_a_new_room_type(self):
        reqs, notes = self._apply("add a study room")
        types = [r["room_type"] for r in reqs["rooms"]]
        self.assertTrue(any("Study" in t for t in types), types)
        self.assertTrue(notes)

    def test_add_an_existing_type_bumps_quantity(self):
        reqs, _ = self._apply("add a bedroom")
        bedroom = next(r for r in reqs["rooms"]
                       if r["room_type"] == "Bedroom")
        self.assertEqual(bedroom["quantity"], 3)

    def test_remove_decrements_before_deleting(self):
        reqs, _ = self._apply("remove a bedroom")
        bedroom = next((r for r in reqs["rooms"]
                        if r["room_type"] == "Bedroom"), None)
        self.assertIsNotNone(bedroom)
        self.assertEqual(bedroom["quantity"], 1)

    def test_floors_and_entrance_and_vastu(self):
        reqs, _ = self._apply("make it two floors")
        self.assertEqual(reqs["number_of_floors"], 2)
        reqs, _ = self._apply("entrance from the east")
        self.assertEqual(reqs["plot_context"]["entrance_side"], "east")
        reqs, _ = self._apply("make it vastu compliant")
        self.assertIs(reqs["vastu_compliant"], True)


class TestOverrides(unittest.TestCase):
    def _room(self, rtype, area):
        from models import EnrichedRoom
        return EnrichedRoom(
            room_id=f"{rtype}_1", room_type=rtype, display_name=rtype.title(),
            target_width_ft=10.0, target_length_ft=area / 10.0,
            target_area_sqft=area, min_width_ft=6.0, min_length_ft=6.0,
            min_area_sqft=40.0, max_area_sqft=area * 1.5,
            ceiling_height_ft=9.0)

    def _plan(self, rooms):
        return type("EP", (), {"rooms": rooms})()

    def test_relative_resize_applies(self):
        room = self._room("kitchen", 100.0)
        ov = plan_edit.overrides_from(
            plan_edit.parse("make the kitchen bigger").intents)
        notes = plan_edit.apply_overrides(self._plan([room]), ov)
        self.assertGreater(room.target_area_sqft, 100.0)
        self.assertTrue(notes)

    def test_absolute_resize_applies(self):
        room = self._room("kitchen", 100.0)
        ov = plan_edit.overrides_from(
            plan_edit.parse("make the kitchen 150 sqft").intents)
        plan_edit.apply_overrides(self._plan([room]), ov)
        self.assertAlmostEqual(room.target_area_sqft, 150.0, delta=1.0)

    def test_resize_cannot_go_below_the_nbc_minimum(self):
        room = self._room("bathroom", 45.0)
        room.min_area_sqft = 40.0
        ov = {"absolute": {"bathroom": 5.0}, "scale": {}, "sector": {}}
        plan_edit.apply_overrides(self._plan([room]), ov)
        self.assertGreaterEqual(room.target_area_sqft, 40.0)

    def test_resize_is_clamped_and_says_so(self):
        room = self._room("bathroom", 45.0)
        ov = {"absolute": {"bathroom": 5000.0}, "scale": {}, "sector": {}}
        notes = plan_edit.apply_overrides(self._plan([room]), ov)
        self.assertLess(room.target_area_sqft, 5000.0)
        self.assertTrue(any("clamped" in n for n in notes))

    def test_move_sets_the_direction(self):
        room = self._room("pooja_room", 40.0)
        ov = plan_edit.overrides_from(
            plan_edit.parse("move the pooja room to the north east").intents)
        plan_edit.apply_overrides(self._plan([room]), ov)
        self.assertEqual(room.preferred_direction, "NE")


class TestEditEndpoint(unittest.TestCase):
    def setUp(self):
        # the endpoint creates the run directory before handing off to the
        # background task, so without this the suite litters the user's
        # output/ with empty run folders
        self._tmp = tempfile.mkdtemp(prefix="planedit")
        self._out = mock.patch("api.server.OUTPUT_DIR", Path(self._tmp))
        self._out.start()
        self.client = TestClient(app)
        self.sid = self.client.post("/api/v1/sessions").json()["session_id"]
        self.run_id = "20260906_090000"
        sessions[self.sid]["requirements"] = {
            "rooms": [{"room_type": "Kitchen", "quantity": 1}],
            "number_of_floors": 1}
        sessions[self.sid]["runs"] = {self.run_id: {
            "run_id": self.run_id,
            "requirements": sessions[self.sid]["requirements"],
            "alternatives": [], "_signature": None}}

    def tearDown(self):
        self._out.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _edit(self, text, preview=False):
        return self.client.post(
            f"/api/v1/runs/{self.run_id}/edit",
            json={"session_id": self.sid, "text": text, "preview": preview})

    def test_preview_explains_without_generating(self):
        r = self._edit("make the kitchen bigger", preview=True)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["understood"])
        self.assertTrue(body["preview"])
        self.assertIn("kitchen", body["summary"])
        self.assertNotIn("run_id", body)

    def test_an_edit_starts_a_new_run(self):
        with mock.patch("api.server._run_pipeline_task") as task:
            r = self._edit("make the kitchen bigger")
        body = r.json()
        self.assertTrue(body["understood"])
        self.assertNotEqual(body["run_id"], self.run_id)
        self.assertEqual(body["edited_from"], self.run_id)
        task.assert_called_once()

    def test_the_edit_carries_the_previous_seed_and_layout(self):
        """Continuity depends on both: the seed starts the carve in the same
        place, the signature makes selection prefer the familiar result."""
        with mock.patch("api.server._run_pipeline_task") as task:
            self._edit("move the kitchen to the south east")
        opts = task.call_args[0][2]
        self.assertEqual(opts["seed_from"], self.run_id)
        self.assertIn("previous_signature", opts)
        self.assertIn("sector", opts["overrides"])

    def test_unparseable_edit_is_refused_helpfully(self):
        r = self._edit("paint everything turquoise")
        body = r.json()
        self.assertFalse(body["understood"])
        self.assertTrue(body["unparsed"])
        self.assertIn("kitchen bigger", body["message"])

    def test_edit_without_a_brief_is_400(self):
        sid = self.client.post("/api/v1/sessions").json()["session_id"]
        sessions[sid]["requirements"] = None
        r = self.client.post("/api/v1/runs/x/edit",
                             json={"session_id": sid, "text": "bigger kitchen"})
        self.assertEqual(r.status_code, 400)

    def test_partially_understood_edit_reports_the_remainder(self):
        with mock.patch("api.server._run_pipeline_task"):
            body = self._edit("make the kitchen bigger and paint it blue").json()
        self.assertTrue(body["understood"])
        self.assertTrue(body["unparsed"])


if __name__ == "__main__":
    unittest.main()
