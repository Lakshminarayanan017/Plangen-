"""
test_product_loop.py — choose, regenerate, and the preference log.

Also the first endpoint tests this project has had. They drive the real
FastAPI app through TestClient with the session seeded directly, so the loop
is exercised without step 1's LLM or step 2's index — the parts these
endpoints do not touch.

The point of the whole feature: `critic/preferences.jsonl` has been built,
live and empty since M6, because nothing ever asked a user to choose.
Perturbation labels teach the critic to recognise damage; only a real pick
among real options teaches it taste, and taste data cannot be collected
retroactively.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from api.server import app, sessions


def _alternatives(n=3):
    return [{"rank": i, "svg": f"alternative_{i + 1}.svg",
             "score": 100.0 - i, "fidelity": 0.85, "rooms": 6,
             "is_best": i == 0, "highlights": [f"trait {i}"]}
            for i in range(n)]


def _seed_run(client, *, n_alts=3, with_vectors=True, with_sigs=False):
    """A session holding one finished run, without running the pipeline."""
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    run_id = "20260906_120000"
    payload = None
    if with_vectors:
        payload = {"vectors": [[float(i)] * 47 for i in range(n_alts)],
                   "soft_scores": [100.0 - i for i in range(n_alts)]}
    sessions[session_id]["requirements"] = {"rooms": []}
    run = {
        "run_id": run_id,
        "alternatives": _alternatives(n_alts),
        "_preference_vectors": payload,
    }
    if with_sigs:
        # each option sits in a different place, so adopting one is visible
        run["_alternative_signatures"] = [
            {"kitchen": [0.1 * (i + 1), 0.1 * (i + 1), 0.2, 0.2, 1.0]}
            for i in range(n_alts)]
    sessions[session_id]["runs"] = {run_id: run}
    return session_id, run_id


class TestAlternativesEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_lists_every_option(self):
        sid, run_id = _seed_run(self.client)
        r = self.client.get(f"/api/v1/runs/{run_id}/alternatives",
                            params={"session_id": sid})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["alternatives"]), 3)
        self.assertTrue(body["alternatives"][0]["is_best"])
        self.assertIsNone(body["chosen_rank"])

    def test_unknown_run_is_404_not_a_crash(self):
        sid, _ = _seed_run(self.client)
        r = self.client.get("/api/v1/runs/nope/alternatives",
                            params={"session_id": sid})
        self.assertEqual(r.status_code, 404)

    def test_unknown_session_is_404(self):
        r = self.client.get("/api/v1/runs/x/alternatives",
                            params={"session_id": "not-a-session"})
        self.assertEqual(r.status_code, 404)


class TestChooseEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.tmp = tempfile.mkdtemp(prefix="prefs")
        self.log = os.path.join(self.tmp, "preferences.jsonl")

    def _choose(self, sid, run_id, rank, note=""):
        with mock.patch(
                "modules.step4_generate.critic.preferences.DEFAULT_LOG",
                self.log):
            return self.client.post(
                f"/api/v1/runs/{run_id}/choose",
                json={"session_id": sid, "rank": rank, "note": note})

    def test_choice_is_recorded_and_logged(self):
        sid, run_id = _seed_run(self.client)
        r = self._choose(sid, run_id, 2, note="liked the kitchen")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["chosen_rank"], 2)
        self.assertTrue(body["logged"], "the pick was not written to the log")
        self.assertEqual(body["svg"], "alternative_3.svg")

        with open(self.log, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen"], 2)
        self.assertEqual(len(rows[0]["vectors"]), 3)
        self.assertEqual(rows[0]["note"], "liked the kitchen")

    def test_log_is_append_only(self):
        sid, run_id = _seed_run(self.client)
        self._choose(sid, run_id, 0)
        self._choose(sid, run_id, 1)
        with open(self.log, encoding="utf-8") as fh:
            self.assertEqual(len([ln for ln in fh if ln.strip()]), 2)

    def test_logged_rows_are_readable_by_the_critic(self):
        """The log has to be consumable by critic/preferences.py — writing a
        shape it cannot read would be worse than not writing at all."""
        sid, run_id = _seed_run(self.client)
        self._choose(sid, run_id, 1)
        from modules.step4_generate.critic.preferences import (
            pairwise_samples, read_log,
        )
        records = read_log(self.log)
        self.assertEqual(len(records), 1)
        pairs = list(pairwise_samples(records))
        # K candidates yield K-1 ordered (chosen, rejected) pairs
        self.assertEqual(len(pairs), 2)

    def test_rank_out_of_range_is_400(self):
        sid, run_id = _seed_run(self.client)
        for bad in (-1, 3, 99):
            self.assertEqual(self._choose(sid, run_id, bad).status_code, 400)

    def test_choice_survives_a_missing_vector_payload(self):
        """A pick must never be lost because logging could not run."""
        sid, run_id = _seed_run(self.client, with_vectors=False)
        r = self._choose(sid, run_id, 1)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["chosen_rank"], 1)
        self.assertFalse(r.json()["logged"])

    def test_choice_is_visible_afterwards(self):
        sid, run_id = _seed_run(self.client)
        self._choose(sid, run_id, 1)
        body = self.client.get(f"/api/v1/runs/{run_id}/alternatives",
                               params={"session_id": sid}).json()
        self.assertEqual(body["chosen_rank"], 1)


class TestChoosingAdoptsTheLayout(unittest.TestCase):
    """Recording a pick and then editing a DIFFERENT layout would make the
    choice cosmetic — the loop would ask the user to choose and then ignore
    it on the very next instruction."""

    def setUp(self):
        # /choose APPENDS to the real preference log. A test that writes
        # synthetic 47-dim vectors into the file the critic will train on is
        # poisoning it — three such rows were found in the live log.
        self.tmp = tempfile.mkdtemp(prefix="prefs")
        self.patch = mock.patch(
            "modules.step4_generate.critic.preferences.DEFAULT_LOG",
            os.path.join(self.tmp, "preferences.jsonl"))
        self.patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_chosen_option_becomes_the_edit_baseline(self):
        sid, run_id = _seed_run(self.client, with_sigs=True)
        run = sessions[sid]["runs"][run_id]
        run["_signature"] = None
        r = self.client.post(f"/api/v1/runs/{run_id}/choose",
                             json={"session_id": sid, "rank": 2})
        self.assertEqual(r.status_code, 200)
        sig = run["_signature"]
        self.assertTrue(sig, "choosing should adopt that option's layout")
        self.assertAlmostEqual(sig.rooms["kitchen"].cx, 0.3, places=6)

    def test_choose_still_works_without_signatures(self):
        """Older runs in a live session have no per-option signatures; the
        pick must still be recorded rather than 500."""
        sid, run_id = _seed_run(self.client)
        r = self.client.post(f"/api/v1/runs/{run_id}/choose",
                             json={"session_id": sid, "rank": 1})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(sessions[sid]["runs"][run_id]["chosen_rank"], 1)

    def test_the_response_names_the_sheet_to_show(self):
        sid, run_id = _seed_run(self.client, with_sigs=True)
        body = self.client.post(f"/api/v1/runs/{run_id}/choose",
                                json={"session_id": sid, "rank": 1}).json()
        self.assertEqual(body["svg"], "alternative_2.svg")


class TestRegenerateEndpoint(unittest.TestCase):
    def setUp(self):
        # regenerate creates the run directory before the background task
        # runs; redirect it so the suite leaves no empty folders behind
        self._tmp = tempfile.mkdtemp(prefix="planregen")
        self._out = mock.patch("api.server.OUTPUT_DIR", Path(self._tmp))
        self._out.start()
        self.client = TestClient(app)

    def tearDown(self):
        self._out.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_starts_a_distinct_run(self):
        sid, run_id = _seed_run(self.client)
        with mock.patch("api.server._run_pipeline_task") as task:
            r = self.client.post("/api/v1/pipeline/regenerate",
                                 json={"session_id": sid})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["regenerated"])
        self.assertNotEqual(body["run_id"], run_id)
        task.assert_called_once()

    def test_requires_step_one(self):
        sid = self.client.post("/api/v1/sessions").json()["session_id"]
        sessions[sid]["requirements"] = None
        r = self.client.post("/api/v1/pipeline/regenerate",
                             json={"session_id": sid})
        self.assertEqual(r.status_code, 400)

    def test_a_new_run_id_means_new_plans(self):
        """Regenerate relies on the engine seeding itself from the run_id, so
        a different id must produce a different seed — and the same id must
        reproduce, or a shared link would show something else."""
        from api.engine_bridge import _seed_from
        self.assertNotEqual(_seed_from("20260906_120000"),
                            _seed_from("20260906_120001_r"))
        self.assertEqual(_seed_from("same"), _seed_from("same"))


class TestExistingEndpointsStillWork(unittest.TestCase):
    """The first endpoint tests in the project — phase 00 flagged that there
    were none at all."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        body = self.client.get("/api/v1/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("diagnostics", body)

    def test_diagnostics(self):
        body = self.client.get("/api/v1/diagnostics").json()
        self.assertIn("overall", body)
        self.assertIn("checks", body)

    def test_session_lifecycle(self):
        sid = self.client.post("/api/v1/sessions").json()["session_id"]
        self.assertIn(sid, sessions)
        self.client.delete(f"/api/v1/sessions/{sid}")
        self.assertNotIn(sid, sessions)

    def test_config_options(self):
        body = self.client.get("/api/v1/config/options").json()
        self.assertIn("room_types", body)
        self.assertEqual(body["solvers"], ["wall_graph_carver"])


class TestHighlightsAreDifferentiating(unittest.TestCase):
    """Highlights exist to tell options APART. Two bugs are pinned here:
    reporting each option's own numbers (all identical), and handing the same
    superlative to several options at once."""

    def _cand(self, **breakdown):
        from modules.step4_generate.engine.contracts import Candidate, Verdict
        return Candidate(plan=None, proposal=None,
                         verdict=Verdict(hard=[], soft_score=100.0,
                                         breakdown=breakdown))

    def test_superlatives_are_unique(self):
        from api.engine_bridge import _candidate_highlights
        peers = [self._cand(area_drift=d, worst_aspect=a)
                 for d, a in ((0.01, 1.2), (0.05, 1.6), (0.12, 2.4))]
        claimed = [h for c in peers for h in _candidate_highlights(c, peers)]
        self.assertEqual(len(claimed), len(set(claimed)),
                         f"a superlative was claimed twice: {claimed}")

    def test_identical_options_get_no_highlights(self):
        from api.engine_bridge import _candidate_highlights
        peers = [self._cand(area_drift=0.02, worst_aspect=1.5)
                 for _ in range(3)]
        for c in peers:
            self.assertEqual(_candidate_highlights(c, peers), [],
                             "identical options must not be given "
                             "distinguishing traits")

    def test_the_best_option_is_named_as_best(self):
        from api.engine_bridge import _candidate_highlights
        best = self._cand(area_drift=0.01)
        peers = [best, self._cand(area_drift=0.20),
                 self._cand(area_drift=0.30)]
        text = " ".join(_candidate_highlights(best, peers))
        self.assertIn("closest", text)

    def test_a_lone_candidate_has_nothing_to_compare_to(self):
        from api.engine_bridge import _candidate_highlights
        c = self._cand(area_drift=0.02)
        self.assertEqual(_candidate_highlights(c, [c]), [])


if __name__ == "__main__":
    unittest.main()
