"""
test_reward_freeze.py — the reward has to hold still while training runs.

Stage (b) imitates plans the reward chose; stage (c) optimises it directly.
If a weight is retuned or a rule added between building a corpus and training
on it, both silently stop meaning what they meant — the same failure v2 hit
when `--eval-n` changed mid-run. These tests pin the detector.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

from modules.step4_generate.engine import reward as reward_mod
from modules.step4_generate.engine.contracts import EngineConfig


class TestFingerprint(unittest.TestCase):
    def test_stable_across_calls(self):
        self.assertEqual(reward_mod.fingerprint(), reward_mod.fingerprint())

    def test_scoring_weight_changes_it(self):
        base = reward_mod.fingerprint(EngineConfig())
        moved = reward_mod.fingerprint(EngineConfig(w_area_drift=95.0))
        self.assertNotEqual(base, moved)

    def test_vastu_weight_changes_it(self):
        self.assertNotEqual(
            reward_mod.fingerprint(EngineConfig()),
            reward_mod.fingerprint(EngineConfig(w_vastu_sector=9.0)))

    def test_hard_gates_change_it(self):
        """`vastu_hard` and `fsp001_hard` promote soft penalties to
        disqualifications — a different reward, not a tuning detail."""
        for kw in ({"vastu_hard": True}, {"fsp001_hard": True}):
            self.assertNotEqual(reward_mod.fingerprint(EngineConfig()),
                                reward_mod.fingerprint(EngineConfig(**kw)),
                                kw)

    def test_generation_knobs_do_NOT_change_it(self):
        """These change which plans get MADE, not how one is JUDGED. A corpus
        stays valid across them, and treating them as drift would force
        pointless re-gates."""
        for kw in ({"settle_sweeps": 30}, {"realize_attempts": 8},
                   {"cpsat_mode": "repair"}, {"vastu_bias": 0.2},
                   {"top_up_candidates": False}):
            self.assertEqual(reward_mod.fingerprint(EngineConfig()),
                             reward_mod.fingerprint(EngineConfig(**kw)), kw)

    def test_every_weight_is_covered_automatically(self):
        """Every `w_*` field is picked up by pattern, so a newly added weight
        cannot be forgotten from the fingerprint."""
        weights = reward_mod.scoring_weights(EngineConfig())
        cfg = EngineConfig().to_dict()
        for key in cfg:
            if key.startswith("w_"):
                self.assertIn(key, weights, f"{key} missing from the reward")


class TestRuleSignature(unittest.TestCase):
    def test_covers_every_registered_rule(self):
        from modules.step4_generate.engine.rules import RULES
        self.assertEqual(len(reward_mod.rule_signature()), len(RULES))

    def test_records_severity(self):
        sig = reward_mod.rule_signature()
        self.assertTrue(any(s.startswith("VAS-002") and s.endswith("|hard")
                            for s in sig))
        self.assertTrue(any(s.startswith("VAS-001") and s.endswith("|soft")
                            for s in sig))


class TestFreezeAndVerify(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="reward")
        self.path = os.path.join(self.dir, "reward.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_freeze_then_verify_matches(self):
        reward_mod.freeze(self.path)
        self.assertTrue(reward_mod.verify(self.path).matches)

    def test_drift_is_itemised_not_just_flagged(self):
        """"the reward changed" is not actionable; naming the weight is."""
        reward_mod.freeze(self.path)
        drift = reward_mod.verify(
            self.path, config=EngineConfig(w_area_drift=95.0,
                                           w_vastu_sector=9.0))
        self.assertFalse(drift.matches)
        self.assertIn("w_area_drift", drift.weights_changed)
        self.assertEqual(drift.weights_changed["w_area_drift"], (90.0, 95.0))
        self.assertIn("w_vastu_sector", drift.weights_changed)
        text = drift.describe()
        self.assertIn("90.0 -> 95.0", text)
        self.assertIn("NOT comparable", text)

    def test_missing_snapshot_is_reported_not_crashed(self):
        drift = reward_mod.verify(os.path.join(self.dir, "absent.json"))
        self.assertFalse(drift.matches)
        self.assertTrue(drift.notes)

    def test_corrupt_snapshot_is_survivable(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        drift = reward_mod.verify(self.path)
        self.assertFalse(drift.matches)

    def test_snapshot_carries_the_baseline_and_context(self):
        snap = reward_mod.freeze(
            self.path, baseline={"n_briefs": 28, "mean_best_score": 69.5})
        self.assertEqual(snap["harness_baseline"]["mean_best_score"], 69.5)
        self.assertIn("settle_sweeps", snap["generation_context"])
        self.assertNotIn("settle_sweeps", snap["weights"])

    def test_snapshot_is_readable_json(self):
        reward_mod.freeze(self.path)
        with open(self.path, encoding="utf-8") as fh:
            blob = json.load(fh)
        for key in ("version", "fingerprint", "rules", "weights", "critic"):
            self.assertIn(key, blob)

    def test_require_raises_only_when_strict(self):
        """`require` PRINTS the drift banner by design — that is the whole
        point of it in a real run. Captured here, because a test that emits
        "REWARD DRIFT ... NOT comparable" into a passing suite reads as a
        failure to anyone skimming the output (it did, on Colab, where the
        summary line above it had been cut off by `tail`)."""
        reward_mod.freeze(self.path)
        drifted = EngineConfig(w_area_drift=95.0)
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            drift = reward_mod.require(self.path, config=drifted)
        self.assertFalse(drift.matches)
        # the banner still has to be produced — just not leaked to the suite
        self.assertIn("REWARD DRIFT", buffer.getvalue())

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                reward_mod.require(self.path, config=drifted, strict=True)


class TestShippedSnapshot(unittest.TestCase):
    """The snapshot committed with the repo must match the live engine —
    otherwise every training run starts by reporting drift."""

    def test_repo_snapshot_exists_and_matches(self):
        frozen = reward_mod.load()
        self.assertIsNotNone(frozen, "reward_v1.json is missing")
        drift = reward_mod.verify()
        self.assertTrue(
            drift.matches,
            f"the shipped reward snapshot no longer matches the engine:\n"
            f"{drift.describe()}\n"
            f"Re-freeze it, and rebuild any self-play corpus.")

    def test_snapshot_records_a_baseline(self):
        frozen = reward_mod.load()
        self.assertTrue(frozen.get("harness_baseline"),
                        "the frozen reward should carry the harness numbers "
                        "it was measured against")


if __name__ == "__main__":
    unittest.main()
