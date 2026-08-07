"""
Critic tests (M6): the booster, the feature contract, the damage catalog,
preference logging, and the ranking blend.

The load-bearing property is that the critic can never make a plan legal or
illegal — it only reorders what the rules already accepted, and an untrained
or broken critic leaves ranking exactly as it was.
"""

import os
import tempfile
import unittest

import numpy as np

from modules.step4_generate.critic import features as feat
from modules.step4_generate.critic.critic import LearnedCritic
from modules.step4_generate.critic.gbt import (
    GBTConfig, GBTModel, accuracy, log_loss, roc_auc, sigmoid, train_gbt,
)
from modules.step4_generate.critic.perturb import PERTURBATIONS, perturb_all
from modules.step4_generate.critic.preferences import (
    FEATURE_SET_VERSION, PreferenceRecord, agreement, log_choice,
    pairwise_samples, read_log,
)
from modules.step4_generate.engine.contracts import EngineConfig
from modules.step4_generate.engine.orchestrator import blended_score


def _xor_ish(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] > 1.0)).astype(float)
    return X, y


class TestGBT(unittest.TestCase):
    def test_learns_a_nonlinear_boundary(self):
        X, y = _xor_ish()
        model = train_gbt(X[:450], y[:450],
                          config=GBTConfig(n_estimators=150, max_depth=3))
        p = model.predict_proba(X[450:])
        self.assertGreater(roc_auc(y[450:], p), 0.85)
        self.assertGreater(accuracy(y[450:], p), 0.78)

    def test_early_stopping_trims_trees(self):
        X, y = _xor_ish(400, seed=1)
        model = train_gbt(X[:250], y[:250],
                          config=GBTConfig(n_estimators=400,
                                           early_stopping_rounds=5),
                          validation=(X[250:], y[250:]))
        self.assertLess(len(model.trees), 400)
        self.assertIsNotNone(model.train_report["val_logloss"])

    def test_importance_finds_the_real_features(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(500, 6))
        y = (X[:, 4] > 0.3).astype(float)
        model = train_gbt(X, y, config=GBTConfig(n_estimators=60),
                          feature_names=[f"c{i}" for i in range(6)])
        self.assertEqual(max(model.gain_importance(),
                             key=model.gain_importance().get), "c4")

    def test_json_roundtrip_reproduces_predictions(self):
        X, y = _xor_ish(300, seed=5)
        model = train_gbt(X, y, config=GBTConfig(n_estimators=40))
        with tempfile.TemporaryDirectory() as tmp:
            path = model.save(os.path.join(tmp, "m.json"))
            loaded = GBTModel.load(path)
        self.assertLess(float(np.abs(loaded.predict_proba(X)
                                     - model.predict_proba(X)).max()), 1e-5)

    def test_constant_features_are_never_split_on(self):
        X = np.zeros((200, 3))
        X[:, 1] = np.arange(200) / 200.0
        y = (X[:, 1] > 0.5).astype(float)
        model = train_gbt(X, y, config=GBTConfig(n_estimators=30),
                          feature_names=["const_a", "real", "const_b"])
        self.assertEqual(set(model.gain_importance()) - {"real"}, set())

    def test_metrics_behave(self):
        y = np.array([0.0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        self.assertAlmostEqual(roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertGreater(log_loss(y, np.array([0.9, 0.9, 0.1, 0.1])),
                           log_loss(y, np.array([0.1, 0.1, 0.9, 0.9])))

    def test_sigmoid_does_not_overflow(self):
        self.assertTrue(np.isfinite(sigmoid(np.array([-1e6, 1e6]))).all())


class TestFeatureContract(unittest.TestCase):
    def test_names_are_unique_and_counted(self):
        self.assertEqual(len(set(feat.FEATURE_NAMES)), feat.N_FEATURES)

    def test_extract_returns_the_declared_length(self):
        import logging

        from modules.step4_generate.engine.orchestrator import Orchestrator
        from ml.harness.briefs import golden_briefs
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)
        cand = Orchestrator().generate(golden_briefs(k=2)[0]).best
        vector = feat.extract(cand.plan, cand.request, cand.room_ids,
                              cand.verdict)
        self.assertEqual(vector.shape, (feat.N_FEATURES,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertEqual(len(feat.as_dict(vector)), feat.N_FEATURES)


class TestPerturbations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        from modules.step4_generate.engine.orchestrator import Orchestrator
        from ml.harness.briefs import golden_briefs
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)
        cls.cand = Orchestrator().generate(golden_briefs(k=4)[7]).best

    def test_damage_is_applicable_and_named(self):
        damaged = perturb_all(self.cand.plan, self.cand.request,
                              self.cand.room_ids, seed=1)
        self.assertGreaterEqual(len(damaged), 3)
        for d in damaged:
            self.assertIn(d.kind, PERTURBATIONS)
            self.assertTrue(d.detail)

    def test_the_original_is_never_mutated(self):
        before = self.cand.plan.grid.copy()
        n_openings = len(self.cand.plan.openings)
        perturb_all(self.cand.plan, self.cand.request, self.cand.room_ids,
                    seed=2)
        self.assertTrue((self.cand.plan.grid == before).all())
        self.assertEqual(len(self.cand.plan.openings), n_openings)

    def test_geometry_stays_structurally_valid(self):
        """Damage must be architectural, not corruption — otherwise the
        critic learns to detect broken files, not bad plans."""
        for d in perturb_all(self.cand.plan, self.cand.request,
                             self.cand.room_ids, seed=3):
            self.assertTrue(d.plan.verify(), d.kind)

    def test_damage_is_deterministic(self):
        a = perturb_all(self.cand.plan, self.cand.request,
                        self.cand.room_ids, seed=9)
        b = perturb_all(self.cand.plan, self.cand.request,
                        self.cand.room_ids, seed=9)
        self.assertEqual([d.detail for d in a], [d.detail for d in b])


class TestBlend(unittest.TestCase):
    def test_no_critic_leaves_the_score_alone(self):
        self.assertEqual(blended_score(80.0, None, 0.4), 80.0)
        self.assertEqual(blended_score(80.0, 10.0, 0.0), 80.0)

    def test_weight_means_what_it_says(self):
        self.assertAlmostEqual(blended_score(80.0, 40.0, 0.4), 64.0)
        self.assertAlmostEqual(blended_score(80.0, 40.0, 1.0), 40.0)

    def test_default_weight_is_documented(self):
        self.assertAlmostEqual(EngineConfig().critic_weight, 0.4)

    def test_untrained_critic_is_absent_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.json")
            self.assertIsNone(LearnedCritic.load_if_available(missing))

    def test_feature_set_mismatch_is_refused_loudly(self):
        model = GBTModel(trees=[], base_score=0.0,
                         feature_names=["only", "two"])
        with self.assertRaises(ValueError):
            LearnedCritic(model)

    def test_a_failing_critic_does_not_cost_a_candidate_its_rank(self):
        import logging

        from modules.step4_generate.engine.orchestrator import Orchestrator
        from ml.harness.briefs import golden_briefs
        logging.getLogger("PlanGen.Engine").setLevel(logging.CRITICAL)

        class Exploding:
            def score_candidate(self, cand):
                raise RuntimeError("boom")

        brief = golden_briefs(k=3)[3]
        plain = Orchestrator().generate(brief)
        with_critic = Orchestrator(critic=Exploding()).generate(brief)
        self.assertEqual(len(plain.ranked), len(with_critic.ranked))
        self.assertEqual(plain.best.verdict.soft_score,
                         with_critic.best.verdict.soft_score)


class TestPreferenceLog(unittest.TestCase):
    @staticmethod
    def _record(chosen, vectors, scores=None):
        return PreferenceRecord(
            brief="b", chosen=chosen, vectors=vectors,
            soft_scores=scores or [0.0] * len(vectors), timestamp="t",
            version=FEATURE_SET_VERSION)

    def test_jsonl_roundtrip(self):
        rec = self._record(1, [[1.0, 2.0], [3.0, 4.0]], [10.0, 20.0])
        back = PreferenceRecord.from_json(rec.to_json())
        self.assertEqual(back.chosen, 1)
        self.assertEqual(back.vectors, [[1.0, 2.0], [3.0, 4.0]])

    def test_pairwise_samples_yield_k_minus_one_pairs(self):
        rec = self._record(0, [[1.0], [2.0], [3.0]])
        self.assertEqual(len(list(pairwise_samples([rec]))), 2)

    def test_records_from_another_feature_set_are_skipped(self):
        rec = self._record(0, [[1.0], [2.0]])
        rec.version = FEATURE_SET_VERSION + 99
        self.assertEqual(list(pairwise_samples([rec])), [])

    def test_agreement_reports_critic_soft_and_chance(self):
        recs = [self._record(0, [[5.0], [1.0]], [9.0, 1.0]),
                self._record(1, [[1.0], [5.0]], [1.0, 9.0])]
        stats = agreement(recs, lambda v: float(v[0]))
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["critic_top1"], 1.0)
        self.assertAlmostEqual(stats["soft_top1"], 1.0)
        self.assertAlmostEqual(stats["random_top1"], 0.5)

    def test_log_choice_appends_and_reads_back(self):
        import logging

        from modules.step4_generate.engine.orchestrator import Orchestrator
        from ml.harness.briefs import golden_briefs
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)
        result = Orchestrator().generate(golden_briefs(k=3)[10])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prefs.jsonl")
            log_choice(result.ranked, 0, brief="t", path=path)
            log_choice(result.ranked, min(1, len(result.ranked) - 1),
                       brief="t", path=path)
            records = read_log(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[0].vectors), len(result.ranked))
        self.assertEqual(len(records[0].vectors[0]), feat.N_FEATURES)

    def test_out_of_range_pick_is_an_error(self):
        with self.assertRaises(IndexError):
            log_choice([], 0)


if __name__ == "__main__":
    unittest.main()
