"""
test_tuning.py — Phase-A selector-tuning tests.

Two kinds of test:
  • Correctness of the linear decomposition against the REAL engine — the
    assumption the whole tuner rests on (soft_score == 100 + w·g). Runs the
    engine on two small briefs; the reconstruction must match the
    validator to within its 2-dp rounding.
  • Behaviour of the tuner on SYNTHETIC data with a known ground-truth
    weight vector — proves the optimizer actually recovers a selector that
    agrees with the labels, and that CV prefers the honest solution. No
    human data is fabricated: the synthetic labels are generated from a
    declared w_true purely to test the math.
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.step4_generate.engine.contracts import EngineConfig
from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs
from ml.tuning.corpus import (
    BriefCorpus, CandidateRecord, Corpus,
)
from ml.tuning.features import (
    WEIGHT_KEYS, extract_features, reconstruct_score, room_ids_for,
    with_weights,
)
from ml.tuning.labeling import BriefPref, Preferences, render_labeling_html
from ml.tuning.tuner import (
    build_matrices, fit_weights, pairwise_agreement, top1_agreement, tune,
    _pairs_from_pref,
)


# ── the load-bearing correctness test ────────────────────────────────────────

class TestFeatureReconstruction(unittest.TestCase):
    """soft_score(w) must equal 100 + w·g on real engine candidates."""

    @classmethod
    def setUpClass(cls):
        cls.config = EngineConfig()
        orch = Orchestrator(config=cls.config)
        # two of the smallest briefs keep this fast but real
        small = [b for b in golden_briefs(k=3)
                 if b.name in ("20x30_S_1bhk", "35x35_S_2bhk")]
        cls.samples = []
        for req in small:
            for cand in orch.generate(req).ranked:
                cls.samples.append((req, cand))

    def test_reconstructs_validator_score(self):
        self.assertTrue(self.samples, "engine produced no candidates")
        for req, cand in self.samples:
            rids = room_ids_for(cand.plan, req)
            g = extract_features(cand.plan, req, rids, self.config)
            recon = reconstruct_score(g, self.config)
            self.assertAlmostEqual(
                recon, cand.verdict.soft_score, delta=0.02,
                msg=f"{req.name}: recon {recon} vs "
                    f"{cand.verdict.soft_score}")

    def test_reconstructs_under_perturbed_weights(self):
        """Linearity must hold for arbitrary weights, not just defaults."""
        rng = np.random.default_rng(0)
        for req, cand in self.samples[:4]:
            rids = room_ids_for(cand.plan, req)
            g = extract_features(cand.plan, req, rids, self.config)
            weights = {k: float(rng.uniform(0, 20)) for k in WEIGHT_KEYS}
            probe_cfg = with_weights(self.config, weights)
            recon = reconstruct_score(g, probe_cfg)
            # ground truth: re-score the same plan under probe_cfg
            from modules.step4_generate.engine.validator import BasicValidator
            truth = BasicValidator(probe_cfg).check(cand.plan, req, rids)
            self.assertAlmostEqual(recon, truth.soft_score, delta=0.02)


# ── synthetic corpus helpers ─────────────────────────────────────────────────

def _synthetic_corpus(n_briefs=16, k=4, seed=0):
    """A corpus of random feature vectors and a declared ground-truth weight
    vector w_true; the best candidate per brief is argmax w_true·g."""
    rng = np.random.default_rng(seed)
    w_true = rng.uniform(0.4, 2.0, size=len(WEIGHT_KEYS))
    # neutral prior: all-ones weights, so defaults disagree with w_true
    base = with_weights(EngineConfig(), {k_: 1.0 for k_ in WEIGHT_KEYS})

    briefs, best_ids = [], {}
    for b in range(n_briefs):
        cands = []
        G = rng.normal(0, 1, size=(k, len(WEIGHT_KEYS)))
        scores = G @ w_true
        best_row = int(np.argmax(scores))
        for r in range(k):
            cid = f"brief{b}#{r}"
            cands.append(CandidateRecord(
                cand_id=cid, rank=r, base_score=0.0, fidelity=1.0,
                features={key: float(G[r, i])
                          for i, key in enumerate(WEIGHT_KEYS)},
                breakdown={}))
        best_ids[f"brief{b}"] = f"brief{b}#{best_row}"
        briefs.append(BriefCorpus(name=f"brief{b}", request={},
                                  candidates=cands))
    corpus = Corpus(base_config=base.to_dict(), k=k, seed=seed, briefs=briefs)
    prefs = Preferences(by_brief={n: BriefPref(best=bid)
                                  for n, bid in best_ids.items()})
    return corpus, prefs, w_true, base


class TestTunerRecoversWeights(unittest.TestCase):

    def test_in_sample_fit_agrees_with_labels(self):
        corpus, prefs, _w_true, base = _synthetic_corpus()
        matrices = build_matrices(corpus, prefs)
        w0 = np.array([getattr(base, k) for k in WEIGHT_KEYS])
        w = fit_weights(matrices, w0, lam=0.0)
        # a linear model over 20 features must separate 4-way picks cleanly
        self.assertGreaterEqual(top1_agreement(matrices, w), 0.95)
        self.assertGreaterEqual(pairwise_agreement(matrices, w), 0.95)

    def test_tune_beats_wrong_defaults(self):
        # neutral all-ones defaults are a mediocre selector for w_true; the
        # tuned, cross-validated selector must clearly learn a better one.
        # (A strictly-positive baseline can't be driven near-chance in the
        # positive orthant, so we assert improvement + a high absolute bar,
        # not an artificially low baseline.)
        corpus, prefs, _w_true, base = _synthetic_corpus(n_briefs=20, k=5)
        result = tune(corpus, prefs, base_config=base)
        self.assertGreaterEqual(result.tuned_top1_cv, result.base_top1)
        self.assertGreaterEqual(result.tuned_top1_cv, 0.8)

    def test_weights_stay_nonnegative(self):
        corpus, prefs, _w_true, base = _synthetic_corpus()
        result = tune(corpus, prefs, base_config=base)
        for k, v in result.weights.items():
            self.assertGreaterEqual(v, 0.0, f"{k} went negative")

    def test_sparse_labels_warn_on_collapsed_weights(self):
        """Real-world regression: with few labeled briefs relative to 20
        free weights, an unregularized/lightly-regularized fit can drive
        many weights to exactly zero without any held-out score penalty.
        That must surface as an explicit, actionable note — never ship
        silently — because a zeroed weight there means 'no gradient,' not
        'this rule doesn't matter.'"""
        corpus, prefs, _w_true, base = _synthetic_corpus(n_briefs=3, k=3)
        result = tune(corpus, prefs, base_config=base)
        if len(result.zeroed_weights) >= len(WEIGHT_KEYS) // 3:
            self.assertTrue(
                any("collapsed to zero" in n for n in result.notes),
                "many weights collapsed but no warning note was emitted")
            self.assertTrue(
                any("DO NOT wire" in n for n in result.notes),
                "collapsed-weight note must tell the operator not to "
                "deploy the fit as-is")


# ── preference → pairs ───────────────────────────────────────────────────────

class TestPairsFromPref(unittest.TestCase):

    def setUp(self):
        self.index = {"a": 0, "b": 1, "c": 2, "d": 3}

    def test_best_only(self):
        pairs = _pairs_from_pref(BriefPref(best="b"), self.index)
        self.assertEqual(set(pairs), {(1, 0), (1, 2), (1, 3)})

    def test_best_with_bad(self):
        pairs = _pairs_from_pref(BriefPref(best="b", bad=["d"]), self.index)
        # best≻others plus everyone-acceptable≻d
        self.assertIn((1, 3), pairs)
        self.assertIn((0, 3), pairs)
        self.assertIn((2, 3), pairs)
        self.assertNotIn((3, 3), pairs)

    def test_ranking(self):
        pairs = _pairs_from_pref(BriefPref(ranking=["a", "b", "c"]),
                                 self.index)
        self.assertEqual(set(pairs), {(0, 1), (0, 2), (1, 2)})

    def test_stale_ids_skipped(self):
        pairs = _pairs_from_pref(BriefPref(best="zzz"), self.index)
        self.assertEqual(pairs, [])


# ── serialization + UI smoke ─────────────────────────────────────────────────

class TestSerialization(unittest.TestCase):

    def test_corpus_roundtrip(self):
        corpus, _prefs, _w, _base = _synthetic_corpus(n_briefs=3, k=3)
        restored = Corpus.from_json(corpus.to_json(include_svg=False))
        self.assertEqual(len(restored.briefs), 3)
        self.assertEqual(restored.briefs[0].candidates[0].features,
                         corpus.briefs[0].candidates[0].features)

    def test_preferences_from_json(self):
        prefs = Preferences.from_json({
            "corpus_seed": 7,
            "labels": {"b0": {"best": "b0#1", "bad": ["b0#2"]}}})
        self.assertEqual(prefs.corpus_seed, 7)
        self.assertEqual(prefs.for_brief("b0").best, "b0#1")
        self.assertIsNone(prefs.for_brief("missing"))

    def test_labeling_html_smoke(self):
        corpus, _p, _w, _b = _synthetic_corpus(n_briefs=2, k=2)
        for b in corpus.briefs:          # UI needs an SVG per candidate
            for c in b.candidates:
                c.svg = "<svg><rect/></svg>"
        import tempfile
        path = render_labeling_html(
            corpus, path=tempfile.mktemp(suffix=".html"))
        with open(path, encoding="utf-8") as f:
            page = f.read()
        self.assertIn("preference labeling", page)
        self.assertIn("brief0", page)
        self.assertNotIn("/*__DATA__*/", page)   # placeholder was replaced


if __name__ == "__main__":
    unittest.main(verbosity=2)
