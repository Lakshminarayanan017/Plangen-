"""
Tier-2 confidence calibration and the merge gate (M5 close-out).

The bug this file exists to prevent: the fallback threshold was compared
against the model's top-1 cell probability, which the training objective
caps at ~0.159 (sigma=1 Gaussian label smoothing over 1024 cells). A tau of
0.35 on that quantity is unreachable BY CONSTRUCTION, so a perfectly trained
model would have fallen back on every request forever, silently.
"""

import unittest

import numpy as np

from ml.tier2_placer import masked_decode as md
from ml.tier2_placer.export import checkpoint_summary
from ml.tier2_placer.merge_gate import evaluate
from ml.tier2_placer.tier2_placer import DEFAULT_TAU
from ml.tier2_placer.train import eval_key_for


def _gaussian_field(grid=md.SEED_GRID, sigma=1.0, center=None):
    """The posterior a PERFECTLY fitted model produces: the same Gaussian
    the loss uses as its soft target."""
    cr, cc = center or (grid // 2, grid // 2)
    rows = np.arange(grid).reshape(grid, 1)
    cols = np.arange(grid).reshape(1, grid)
    d2 = (rows - cr) ** 2 + (cols - cc) ** 2
    field = np.exp(-d2 / (2 * sigma ** 2))
    return (field / field.sum()).reshape(-1)


class TestConfidenceCeiling(unittest.TestCase):
    def test_documented_ceilings_match_the_training_objective(self):
        probs = _gaussian_field()
        best = int(np.argmax(probs))
        self.assertAlmostEqual(float(probs.max()), md.PERFECT_TOP1, places=3)
        self.assertAlmostEqual(md.neighborhood_confidence(probs, best),
                               md.PERFECT_NEIGHBORHOOD, places=3)

    def test_the_default_tau_is_reachable(self):
        """A perfect fit must clear tau with room to spare; on the old
        top-1 metric it could not clear it at all."""
        probs = _gaussian_field()
        best = int(np.argmax(probs))
        self.assertGreater(md.neighborhood_confidence(probs, best),
                           DEFAULT_TAU)
        self.assertLess(md.PERFECT_TOP1, DEFAULT_TAU)

    def test_a_flat_posterior_scores_near_zero(self):
        flat = np.full(md.SEED_GRID ** 2, 1.0 / md.SEED_GRID ** 2)
        self.assertLess(md.neighborhood_confidence(flat, 0), 0.01)

    def test_a_spike_scores_one(self):
        spike = np.zeros(md.SEED_GRID ** 2)
        spike[500] = 1.0
        self.assertAlmostEqual(md.neighborhood_confidence(spike, 500), 1.0)

    def test_confidence_rises_as_the_model_sharpens(self):
        best = int(np.argmax(_gaussian_field(sigma=1.0)))
        vague = md.neighborhood_confidence(_gaussian_field(sigma=4.0), best)
        sharp = md.neighborhood_confidence(_gaussian_field(sigma=0.6), best)
        self.assertLess(vague, sharp)

    def test_corner_cells_do_not_crash(self):
        probs = _gaussian_field(center=(0, 0))
        self.assertGreater(md.neighborhood_confidence(probs, 0), 0.0)


class _StubBackend:
    """A model whose posterior is a fixed Gaussian — lets the decoder be
    tested without weights."""

    cell_count = md.SEED_GRID ** 2
    size_count = 40

    def __init__(self, sigma=1.0):
        self.field = np.log(np.maximum(_gaussian_field(sigma=sigma), 1e-12))

    def encode(self, arrays):
        return None

    def decode_step(self, memory, arrays, prev_cell, prev_size):
        n = prev_cell.shape[0]
        return (np.tile(self.field, (n, 1)),
                np.zeros((n, self.size_count)))


class TestDecodeResult(unittest.TestCase):
    def _arrays(self, n_rooms=3):
        return {"n_rooms": n_rooms,
                "boundary": np.ones((2, 64, 64), dtype=np.float32),
                "type_ids": np.zeros(n_rooms, dtype=np.int64),
                "zone_ids": np.zeros(n_rooms, dtype=np.int64)}

    def test_both_confidences_are_reported(self):
        result = md.sample(_StubBackend(), self._arrays(), greedy=True)
        self.assertGreater(result.confidence, result.top1_confidence)
        self.assertGreater(result.confidence, DEFAULT_TAU)

    def test_temperature_does_not_change_the_reported_confidence(self):
        """A high-temperature variant explores more; it is not a less
        certain model, and scoring it as one would make diversity look like
        weakness. Checked on a ONE-room program, where the two runs share
        the same posterior — with more rooms the sampled trajectories
        legitimately diverge and the later posteriors differ."""
        arrays = self._arrays(n_rooms=1)
        cold = md.sample(_StubBackend(), arrays, temperature=1.0,
                         rng=np.random.default_rng(0))
        hot = md.sample(_StubBackend(), arrays, temperature=5.0,
                        rng=np.random.default_rng(0))
        self.assertAlmostEqual(cold.confidence, hot.confidence, places=6)
        self.assertAlmostEqual(cold.top1_confidence, hot.top1_confidence,
                               places=6)


class TestEvalKey(unittest.TestCase):
    def test_key_changes_with_the_eval_set(self):
        self.assertNotEqual(eval_key_for(50, 443), eval_key_for(443, 443))

    def test_key_is_stable_when_eval_n_exceeds_the_split(self):
        self.assertEqual(eval_key_for(1000, 443), eval_key_for(443, 443))


class TestCheckpointSummary(unittest.TestCase):
    def test_reports_the_epoch_the_best_weights_came_from(self):
        ckpt = {"next_epoch": 24, "best_score": 70.82, "best_model": {},
                "log": [{"epoch": 0, "mean_soft_score": 67.95},
                        {"epoch": 2, "mean_soft_score": 70.82},
                        {"epoch": 10, "mean_soft_score": 56.08}]}
        info = checkpoint_summary(ckpt)
        self.assertEqual(info["epochs_done"], 24)
        self.assertEqual(info["best_epoch"], 2)
        self.assertTrue(info["has_best_model"])

    def test_handles_a_checkpoint_with_no_evaluations(self):
        info = checkpoint_summary({"next_epoch": 1, "log": []})
        self.assertIsNone(info["best_epoch"])
        self.assertFalse(info["has_best_model"])


class TestMergeGateVerdict(unittest.TestCase):
    @staticmethod
    def _arm(scores, fidelity=0.85):
        rows = [{"brief": f"b{i}", "kept": 1 if s else 0, "best_score": s,
                 "fidelity": fidelity if s else 0.0}
                for i, s in enumerate(scores)]
        planned = [r for r in rows if r["kept"]]
        return {"rows": rows, "n_briefs": len(rows),
                "briefs_with_plan": len(planned),
                "mean_best_score": round(sum(scores) / len(scores), 2),
                "mean_fidelity": fidelity}

    def test_a_clear_improvement_passes(self):
        verdict = evaluate(self._arm([70.0, 60.0, 80.0]),
                           self._arm([75.0, 63.0, 85.0]))
        self.assertTrue(verdict["passed"])
        self.assertGreater(verdict["score_delta"], 0)

    def test_one_big_regression_fails_even_if_the_mean_rises(self):
        verdict = evaluate(self._arm([70.0, 60.0, 80.0]),
                           self._arm([95.0, 50.0, 85.0]))
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["regressions"])

    def test_low_fidelity_fails(self):
        verdict = evaluate(self._arm([70.0, 60.0]),
                           self._arm([75.0, 65.0], fidelity=0.5))
        self.assertFalse(verdict["passed"])

    def test_losing_a_plan_fails(self):
        verdict = evaluate(self._arm([70.0, 60.0]), self._arm([95.0, 0.0]))
        self.assertFalse(verdict["passed"])

    def test_every_check_is_reported_pass_or_fail(self):
        verdict = evaluate(self._arm([70.0]), self._arm([75.0]))
        self.assertEqual(len(verdict["checks"]), 4)
        for check in verdict["checks"]:
            self.assertIn("pass", check)
            self.assertIn("detail", check)


if __name__ == "__main__":
    unittest.main()
