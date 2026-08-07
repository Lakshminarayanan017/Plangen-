"""
critic.py — Tier 5: the learned second opinion on ranking.

The reviewer decides what is ALLOWED (hard rules) and gives a defensible
score. The critic decides what is BETTER among plans the reviewer already
accepted. It never vetoes: a plan the rules disqualify stays disqualified no
matter how much the critic likes it, and a critic that has not been trained
(no weights on disk) leaves ranking exactly as it was.

Blend (implementation_plan_v2.md §4.3):

    rank = (1 - w) * soft_score + w * 100 * critic_probability

with w = EngineConfig.critic_weight (0.4 by default, 0 disables). The critic
is scaled to the soft score's 0-100 range so the weight means what it says.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from modules.step4_generate.critic import features as feat
from modules.step4_generate.critic.gbt import GBTModel
from modules.step4_generate.engine.contracts import Candidate, EngineConfig

log = logging.getLogger("PlanGen.Critic")

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
DEFAULT_MODEL_PATH = os.path.join(WEIGHTS_DIR, "critic_gbt.json")


class LearnedCritic:
    """Scores a candidate in [0, 100]. Feature order is verified against the
    trained model at construction — a critic silently reading the wrong
    columns is worse than no critic."""

    def __init__(self, model: GBTModel,
                 config: Optional[EngineConfig] = None):
        if model.feature_names and model.feature_names != feat.FEATURE_NAMES:
            raise ValueError(
                "critic model was trained on a different feature set "
                f"({len(model.feature_names)} features) than this build "
                f"provides ({feat.N_FEATURES}) — retrain it "
                f"(python -m critic.train)")
        self.model = model
        self.config = config or EngineConfig()

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH,
             config: Optional[EngineConfig] = None) -> "LearnedCritic":
        return cls(GBTModel.load(path), config)

    @classmethod
    def load_if_available(cls, path: str = DEFAULT_MODEL_PATH,
                          config: Optional[EngineConfig] = None
                          ) -> Optional["LearnedCritic"]:
        """None when there are no trained weights — the engine then ranks by
        the rules alone, which is the documented default, not a failure."""
        if not os.path.exists(path):
            log.info("no critic weights at %s; ranking by rules only", path)
            return None
        try:
            return cls.load(path, config)
        except Exception as exc:
            log.warning("critic weights at %s unusable (%s); "
                        "ranking by rules only", path, exc)
            return None

    # ── scoring ──────────────────────────────────────────────────────────
    def score_candidate(self, candidate: Candidate) -> float:
        """0-100. Requires a candidate that carries its request/room_ids."""
        if candidate.request is None or not candidate.room_ids:
            raise ValueError("candidate has no request/room_ids — it did "
                             "not come from the orchestrator")
        vector = feat.extract(candidate.plan, candidate.request,
                              candidate.room_ids, candidate.verdict,
                              self.config)
        return 100.0 * self.model.score_one(vector)

    def score_batch(self, candidates: List[Candidate]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        X = np.vstack([
            feat.extract(c.plan, c.request, c.room_ids, c.verdict,
                         self.config) for c in candidates])
        return 100.0 * self.model.predict_proba(X)

    # ── explanation ──────────────────────────────────────────────────────
    def explain_candidate(self, candidate: Candidate,
                          top: int = 6) -> Dict[str, float]:
        """The candidate's values for the features the model leans on most —
        so a ranking decision can always be shown, not just asserted."""
        vector = feat.extract(candidate.plan, candidate.request,
                              candidate.room_ids, candidate.verdict,
                              self.config)
        named = feat.as_dict(vector)
        return {name: named.get(name, 0.0)
                for name in list(self.model.gain_importance())[:top]}


# The blend itself lives in the engine (engine.orchestrator.blended_score),
# so the engine never imports the critic. Re-exported here because this is
# where a reader looks for it.
from modules.step4_generate.engine.orchestrator import blended_score          # noqa: E402,F401
