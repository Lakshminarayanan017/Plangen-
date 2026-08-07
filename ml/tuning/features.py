"""
features.py — exact linear decomposition of a plan's soft score.

The whole tuner rests on one structural fact: for a FIXED plan, every soft
rule contributes `weight * measure` to the score, so

    soft_score(w) = 100  -  Σ_penalty w_k · m_k  +  Σ_bonus w_k · m_k
                  = 100  +  w · g

where g_k = -m_k for a penalty weight and +m_k for a bonus weight (m_k is
the *transformed* measure the rule multiplies its weight by — the hinges,
caps and scale factors are already folded into m_k). g is the plan's
"feature vector" in weight space; it does not depend on the weights.

We recover g WITHOUT editing the engine: re-score the fixed plan with a
probe config that sets exactly one weight to a large value and the rest to
zero. Then g_k = (score_probe - 100) / probe. Because the score is exactly
affine in the weights, this is exact (we read the unrounded 100-penalty
+bonus directly, not the validator's 2-dp rounded field). The plan's
geometry is computed once and reused across all 20 probes via the
ReviewContext caches, so extraction is cheap.

Why not instrument the rules instead? Editing 20 penalty sites in the
engine's hot path is higher risk for zero functional gain; this module is
purely additive and the reconstruction test (tests/test_tuning.py) proves
it reproduces the validator's score on every real candidate.

Note FSP-001 is registered "hard" but adds a `w_freespace` *soft* penalty
when config.fsp001_hard is False — so we must run the FULL rule set, not
only soft-severity rules, or w_freespace would read as zero.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from modules.step4_generate.core.grid_plan import GridPlan
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest
from modules.step4_generate.engine.rules import RULES, ReviewContext

# The 20 tunable soft-score weights, in a fixed canonical order. This order
# defines the layout of every feature/weight vector the tuner builds.
WEIGHT_KEYS: List[str] = [
    "w_area_drift", "w_narrow", "w_aspect", "w_nbc",
    "w_wide_bonus", "w_privacy_door", "w_wet_social", "w_circ_depth",
    "w_daylight", "w_window", "w_freespace", "w_circ_waste",
    "w_passage_width", "w_hub_centrality", "w_wall_efficiency", "w_wall_jog",
    "w_toothpick", "w_hinge_stub", "w_swing_direction", "w_cross_vent",
]

# Weights whose rule adds to ctx.bonus (raise the score) rather than penalty.
BONUS_KEYS = frozenset({"w_wide_bonus", "w_cross_vent"})

# Probe magnitude: large enough that (score - 100) carries the feature with
# full float precision; the arithmetic is exact regardless of the value.
_PROBE = 1000.0

# All 20 weights zeroed — the base for a single-weight probe.
_ZERO_WEIGHTS = {k: 0.0 for k in WEIGHT_KEYS}


def room_ids_for(plan: GridPlan, request: EngineRequest) -> Dict[str, int]:
    """Reconstruct the name→grid-id map the validator needs from a plan.

    Room names are unique within a request (EngineRequest.validate enforces
    it) and the realizer names each face after its spec, so this recovers
    exactly the room_ids the orchestrator passed to the validator.
    """
    return {spec.name: plan.find_room(spec.name).id for spec in request.rooms}


def extract_features(plan: GridPlan, request: EngineRequest,
                     room_ids: Dict[str, int],
                     base_config: EngineConfig) -> Dict[str, float]:
    """Return the signed feature vector g (keyed by weight name) such that
    `soft_score(w) = 100 + Σ_k w_k · g[k]` for this plan.

    `base_config` supplies every NON-weight field (notably fsp001_hard,
    which routes FSP-001 between hard and soft); only the 20 weights are
    overridden per probe.
    """
    # One context, geometry computed lazily and then reused across probes.
    ctx = ReviewContext(plan=plan, request=request, room_ids=room_ids,
                         config=base_config)
    features: Dict[str, float] = {}
    for key in WEIGHT_KEYS:
        probe_cfg = replace(base_config, **{**_ZERO_WEIGHTS, key: _PROBE})
        raw = _soft_total(ctx, probe_cfg)
        features[key] = (raw - 100.0) / _PROBE
    return features


def reconstruct_score(features: Dict[str, float],
                      config: EngineConfig) -> float:
    """Rebuild soft_score from a feature vector and a weight config:
    `100 + Σ_k w_k · g[k]`. Matches BasicValidator (pre-rounding)."""
    return 100.0 + sum(getattr(config, k) * features.get(k, 0.0)
                       for k in WEIGHT_KEYS)


def weight_vector(config: EngineConfig) -> List[float]:
    """The config's 20 weights as a plain list in WEIGHT_KEYS order."""
    return [getattr(config, k) for k in WEIGHT_KEYS]


def with_weights(config: EngineConfig, weights: Dict[str, float]
                 ) -> EngineConfig:
    """Copy of `config` with the given weights overridden (others kept)."""
    return replace(config, **{k: float(v) for k, v in weights.items()
                              if k in set(WEIGHT_KEYS)})


def _soft_total(ctx: ReviewContext, config: EngineConfig) -> float:
    """Run the full rule set on the shared context under `config` and return
    the unrounded 100 - penalty + bonus. Resets the mutable accumulators so
    the cached geometry (shared walls, opening graph, free component) is the
    only state carried between probes."""
    ctx.config = config
    ctx.penalty = 0.0
    ctx.bonus = 0.0
    ctx.hard = []
    ctx.breakdown = {}
    for _rule_id, _severity, fn in RULES:
        fn(ctx)
    return 100.0 - ctx.penalty + ctx.bonus
