"""
semantic_matcher.py
===================
Production-grade Step 2 — Smart Semantic Matcher.

Replaces the placeholder matcher.py.

Given a BuildingRequirements, finds the K most similar real floor
plans from the CubiCasa5K index using cosine similarity on 28-dim
feature vectors (no GPU, no FAISS — pure NumPy, sub-ms on 5000 plans),
then builds a rich KnowledgeBundle from the matched plans' statistics.

Usage
-----
    matcher = SemanticMatcher()
    bundle = matcher.fetch_patterns(requirements)
    print(bundle.summary())

CLI smoke-test
--------------
    python -m modules.step2_match.semantic_matcher --test "3BHK 30x40 east-facing"
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import BuildingRequirements, KnowledgeBundle, MatchedPlanRef
from modules.step2_match.feature_encoder import FeatureEncoder
from modules.step2_match.stats_aggregator import StatsAggregator

log = logging.getLogger(__name__)

BASE      = Path(__file__).parents[2]
INDEX_DIR = BASE / "extracted data" / "plan_index"

# Number of nearest neighbours to retrieve
TOP_K = 15


class SemanticMatcher:
    """
    Production-grade semantic plan retrieval engine.

    Init cost  : ~50ms (loads 4983 × 28 float32 matrix + metadata)
    Query cost : <1ms  (cosine similarity via NumPy dot product)

    Architecture
    ------------
    1. FeatureEncoder  → BuildingRequirements → 28-dim query vector
    2. Cosine search   → top-K plan keys + similarity scores
    3. StatsAggregator → matched plan stats → KnowledgeBundle
    """

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        self._ready = False

        self._vectors:       Optional[np.ndarray]  = None   # (N, 28) float32
        self._plan_keys:     List[str]              = []
        self._metadata_list: List[Dict[str, Any]]  = []
        self._total_plans:   int                    = 0
        self._index_version: str                    = "unknown"

        self._encoder    = FeatureEncoder()
        self._aggregator = StatsAggregator()

        self._load_index()

    # ── Public API ─────────────────────────────────────────────────────

    def fetch_patterns(self, reqs: BuildingRequirements) -> KnowledgeBundle:
        """
        Main entry point for Step 2.

        Args:
            reqs: Parsed BuildingRequirements from Step 1.

        Returns:
            KnowledgeBundle with matched plan stats, room distributions,
            adjacency weights, zone probabilities, Vastu rules, and
            NBC standards.
        """
        t0 = time.perf_counter()
        log.info("Step 2 — Semantic Matcher: searching %d plans …", self._total_plans)

        if not self._ready:
            log.warning(
                "Plan index not found at %s. "
                "Run: python -m modules.data_prep.plan_indexer",
                INDEX_DIR,
            )
            return self._fallback_bundle(reqs)

        # ── 1. Encode query ─────────────────────────────────────────────
        query_vec = self._encoder.encode(reqs)           # shape (28,)
        # Capture raw BHK numeric BEFORE norm_params are applied so it can be
        # compared directly against the raw _bhk_numerics array stored at load time.
        raw_query_bhk = self._encoder._infer_bhk_numeric(reqs)

        # ── 2. Cosine similarity search ─────────────────────────────────
        matched_refs, similarities = self._cosine_search(query_vec, raw_query_bhk)

        retrieval_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "  Found %d matches in %.1fms (top similarity: %.3f)",
            len(matched_refs), retrieval_ms,
            similarities[0] if similarities else 0,
        )

        # ── 3. Build KnowledgeBundle ────────────────────────────────────
        bundle = self._aggregator.build_bundle(
            reqs          = reqs,
            matched_refs  = matched_refs,
            total_in_index = self._total_plans,
            retrieval_ms  = round(retrieval_ms, 2),
            index_version = self._index_version,
        )

        total_ms = (time.perf_counter() - t0) * 1000
        log.info("Step 2 complete in %.1fms — %s", total_ms, bundle.summary())
        return bundle

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def total_plans(self) -> int:
        return self._total_plans

    # ── Index loading ──────────────────────────────────────────────────

    def _load_index(self) -> None:
        """Load the pre-built plan index from disk."""
        vectors_file = INDEX_DIR / "plan_vectors.npy"
        metadata_file = INDEX_DIR / "plan_metadata.json"
        meta_file = INDEX_DIR / "index_metadata.json"

        if not vectors_file.exists() or not metadata_file.exists():
            log.warning(
                "Plan index not found. "
                "Build it first: python -m modules.data_prep.plan_indexer"
            )
            return

        try:
            t0 = time.perf_counter()

            # Load feature matrix
            self._vectors = np.load(vectors_file)     # (N, 28) float32
            # Pre-normalise rows for fast cosine similarity (dot product only)
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            self._vectors_normed = (self._vectors / norms).astype(np.float32)

            # Load metadata
            with metadata_file.open() as f:
                data = json.load(f)
            self._plan_keys     = data["plan_keys"]
            self._metadata_list = data["metadata"]

            # Load index metadata
            if meta_file.exists():
                with meta_file.open() as f:
                    idx_meta = json.load(f)
                self._index_version = idx_meta.get("version", "unknown")
                self._total_plans   = idx_meta.get("total_plans", len(self._plan_keys))

                # Pre-build vectorised BHK numeric array for fast post-scoring.
            # Storing it here avoids repeated string lookups on every query.
            _BHK_NUM = {"1bhk":0.25,"2bhk":0.50,"3bhk":0.75,"4bhk":1.00,"4bhk+":1.00,
                        "studio":0.0,"1":0.25,"2":0.50,"3":0.75,"4":1.00}
            self._bhk_numerics = np.array(
                [_BHK_NUM.get(str(m.get("bhk","unknown")).lower(), 0.5)
                 for m in self._metadata_list],
                dtype=np.float32,
            )   # shape (N,)

            self._ready = True
            load_ms = (time.perf_counter() - t0) * 1000
            log.info(
                "Plan index loaded: %d plans, dim=%d, %.1fms",
                self._total_plans, self._vectors.shape[1], load_ms,
            )

        except Exception as e:
            log.error("Failed to load plan index: %s", e)
            self._ready = False

    # ── Cosine similarity search ───────────────────────────────────────

    def _cosine_search(
        self,
        query_vec:     np.ndarray,
        raw_query_bhk: float = 0.0,
    ) -> Tuple[List[MatchedPlanRef], List[float]]:
        """
        Find the top-K most similar plans using cosine similarity,
        augmented with a BHK match bonus.

        BHK boost rationale
        -------------------
        In the 28-dim feature space, BHK occupies only 1 dimension (~3.6%
        weight in cosine similarity). Yet BHK is architecturally the most
        important predictor of room sizes, circulation needs, and floor area.
        A 3BHK query matched against 1BHK plans produces meaninglessly small
        room size distributions.

        We apply an additive BHK bonus AFTER computing cosine similarity:
          exact BHK match  (delta = 0)     → +0.06
          BHK off by 1 step (delta ≤ 0.26) → +0.02
          BHK mismatch (delta > 0.26)      → +0.00

        This is additive, not multiplicative, so it cannot overrule a very
        strong cosine match — it acts as a tiebreaker and slight re-ranking
        to bring BHK-correct plans to the top of the list.

        Pure NumPy — runs in <1ms for 5000 plans.

        Returns:
            (matched_refs, similarity_scores)
        """
        # Normalise query vector
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-8:
            q_norm = 1.0
        query_normed = (query_vec / q_norm).astype(np.float32)

        # Cosine similarity = dot product on pre-normalised vectors
        similarities = self._vectors_normed @ query_normed    # shape (N,)

        # ── BHK boost — vectorised, no Python loop ───────────────────
        # Compare raw_query_bhk (0.25 / 0.50 / 0.75 / 1.0) against the
        # pre-built _bhk_numerics array loaded from index metadata.
        if raw_query_bhk > 0.0 and hasattr(self, "_bhk_numerics"):
            bhk_delta   = np.abs(self._bhk_numerics - raw_query_bhk)
            bhk_bonus   = np.where(bhk_delta < 0.01, 0.06,
                          np.where(bhk_delta < 0.26, 0.02,
                                   0.0)).astype(np.float32)
            similarities = similarities + bhk_bonus  # new array, original untouched

        # Get top-K by BHK-boosted score (re-sort so boost affects ranking)
        k = min(self.top_k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        matched_refs: List[MatchedPlanRef] = []
        sim_scores:   List[float]          = []

        for idx in top_indices:
            # Report the boosted score as similarity_score so StatsAggregator
            # can use it for weighted averaging and quality score calculation.
            sim  = float(similarities[idx])
            meta = self._metadata_list[idx]

            matched_refs.append(MatchedPlanRef(
                plan_key         = self._plan_keys[idx],
                similarity_score = round(max(sim, 0.0), 4),
                bhk              = meta.get("bhk", "unknown"),
                room_count       = int(meta.get("room_count", 0)),
                aspect_ratio     = float(meta.get("aspect_ratio", 1.0)),
                zone_balance     = meta.get("zone_balance", {}),
            ))
            sim_scores.append(sim)

        return matched_refs, sim_scores

    # ── Fallback when index is unavailable ────────────────────────────

    def _fallback_bundle(self, reqs: BuildingRequirements) -> KnowledgeBundle:
        """
        Returns a minimal KnowledgeBundle built from NBC standards only.
        Used when the plan index hasn't been built yet.
        """
        from modules.step2_match.indian_standards import nbc_summary, get_setbacks, get_room_minimums
        from models import RoomStats, AdjacencyRule, RoomSizeDistribution, Setbacks

        log.warning("Using fallback bundle (no matched plans — index not built).")

        room_stats = {}
        room_size_dist = {}
        for room in reqs.rooms:
            rt_norm = room.room_type.lower().replace(" ", "_")
            nbc = get_room_minimums(room.room_type)
            room_stats[rt_norm] = RoomStats(
                min_width   = nbc["min_width_ft"],
                min_length  = nbc["min_length_ft"],
                target_area = nbc["min_area_sqft"] * 1.25,
            )
            room_size_dist[rt_norm] = RoomSizeDistribution(
                room_type        = rt_norm,
                min_width_ft     = nbc["min_width_ft"],
                p25_width_ft     = nbc["min_width_ft"] * 1.1,
                median_width_ft  = nbc["min_width_ft"] * 1.2,
                p75_width_ft     = nbc["min_width_ft"] * 1.4,
                max_width_ft     = nbc["min_width_ft"] * 2.0,
                min_area_sqft    = nbc["min_area_sqft"],
                p25_area_sqft    = nbc["min_area_sqft"] * 1.1,
                median_area_sqft = nbc["min_area_sqft"] * 1.25,
                p75_area_sqft    = nbc["min_area_sqft"] * 1.5,
                max_area_sqft    = nbc["min_area_sqft"] * 2.5,
                sample_count     = 0,
                source           = "nbc_fallback",
            )

        plot_area = (
            reqs.plot_dimensions.total_area_sqft
            if reqs.plot_dimensions and reqs.plot_dimensions.total_area_sqft
            else 1200.0
        )
        sb = get_setbacks(plot_area)

        return KnowledgeBundle(
            original_requirements   = reqs,
            room_stats              = room_stats,
            room_size_distributions = room_size_dist,
            nbc_standards           = nbc_summary(),
            setbacks_recommended    = Setbacks(**sb),
            index_version           = "fallback",
            total_plans_in_index    = 0,
        )

    # ── Introspection / debugging ──────────────────────────────────────

    def explain_query(self, reqs: BuildingRequirements) -> Dict[str, Any]:
        """
        Explain how a BuildingRequirements is encoded.
        Useful for debugging why certain plans are matched.
        """
        vec = self._encoder.encode(reqs)
        return {
            "feature_vector":  self._encoder.describe(vec),
            "top_k":           self.top_k,
            "total_plans":     self._total_plans,
            "index_version":   self._index_version,
            "index_ready":     self._ready,
        }


# ── Backward compatibility alias ───────────────────────────────────────
# Keeps old code using PatternMatcher working during migration
class PatternMatcher(SemanticMatcher):
    """Backward-compatible alias for SemanticMatcher."""
    pass


# ── CLI smoke-test ─────────────────────────────────────────────────────

def _run_test(description: str) -> None:
    """Quick CLI test of the matcher."""
    from models import (
        BuildingRequirements, PlotDimensions, PlotContext,
        RoomRequirement, Direction,
    )

    print(f"\n{'='*60}")
    print(f"SemanticMatcher Smoke Test")
    print(f"Query: '{description}'")
    print(f"{'='*60}")

    reqs = BuildingRequirements(
        plot_dimensions = PlotDimensions(length=40.0, width=30.0, unit="ft"),
        plot_context    = PlotContext(
            entrance_side   = Direction.EAST,
            north_direction = Direction.NORTH,
        ),
        number_of_floors = 2,
        vastu_compliant  = True,
        rooms = [
            RoomRequirement(room_type="Master Bedroom", quantity=1),
            RoomRequirement(room_type="Bedroom",        quantity=2),
            RoomRequirement(room_type="Kitchen",        quantity=1),
            RoomRequirement(room_type="Living Room",    quantity=1),
            RoomRequirement(room_type="Dining Room",    quantity=1),
            RoomRequirement(room_type="Bathroom",       quantity=2),
            RoomRequirement(room_type="Pooja Room",     quantity=1),
            RoomRequirement(room_type="Staircase",      quantity=1),
        ],
    )

    matcher = SemanticMatcher()

    explain = matcher.explain_query(reqs)
    print("\nQuery encoding:")
    for k, v in explain["feature_vector"].items():
        if v > 0:
            print(f"  {k:<30} = {v:.4f}")

    bundle = matcher.fetch_patterns(reqs)
    s = bundle.summary()
    print(f"\nKnowledgeBundle summary:")
    for k, v in s.items():
        print(f"  {k:<30} = {v}")

    print(f"\nTop 5 matched plans:")
    for ref in bundle.matched_plans[:5]:
        print(f"  {ref.plan_key:<50} sim={ref.similarity_score:.4f}  bhk={ref.bhk}")

    print(f"\nRoom size distributions:")
    for rt, dist in bundle.room_size_distributions.items():
        print(f"  {rt:<25}  "
              f"min={dist.min_area_sqft:.0f} "
              f"med={dist.median_area_sqft:.0f} "
              f"max={dist.max_area_sqft:.0f} sqft  |  "
              f"w: {dist.min_width_ft:.1f}–{dist.median_width_ft:.1f}–{dist.max_width_ft:.1f} ft")

    print(f"\nAdjacency weights (strongest pairs):")
    all_pairs = [
        (a, b, w)
        for a, nbrs in bundle.adjacency_weights.items()
        for b, w in nbrs.items()
    ]
    for a, b, w in sorted(all_pairs, key=lambda x: -x[2])[:8]:
        print(f"  {a:<20} ↔ {b:<20} weight={w:.2f}")

    print(f"\nSetbacks recommended: "
          f"F={bundle.setbacks_recommended.front}ft "
          f"R={bundle.setbacks_recommended.rear}ft "
          f"L={bundle.setbacks_recommended.left}ft "
          f"R={bundle.setbacks_recommended.right}ft")

    print(f"\nVastu enabled: {bool(bundle.vastu_rules_applied)}")
    print(f"Retrieval time: {bundle.retrieval_time_ms:.1f}ms")
    print("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    desc = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "3BHK 30x40 east-facing vastu"
    _run_test(desc)
