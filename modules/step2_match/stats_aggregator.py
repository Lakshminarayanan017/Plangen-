"""
stats_aggregator.py
===================
Aggregates statistical data from the top-K matched plans and the
global index datasets into a rich KnowledgeBundle.

Called by SemanticMatcher after the similarity search returns K results.
Produces per-room size distributions, adjacency weights filtered to
the requested rooms, zone probabilities, and circulation benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import (
    AdjacencyRule,
    BuildingRequirements,
    KnowledgeBundle,
    MatchedPlanRef,
    RoomSizeDistribution,
    RoomStats,
    Setbacks,
)
from modules.step2_match.indian_standards import (
    get_room_minimums,
    get_setbacks,
    nbc_summary,
)

BASE      = Path(__file__).parents[2]
INDEX_DIR = BASE / "extracted data" / "plan_index"
DATA_DIR  = BASE / "extracted data"   # all dataset JSONs live here


class StatsAggregator:
    """
    Aggregates global index statistics + matched plan data →
    a fully-populated KnowledgeBundle.

    Loads once at init; called per request.
    """

    def __init__(self) -> None:
        self._room_stats    = self._load_json(INDEX_DIR / "room_stats_by_bhk.json")
        self._adj_weights   = self._load_json(INDEX_DIR / "adjacency_weights.json")
        self._zone_probs    = self._load_json(INDEX_DIR / "zone_probs.json")
        self._circ_meta     = self._load_json(INDEX_DIR / "circulation_meta.json")
        self._vastu_rules   = self._load_json(DATA_DIR  / "vastu_rules.json")
        self._nbc_standards = nbc_summary()
        self._nbc_plot_regs = self._load_nbc_plot_regulations()

    # ── Public API ─────────────────────────────────────────────────────

    def build_bundle(
        self,
        reqs:          BuildingRequirements,
        matched_refs:  List[MatchedPlanRef],
        total_in_index: int,
        retrieval_ms:   float,
        index_version:  str,
    ) -> KnowledgeBundle:
        """
        Build the complete KnowledgeBundle from:
          • User requirements
          • List of matched plan references
          • Global index statistics
        """
        bundle = KnowledgeBundle(
            original_requirements = reqs,
            matched_plans         = matched_refs,
            match_quality_score   = self._compute_quality_score(matched_refs),
            total_plans_in_index  = total_in_index,
            retrieval_time_ms     = retrieval_ms,
            index_version         = index_version,
        )

        # ── Room size distributions ─────────────────────────────────────
        requested_types = self._get_requested_room_types(reqs)
        bundle.room_size_distributions = self._build_room_distributions(
            requested_types, matched_refs
        )

        # ── Legacy room_stats (backward compat) ────────────────────────
        bundle.room_stats = {
            rt: RoomStats(
                min_width   = dist.min_width_ft,
                min_length  = dist.min_width_ft,   # approximate
                target_area = dist.median_area_sqft,
            )
            for rt, dist in bundle.room_size_distributions.items()
        }

        # ── Adjacency weights ──────────────────────────────────────────
        bundle.adjacency_weights = self._build_adjacency(requested_types)

        # ── Legacy adjacency_rules (backward compat) ───────────────────
        bundle.adjacency_rules = [
            AdjacencyRule(room_a=a, room_b=b, relation="adjacent", weight=min(w, 10.0))
            for a, neighbours in bundle.adjacency_weights.items()
            for b, w in neighbours.items()
            if w >= 1.0   # only strong connections in legacy list
        ]

        # ── Zone probabilities ─────────────────────────────────────────
        bundle.zone_probabilities = self._build_zone_probs(requested_types)

        # ── Circulation benchmarks ─────────────────────────────────────
        bundle.circulation_benchmarks = dict(self._circ_meta)

        # ── Floor distribution suggestions ─────────────────────────────
        bundle.floor_distribution_suggestions = self._build_floor_dist(reqs)

        # ── Vastu rules ────────────────────────────────────────────────
        if reqs.vastu_compliant:
            bundle.vastu_rules_applied = self._vastu_rules

        # ── NBC standards ──────────────────────────────────────────────
        bundle.nbc_standards      = self._nbc_standards
        bundle.nbc_plot_regulations = self._nbc_plot_regs

        # ── Recommended setbacks ───────────────────────────────────────
        plot_area = (
            reqs.plot_dimensions.total_area_sqft
            if reqs.plot_dimensions and reqs.plot_dimensions.total_area_sqft
            else 1200.0   # default 1200 sqft plot
        )
        sb = get_setbacks(plot_area)
        bundle.setbacks_recommended = Setbacks(
            front=sb["front"], rear=sb["rear"],
            left=sb["left"],   right=sb["right"],
        )

        return bundle

    # ── Private helpers ────────────────────────────────────────────────

    def _build_room_distributions(
        self,
        requested_types:  List[str],
        matched_refs:     List[MatchedPlanRef],
    ) -> Dict[str, RoomSizeDistribution]:
        """
        Build RoomSizeDistribution for each requested room type.
        Uses global room_stats (from learned_patterns + NBC minimums).
        The matched_refs BHK weighting adjusts the estimates slightly.
        """
        distributions: Dict[str, RoomSizeDistribution] = {}

        # Compute a BHK adjustment factor from matched plans
        bhk_factor = self._bhk_scale_factor(matched_refs)

        for room_type in requested_types:
            rt_norm = room_type.lower().replace(" ", "_")

            # Look up raw stats
            raw = self._room_stats.get(rt_norm)
            if raw is None:
                # Try fuzzy match
                for key in self._room_stats:
                    if key in rt_norm or rt_norm in key:
                        raw = self._room_stats[key]
                        break

            nbc_min = get_room_minimums(room_type)

            if raw:
                # Scale by BHK factor and ensure NBC minimums are respected
                min_w = max(float(raw.get("min_width_ft",    7.0)) * bhk_factor,
                            nbc_min["min_width_ft"])
                p25_w = max(float(raw.get("p25_width_ft",    9.0)) * bhk_factor,
                            nbc_min["min_width_ft"])
                med_w = max(float(raw.get("median_width_ft",10.0)) * bhk_factor,
                            nbc_min["min_width_ft"])
                p75_w = max(float(raw.get("p75_width_ft",   12.0)) * bhk_factor,
                            nbc_min["min_width_ft"])
                max_w = max(float(raw.get("max_width_ft",   16.0)) * bhk_factor,
                            nbc_min["min_width_ft"])

                min_a = max(float(raw.get("min_area_sqft",  64.0)) * bhk_factor**2,
                            nbc_min["min_area_sqft"])
                p25_a = max(float(raw.get("p25_area_sqft",  80.0)) * bhk_factor**2,
                            nbc_min["min_area_sqft"])
                med_a = max(float(raw.get("median_area_sqft", 100.0)) * bhk_factor**2,
                            nbc_min["min_area_sqft"])
                p75_a = max(float(raw.get("p75_area_sqft", 130.0)) * bhk_factor**2,
                            nbc_min["min_area_sqft"])
                max_a = max(float(raw.get("max_area_sqft", 200.0)) * bhk_factor**2,
                            nbc_min["min_area_sqft"])
                samples = int(raw.get("sample_count", 0))
            else:
                # Pure NBC fallback
                min_w = nbc_min["min_width_ft"]
                p25_w = nbc_min["min_width_ft"] * 1.1
                med_w = nbc_min["min_width_ft"] * 1.2
                p75_w = nbc_min["min_width_ft"] * 1.4
                max_w = nbc_min["min_width_ft"] * 2.0
                min_a = nbc_min["min_area_sqft"]
                p25_a = nbc_min["min_area_sqft"] * 1.1
                med_a = nbc_min["min_area_sqft"] * 1.25
                p75_a = nbc_min["min_area_sqft"] * 1.5
                max_a = nbc_min["min_area_sqft"] * 2.5
                samples = 0

            distributions[rt_norm] = RoomSizeDistribution(
                room_type        = rt_norm,
                min_width_ft     = round(min_w, 2),
                p25_width_ft     = round(p25_w, 2),
                median_width_ft  = round(med_w, 2),
                p75_width_ft     = round(p75_w, 2),
                max_width_ft     = round(max_w, 2),
                min_area_sqft    = round(min_a, 2),
                p25_area_sqft    = round(p25_a, 2),
                median_area_sqft = round(med_a, 2),
                p75_area_sqft    = round(p75_a, 2),
                max_area_sqft    = round(max_a, 2),
                sample_count     = samples,
            )

        return distributions

    def _build_adjacency(self, requested_types: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Return adjacency weights filtered to rooms actually requested.
        Adds indirect connections if a direct one isn't in the index.
        """
        rt_set = {rt.lower().replace(" ", "_") for rt in requested_types}
        result: Dict[str, Dict[str, float]] = {}

        for a in rt_set:
            # Find this room's neighbours in the index
            neighbours = self._adj_weights.get(a, {})
            filtered = {
                b: w
                for b, w in neighbours.items()
                if b in rt_set and b != a
            }
            if filtered:
                result[a] = filtered

        return result

    def _build_zone_probs(self, requested_types: List[str]) -> Dict[str, Dict[str, float]]:
        """Zone probability dict filtered to requested room types."""
        result = {}
        for room_type in requested_types:
            rt_norm = room_type.lower().replace(" ", "_")
            if rt_norm in self._zone_probs:
                result[rt_norm] = self._zone_probs[rt_norm]
            else:
                # Fuzzy match
                for key in self._zone_probs:
                    if key in rt_norm or rt_norm in key:
                        result[rt_norm] = self._zone_probs[key]
                        break
                else:
                    result[rt_norm] = {"front": 0.33, "middle": 0.34, "back": 0.33}
        return result

    def _build_floor_dist(self, reqs: BuildingRequirements) -> Dict[str, int]:
        """Default floor distribution based on Indian residential norms."""
        floors = reqs.number_of_floors or 1
        dist: Dict[str, int] = {}

        for room in reqs.rooms:
            rt = room.room_type.lower().replace(" ", "_")

            if "car_parking" in rt or "car parking" in rt:
                dist[room.room_type] = 0
            elif "living" in rt or "drawing" in rt or "dining" in rt:
                dist[room.room_type] = 0
            elif "kitchen" in rt:
                dist[room.room_type] = 0
            elif "pooja" in rt or "puja" in rt:
                dist[room.room_type] = 0
            elif "utility" in rt or "servant" in rt:
                dist[room.room_type] = 0
            elif "master" in rt and floors > 1:
                dist[room.room_type] = 1
            elif "bedroom" in rt and floors > 1:
                dist[room.room_type] = 1 if floors == 2 else 2
            elif "study" in rt and floors > 1:
                dist[room.room_type] = 1
            else:
                dist[room.room_type] = 0

        return dist

    def _get_requested_room_types(self, reqs: BuildingRequirements) -> List[str]:
        """Flatten all room types from requirements, expanding quantities."""
        types: List[str] = []
        seen: set = set()
        for room in reqs.rooms:
            rt_norm = room.room_type.lower().replace(" ", "_")
            if rt_norm not in seen:
                types.append(room.room_type)
                seen.add(rt_norm)
        return types

    def _bhk_scale_factor(self, matched_refs: List[MatchedPlanRef]) -> float:
        """
        Compute a size-scaling factor based on matched plans' BHK types.
        3BHK matched plans → factor ~1.0
        2BHK matched plans → factor ~0.85
        4BHK+ matched plans → factor ~1.15
        """
        if not matched_refs:
            return 1.0

        BHK_SCALE = {
            "1BHK": 0.75, "2BHK": 0.90, "3BHK": 1.0,
            "4BHK": 1.1, "4BHK+": 1.15,
        }
        # Weighted average by similarity score
        total_weight = sum(r.similarity_score for r in matched_refs)
        if total_weight < 1e-6:
            return 1.0

        weighted_scale = sum(
            BHK_SCALE.get(r.bhk, 1.0) * r.similarity_score
            for r in matched_refs
        )
        return round(weighted_scale / total_weight, 3)

    def _compute_quality_score(self, matched_refs: List[MatchedPlanRef]) -> float:
        """Average similarity of top-K matches, as a quality indicator."""
        if not matched_refs:
            return 0.0
        return round(
            sum(r.similarity_score for r in matched_refs) / len(matched_refs), 4
        )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_nbc_plot_regulations(self) -> Dict[str, Any]:
        """Load NBC plot regulations from extracted_data if available."""
        nbc_plot = BASE / "sources" / "extracted_data" / "nbc_plot_regulations.json"
        if nbc_plot.exists():
            with nbc_plot.open() as f:
                return json.load(f)
        # Hardcoded essentials if file not present
        return {
            "max_ground_coverage": 0.60,
            "far_residential":     1.5,
            "max_floors":          3,
            "setback_table":       "see indian_standards.py",
        }
