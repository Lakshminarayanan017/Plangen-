import json
import logging
from pathlib import Path
from typing import Any, Dict

from models import BuildingRequirements, KnowledgeBundle, RoomStats, AdjacencyRule

logger = logging.getLogger(__name__)

# Locate the sources directory
SOURCES_DIR = Path(__file__).parent.parent.parent / "sources"

class PatternMatcher:
    """
    Step 2: Fetch Matching Patterns.
    Takes parsed requirements and enriches them with statistical patterns,
    architectural standards, and optional Vastu rules.
    """

    def __init__(self):
        self.learned_patterns = self._load_json("learned_patterns.json")
        self.indian_standards = self._load_json("indian_standards.json")
        self.vastu_rules = self._load_json("vastu_rules.json")

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Utility to load mock JSON databases."""
        filepath = SOURCES_DIR / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        logger.warning(f"Database file {filename} not found.")
        return {}

    def fetch_patterns(self, reqs: BuildingRequirements) -> KnowledgeBundle:
        """
        Main execution for Step 2.
        Builds and returns the KnowledgeBundle.
        """
        logger.info("Step 2: Fetching matching patterns and standards...")

        # 1. Initialize empty bundle with original requirements
        bundle = KnowledgeBundle(original_requirements=reqs)

        # 2. Extract Room Stats for requested rooms
        room_averages = self.learned_patterns.get("room_averages", {})
        for room_req in reqs.rooms:
            room_type = room_req.room_type
            if room_type in room_averages:
                # Convert dict to Pydantic RoomStats
                bundle.room_stats[room_type] = RoomStats(**room_averages[room_type])
            else:
                # Fallback / default size if room pattern not found
                bundle.room_stats[room_type] = RoomStats(
                    min_width=8.0, min_length=8.0, target_area=64.0
                )

        # 3. Extract Adjacencies for requested rooms
        common_adj = self.learned_patterns.get("common_adjacencies", [])
        requested_room_types = {r.room_type for r in reqs.rooms}
        for adj in common_adj:
            # Only add the rule if both rooms are actually requested by the user
            if adj["room_a"] in requested_room_types and adj["room_b"] in requested_room_types:
                bundle.adjacency_rules.append(AdjacencyRule(**adj))

        # 4. Floor distribution suggestions
        # Apply only if the building is multi-story (G+1 or G+2)
        if reqs.number_of_floors and reqs.number_of_floors > 1:
            dist = self.learned_patterns.get("typical_floor_distribution", {})
            for room_type, floor in dist.items():
                if room_type in requested_room_types:
                    bundle.floor_distribution_suggestions[room_type] = floor

        # 5. Apply Indian Standards
        bundle.standards_applied = self.indian_standards

        # 6. Apply Vastu Rules conditionally
        if reqs.vastu_compliant:
            logger.info("Vastu compliance requested. Loading Vastu rules...")
            bundle.vastu_rules_applied = self.vastu_rules
        else:
            logger.info("Vastu compliance NOT requested. Skipping Vastu rules.")

        logger.info("Knowledge Bundle successfully created.")
        return bundle
