"""
Interactive Detail Gatherer — Continuous Question-Answer Loop Engine
====================================================================
Production-grade module that inspects partially-extracted BuildingRequirements
and drives a priority-ordered interactive loop to fill in missing information.

Design
------
* Operates BETWEEN the initial parse/image-analysis and the final gatekeeper
  validation — it's a pre-validation enrichment loop.
* Priority-ordered: Tier 1 (blockers) are asked first, then Tier 2 (important),
  then Tier 3 (nice-to-have). The loop stops once Tier 1 is satisfied.
* Generates ONE question at a time using the architect persona (Gemini Flash).
* After each user answer, merges the response into the existing JSON using
  the existing merge_and_reparse infrastructure.
* Stateful per-session — tracks which questions have been asked to avoid repeats.
* Integrates seamlessly with Module1Pipeline's session management.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import IntEnum

from pydantic import BaseModel
from dotenv import load_dotenv

# Official Google GenAI SDK
from google import genai
from google.genai import types

# Project models
from models import BuildingRequirements

logger = logging.getLogger("PlanGen.InteractiveGatherer")


# =====================================================================
# 1. PRIORITY TIER DEFINITIONS
# =====================================================================
class Tier(IntEnum):
    """Priority tiers for missing information, ordered by criticality."""
    TIER_1 = 1  # Critical blockers — system CANNOT proceed without these
    TIER_2 = 2  # Important — significantly improve the plan quality
    TIER_3 = 3  # Nice-to-have — system has sensible defaults


class MissingField:
    """
    Represents a single missing field with its priority and metadata.
    """

    def __init__(
        self,
        field_key: str,
        display_name: str,
        description: str,
        tier: Tier,
    ):
        self.field_key = field_key
        self.display_name = display_name
        self.description = description
        self.tier = tier

    def __repr__(self):
        return f"MissingField({self.field_key}, Tier {self.tier})"


# =====================================================================
# 2. REQUIREMENTS INSPECTOR
# =====================================================================
class RequirementsInspector:
    """
    Inspects a BuildingRequirements dict and determines which fields
    are missing, categorized by priority tier.

    This is the brain of the interactive loop — it knows EXACTLY what
    data is essential, what's important, and what's optional.
    """

    @staticmethod
    def inspect(data: Dict[str, Any]) -> List[MissingField]:
        """
        Analyze the current BuildingRequirements data and return an
        ordered list of missing fields (Tier 1 first, then Tier 2, then Tier 3).

        Args:
            data: BuildingRequirements as a dict.

        Returns:
            List of MissingField objects, sorted by priority (most critical first).
        """
        missing: List[MissingField] = []

        # ---- TIER 1: SYSTEM BLOCKERS ----

        # 1a. Plot dimensions — both length AND width needed
        plot_dims = data.get("plot_dimensions")
        if plot_dims is None:
            missing.append(MissingField(
                field_key="plot_dimensions",
                display_name="Plot Dimensions",
                description=(
                    "The physical size of the plot (length and width in feet). "
                    "Without this, it is impossible to determine the building footprint, "
                    "setbacks, or room proportions."
                ),
                tier=Tier.TIER_1,
            ))
        else:
            if plot_dims.get("length") is None and plot_dims.get("width") is None:
                # Has plot_dimensions object but no actual values
                if plot_dims.get("total_area_sqft") is None:
                    missing.append(MissingField(
                        field_key="plot_dimensions",
                        display_name="Plot Dimensions",
                        description=(
                            "The plot dimensions object exists but has no length, width, or area. "
                            "We need at least the plot size (e.g., 30x40 ft)."
                        ),
                        tier=Tier.TIER_1,
                    ))
                # If only area is present, we still need length+width
                elif plot_dims.get("length") is None or plot_dims.get("width") is None:
                    missing.append(MissingField(
                        field_key="plot_dimensions.boundaries",
                        display_name="Complete Plot Boundaries",
                        description=(
                            f"We have the total area ({plot_dims.get('total_area_sqft')} sqft) "
                            "but need both length and width to plan the layout. "
                            "Area alone doesn't tell us the plot shape."
                        ),
                        tier=Tier.TIER_1,
                    ))
            elif plot_dims.get("length") is None or plot_dims.get("width") is None:
                # Only one dimension present
                known = "length" if plot_dims.get("length") else "width"
                known_val = plot_dims.get("length") or plot_dims.get("width")
                missing_side = "width" if known == "length" else "length"
                missing.append(MissingField(
                    field_key=f"plot_dimensions.{missing_side}",
                    display_name=f"Plot {missing_side.title()}",
                    description=(
                        f"We have the plot {known} ({known_val} ft) but need the "
                        f"{missing_side} to complete the boundary for layout planning."
                    ),
                    tier=Tier.TIER_1,
                ))

        # 1b. Rooms — at least ONE room must be specified
        rooms = data.get("rooms", [])
        if not rooms or len(rooms) == 0:
            missing.append(MissingField(
                field_key="rooms",
                display_name="Room Configuration",
                description=(
                    "No rooms have been specified. We need at least the basic room layout "
                    "(e.g., 2BHK, 3BHK, or a custom list of rooms)."
                ),
                tier=Tier.TIER_1,
            ))

        # 1c. Road-facing direction
        plot_ctx = data.get("plot_context")
        if plot_ctx is None or not plot_ctx.get("road_facing_sides"):
            missing.append(MissingField(
                field_key="plot_context.road_facing_sides",
                display_name="Plot Facing Direction",
                description=(
                    "Which side(s) of the plot face the road. This determines entrance "
                    "placement, natural ventilation flow, and Vastu orientation."
                ),
                tier=Tier.TIER_1,
            ))

        # ---- TIER 2: IMPORTANT ----

        # 2a. Number of floors
        if data.get("number_of_floors") is None:
            missing.append(MissingField(
                field_key="number_of_floors",
                display_name="Number of Floors",
                description=(
                    "Total number of floors (Ground only, G+1, or G+2). Impacts structural "
                    "design and room distribution across floors."
                ),
                tier=Tier.TIER_2,
            ))

        # 2b. Vastu compliance
        if data.get("vastu_compliant") is None:
            missing.append(MissingField(
                field_key="vastu_compliant",
                display_name="Vastu Compliance",
                description=(
                    "Whether to follow Vastu Shastra principles for room orientations "
                    "(kitchen placement, master bedroom direction, entrance position)."
                ),
                tier=Tier.TIER_2,
            ))

        # 2c. Entrance side (if road-facing is known but entrance is not)
        if plot_ctx and plot_ctx.get("road_facing_sides") and not plot_ctx.get("entrance_side"):
            missing.append(MissingField(
                field_key="plot_context.entrance_side",
                display_name="Entrance Side",
                description=(
                    "The preferred main entrance direction. Usually the same as the "
                    "primary road-facing side, but some clients prefer a side entrance."
                ),
                tier=Tier.TIER_2,
            ))

        # ---- TIER 3: NICE-TO-HAVE ----

        # 3a. Parking
        if data.get("parking_type") is None:
            missing.append(MissingField(
                field_key="parking_type",
                display_name="Parking Type",
                description=(
                    "Type of vehicle parking: stilt, garage, marked area, or none."
                ),
                tier=Tier.TIER_3,
            ))

        # 3b. Building type
        if data.get("building_type") is None:
            missing.append(MissingField(
                field_key="building_type",
                display_name="Building Type",
                description=(
                    "Type of home: villa, bungalow, independent house, duplex, etc."
                ),
                tier=Tier.TIER_3,
            ))

        # 3c. Architectural style
        if data.get("architectural_style") is None:
            missing.append(MissingField(
                field_key="architectural_style",
                display_name="Architectural Style",
                description=(
                    "Design style preference: modern, contemporary, traditional, etc."
                ),
                tier=Tier.TIER_3,
            ))

        # Sort by tier (already in order from above, but enforce)
        missing.sort(key=lambda m: m.tier)

        return missing


# =====================================================================
# 3. QUESTION GENERATOR
# =====================================================================
class QuestionGenerator:
    """
    Generates architect-persona questions for specific missing fields
    using Gemini Flash for natural, warm, contextual responses.
    """

    def __init__(self, config):
        """
        Args:
            config: ParserConfig (from the main parser) — provides key rotator.
        """
        self.config = config
        self._question_template: Optional[str] = None

    @property
    def question_template(self) -> str:
        """Lazy-load the question prompt template."""
        if self._question_template is None:
            from docs.prompts.loader import load_prompt
            self._question_template = load_prompt("step1_interactive_question.md")
        return self._question_template

    def generate_question(
        self,
        current_data: Dict[str, Any],
        missing_field: MissingField,
    ) -> str:
        """
        Generate a single architect-persona question for the given missing field.

        Args:
            current_data: The current BuildingRequirements dict.
            missing_field: The MissingField to ask about.

        Returns:
            The question string (plain text, 2-4 sentences).
        """
        # Prepare the template context
        prompt = self.question_template.format(
            current_data=json.dumps(current_data, indent=2, default=str),
            missing_field=missing_field.display_name,
            field_description=missing_field.description,
            priority_tier=f"Tier {missing_field.tier}",
        )

        # Call Gemini Flash for a warm, persona-driven question
        for _attempt in range(self.config.key_rotator.key_count + 1):
            client, slot_idx = self.config.key_rotator.get_client()
            try:
                response = client.models.generate_content(
                    model=self.config.FLASH_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.6,  # Slightly warm for conversational tone
                        max_output_tokens=250,
                    ),
                )
                return response.text.strip()

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or ("resource" in error_str and "exhausted" in error_str):
                    logger.warning(
                        "429 rate-limit on question generation (slot %d): %s",
                        slot_idx,
                        e,
                    )
                    self.config.key_rotator.report_rate_limited(slot_idx)
                    continue
                logger.error("Failed to generate question: %s", e)
                break

        # Fallback: static question if LLM call fails
        return self._fallback_question(missing_field)

    @staticmethod
    def _fallback_question(missing_field: MissingField) -> str:
        """Generate a simple static fallback question when LLM fails."""
        fallbacks = {
            "plot_dimensions": (
                "Namaste. To begin the layout, could you share your plot dimensions? "
                "For example, is it a 30x40 plot, or perhaps a different size?"
            ),
            "plot_dimensions.boundaries": (
                "I have your plot area, but I need the exact length and width "
                "to plan the rooms properly. Could you share those?"
            ),
            "rooms": (
                "What room configuration are you looking for? A 2BHK, 3BHK, "
                "or a custom set of rooms?"
            ),
            "plot_context.road_facing_sides": (
                "Which direction does your plot face? Is the road on the north side, "
                "south, east, or west?"
            ),
            "number_of_floors": (
                "How many floors are you planning? Ground only, G+1, or G+2?"
            ),
            "vastu_compliant": (
                "Would you like the layout to follow Vastu Shastra principles?"
            ),
            "plot_context.entrance_side": (
                "Where would you prefer the main entrance — on the road-facing side?"
            ),
            "parking_type": (
                "Do you need vehicle parking? Stilt, garage, or open parking?"
            ),
            "building_type": (
                "What type of home are you envisioning — a villa, bungalow, duplex, "
                "or independent house?"
            ),
            "architectural_style": (
                "Do you have a design style preference? Modern, contemporary, traditional?"
            ),
        }
        # Try exact match first, then prefix match
        if missing_field.field_key in fallbacks:
            return fallbacks[missing_field.field_key]

        for key, question in fallbacks.items():
            if missing_field.field_key.startswith(key):
                return question

        return f"Could you share details about {missing_field.display_name}?"


# =====================================================================
# 4. INTERACTIVE GATHERER ORCHESTRATOR
# =====================================================================
class InteractiveGatherer:
    """
    Orchestrates the interactive question-answer loop.

    This is the main entry point that:
      1. Inspects the current data for missing fields
      2. Determines the next question to ask (priority-ordered)
      3. Generates the question using the architect persona
      4. Returns control to the caller to get the user's answer
      5. After receiving the answer, merges it and re-inspects

    The loop continues until all Tier 1 fields are filled, then asks
    up to MAX_TIER2_QUESTIONS Tier 2 questions before finishing.

    Usage:
        gatherer = InteractiveGatherer(parser_config)

        # Check what's needed
        result = gatherer.get_next_action(current_data, session_id)
        # result = {"action": "ask", "question": "...", "field": "...", "tier": 1}
        # or
        # result = {"action": "complete", "data": {...}}

        # After user answers
        updated_data = gatherer.process_answer(answer, current_data, session_id)
    """

    # Maximum number of Tier 2 questions to ask before proceeding
    MAX_TIER2_QUESTIONS: int = 3

    # Maximum number of Tier 3 questions (usually 0 — we skip Tier 3)
    MAX_TIER3_QUESTIONS: int = 0

    # Maximum total interactive turns to prevent infinite loops
    MAX_TOTAL_TURNS: int = 8

    def __init__(self, config):
        """
        Args:
            config: ParserConfig from the main parser pipeline.
        """
        self.config = config
        self.inspector = RequirementsInspector()
        self.question_gen = QuestionGenerator(config)

        # Session-level tracking
        self._session_state: Dict[str, Dict[str, Any]] = {}

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or initialize session tracking state."""
        if session_id not in self._session_state:
            self._session_state[session_id] = {
                "asked_fields": set(),      # Fields already asked about
                "tier2_asked": 0,           # Count of Tier 2 questions asked
                "tier3_asked": 0,           # Count of Tier 3 questions asked
                "total_turns": 0,           # Total question turns
            }
        return self._session_state[session_id]

    def get_next_action(
        self,
        current_data: Dict[str, Any],
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Determine the next action: ask a question or proceed to completion.

        Args:
            current_data: Current BuildingRequirements dict.
            session_id: Session identifier for tracking state.

        Returns:
            Dict with one of two structures:
            - {"action": "ask", "question": str, "field": str, "tier": int, "missing_summary": dict}
            - {"action": "complete", "data": dict, "completion_reason": str}
        """
        session = self._get_session(session_id)

        # Safety valve: prevent infinite loops
        if session["total_turns"] >= self.MAX_TOTAL_TURNS:
            logger.warning(
                "Max interactive turns (%d) reached. Proceeding with available data.",
                self.MAX_TOTAL_TURNS,
            )
            return {
                "action": "complete",
                "data": current_data,
                "completion_reason": "max_turns_reached",
            }

        # Inspect current state
        missing_fields = self.inspector.inspect(current_data)

        if not missing_fields:
            logger.info("All fields are populated. Proceeding to next stage.")
            return {
                "action": "complete",
                "data": current_data,
                "completion_reason": "all_fields_filled",
            }

        # Categorize missing fields
        tier1_missing = [f for f in missing_fields if f.tier == Tier.TIER_1]
        tier2_missing = [f for f in missing_fields if f.tier == Tier.TIER_2]
        tier3_missing = [f for f in missing_fields if f.tier == Tier.TIER_3]

        # Build missing summary for caller visibility
        missing_summary = {
            "tier1": [f.display_name for f in tier1_missing],
            "tier2": [f.display_name for f in tier2_missing],
            "tier3": [f.display_name for f in tier3_missing],
        }

        # --- TIER 1: Must ask all ---
        if tier1_missing:
            # Find the first Tier 1 field we haven't asked about yet
            next_field = None
            for field in tier1_missing:
                if field.field_key not in session["asked_fields"]:
                    next_field = field
                    break

            if next_field is None:
                # We've asked about all Tier 1 fields but they're still missing
                # Re-ask the first one (user may have given incomplete answer)
                next_field = tier1_missing[0]
                logger.info(
                    "Re-asking Tier 1 field '%s' — still missing after previous answer.",
                    next_field.display_name,
                )

            # Generate the question
            question = self.question_gen.generate_question(current_data, next_field)
            session["asked_fields"].add(next_field.field_key)
            session["total_turns"] += 1

            return {
                "action": "ask",
                "question": question,
                "field": next_field.field_key,
                "field_name": next_field.display_name,
                "tier": next_field.tier,
                "missing_summary": missing_summary,
            }

        # --- TIER 2: Ask up to MAX_TIER2_QUESTIONS ---
        if tier2_missing and session["tier2_asked"] < self.MAX_TIER2_QUESTIONS:
            next_field = None
            for field in tier2_missing:
                if field.field_key not in session["asked_fields"]:
                    next_field = field
                    break

            if next_field:
                question = self.question_gen.generate_question(current_data, next_field)
                session["asked_fields"].add(next_field.field_key)
                session["tier2_asked"] += 1
                session["total_turns"] += 1

                return {
                    "action": "ask",
                    "question": question,
                    "field": next_field.field_key,
                    "field_name": next_field.display_name,
                    "tier": next_field.tier,
                    "missing_summary": missing_summary,
                }

        # --- TIER 3: Ask up to MAX_TIER3_QUESTIONS (default 0 = skip) ---
        if tier3_missing and session["tier3_asked"] < self.MAX_TIER3_QUESTIONS:
            next_field = None
            for field in tier3_missing:
                if field.field_key not in session["asked_fields"]:
                    next_field = field
                    break

            if next_field:
                question = self.question_gen.generate_question(current_data, next_field)
                session["asked_fields"].add(next_field.field_key)
                session["tier3_asked"] += 1
                session["total_turns"] += 1

                return {
                    "action": "ask",
                    "question": question,
                    "field": next_field.field_key,
                    "field_name": next_field.display_name,
                    "tier": next_field.tier,
                    "missing_summary": missing_summary,
                }

        # --- ALL TIERS SATISFIED OR QUOTA REACHED ---
        completion_reasons = []
        if not tier1_missing:
            completion_reasons.append("all_tier1_filled")
        if session["tier2_asked"] >= self.MAX_TIER2_QUESTIONS:
            completion_reasons.append("tier2_quota_reached")
        if session["tier3_asked"] >= self.MAX_TIER3_QUESTIONS:
            completion_reasons.append("tier3_quota_reached")

        logger.info(
            "Interactive gathering complete. Reasons: %s",
            ", ".join(completion_reasons),
        )

        return {
            "action": "complete",
            "data": current_data,
            "completion_reason": ", ".join(completion_reasons) or "all_questions_asked",
        }

    def get_missing_summary(self, current_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Get a summary of all missing fields, categorized by tier.
        Useful for debugging and status reporting.
        """
        missing_fields = self.inspector.inspect(current_data)
        return {
            "tier1": [f.display_name for f in missing_fields if f.tier == Tier.TIER_1],
            "tier2": [f.display_name for f in missing_fields if f.tier == Tier.TIER_2],
            "tier3": [f.display_name for f in missing_fields if f.tier == Tier.TIER_3],
        }

    def clear_session(self, session_id: str = "default") -> None:
        """Clear tracking state for a session."""
        self._session_state.pop(session_id, None)


# =====================================================================
# 5. LOCAL VERIFICATION RUNNER
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("INTERACTIVE GATHERER — Unit Verification")
    print("=" * 60)

    # Test the inspector
    inspector = RequirementsInspector()

    # Test 1: Completely empty data
    print("\n--- Test 1: Empty data ---")
    empty_data = BuildingRequirements().model_dump()
    missing = inspector.inspect(empty_data)
    tier1_count = len([m for m in missing if m.tier == Tier.TIER_1])
    tier2_count = len([m for m in missing if m.tier == Tier.TIER_2])
    tier3_count = len([m for m in missing if m.tier == Tier.TIER_3])
    print(f"Tier 1 missing: {tier1_count} fields")
    print(f"Tier 2 missing: {tier2_count} fields")
    print(f"Tier 3 missing: {tier3_count} fields")
    assert tier1_count == 3, f"Expected 3 Tier 1 missing, got {tier1_count}"
    print("[PASS] Empty data correctly identifies 3 Tier-1 blockers")

    # Test 2: Partial data (has plot + rooms, missing direction)
    print("\n--- Test 2: Partial data ---")
    partial_data = {
        "plot_dimensions": {"length": 40, "width": 30, "unit": "ft", "total_area_sqft": 1200},
        "plot_context": {"shape": "rectangular", "road_facing_sides": [], "north_direction": None, "entrance_side": None, "image_source_notes": None},
        "setbacks": None,
        "number_of_floors": None,
        "rooms": [{"room_type": "Bedroom", "quantity": 3, "specific_requirements": None, "preferred_floor": None}],
        "vastu_compliant": None,
        "parking_type": None,
        "include_furniture": None,
        "architectural_style": None,
        "building_type": None,
        "additional_notes": None,
    }
    missing = inspector.inspect(partial_data)
    tier1_fields = [m.field_key for m in missing if m.tier == Tier.TIER_1]
    print(f"Tier 1 still missing: {tier1_fields}")
    assert "plot_context.road_facing_sides" in tier1_fields
    print("[PASS] Correctly identifies missing road-facing direction as Tier 1")

    # Test 3: Complete Tier 1 data
    print("\n--- Test 3: Complete Tier 1 data ---")
    complete_t1 = dict(partial_data)
    complete_t1["plot_context"]["road_facing_sides"] = ["north"]
    missing = inspector.inspect(complete_t1)
    tier1_remaining = [m for m in missing if m.tier == Tier.TIER_1]
    print(f"Tier 1 remaining: {len(tier1_remaining)} (should be 0)")
    assert len(tier1_remaining) == 0
    tier2_fields = [m.field_key for m in missing if m.tier == Tier.TIER_2]
    print(f"Tier 2 still missing: {tier2_fields}")
    print("[PASS] All Tier 1 satisfied — system can proceed to Tier 2")

    # Test 4: Full data — nothing missing
    print("\n--- Test 4: Fully populated data ---")
    full_data = dict(complete_t1)
    full_data["number_of_floors"] = 2
    full_data["vastu_compliant"] = True
    full_data["parking_type"] = "stilt"
    full_data["building_type"] = "villa"
    full_data["architectural_style"] = "modern"
    full_data["plot_context"]["entrance_side"] = "north"
    missing = inspector.inspect(full_data)
    print(f"Total missing: {len(missing)} (should be 0)")
    assert len(missing) == 0
    print("[PASS] Fully populated data — no missing fields")

    print("\n" + "=" * 60)
    print("ALL INTERACTIVE GATHERER VERIFICATION CHECKS PASSED")
    print("=" * 60)
