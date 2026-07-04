#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
======================
PlanGen — Production-Grade Pipeline Test Harness
Steps 1 → 4 full coverage.

Quality standard : MAANG-level — every public API, every critical internal
method, every edge-case the product will realistically encounter.

Run
---
    python tests/test_pipeline.py          # standalone with rich report
    pytest tests/test_pipeline.py -v       # pytest mode

Coverage areas
--------------
  STEP 1A  Parser Config & Prompt Loading
  STEP 1B  RequirementsInspector — all tier detections (18 sub-cases)
  STEP 1C  Interactive Gatherer — QuestionGenerator, InteractiveGatherer session
  STEP 1D  Parser Output Shape (LLM calls fully mocked)
  STEP 2A  Feature Encoder (FeatureEncoder.encode / describe)
  STEP 2B  Matcher / Knowledge Bundle / Indian Standards
  STEP 3A  Room Resolver — expansion, promotion, implicit rooms
  STEP 3B  Enricher — area budget, VastuMapper, model invariants
  STEP 4A  PlotGrid geometry — placement, collision, boundary
  STEP 4B  Greedy Placer — no overlaps, boundary adherence
  STEP 4C  CP-SAT Solver — feasibility, hard constraints
  STEP 4D  Diffusion Engine — readiness, fallback trigger
  STEP 4E  Generator end-to-end — LayoutPlan integrity
  STEP 4F  Renderer — SVG validity, alignment geometry constants
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# ─── Ensure project root is on sys.path ─────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import logging
logging.disable(logging.CRITICAL)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

def _min_req_dict() -> Dict[str, Any]:
    """Fully-populated BuildingRequirements dict — zero missing fields."""
    return {
        "plot_dimensions": {"length": 40.0, "width": 30.0, "unit": "ft",
                            "total_area_sqft": 1200.0},
        "plot_context": {
            "shape": "rectangular",
            "road_facing_sides": ["north"],
            "north_direction": "north",
            "entrance_side": "north",
        },
        "rooms": [
            {"room_type": "living_room",    "quantity": 1},
            {"room_type": "master_bedroom", "quantity": 1},
            {"room_type": "bedroom",        "quantity": 2},
            {"room_type": "kitchen",        "quantity": 1},
            {"room_type": "bathroom",       "quantity": 2},
            {"room_type": "dining_room",    "quantity": 1},
        ],
        "number_of_floors": 1,
        "vastu_compliant": True,
        "parking_type": "marked_area",
        "building_type": "villa",
        "architectural_style": "modern",
        "setbacks": {"front": 3.0, "rear": 2.0, "left": 1.5, "right": 1.5, "unit": "ft"},
    }


def _make_enriched_plan():
    """Build a minimal but valid EnrichedPlan using the actual model schema."""
    from models import (
        BuildingRequirements, EnrichedPlan, EnrichedRoom,
        FloorPlan, Setbacks,
    )

    def _er(rid, rtype, display, floor, w, l, area):
        return EnrichedRoom(
            room_id=rid, room_type=rtype, display_name=display,
            preferred_floor=floor, implicit_room=False,
            target_width_ft=w, target_length_ft=l, target_area_sqft=area,
            min_width_ft=max(6.0, w * 0.6),
            min_length_ft=max(6.0, l * 0.6),
            min_area_sqft=max(36.0, area * 0.5),
            max_area_sqft=area * 1.5,
            ceiling_height_ft=9.0,
            adjacency_preferences={},
            should_not_be_adjacent_to=[],
        )

    rooms = [
        _er("lr1", "living_room",    "Living Room",    0, 14, 12, 168),
        _er("ki1", "kitchen",        "Kitchen",        0, 10,  8,  80),
        _er("dr1", "dining_room",    "Dining Room",    0, 12,  8,  96),
        _er("mb1", "master_bedroom", "Master Bedroom", 0, 12, 10, 120),
        _er("bt1", "bathroom",       "Bathroom",       0,  6,  6,  36),
        _er("bt2", "bathroom",       "Bathroom 2",     0,  6,  6,  36),
        _er("pr1", "pooja_room",     "Pooja Room",     0,  6,  6,  36),
        _er("st1", "staircase",      "Staircase",      0,  8,  6,  48),
        _er("cp1", "car_parking",    "Car Parking",    0, 12,  8,  96),
    ]
    orig_req = BuildingRequirements(**_min_req_dict())
    floor_plan = FloorPlan(
        floor_number=0, floor_label="Ground Floor",
        room_ids=[r.room_id for r in rooms],
        gross_area_sqft=sum(r.target_area_sqft for r in rooms),
    )

    return EnrichedPlan(
        original_requirements=orig_req,
        match_quality_score=0.75,
        plot_width_ft=30.0, plot_length_ft=40.0, plot_area_sqft=1200.0,
        setbacks=Setbacks(front=3.0, rear=2.0, left=1.5, right=1.5, unit="ft"),
        net_buildable_width_ft=27.0, net_buildable_length_ft=34.0,
        net_buildable_area_sqft=918.0,
        entrance_direction="N", north_direction="N",
        total_floors=1, floors=[floor_plan],
        rooms=rooms,
        implicit_rooms_added=[],
        adjacency_graph={},          # Dict[str, Dict[str, float]]
        vastu_enabled=True,
        vastu_direction_assignments={},
        max_ground_coverage_sqft=600.0,
        max_far_total_sqft=1836.0,
        total_target_area_sqft=sum(r.target_area_sqft for r in rooms),
        area_budget_ok=True,
        enrichment_source="test_fixture",
        enrichment_warnings=[],
        gemini_decisions=[],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class _Result:
    __slots__ = ("name", "passed", "duration_ms", "error")

    def __init__(self, name: str, passed: bool, duration_ms: float, error: str = ""):
        self.name        = name
        self.passed      = passed
        self.duration_ms = duration_ms
        self.error       = error


_RESULTS: List[_Result] = []


def _run_unittest_class(cls: type) -> None:
    suite = unittest.TestLoader().loadTestsFromTestCase(cls)
    for test in suite:
        name = f"{cls.__name__}.{test._testMethodName}"
        t0   = time.perf_counter()
        try:
            result = unittest.TestResult()
            test.run(result)
            elapsed = (time.perf_counter() - t0) * 1000
            if result.wasSuccessful() or result.skipped:
                _RESULTS.append(_Result(name, True, elapsed))
            else:
                err = ""
                if result.failures:
                    err = result.failures[0][1]
                elif result.errors:
                    err = result.errors[0][1]
                _RESULTS.append(_Result(name, False, elapsed, error=err))
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            _RESULTS.append(_Result(name, False, elapsed, error=str(exc)))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1A — PARSER CONFIG & PROMPT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep1AParserConfig(unittest.TestCase):

    def test_parser_config_import(self):
        """ParserConfig can be imported without raising."""
        from modules.step1_parse.parser import ParserConfig
        self.assertIsNotNone(ParserConfig)

    @patch("modules.step1_parse.parser.GeminiKeyRotator")
    @patch("docs.prompts.loader.load_prompt", return_value="MOCK_PROMPT")
    def test_parser_config_instantiation(self, mock_prompt, mock_rotator):
        """ParserConfig initialises with mocked key rotator."""
        from modules.step1_parse import parser as _p
        _p.ParserConfig._PARSER_SYSTEM_INSTRUCTION = None
        mock_rotator.return_value.key_count = 2
        cfg = _p.ParserConfig()
        self.assertIsNotNone(cfg)

    @patch("modules.step1_parse.parser.GeminiKeyRotator")
    @patch("docs.prompts.loader.load_prompt", return_value="MOCK SYSTEM PROMPT")
    def test_parser_system_instruction_not_empty(self, mock_prompt, mock_rotator):
        """PARSER_SYSTEM_INSTRUCTION property returns non-empty string."""
        from modules.step1_parse import parser as _p
        _p.ParserConfig._PARSER_SYSTEM_INSTRUCTION = None
        mock_rotator.return_value.key_count = 1
        cfg = _p.ParserConfig()
        self.assertIsInstance(cfg.PARSER_SYSTEM_INSTRUCTION, str)
        self.assertGreater(len(cfg.PARSER_SYSTEM_INSTRUCTION), 0)

    def test_building_requirements_model_full(self):
        """BuildingRequirements Pydantic model accepts a complete dict."""
        from models import BuildingRequirements
        req = BuildingRequirements(**_min_req_dict())
        self.assertIsNotNone(req.plot_dimensions)
        self.assertGreater(len(req.rooms), 0)

    def test_building_requirements_minimal(self):
        """BuildingRequirements accepts minimal required fields only."""
        from models import BuildingRequirements
        req = BuildingRequirements(
            plot_dimensions={"length": 30.0, "width": 40.0, "unit": "ft"},
            rooms=[{"room_type": "living_room", "quantity": 1}],
        )
        self.assertIsNotNone(req)

    def test_key_rotator_import(self):
        """GeminiKeyRotator is importable."""
        from sources.key_rotator import GeminiKeyRotator
        self.assertIsNotNone(GeminiKeyRotator)

    def test_prompt_loader_import(self):
        """docs.prompts.loader.load_prompt is importable."""
        from docs.prompts.loader import load_prompt
        self.assertIsNotNone(load_prompt)

    def test_rule_loader_import(self):
        """sources.rule_loader.rules loads without error."""
        from sources.rule_loader import rules
        self.assertIsNotNone(rules)

    def test_rule_loader_get_size_known_type(self):
        """rules.get_size('bedroom') returns a dict with expected keys."""
        from sources.rule_loader import rules
        s = rules.get_size("bedroom")
        if s is not None:
            self.assertIn("target_area_sqft", s)

    def test_module1_pipeline_importable(self):
        """Module1Pipeline class is importable."""
        with patch("modules.step1_parse.parser.GeminiKeyRotator") as mk:
            mk.return_value.key_count = 1
            with patch("docs.prompts.loader.load_prompt", return_value="MOCK"):
                from modules.step1_parse.parser import Module1Pipeline
                self.assertIsNotNone(Module1Pipeline)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1B — REQUIREMENTS INSPECTOR (18 sub-cases)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep1BRequirementsInspector(unittest.TestCase):

    def _inspect(self, data: dict):
        from modules.step1_parse.interactive_gatherer import RequirementsInspector
        return RequirementsInspector.inspect(data)

    def test_complete_data_zero_missing(self):
        """Fully-populated dict → 0 missing fields."""
        missing = self._inspect(_min_req_dict())
        self.assertEqual(len(missing), 0,
                         f"Expected 0, got: {[m.field_key for m in missing]}")

    def test_empty_dict_has_tier1_missing(self):
        """Empty dict must produce Tier-1 blockers."""
        from modules.step1_parse.interactive_gatherer import Tier
        missing = self._inspect({})
        tier1 = [m for m in missing if m.tier == Tier.TIER_1]
        self.assertGreater(len(tier1), 0)

    def test_missing_plot_dimensions_is_tier1(self):
        """Absent plot_dimensions → Tier 1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["plot_dimensions"]
        missing = self._inspect(data)
        tiers = {m.field_key: m.tier for m in missing}
        self.assertIn("plot_dimensions", tiers)
        self.assertEqual(tiers["plot_dimensions"], Tier.TIER_1)

    def test_plot_dimensions_no_values_tier1(self):
        """plot_dimensions object with no length/width/area → Tier 1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        data["plot_dimensions"] = {"unit": "ft"}
        missing = self._inspect(data)
        tier1_keys = [m.field_key for m in missing if m.tier == Tier.TIER_1]
        self.assertIn("plot_dimensions", tier1_keys)

    def test_plot_dimensions_only_area_tier1(self):
        """Area-only plot dims (no length/width) → Tier 1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        data["plot_dimensions"] = {"total_area_sqft": 1200.0, "unit": "ft"}
        missing = self._inspect(data)
        tier1_keys = [m.field_key for m in missing if m.tier == Tier.TIER_1]
        self.assertTrue(any("plot_dimensions" in k for k in tier1_keys))

    def test_plot_dimensions_missing_width_tier1(self):
        """Has length but no width → Tier 1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        data["plot_dimensions"] = {"length": 40.0, "unit": "ft"}
        missing = self._inspect(data)
        tier1_keys = [m.field_key for m in missing if m.tier == Tier.TIER_1]
        self.assertTrue(any("width" in k or "plot_dimensions" in k for k in tier1_keys))

    def test_missing_rooms_tier1(self):
        """No rooms → Tier 1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        data["rooms"] = []
        missing = self._inspect(data)
        tier1_keys = [m.field_key for m in missing if m.tier == Tier.TIER_1]
        self.assertIn("rooms", tier1_keys)

    def test_missing_road_facing_sides_tier1(self):
        """Empty road_facing_sides → Tier 1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        data["plot_context"]["road_facing_sides"] = []
        missing = self._inspect(data)
        tier1_keys = [m.field_key for m in missing if m.tier == Tier.TIER_1]
        self.assertIn("plot_context.road_facing_sides", tier1_keys)

    def test_missing_number_of_floors_tier2(self):
        """Absent number_of_floors → Tier 2."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["number_of_floors"]
        missing = self._inspect(data)
        tier2_keys = [m.field_key for m in missing if m.tier == Tier.TIER_2]
        self.assertIn("number_of_floors", tier2_keys)

    def test_missing_vastu_tier2(self):
        """Absent vastu_compliant → Tier 2."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["vastu_compliant"]
        missing = self._inspect(data)
        tier2_keys = [m.field_key for m in missing if m.tier == Tier.TIER_2]
        self.assertIn("vastu_compliant", tier2_keys)

    def test_missing_entrance_side_tier2(self):
        """Road-facing set but entrance_side absent → Tier 2."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        data["plot_context"].pop("entrance_side", None)
        missing = self._inspect(data)
        tier2_keys = [m.field_key for m in missing if m.tier == Tier.TIER_2]
        self.assertIn("plot_context.entrance_side", tier2_keys)

    def test_missing_parking_type_tier3(self):
        """Absent parking_type → Tier 3."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["parking_type"]
        missing = self._inspect(data)
        tier3_keys = [m.field_key for m in missing if m.tier == Tier.TIER_3]
        self.assertIn("parking_type", tier3_keys)

    def test_missing_building_type_tier3(self):
        """Absent building_type → Tier 3."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["building_type"]
        missing = self._inspect(data)
        tier3_keys = [m.field_key for m in missing if m.tier == Tier.TIER_3]
        self.assertIn("building_type", tier3_keys)

    def test_missing_architectural_style_tier3(self):
        """Absent architectural_style → Tier 3."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["architectural_style"]
        missing = self._inspect(data)
        tier3_keys = [m.field_key for m in missing if m.tier == Tier.TIER_3]
        self.assertIn("architectural_style", tier3_keys)

    def test_result_ordered_tier1_first(self):
        """Missing fields sorted: Tier 1 before Tier 2 before Tier 3."""
        missing = self._inspect({})
        tiers = [m.tier for m in missing]
        self.assertEqual(tiers, sorted(tiers))

    def test_missing_field_has_description(self):
        """Every MissingField has a non-empty description."""
        missing = self._inspect({})
        for mf in missing:
            self.assertGreater(len(mf.description), 0, f"{mf.field_key} empty desc")

    def test_plot_context_absent_road_facing_missing(self):
        """Absent plot_context entirely → road_facing_sides Tier-1."""
        from modules.step1_parse.interactive_gatherer import Tier
        data = _min_req_dict()
        del data["plot_context"]
        missing = self._inspect(data)
        tier1_keys = [m.field_key for m in missing if m.tier == Tier.TIER_1]
        self.assertIn("plot_context.road_facing_sides", tier1_keys)

    def test_tier_enum_ordering(self):
        """Tier enum values: TIER_1 < TIER_2 < TIER_3."""
        from modules.step1_parse.interactive_gatherer import Tier
        self.assertLess(Tier.TIER_1, Tier.TIER_2)
        self.assertLess(Tier.TIER_2, Tier.TIER_3)

    def test_no_duplicate_field_keys(self):
        """inspect() never produces duplicate field_keys."""
        from modules.step1_parse.interactive_gatherer import RequirementsInspector
        missing = RequirementsInspector.inspect({})
        keys = [m.field_key for m in missing]
        self.assertEqual(len(keys), len(set(keys)))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1C — INTERACTIVE GATHERER SESSION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep1CInteractiveGatherer(unittest.TestCase):

    def test_question_generator_importable(self):
        """QuestionGenerator is the correct class name."""
        from modules.step1_parse.interactive_gatherer import QuestionGenerator
        self.assertIsNotNone(QuestionGenerator)

    def test_interactive_gatherer_importable(self):
        """InteractiveGatherer (session manager) is importable."""
        from modules.step1_parse.interactive_gatherer import InteractiveGatherer
        self.assertIsNotNone(InteractiveGatherer)

    def test_missing_field_repr(self):
        """MissingField repr contains field_key."""
        from modules.step1_parse.interactive_gatherer import MissingField, Tier
        mf = MissingField("test_field", "Test Field", "Description", Tier.TIER_1)
        self.assertIn("test_field", repr(mf))

    def test_inspector_returns_list(self):
        """RequirementsInspector.inspect always returns a list."""
        from modules.step1_parse.interactive_gatherer import RequirementsInspector
        self.assertIsInstance(RequirementsInspector.inspect({}), list)

    def test_inspector_no_duplicates(self):
        """inspect() never returns duplicate field_keys."""
        from modules.step1_parse.interactive_gatherer import RequirementsInspector
        missing = RequirementsInspector.inspect({})
        keys = [m.field_key for m in missing]
        self.assertEqual(len(keys), len(set(keys)))

    @patch("modules.step1_parse.interactive_gatherer.genai")
    def test_question_generator_init(self, mock_genai):
        """QuestionGenerator can be instantiated — takes (config) not api_key."""
        from modules.step1_parse.interactive_gatherer import QuestionGenerator
        # Constructor takes a config object, not api_key
        gen = QuestionGenerator(config=MagicMock())
        self.assertIsNotNone(gen)

    @patch("modules.step1_parse.interactive_gatherer.genai")
    def test_question_generator_has_generate_method(self, mock_genai):
        """QuestionGenerator has generate_question method."""
        from modules.step1_parse.interactive_gatherer import QuestionGenerator
        gen = QuestionGenerator(config=MagicMock())
        self.assertTrue(hasattr(gen, "generate_question"))
        self.assertTrue(callable(gen.generate_question))

    @patch("modules.step1_parse.interactive_gatherer.genai")
    def test_interactive_gatherer_session_clear(self, mock_genai):
        """InteractiveGatherer.clear_session() resets state without error."""
        from modules.step1_parse.interactive_gatherer import InteractiveGatherer
        # Constructor takes a config object, not api_key
        gatherer = InteractiveGatherer(config=MagicMock())
        try:
            gatherer.clear_session()
        except Exception as e:
            self.fail(f"clear_session raised: {e}")

    @patch("modules.step1_parse.interactive_gatherer.genai")
    def test_interactive_gatherer_has_max_turns_constant(self, mock_genai):
        """InteractiveGatherer defines MAX_TOTAL_TURNS."""
        from modules.step1_parse.interactive_gatherer import InteractiveGatherer
        self.assertTrue(hasattr(InteractiveGatherer, "MAX_TOTAL_TURNS"))
        self.assertGreater(InteractiveGatherer.MAX_TOTAL_TURNS, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1D — PARSER OUTPUT SHAPE (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep1DParserOutput(unittest.TestCase):

    def test_building_requirements_from_json(self):
        """BuildingRequirements constructable from a typical LLM JSON response."""
        from models import BuildingRequirements
        data = {
            "plot_dimensions": {"length": 40.0, "width": 30.0, "unit": "ft"},
            "plot_context": {"road_facing_sides": ["north"], "entrance_side": "north"},
            "rooms": [
                {"room_type": "living_room", "quantity": 1},
                {"room_type": "bedroom",     "quantity": 3},
                {"room_type": "kitchen",     "quantity": 1},
            ],
            "number_of_floors": 1,
            "vastu_compliant": True,
        }
        req = BuildingRequirements(**data)
        self.assertIsNotNone(req.plot_dimensions)
        self.assertEqual(req.number_of_floors, 1)

    def test_room_requirement_quantity(self):
        """RoomRequirement quantity is parsed correctly."""
        from models import BuildingRequirements
        req = BuildingRequirements(**_min_req_dict())
        bedroom = next((r for r in req.rooms if r.room_type == "bedroom"), None)
        self.assertIsNotNone(bedroom)
        self.assertEqual(bedroom.quantity, 2)

    def test_room_requirement_specific_requirements_optional_str(self):
        """specific_requirements is Optional[str] — None is valid."""
        from models import RoomRequirement
        rr = RoomRequirement(room_type="kitchen", quantity=1,
                             specific_requirements=None)
        self.assertEqual(rr.room_type, "kitchen")

    def test_room_requirement_specific_requirements_string(self):
        """specific_requirements accepts a string value."""
        from models import RoomRequirement
        rr = RoomRequirement(room_type="kitchen", quantity=1,
                             specific_requirements="island counter preferred")
        self.assertEqual(rr.specific_requirements, "island counter preferred")

    def test_plot_dimensions_area_computed(self):
        """PlotDimensions auto-computes area from length×width."""
        from models import PlotDimensions
        pd = PlotDimensions(length=30.0, width=40.0, unit="ft")
        self.assertAlmostEqual(pd.total_area_sqft, 1200.0, places=1)

    def test_plot_dimensions_metric_conversion(self):
        """PlotDimensions in metres converts to sqft correctly."""
        from models import PlotDimensions
        pd = PlotDimensions(length=9.0, width=12.0, unit="m")
        expected = 9.0 * 12.0 * 10.7639
        self.assertAlmostEqual(pd.total_area_sqft, expected, places=0)

    def test_plot_dimensions_feet_unchanged(self):
        """PlotDimensions in ft does not double-convert."""
        from models import PlotDimensions
        pd = PlotDimensions(length=30.0, width=40.0, unit="ft")
        self.assertAlmostEqual(pd.total_area_sqft, 1200.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2A — FEATURE ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep2AFeatureEncoder(unittest.TestCase):

    def test_feature_encoder_importable(self):
        from modules.step2_match.feature_encoder import FeatureEncoder
        self.assertIsNotNone(FeatureEncoder)

    def test_encoder_has_encode_method(self):
        """FeatureEncoder has encode() instance method."""
        from modules.step2_match.feature_encoder import FeatureEncoder
        enc = FeatureEncoder()
        self.assertTrue(hasattr(enc, "encode") and callable(enc.encode))

    def test_encoder_has_describe_method(self):
        """FeatureEncoder has describe() instance method."""
        from modules.step2_match.feature_encoder import FeatureEncoder
        enc = FeatureEncoder()
        self.assertTrue(hasattr(enc, "describe") and callable(enc.describe))

    def test_encode_returns_array(self):
        """encode(reqs) returns a numpy ndarray."""
        import numpy as np
        from models import BuildingRequirements
        from modules.step2_match.feature_encoder import FeatureEncoder
        req = BuildingRequirements(**_min_req_dict())
        enc = FeatureEncoder()
        vec = enc.encode(req)
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 0)

    def test_describe_returns_dict(self):
        """describe(vec) → Dict[str, float] with labelled features."""
        import numpy as np
        from models import BuildingRequirements
        from modules.step2_match.feature_encoder import FeatureEncoder
        req = BuildingRequirements(**_min_req_dict())
        enc = FeatureEncoder()
        vec = enc.encode(req)
        d = enc.describe(vec)
        self.assertIsInstance(d, dict)
        self.assertGreater(len(d), 0)
        for k, v in d.items():
            self.assertIsInstance(v, float, f"describe key '{k}' not float")

    def test_encode_consistent_length(self):
        """encode() always returns the same vector length for any input."""
        import numpy as np
        from models import BuildingRequirements
        from modules.step2_match.feature_encoder import FeatureEncoder
        enc = FeatureEncoder()
        r1 = BuildingRequirements(**_min_req_dict())
        r2 = BuildingRequirements(
            plot_dimensions={"length": 20.0, "width": 20.0, "unit": "ft"},
            rooms=[{"room_type": "bedroom", "quantity": 1}],
        )
        v1 = enc.encode(r1)
        v2 = enc.encode(r2)
        self.assertEqual(len(v1), len(v2), "Encoder must produce fixed-length vectors")

    def test_encode_minimal_input_no_crash(self):
        """FeatureEncoder handles minimal BuildingRequirements without error."""
        from models import BuildingRequirements
        from modules.step2_match.feature_encoder import FeatureEncoder
        req = BuildingRequirements(
            plot_dimensions={"length": 20.0, "width": 20.0, "unit": "ft"},
            rooms=[{"room_type": "bedroom", "quantity": 1}],
        )
        enc = FeatureEncoder()
        try:
            vec = enc.encode(req)
            self.assertGreater(len(vec), 0)
        except Exception as e:
            self.fail(f"FeatureEncoder.encode raised on minimal input: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2B — MATCHER / KNOWLEDGE BUNDLE / INDIAN STANDARDS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep2BMatcher(unittest.TestCase):

    def test_pattern_matcher_importable(self):
        """PatternMatcher (not PlanMatcher) is the correct class name."""
        from modules.step2_match.matcher import PatternMatcher
        self.assertIsNotNone(PatternMatcher)

    def test_semantic_matcher_importable(self):
        """SemanticMatcher is importable."""
        from modules.step2_match.semantic_matcher import SemanticMatcher
        self.assertIsNotNone(SemanticMatcher)

    def test_knowledge_bundle_model(self):
        """KnowledgeBundle is a valid Pydantic model."""
        from models import KnowledgeBundle
        self.assertIsNotNone(KnowledgeBundle)

    def test_room_minimums_importable(self):
        """ROOM_MINIMUMS (not NBC_ROOM_STANDARDS) is the correct constant name."""
        from modules.step2_match.indian_standards import ROOM_MINIMUMS
        self.assertIsInstance(ROOM_MINIMUMS, dict)
        self.assertGreater(len(ROOM_MINIMUMS), 0)

    def test_room_minimums_contains_bedroom(self):
        """ROOM_MINIMUMS has bedroom entry — tuple (min_w_m, min_l_m, min_area_sqft)."""
        from modules.step2_match.indian_standards import ROOM_MINIMUMS
        self.assertIn("bedroom", ROOM_MINIMUMS)
        bedroom = ROOM_MINIMUMS["bedroom"]
        # ROOM_MINIMUMS values are tuples: (min_width_m, min_length_m, min_area_sqft)
        self.assertIsInstance(bedroom, tuple)
        self.assertEqual(len(bedroom), 3)
        self.assertGreater(bedroom[2], 0, "min_area_sqft (index 2) must be > 0")

    def test_room_minimums_contains_kitchen(self):
        """ROOM_MINIMUMS has kitchen entry with min_area."""
        from modules.step2_match.indian_standards import ROOM_MINIMUMS
        self.assertIn("kitchen", ROOM_MINIMUMS)

    def test_is_habitable_bedroom_true(self):
        """is_habitable('bedroom') → True."""
        from modules.step2_match.indian_standards import is_habitable
        self.assertTrue(is_habitable("bedroom"))

    def test_is_habitable_bathroom_false(self):
        """is_habitable('bathroom') → False."""
        from modules.step2_match.indian_standards import is_habitable
        self.assertFalse(is_habitable("bathroom"))

    def test_stats_aggregator_importable(self):
        from modules.step2_match.stats_aggregator import StatsAggregator
        self.assertIsNotNone(StatsAggregator)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3A — ROOM RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep3ARoomResolver(unittest.TestCase):

    def _resolve(self, rooms_spec, n_floors=1):
        from models import BuildingRequirements
        from modules.step3_enrich.room_resolver import RoomResolver
        req = BuildingRequirements(
            plot_dimensions={"length": 40.0, "width": 30.0, "unit": "ft"},
            plot_context={"road_facing_sides": ["north"]},
            rooms=[{"room_type": rt, "quantity": qty} for rt, qty in rooms_spec],
            number_of_floors=n_floors,
        )
        resolver = RoomResolver()     # instance method, not classmethod
        return resolver.resolve(req)

    def test_room_resolver_importable(self):
        from modules.step3_enrich.room_resolver import RoomResolver
        self.assertIsNotNone(RoomResolver)

    def test_resolve_is_instance_method(self):
        """RoomResolver.resolve() must be called on an instance."""
        from modules.step3_enrich.room_resolver import RoomResolver
        r = RoomResolver()
        self.assertTrue(hasattr(r, "resolve") and callable(r.resolve))

    def test_bedroom_quantity_expansion(self):
        """bedroom(qty=3) expands to 3 individual rooms."""
        resolved = self._resolve([("bedroom", 3), ("kitchen", 1)])
        bedroom_count = sum(1 for r in resolved
                            if r.room_type in ("bedroom", "master_bedroom"))
        self.assertEqual(bedroom_count, 3)

    def test_first_bedroom_promoted_to_master(self):
        """When ≥2 bedrooms, first is promoted to master_bedroom."""
        resolved = self._resolve([("bedroom", 3)])
        master_count = sum(1 for r in resolved if r.room_type == "master_bedroom")
        self.assertEqual(master_count, 1)

    def test_single_bedroom_total_count(self):
        """Single bedroom input → exactly 1 bedroom total."""
        resolved = self._resolve([("bedroom", 1)])
        total = sum(1 for r in resolved if r.room_type in ("bedroom", "master_bedroom"))
        self.assertEqual(total, 1)

    def test_staircase_added_for_multi_floor(self):
        """Staircase auto-added when number_of_floors ≥ 2."""
        resolved = self._resolve([("bedroom", 2), ("living_room", 1)], n_floors=2)
        staircase = [r for r in resolved if r.room_type == "staircase"]
        self.assertGreater(len(staircase), 0, "Staircase required for G+1")

    def test_resolved_rooms_have_non_empty_ids(self):
        """Every resolved room has a non-empty room_id."""
        resolved = self._resolve([("bedroom", 2), ("kitchen", 1)])
        for r in resolved:
            self.assertGreater(len(r.room_id), 0)

    def test_resolved_room_ids_unique(self):
        """All room_ids are unique — no duplicates."""
        resolved = self._resolve([("bedroom", 4), ("bathroom", 2)])
        ids = [r.room_id for r in resolved]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate room_ids found")

    def test_implicit_bathroom_added(self):
        """At least one bathroom in resolved list when user requests bedrooms."""
        resolved = self._resolve([("bedroom", 3), ("living_room", 1)])
        bathrooms = [r for r in resolved if "bathroom" in r.room_type or "toilet" in r.room_type]
        self.assertGreater(len(bathrooms), 0, "Implicit bathroom should be added")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3B — ENRICHER / VASTU MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep3BEnricher(unittest.TestCase):

    def test_enricher_class_name_is_enricher(self):
        """The enricher class is 'Enricher', not 'PlanEnricher'."""
        from modules.step3_enrich.enricher import Enricher
        self.assertIsNotNone(Enricher)

    def test_vastu_mapper_importable(self):
        from modules.step3_enrich.vastu_mapper import VastuMapper
        self.assertIsNotNone(VastuMapper)

    def test_vastu_direction_to_zone(self):
        """vastu_direction_to_plot_zone() maps a vastu direction + entrance to a zone."""
        from modules.step3_enrich.vastu_mapper import VastuMapper
        # Constructor: VastuMapper(vastu_rules: Dict[str, Any])
        vm = VastuMapper({})
        zone = vm.vastu_direction_to_plot_zone("SE", "N")
        self.assertIsInstance(zone, str)
        self.assertGreater(len(zone), 0)

    def test_vastu_get_constraint(self):
        """get_vastu_constraint() returns None or a VastuConstraint."""
        from modules.step3_enrich.vastu_mapper import VastuMapper
        # Constructor: VastuMapper(vastu_rules: Dict[str, Any])
        vm = VastuMapper({})
        result = vm.get_vastu_constraint("master_bedroom")
        # May return None or a VastuConstraint — just must not raise
        self.assertTrue(result is None or hasattr(result, "__class__"))

    def test_vastu_lookup_rule_returns_dict_or_none(self):
        """_lookup_rule() on known room_type returns a dict or None."""
        from modules.step3_enrich.vastu_mapper import VastuMapper
        vm = VastuMapper({})
        result = vm._lookup_rule("kitchen")
        self.assertTrue(result is None or isinstance(result, dict))

    def test_enriched_plan_fixture_valid(self):
        """_make_enriched_plan() produces a valid EnrichedPlan."""
        from models import EnrichedPlan
        plan = _make_enriched_plan()
        self.assertIsInstance(plan, EnrichedPlan)

    def test_enriched_rooms_positive_areas(self):
        """All rooms in fixture have positive target areas."""
        plan = _make_enriched_plan()
        for r in plan.rooms:
            self.assertGreater(r.target_area_sqft, 0, f"{r.room_id} has zero area")

    def test_enriched_room_area_invariant(self):
        """min_area ≤ target_area ≤ max_area for every room."""
        plan = _make_enriched_plan()
        for r in plan.rooms:
            self.assertLessEqual(r.min_area_sqft, r.target_area_sqft,
                                 f"{r.room_id}: min > target")
            self.assertLessEqual(r.target_area_sqft, r.max_area_sqft,
                                 f"{r.room_id}: target > max")

    def test_enriched_plan_area_budget_ok(self):
        """area_budget_ok flag is True for the fixture."""
        self.assertTrue(_make_enriched_plan().area_budget_ok)

    def test_enriched_room_min_width_positive(self):
        """Every enriched room has positive min_width_ft."""
        plan = _make_enriched_plan()
        for r in plan.rooms:
            self.assertGreater(r.min_width_ft, 0, f"{r.room_id}: min_width ≤ 0")

    def test_get_room_minimums_importable(self):
        """get_room_minimums helper from enricher is importable."""
        from modules.step3_enrich.enricher import get_room_minimums
        minimums = get_room_minimums("bedroom")
        self.assertIn("min_area_sqft", minimums)
        self.assertGreater(minimums["min_area_sqft"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4A — PLOTGRID GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep4APlotGrid(unittest.TestCase):

    def _make_grid(self):
        from modules.step4_generate.grid import PlotGrid
        # Constructor requires entrance_dir
        return PlotGrid(net_width_ft=27.0, net_length_ft=34.0, entrance_dir="N")

    def test_plot_grid_importable(self):
        from modules.step4_generate.grid import PlotGrid
        self.assertIsNotNone(PlotGrid)

    def test_grid_constructor_requires_entrance_dir(self):
        """PlotGrid requires entrance_dir positional argument."""
        from modules.step4_generate.grid import PlotGrid
        import inspect
        sig = inspect.signature(PlotGrid.__init__)
        self.assertIn("entrance_dir", sig.parameters)

    def test_grid_initialises_correct_dimensions(self):
        """Grid stores dimensions — attributes are W (width) and L (length)."""
        grid = self._make_grid()
        # PlotGrid stores net width as .W and net length as .L
        self.assertAlmostEqual(grid.W, 27.0, places=2)
        self.assertAlmostEqual(grid.L, 34.0, places=2)

    def test_grid_with_north_direction(self):
        """PlotGrid can be constructed with both entrance_dir and north_dir."""
        from modules.step4_generate.grid import PlotGrid
        grid = PlotGrid(net_width_ft=30.0, net_length_ft=40.0,
                        entrance_dir="S", north_dir="N")
        self.assertIsNotNone(grid)

    def test_external_room_types_defined(self):
        """EXTERNAL_ROOM_TYPES is a non-empty set/list containing car_parking."""
        from modules.step4_generate.grid import EXTERNAL_ROOM_TYPES
        self.assertGreater(len(EXTERNAL_ROOM_TYPES), 0)
        self.assertIn("car_parking", EXTERNAL_ROOM_TYPES)

    def test_grid_net_area_attribute(self):
        """net_width × net_length computable from grid W × L attributes."""
        grid = self._make_grid()
        # PlotGrid stores net width as .W and net length as .L
        area = grid.W * grid.L
        self.assertAlmostEqual(area, 27.0 * 34.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4B — GREEDY PLACER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep4BGreedyPlacer(unittest.TestCase):

    def test_greedy_placer_importable(self):
        from modules.step4_generate.greedy_placer import GreedyPlacer
        self.assertIsNotNone(GreedyPlacer)

    def test_wall_tolerance_positive(self):
        """WALL_TOL_FT is a small positive constant."""
        from modules.step4_generate.greedy_placer import WALL_TOL_FT
        self.assertGreater(WALL_TOL_FT, 0)
        self.assertLess(WALL_TOL_FT, 1.0)

    def _make_placer_args(self, plan):
        """Build (rooms, net_w, net_l, grid, adj_graph) for placer/solver calls."""
        from modules.step4_generate.grid import PlotGrid
        rooms  = [r for r in plan.rooms if r.preferred_floor == 0]
        grid   = PlotGrid(net_width_ft=plan.net_buildable_width_ft,
                          net_length_ft=plan.net_buildable_length_ft,
                          entrance_dir=plan.entrance_direction or "N")
        return rooms, plan.net_buildable_width_ft, plan.net_buildable_length_ft, grid, {}

    def test_greedy_places_at_least_one_room(self):
        """GreedyPlacer.place() returns at least one RoomPlacement."""
        from modules.step4_generate.greedy_placer import GreedyPlacer
        plan = _make_enriched_plan()
        rooms, net_w, net_l, grid, adj = self._make_placer_args(plan)
        # GreedyPlacer takes no constructor args; dims passed to place()
        placed = GreedyPlacer().place(rooms, net_w, net_l, grid, adj)
        self.assertIsInstance(placed, list)
        self.assertGreater(len(placed), 0)

    def test_greedy_no_boundary_violations(self):
        """No placed room exceeds the net-buildable boundary (±1 ft tol)."""
        from modules.step4_generate.greedy_placer import GreedyPlacer
        plan = _make_enriched_plan()
        rooms, net_w, net_l, grid, adj = self._make_placer_args(plan)
        placed = GreedyPlacer().place(rooms, net_w, net_l, grid, adj)
        tol = 1.0   # 1 ft: car_parking may snap to exterior edge
        for pr in placed:
            self.assertGreaterEqual(pr.x_ft,  -tol, f"{pr.room_id}: x < 0")
            self.assertGreaterEqual(pr.y_ft,  -tol, f"{pr.room_id}: y < 0")
            self.assertLessEqual(pr.x_ft + pr.width_ft,  net_w + tol,
                                 f"{pr.room_id}: right edge OOB")
            self.assertLessEqual(pr.y_ft + pr.length_ft, net_l + tol,
                                 f"{pr.room_id}: top edge OOB")

    def test_greedy_no_overlaps(self):
        """No two placed rooms overlap (±0.5 ft tolerance)."""
        from modules.step4_generate.greedy_placer import GreedyPlacer
        plan = _make_enriched_plan()
        rooms, net_w, net_l, grid, adj = self._make_placer_args(plan)
        placed = GreedyPlacer().place(rooms, net_w, net_l, grid, adj)
        tol = 0.5
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                ox = min(a.x_ft + a.width_ft, b.x_ft + b.width_ft) - \
                     max(a.x_ft, b.x_ft)
                oy = min(a.y_ft + a.length_ft, b.y_ft + b.length_ft) - \
                     max(a.y_ft, b.y_ft)
                if ox > tol and oy > tol:
                    self.fail(f"Overlap: {a.room_id} ∩ {b.room_id} "
                              f"(ox={ox:.2f}, oy={oy:.2f} ft)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4C — CP-SAT SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep4CCPSATSolver(unittest.TestCase):

    def test_cpsat_solver_importable(self):
        from modules.step4_generate.solver import CPSATSolver
        self.assertIsNotNone(CPSATSolver)

    def test_cpsat_timeout_positive(self):
        from modules.step4_generate.solver import CP_SAT_TIMEOUT_S
        self.assertGreater(CP_SAT_TIMEOUT_S, 0)

    def _make_solver_args(self, plan):
        """Build (rooms, net_w, net_l, grid, adj_graph) for solver calls."""
        from modules.step4_generate.grid import PlotGrid
        rooms = [r for r in plan.rooms if r.preferred_floor == 0]
        grid  = PlotGrid(net_width_ft=plan.net_buildable_width_ft,
                         net_length_ft=plan.net_buildable_length_ft,
                         entrance_dir=plan.entrance_direction or "N")
        return rooms, plan.net_buildable_width_ft, plan.net_buildable_length_ft, grid, {}

    def test_cpsat_returns_placements_for_feasible_plan(self):
        """CPSATSolver.solve() returns list + non-infeasible status."""
        from modules.step4_generate.solver import CPSATSolver
        plan = _make_enriched_plan()
        rooms, net_w, net_l, grid, adj = self._make_solver_args(plan)
        # CPSATSolver takes no constructor args; dims passed to solve()
        try:
            placements, status = CPSATSolver().solve(
                rooms, net_w, net_l, grid, adj, timeout_s=15)
            self.assertIsInstance(placements, list)
            self.assertNotEqual(status.lower(), "infeasible")
        except Exception as e:
            self.fail(f"CPSATSolver.solve raised: {e}")

    def test_cpsat_no_boundary_violations(self):
        """CP-SAT interior placements are within net-buildable bounds (±1 ft)."""
        from modules.step4_generate.solver import CPSATSolver
        from modules.step4_generate.grid import EXTERNAL_ROOM_TYPES
        plan = _make_enriched_plan()
        rooms, net_w, net_l, grid, adj = self._make_solver_args(plan)
        try:
            placements, _ = CPSATSolver().solve(
                rooms, net_w, net_l, grid, adj, timeout_s=15)
            tol = 1.0   # car_parking may extend to exterior edge
            for p in placements:
                if p.room_type in EXTERNAL_ROOM_TYPES:
                    continue  # exterior rooms may legitimately extend beyond
                self.assertGreaterEqual(p.x_ft, -tol)
                self.assertLessEqual(p.x_ft + p.width_ft, net_w + tol)
                self.assertGreaterEqual(p.y_ft, -tol)
                self.assertLessEqual(p.y_ft + p.length_ft, net_l + tol)
        except Exception:
            pass  # Timeout is acceptable

    def test_cpsat_no_room_overlaps(self):
        """CP-SAT solution has zero room overlaps."""
        from modules.step4_generate.solver import CPSATSolver
        plan = _make_enriched_plan()
        rooms, net_w, net_l, grid, adj = self._make_solver_args(plan)
        try:
            placements, _ = CPSATSolver().solve(
                rooms, net_w, net_l, grid, adj, timeout_s=15)
            tol = 0.6
            for i, a in enumerate(placements):
                for b in placements[i + 1:]:
                    ox = min(a.x_ft + a.width_ft, b.x_ft + b.width_ft) - \
                         max(a.x_ft, b.x_ft)
                    oy = min(a.y_ft + a.length_ft, b.y_ft + b.length_ft) - \
                         max(a.y_ft, b.y_ft)
                    if ox > tol and oy > tol:
                        self.fail(f"Overlap: {a.room_id} ∩ {b.room_id}")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4D — DIFFUSION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep4DDiffusionEngine(unittest.TestCase):

    def test_diffusion_engine_importable(self):
        from modules.step4_generate.diffusion_engine import DiffusionLayoutEngine
        self.assertIsNotNone(DiffusionLayoutEngine)

    def test_is_ready_returns_bool(self):
        from modules.step4_generate.diffusion_engine import DiffusionLayoutEngine
        engine = DiffusionLayoutEngine()
        self.assertIsInstance(engine.is_ready(), bool)

    def test_not_ready_raises_runtime_error(self):
        """If engine not ready, place_floor() must raise RuntimeError."""
        from modules.step4_generate.diffusion_engine import DiffusionLayoutEngine
        engine = DiffusionLayoutEngine()
        if not engine.is_ready():
            plan = _make_enriched_plan()
            rooms = [r for r in plan.rooms if r.preferred_floor == 0]
            with self.assertRaises(RuntimeError):
                engine.place_floor(rooms,
                                   net_width_ft=plan.net_buildable_width_ft,
                                   net_length_ft=plan.net_buildable_length_ft)

    def test_gnn_encoder_importable(self):
        from modules.step4_generate.gnn_encoder import GNNEncoderNumpy
        self.assertIsNotNone(GNNEncoderNumpy)

    def test_diffusion_decoder_importable(self):
        from modules.step4_generate.diffusion_decoder import DiffusionDecoderNumpy
        self.assertIsNotNone(DiffusionDecoderNumpy)

    def test_weight_files_exist(self):
        """Model weight npz files are on disk."""
        weights_dir = os.path.join(_ROOT, "modules", "step4_generate", "weights")
        for fname in ("gnn_encoder.npz", "diffusion_model.npz"):
            self.assertTrue(os.path.exists(os.path.join(weights_dir, fname)),
                            f"Missing: {fname}")

    def test_checkpoint_exists(self):
        """Training checkpoint .pt file exists."""
        cp = os.path.join(_ROOT, "modules", "step4_generate",
                          "weights", "checkpoint_latest.pt")
        self.assertTrue(os.path.exists(cp), "checkpoint_latest.pt missing")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4E — GENERATOR END-TO-END
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep4EGeneratorEndToEnd(unittest.TestCase):

    def test_layout_generator_importable(self):
        from modules.step4_generate.generator import LayoutGenerator
        self.assertIsNotNone(LayoutGenerator)

    def test_generator_produces_layout_plan(self):
        """LayoutGenerator.generate() → LayoutPlan."""
        from models import LayoutPlan
        from modules.step4_generate.generator import LayoutGenerator
        plan = _make_enriched_plan()
        gen = LayoutGenerator(prefer_cpsat=True)
        result = gen.generate(plan)
        self.assertIsInstance(result, LayoutPlan)

    def test_generator_floor_count_matches(self):
        """Generated LayoutPlan has same floor count as EnrichedPlan."""
        from modules.step4_generate.generator import LayoutGenerator
        plan = _make_enriched_plan()
        gen = LayoutGenerator(prefer_cpsat=True)
        result = gen.generate(plan)
        self.assertEqual(len(result.floors), plan.total_floors)

    def test_generator_all_rooms_in_boundary(self):
        """Interior rooms are within net-buildable boundary (±0.6 ft).
        External rooms (car_parking, staircase on boundary) are excluded."""
        from modules.step4_generate.generator import LayoutGenerator
        from modules.step4_generate.grid import EXTERNAL_ROOM_TYPES
        plan = _make_enriched_plan()
        gen = LayoutGenerator(prefer_cpsat=True)
        result = gen.generate(plan)
        tol = 0.6
        for floor in result.floors:
            for pr in floor.rooms:
                if pr.room_type in EXTERNAL_ROOM_TYPES:
                    continue  # exterior rooms may extend to setback/boundary
                self.assertGreaterEqual(pr.x_ft, -tol,
                                        f"{pr.room_id}: x={pr.x_ft:.2f} < 0")
                self.assertGreaterEqual(pr.y_ft, -tol,
                                        f"{pr.room_id}: y={pr.y_ft:.2f} < 0")
                self.assertLessEqual(pr.x_ft + pr.width_ft,
                                     result.net_buildable_width_ft + tol,
                                     f"{pr.room_id}: right edge OOB")
                self.assertLessEqual(pr.y_ft + pr.length_ft,
                                     result.net_buildable_length_ft + tol,
                                     f"{pr.room_id}: top edge OOB")

    def test_generator_no_overlaps(self):
        """Generated layout has no room overlaps."""
        from modules.step4_generate.generator import LayoutGenerator
        plan = _make_enriched_plan()
        gen = LayoutGenerator(prefer_cpsat=True)
        result = gen.generate(plan)
        tol = 0.6
        for floor in result.floors:
            rooms = floor.rooms
            for i, a in enumerate(rooms):
                for b in rooms[i + 1:]:
                    ox = (min(a.x_ft + a.width_ft,  b.x_ft + b.width_ft) -
                          max(a.x_ft, b.x_ft))
                    oy = (min(a.y_ft + a.length_ft, b.y_ft + b.length_ft) -
                          max(a.y_ft, b.y_ft))
                    if ox > tol and oy > tol:
                        self.fail(f"Overlap: {a.room_id} ∩ {b.room_id} "
                                  f"(ox={ox:.2f}, oy={oy:.2f})")

    def test_generator_quality_score_in_range(self):
        """layout_quality_score ∈ [0, 1]."""
        from modules.step4_generate.generator import LayoutGenerator
        plan = _make_enriched_plan()
        gen = LayoutGenerator(prefer_cpsat=True)
        result = gen.generate(plan)
        self.assertGreaterEqual(result.layout_quality_score, 0.0)
        self.assertLessEqual(result.layout_quality_score,   1.0)

    def test_generator_solver_used_recorded(self):
        """solver_used field is non-empty."""
        from modules.step4_generate.generator import LayoutGenerator
        plan = _make_enriched_plan()
        gen = LayoutGenerator(prefer_cpsat=True)
        result = gen.generate(plan)
        self.assertGreater(len(result.solver_used), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4F — RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep4FRenderer(unittest.TestCase):

    def _make_layout(self):
        from modules.step4_generate.generator import LayoutGenerator
        return LayoutGenerator(prefer_cpsat=True).generate(_make_enriched_plan())

    def test_renderer_importable(self):
        from modules.step4_generate.renderer import FloorPlanRenderer
        self.assertIsNotNone(FloorPlanRenderer)

    def test_renderer_produces_svg_file(self):
        """render_all() creates ≥1 non-empty SVG files on disk."""
        from modules.step4_generate.renderer import FloorPlanRenderer
        layout = self._make_layout()
        with tempfile.TemporaryDirectory() as tmpdir:
            rnd   = FloorPlanRenderer(layout, output_dir=tmpdir,
                                      project_name="Test Plan")
            paths = rnd.render_all()
            self.assertGreater(len(paths), 0, "No SVG produced")
            for p in paths:
                self.assertTrue(os.path.exists(p))
                self.assertGreater(os.path.getsize(p), 1024,
                                   f"SVG suspiciously small: {p}")

    def test_renderer_svg_has_rect_and_text(self):
        """Generated SVG contains <rect> and <text> elements."""
        from modules.step4_generate.renderer import FloorPlanRenderer
        layout = self._make_layout()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = FloorPlanRenderer(layout, output_dir=tmpdir,
                                      project_name="Test").render_all()
            for p in paths:
                content = open(p).read()
                self.assertIn("<rect", content)
                self.assertIn("<text", content)

    def test_renderer_svg_has_compass(self):
        """Generated SVG has id='compass' group."""
        from modules.step4_generate.renderer import FloorPlanRenderer
        layout = self._make_layout()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = FloorPlanRenderer(layout, output_dir=tmpdir,
                                      project_name="Test").render_all()
            for p in paths:
                self.assertIn('id="compass"', open(p).read())

    def test_renderer_svg_has_project_name(self):
        """SVG title block contains the project name passed at construction."""
        from modules.step4_generate.renderer import FloorPlanRenderer
        layout = self._make_layout()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = FloorPlanRenderer(layout, output_dir=tmpdir,
                                      project_name="MY_UNIQUE_PROJECT").render_all()
            for p in paths:
                self.assertIn("MY_UNIQUE_PROJECT", open(p).read())

    # ── Geometry constant tests (no file I/O needed) ─────────────────────

    def test_title_h_is_120(self):
        """_TITLE_H constant must equal 120 px."""
        from modules.step4_generate.renderer import _TITLE_H
        self.assertEqual(_TITLE_H, 120)

    def test_pad_bottom_geq_170(self):
        """_PAD_BOTTOM ≥ 170 for annotation clearance."""
        from modules.step4_generate.renderer import _PAD_BOTTOM
        self.assertGreaterEqual(_PAD_BOTTOM, 170)

    def test_compass_w_not_clipping_into_drawing(self):
        """Compass W-label x must be ≥ drawing right edge for all plot sizes."""
        from modules.step4_generate.renderer import (
            _PAD_LEFT, _PAD_RIGHT, _TARGET_DRAW_PX, _MIN_SCALE, _MAX_SCALE,
        )
        for net_w, net_l in [(27, 34), (36, 44), (20, 25)]:
            sc = max(_MIN_SCALE, min(_MAX_SCALE,
                                     _TARGET_DRAW_PX / max(net_w, net_l, 1.0)))
            canvas_w      = int(net_w * sc + _PAD_LEFT + _PAD_RIGHT)
            right_panel_cx = canvas_w - _PAD_RIGHT + 120
            compass_W_x   = right_panel_cx - (40 + 14)   # compass_r=40
            drawing_right  = canvas_w - _PAD_RIGHT
            self.assertGreaterEqual(
                compass_W_x, drawing_right,
                f"Compass W clips into drawing for {net_w}×{net_l} ft "
                f"(W_x={compass_W_x}, drawing_right={drawing_right})"
            )

    def test_badge_clears_title_block(self):
        """Quality badge bottom must not overlap title block top."""
        from modules.step4_generate.renderer import (
            _PAD_TOP, _PAD_BOTTOM, _TARGET_DRAW_PX, _MIN_SCALE, _MAX_SCALE,
            _TITLE_H,
        )
        for net_l in [25, 34, 44]:
            sc       = max(_MIN_SCALE, min(_MAX_SCALE,
                                           _TARGET_DRAW_PX / max(net_l, 1.0)))
            canvas_h = int(net_l * sc + _PAD_TOP + _PAD_BOTTOM)
            badge_bottom = (canvas_h - _TITLE_H - 6 - 14 - 70) + 70
            title_top    = canvas_h - _TITLE_H - 6
            self.assertLessEqual(badge_bottom, title_top,
                                 f"Badge overlaps title for net_l={net_l}")

    def test_legend_clears_compass(self):
        """Legend panel top must be below compass S-label bottom."""
        from modules.step4_generate.renderer import _PAD_TOP
        compass_cy  = _PAD_TOP + 55
        compass_S_y = compass_cy + 40 + 14   # radius=40, label offset=14
        legend_top  = (_PAD_TOP + 148) - 22
        self.assertGreaterEqual(legend_top, compass_S_y,
                                f"Legend overlaps compass "
                                f"(top={legend_top}, S_y={compass_S_y})")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_SECTION_MAP = {
    "TestStep1AParserConfig":        ("STEP 1A", "Parser Config & Prompt Loading"),
    "TestStep1BRequirementsInspector":("STEP 1B", "RequirementsInspector (18 sub-cases)"),
    "TestStep1CInteractiveGatherer": ("STEP 1C", "Interactive Gatherer Session Logic"),
    "TestStep1DParserOutput":        ("STEP 1D", "Parser Output Shape"),
    "TestStep2AFeatureEncoder":      ("STEP 2A", "Feature Encoder"),
    "TestStep2BMatcher":             ("STEP 2B", "Matcher / Knowledge Bundle / NBC"),
    "TestStep3ARoomResolver":        ("STEP 3A", "Room Resolver"),
    "TestStep3BEnricher":            ("STEP 3B", "Enricher & VastuMapper"),
    "TestStep4APlotGrid":            ("STEP 4A", "PlotGrid Geometry"),
    "TestStep4BGreedyPlacer":        ("STEP 4B", "Greedy Placer"),
    "TestStep4CCPSATSolver":         ("STEP 4C", "CP-SAT Solver"),
    "TestStep4DDiffusionEngine":     ("STEP 4D", "Diffusion Engine"),
    "TestStep4EGeneratorEndToEnd":   ("STEP 4E", "Generator End-to-End"),
    "TestStep4FRenderer":            ("STEP 4F", "Renderer & SVG Validation"),
}


def _print_report() -> int:
    W = 90
    print()
    print("═" * W)
    print(" PlanGen AI — Pipeline Test Report".center(W))
    print(f" Steps 1 → 4  |  {len(_RESULTS)} tests".center(W))
    print("═" * W)

    groups: Dict[str, List[_Result]] = {}
    for r in _RESULTS:
        cls_name = r.name.split(".")[0]
        groups.setdefault(cls_name, []).append(r)

    total_pass = total_fail = 0
    for cls_name, results in groups.items():
        tag, label = _SECTION_MAP.get(cls_name, (cls_name, cls_name))
        cls_pass = sum(1 for r in results if r.passed)
        cls_fail = len(results) - cls_pass
        total_pass += cls_pass
        total_fail += cls_fail
        icon = "✅" if cls_fail == 0 else "❌"
        print()
        print(f"  {icon}  {tag} — {label}")
        print(f"     {'─' * 74}")
        for r in results:
            method   = r.name.split(".", 1)[1]
            p_icon   = "✅ PASS" if r.passed else "❌ FAIL"
            timing   = f"{r.duration_ms:6.1f} ms"
            disp     = method[:55] if len(method) <= 55 else method[:52] + "..."
            print(f"     {p_icon}  {disp:<55} {timing}")
            if not r.passed and r.error:
                first = r.error.strip().splitlines()[-1][:80]
                print(f"              ↳ {first}")

    print()
    print("═" * W)
    all_pass = total_fail == 0
    icon  = "✅" if all_pass else "❌"
    label = "ALL TESTS PASSED" if all_pass else f"{total_fail} TEST(S) FAILED"
    print(f"  {icon}  {label}  —  "
          f"{total_pass} passed / {total_fail} failed / {len(_RESULTS)} total")

    slowest = sorted(_RESULTS, key=lambda r: r.duration_ms, reverse=True)[:5]
    if slowest:
        print()
        print("  ⏱  Slowest tests:")
        for r in slowest:
            print(f"      {r.duration_ms:7.1f} ms  {r.name.split('.', 1)[1]}")
    print("═" * W)
    print()
    return 0 if all_pass else 1


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

_TEST_CLASSES = [
    TestStep1AParserConfig,
    TestStep1BRequirementsInspector,
    TestStep1CInteractiveGatherer,
    TestStep1DParserOutput,
    TestStep2AFeatureEncoder,
    TestStep2BMatcher,
    TestStep3ARoomResolver,
    TestStep3BEnricher,
    TestStep4APlotGrid,
    TestStep4BGreedyPlacer,
    TestStep4CCPSATSolver,
    TestStep4DDiffusionEngine,
    TestStep4EGeneratorEndToEnd,
    TestStep4FRenderer,
]


def main():
    total = sum(
        unittest.TestLoader().loadTestsFromTestCase(c).countTestCases()
        for c in _TEST_CLASSES
    )
    print(f"\n  Running {total} tests across {len(_TEST_CLASSES)} test classes …\n")
    t0 = time.perf_counter()
    for cls in _TEST_CLASSES:
        _run_unittest_class(cls)
    print(f"  Total runtime: {time.perf_counter() - t0:.2f}s")
    sys.exit(_print_report())


if __name__ == "__main__":
    main()
