"""
Step 2 — Smart Semantic Matcher
================================
Finds the K most similar real floor plans from the CubiCasa5K index,
then assembles a rich KnowledgeBundle of room stats, adjacency weights,
zone probabilities, NBC standards, and (optionally) Vastu rules.

Primary exports
---------------
    SemanticMatcher   — production-grade matcher (use this)
    PatternMatcher    — backward-compatible alias
"""

from modules.step2_match.semantic_matcher import SemanticMatcher, PatternMatcher

__all__ = ["SemanticMatcher", "PatternMatcher"]
