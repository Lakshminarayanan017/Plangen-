"""
matcher.py
==========
BACKWARD COMPATIBILITY SHIM — do not import directly.

All new code should import from semantic_matcher.py:
    from modules.step2_match.semantic_matcher import SemanticMatcher

This file exists only so existing code that imported PatternMatcher
from this file continues to work without modification.
"""

from modules.step2_match.semantic_matcher import SemanticMatcher, PatternMatcher

__all__ = ["SemanticMatcher", "PatternMatcher"]
