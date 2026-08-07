"""
modules/step3_enrich
====================
Step 3 — Enricher: fills all gaps between parsed requirements and
a fully-specified plan ready for the layout generator.

Public API:
    from modules.step3_enrich.enricher import Enricher
    enriched_plan = Enricher().enrich(requirements, knowledge_bundle)
"""

from modules.step3_enrich.enricher import Enricher

__all__ = ["Enricher"]
