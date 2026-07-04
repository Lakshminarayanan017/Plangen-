"""
modules/step4_generate
======================
Step 4 — Layout Generator.

Converts the EnrichedPlan (Step 3 output) into a LayoutPlan with
exact 2-D room coordinates (x, y, width, length in feet).

Public entry point:
    from modules.step4_generate.generator import LayoutGenerator
    layout = LayoutGenerator().generate(enriched_plan)
"""

# Lazy import: generator.py pulls in solver.py / greedy_placer.py / grid.py
# which are NOT included in the trimmed training zip.  Import only when
# LayoutGenerator is actually accessed (i.e., at inference time).
def __getattr__(name: str):
    if name == "LayoutGenerator":
        from modules.step4_generate.generator import LayoutGenerator
        return LayoutGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LayoutGenerator"]
