"""
modules/step4_generate
======================
Step 4 — Layout Generator (the wall-graph partition engine).

Converts the EnrichedPlan (Step 3 output) into carved floor plans with exact
2-D geometry. Rooms are never *placed*; space is *split* by wall insertions,
so overlap is impossible by construction. See README.md in this package for
the full architecture.

Sub-packages
------------
    core/     lattice units, GridPlan (ownership grid, walls, openings), polygon
    carve/    hub-first carving, squeeze-and-settle optimizer, stairs
    engine/   orchestrator, contracts, rules/, cpsat/, multifloor, validator
    critic/   learned critic that ranks candidate plans
    render/   SVG + DXF export
    demos/    runnable visual demos

Entry points
------------
    from modules.step4_generate.engine.orchestrator import Orchestrator
    from modules.step4_generate.engine.multifloor import generate_building

Imports are plain package imports — nothing injects sys.path. Run everything
from the project root.
"""

from modules.step4_generate.room_budget import (  # noqa: F401
    assign_area_fractions,
    assign_generation_order,
)

__all__ = ["assign_generation_order", "assign_area_fractions"]
