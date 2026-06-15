Work Flow on how the user prompt been processed from text to Floor design.

8 - step Pipeline:
    Step 1 — Parse: The user's text gets analyzed. We extract structured data from it

    Step 2 — Fetch Matching Patterns: A small data extraction algorithm to works on with plans similar to the user requirement from our dataset to give idea about the fittings(room size, placements, relationships....)

    Step 3 — Enrich: The system fills in everything the user DIDN'T say using learned data patterns. Room sizes, adjacency preferences, which rooms go on which floor, bathroom attachments, corridor needs — all from CubiCasa/ResPlan patterns + Indian standards.

    Step 4 — Generate: An algorithm takes the enriched requirements and produces a spatial layout — actual room positions, dimensions, wall placements within the plot boundary.

    Step 5 — Detail: Doors, windows, wall thickness, and optionally furniture get placed based on rules and data patterns.

    Step 6 — Validate: The constraint checker runs — minimum sizes, ventilation, Vastu (if enabled), setbacks. If something fails, the generator adjusts and re-runs.

    Step 7 — Alternatives: The generator produces 2-3 variations by making different choices at Step 3 (different room arrangements).

    Step 8 — Render: The validated plan gets drawn as a full architectural-style output and sent to the frontend.


Step 1 - Parse:
    1. User gives prompt
    2. LLM extracts only what's explicit → produces a structured JSON with only what the user said
    3. System checks: is essential info present? (plot size, room list, number of floors)
    4. If essential info is missing → AI asks ONLY those questions (the tier 1 questions we discussed)
    5. Add user's response to enriched data along with the previously extracted data.
    6. Move to Step 2.

Step 2 — FETCH MATCHING PATTERNS
    1. Take the parsed user data (plot size, room list, floor count) from Step 1
    2. Query learned_patterns.json
    3. Find matching patterns from similar real plans:
        - Plans with similar plot dimensions (e.g. 30x40 ± tolerance)
        - Plans with similar room configuration (e.g. 3BHK with pooja room)
        - Plans with similar floor count (G+1)
    4. Retrieve statistical data from matched plans:
        - Average room sizes per room type
        - Common adjacency relationships (which rooms are next to which)
        - Typical floor distribution (what usually goes on ground vs first)
        - Bathroom attachment patterns
        - Corridor/passage percentages
        - Implicit extra rooms commonly present (common bathroom, utility, etc.)
        - Door and window placement tendencies
    5. Also load relevant Indian architectural standards
    (National Building Code minimums, setback norms, ventilation rules)
    6. If Vastu is enabled → load Vastu rulebook for room orientations
    7. Package all this as a "knowledge bundle" for Step 3
    8. Move to Step 3

Step 3 - Enrich
    1. Receive the structured JSON from Step 1 + knowledge bundle from Step 2
    2. Identify gaps — what's NOT in the user's JSON that's needed for generation
    3. For each gap, fill it using the knowledge bundle:
        - Room sizes → from statistical averages
        - Floor distribution → from common patterns
        - Adjacency preferences → from matched plans
        - Bathroom attachments → from attachment patterns
        - Implicit extra rooms → add common bathroom, passage, etc. if missing
    4. Apply Indian architectural standards:
        - Enforce minimum room sizes (e.g., bedroom ≥ 9x9 ft)
        - Enforce setback requirements
        - Ensure ventilation rules (habitable rooms need exterior access)
    5. If Vastu is enabled → apply Vastu rules:
        - Hard Vastu rules → tagged as constraints (must be followed)
        - Soft Vastu rules → tagged as preferences (try to follow)
        - Technical details of which rules are hard vs soft → defined in Vastu rules and technical workflow.
        - Tag these as "Vastu preferences" on each room
    6. Produce the final enriched JSON — has everything needed for generation:
        - Every room has: type, size, floor, adjacency preferences, directional preferences
        - Plot has: boundary, setbacks, entrance side, north direction
        - Global preferences are set (Vastu toggle, style, etc.)
    7. Move to Step 4.

Step 4 - Generate
    1. Receive enriched JSON from Step 3.
    2. Layout Generation - This is the core logic where actual room positions and dimensions are determined within the plot boundary:
        1. Entrance & Staircase: Place entrance(s) and staircase according to entrance side and Vastu (if enabled). Staircase position must be consistent across floors.
        2. Room Placement: Place rooms based on adjacency preferences and floor distribution.
        3. Dimension Allocation: Assign dimensions to rooms using statistical averages (from Step 3) while maintaining desired adjacencies.
        4. Wall & Passage Creation: Generate walls and corridors to create the floor layout.
    3. Add optional furniture based on room types.
    4. Run validation checks (Step 6) - If invalid, adjust and regenerate layout.
    5. Produce initial valid plan.
    6. Generate 2-3 alternative layouts using different random seeds or preference weighting.
    7. Move to Step 5.
    
Step 5 - Detail
    1. Receive the layout JSON from Step 4.
    2. Add structural elements:
        1. Wall thickness (100-150mm typical for Indian construction)
        2. Doors (600-1000mm widths, specified swing directions)
        3. Windows (size and placement based on room type and ventilation needs)
        4. Room labels and dimensions
        5. Area calculations for each room and total
    3. Optionally add furniture based on room types (kitchen counters, bed/wardrobe in bedrooms, etc.).
    4. Ensure staircase aligns correctly across all floors.
    5. Move to Step 6.

Step 6 - Validate
    1. Receive the detailed plan from Step 5.
    2. Run validation checks:
        1. Minimum room sizes
        2. Ventilation and natural light for habitable rooms
        3. Structural integrity (wall connections, load-bearing considerations if applicable)
        4. Vastu compliance (if enabled) - check hard rules first, then soft rules
        5. Setback requirements
        6. Accessibility (door swings, circulation paths)
        7. Kitchen and bathroom functionality
    3. If plan fails any hard rules, regenerate layout (Step 4) with adjusted constraints.
    4. If plan fails soft rules, flag as warnings but allow user to proceed with option to fix.
    5. Once valid, move to Step 7.

Step 7 - Alternatives
    1. Receive the validated plan from Step 6.
    2. Generate 2-3 alternative layouts using different room arrangements, orientations, or minor variations in dimensions.
    3. Each alternative should be valid and compete with the primary plan.
    4. Provide these alternatives to the user for selection.
    5. Move to Step 8.