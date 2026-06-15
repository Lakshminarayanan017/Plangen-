Complete Project Idealization Summary
    Product: AI-powered conversational floor plan generator for Indian homeowners

    Target user: Indian homeowner who has a plot and a rough idea of what they want. They'd normally go to an architect — this gives them a professional-quality starting point instantly.

THE USER FLOW:

    Step 1 — Conversation: User types what they want in plain English. AI (professional, formal tone — like a senior architect) asks only essential questions about plot details and room requirements. All text-based, no buttons. Once it has enough info, it generates.

    Step 2 — First plan: The system produces the best possible plan it can — not a draft. Uses real architectural data + Indian standards + LLM reasoning. Primary plan shown with alternative layout thumbnails the user can switch to.

    Step 3 — AI suggestions: After generation, AI proactively suggests improvements the user might not have thought of (balcony, Vastu adjustments, etc.)

    Step 4 — Refinement via chat: User types changes → AI highlights what will change → user approves → applied. Changes are surgical — only what was asked.

    Step 5 — Manual editing: User can also directly edit — drag rooms, drag walls to resize, add/remove doors and windows, change room types. Pure free movement, no snapping. Subtle validation indicators (red outline, yellow icon) — no interruptions.

    Step 6 — Repeat 4 & 5 until perfect. Full undo/redo history throughout.
    Step 7 — Export: DXF + PDF + PNG

LAYOUT:
    Side-by-side — chat on left, plan on right.

MULTI-FLOOR:

    1. Up to G+2 maximum
    2. System suggests room distribution across floors, user can change
    3. System suggests staircase position, user can override
    4. Staircase aligns across all floors
    5. Primary floor displayed, others as thumbnails — click to switch


KNOWLEDGE SYSTEM (3 layers):

    Layer 1 — Data: CubiCasa5K + ResPlan + Indian-specific datasets (scraped from IndiaFloorPlans, Naksha Dekho etc.)
    Layer 2 — Rules: Indian National Building Code, municipal bylaws, standard Vastu guidelines
    Layer 3 — LLM reasoning: Claude API as the architect brain — understands conversation, fills knowledge gaps, evaluates plans, suggests improvements


PLOT INPUT:

    1. User describes plot (size, shape) in text
    2. System extracts: boundary, road-facing side(s), north direction, nearby structure/setbacks
    3. Common Indian plot sizes supported (30x40, 20x30, 40x60 etc.)
    4. Photo/sketch detection — deferred to later


FEATURES:

    1. Vastu compliance — toggle on/off
    2. Furniture placement — toggle on/off
    3. 3D walkthrough — included (likely later phase but part of the vision)
    4. Building setbacks — system suggests standard values, user can override
    5. Car parking — user chooses type (stilt, garage, marked area)
    6. Impossible requirements — honest feedback + alternatives suggested
    7. User accounts with saved projects
    8. Multiple plan alternatives — one primary + thumbnails of options


OUTPUT QUALITY:
 
    Full architectural drawing style — thick outer walls, thin inner walls, door arcs, window marks, dimensions, room names, area in sqft, hatching, symbols
    Quality target: equal to or better than professional architect output


QUALITY PRINCIPLES (non-negotiable):

    1.Every habitable room gets natural light/ventilation
    2.Room sizes feel spacious, not cramped
    3.Minimal wasted corridor space
    4.Good room flow — kitchen near dining, bedrooms away from noise


BUSINESS:

    1. Free tool, monetize later
    2. Solo dev (may get help later)
    3. English only for now, other languages later


DEFERRED TO LATER:

    1. Photo/sketch plot detection
    2. Cost estimation
    3. Material quantity lists
    4. Architect/contractor marketplace
    5. Sharing/collaboration
    6. Multi-language support
    7. Mobile app