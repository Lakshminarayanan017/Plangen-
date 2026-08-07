# How to Build PlanGen: The Architecture Decision

## The Core Question You're Asking

> "Are we training a model with my data, or is Gemini generating the plan, or is it something in between?"

Let me answer this directly, then break it down.

---

## What Gemini CAN and CANNOT Do

### ✅ What Gemini is GREAT at (what we're already using it for):
- **Understanding natural language** → "I want a 3BHK house with Vastu"
- **Extracting structured data from text** → JSON with rooms, dimensions, preferences
- **Analyzing images** → Reading sketches, photos, plot plans
- **Generating conversational responses** → Architect persona, questions

### ❌ What Gemini CANNOT do well:
- **Generate precise spatial layouts** → It cannot say "put bedroom at x=5ft, y=10ft, width=12ft, height=10ft" reliably
- **Respect physical constraints** → Rooms must fit within plot boundaries, walls must align, rooms can't overlap
- **Produce consistent architectural drawings** → If you ask it to generate a floor plan image, it will make something that LOOKS like a plan but is architecturally meaningless (wrong dimensions, overlapping rooms, impossible layouts)
- **Do geometry** → LLMs are fundamentally bad at spatial reasoning and math

> [!CAUTION]
> **If you ask Gemini to "generate a floor plan", it will produce beautiful-looking garbage.** The rooms won't add up to the plot area, walls won't align, dimensions will be wrong. This is the #1 mistake people make.

---

## Your Data is the REAL Competitive Advantage

Look at what you already have — this is **incredibly valuable**:

| Dataset | What It Contains | Size |
|---|---|---|
| `rooms_extracted.json` | Every room from ~5000 real plans with exact positions, dimensions, areas | **3.3 GB** |
| `zone_patterns.json` | Where each room type typically sits (front/middle/back, left/center/right) | 145 KB |
| `zone_patterns_features.json` | Extended zone features per room type | 1.8 MB |
| `learned_patterns.json` | Adjacency weights, door ratios, connection frequencies between room pairs | 72 KB |
| `circulation_patterns.json` | Graph depth, journey efficiency (entry→kitchen = 1.34 hops avg), privacy violations | 8 KB |
| `normalized_extraction.json` | Full normalized plan database | **396 MB** |
| `full_extraction.json` | Raw extraction from all plans | **389 MB** |

**This is not "just data". This is the statistical DNA of ~5000 real floor plans.** No LLM has this. This is what makes YOUR system different from someone just asking ChatGPT to "draw a house".

---

## The Three Approaches

### 🅰️ Approach A: Pure Algorithmic (No ML Training)

```
User Requirements → Enrich with your data → Constraint Solver → Floor Plan
```

**How it works:**
1. Take enriched requirements (room list, sizes, adjacency preferences)
2. Use **your zone_patterns** to decide WHERE each room goes (kitchen goes in middle zone, 64% of the time)
3. Use **your adjacency_weights** to decide WHICH rooms should be neighbors (kitchen↔living_room: weight 4.34)
4. Use a **constraint-based space-partitioning algorithm** to physically place rooms:
   - Start with the plot rectangle
   - Place entrance based on road side
   - Recursively split the remaining space into rooms
   - Use techniques like: Binary Space Partitioning (BSP), Treemaps, Squarified layouts
5. Validate against Indian standards + Vastu
6. Render as SVG/Canvas drawing

**Your data's role:**
- `zone_patterns.json` → Room placement probabilities (WHERE to put each room)
- `learned_patterns.json` → Adjacency weights (WHICH rooms go next to each other)
- `circulation_patterns.json` → Validate the layout quality (is entry→kitchen ≤ 2 hops?)
- Indian standards → Minimum sizes, setbacks

**Pros:**
- ✅ No ML training needed
- ✅ 100% deterministic — same input = same output
- ✅ Every plan is architecturally valid by construction
- ✅ Can be built in 2-3 weeks
- ✅ Your data directly drives every decision

**Cons:**
- ❌ Plans may look "algorithmic" / rigid / grid-like
- ❌ Won't produce the organic feel of a hand-designed plan
- ❌ Complex L-shaped or irregular plots are harder

---

### 🅱️ Approach B: Hybrid (ML-Guided Placement + Algorithmic Validation) ⭐ RECOMMENDED

```
User Requirements → ML Model predicts room layout → Algorithm refines/validates → Floor Plan
```

**How it works:**
1. **Train a small ML model on your 5000 plans** that predicts:
   - Room positions (x, y coordinates as % of plot)
   - Room sizes (width, height)
   - Room-to-room relationships
2. The model takes as input: plot dimensions, room list, number of floors, Vastu flag
3. The model outputs: a rough layout (room positions + sizes)
4. An **algorithmic refinement step** then:
   - Snaps rooms to grid
   - Fixes overlaps
   - Aligns walls
   - Ensures rooms fit within plot boundary
   - Validates constraints
5. Render the final plan

**What kind of ML model?**

You don't need a massive deep learning model. Options:

| Model Type | Complexity | Training Data Needed | What It Predicts |
|---|---|---|---|
| **Graph Neural Network (GNN)** | Medium | Your room graphs (~5000) | Room adjacency graph → spatial embedding |
| **Conditional VAE** | Medium | Your normalized plans | Given requirements → sample a layout |
| **Simple MLP/Random Forest** | Low | Your zone_patterns + room stats | Per-room: zone, relative position, size |
| **Retrieval + Deformation** | Low | Your full plans | Find the 3 closest plans → warp to fit user's plot |

**Your data's role:**
- `rooms_extracted.json` (3.3 GB) → **THIS IS YOUR TRAINING DATA.** Every room's position, size, and type from 5000 plans
- `zone_patterns.json` → Features for the model (room type → typical zone)
- `learned_patterns.json` → Adjacency graph structure for GNN
- `circulation_patterns.json` → Quality scoring of generated layouts

**Pros:**
- ✅ Plans look more natural/organic (learned from real plans)
- ✅ Your data is fully utilized — it's literally the training set
- ✅ ML handles the "creative" placement, algorithm handles the "engineering"
- ✅ Still architecturally valid (algorithm validates everything)
- ✅ Can generate diverse alternatives by sampling

**Cons:**
- ❌ Requires ML training (but small model, not GPT-scale)
- ❌ Need to preprocess your 3.3GB room data into training format
- ❌ Takes 3-5 weeks to build properly

---

### 🅲 Approach C: Full Generative AI (Gemini/Diffusion)

```
User Requirements → Gemini generates layout JSON → Post-process → Floor Plan
```

**How it works:**
1. Feed enriched requirements + knowledge bundle to Gemini
2. Ask Gemini to output room positions as JSON
3. Post-process and validate

**Pros:**
- ✅ Fastest to prototype (just prompt engineering)
- ✅ Can handle creative/unusual requests

**Cons:**
- ❌ **Gemini will produce invalid layouts** — rooms overlapping, not fitting the plot, wrong dimensions
- ❌ Your datasets are reduced to prompt context (can't feed 3.3GB to an LLM)
- ❌ Non-deterministic — same input gives different (often wrong) output each time
- ❌ No geometric reasoning — LLMs literally cannot do spatial math
- ❌ Expensive API calls for every generation + validation loop
- ❌ **Your valuable data goes mostly unused**

> [!WARNING]
> Approach C wastes your biggest asset — your dataset of 5000 real plans. An LLM can read a few examples in its context window, but it cannot learn from 5000 plans the way a trained model can.

---

## My Recommendation: Approach B (Hybrid)

Here's the architecture I'd recommend:

```
┌──────────────────────────────────────────────────────────────┐
│                    YOUR EXISTING PIPELINE                     │
│                                                              │
│  Step 1 (Parse) ──→ Step 2 (Match) ──→ Step 3 (Enrich)     │
│  [DONE ✅]          [DONE ✅]           [TO BUILD]           │
│                                                              │
│  Uses: Gemini       Uses: Your         Uses: Gemini +       │
│        API          learned_patterns   Your data             │
│                     zone_patterns                            │
│                     circulation_patterns                     │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    THE NEW GENERATION CORE                    │
│                                                              │
│  Step 4 (Generate) ──→ Step 5 (Detail) ──→ Step 6 (Validate)│
│  [THE BIG ONE]         [Algorithm]         [Algorithm]       │
│                                                              │
│  Option 1: Constraint Solver (simpler, faster)               │
│  Option 2: Trained ML Model (better quality)                 │
│                                                              │
│  Uses: rooms_extracted.json (3.3GB training data)            │
│        zone_patterns.json (placement guide)                  │
│        Indian standards (validation rules)                   │
│        Vastu rules (constraint overlay)                      │
└──────────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    RENDERING LAYER                            │
│                                                              │
│  Step 7 (Alternatives) ──→ Step 8 (Render)                  │
│  [Re-run Step 4           [SVG / Canvas /                   │
│   with different seeds]    PNG output]                       │
│                                                              │
│  Gemini's role here: ZERO. This is pure code.               │
└──────────────────────────────────────────────────────────────┘
```

### Where Each Tool Fits:

| Component | Tool | Why |
|---|---|---|
| **Understanding user intent** | Gemini API | LLMs excel at NLP |
| **Image analysis** | Gemini Vision | LLMs excel at image understanding |
| **Interactive Q&A** | Gemini API | LLMs excel at conversation |
| **Enriching requirements** | Gemini + Your Data | Combine LLM reasoning with statistical patterns |
| **Room placement** | **Your trained model OR algorithm** | Geometric/spatial — LLMs can't do this |
| **Wall alignment, grid snapping** | Algorithm (Python code) | Pure math |
| **Validation** | Algorithm (Python code) | Rule checking |
| **Rendering to image** | Code (SVG/Pillow/matplotlib) | Drawing library |

---

## The "Mediator Model" You Were Thinking Of

> *"Are we gonna build a mediator model like this intermediate model will learn all the patterns..."*

**YES — that's exactly Approach B.** Your intuition is right. Here's what that "mediator" looks like:

```
Gemini (NLP brain)                    Your Trained Model (Spatial brain)
     │                                          │
     │  "User wants 3BHK,                      │  "Given 30x40 plot with 3BHK,
     │   30x40 plot,                            │   bedrooms go in back-right zone,
     │   East facing,                           │   kitchen in middle,
     │   Vastu compliant"                       │   living room in front-center"
     │                                          │
     └──────────── ENRICHED JSON ──────────────→│
                                                │
                                         SPATIAL LAYOUT
                                      (room coordinates)
```

- **Gemini** = the LANGUAGE brain (understands what the user wants)
- **Your trained model** = the SPATIAL brain (knows how to arrange rooms, learned from 5000 real plans)
- **Algorithm** = the ENGINEERING brain (validates, fixes, renders)

---

## Practical Next Steps

If you want to go with Approach B, here's the order:

### Phase 1: Build Step 3 (Enrich) — 1 week
The enricher takes Step 1's parsed JSON + Step 2's knowledge bundle and fills in ALL missing details. This is mostly Gemini + your data.

### Phase 2: Build Step 4 (Generate) — Start simple, then add ML
1. **Week 1-2:** Build a simple constraint-based placer (BSP tree / treemap). This gives you a working pipeline end-to-end.
2. **Week 3-4:** Preprocess your `rooms_extracted.json` into training data.
3. **Week 5-6:** Train a small model (GNN or conditional VAE) to predict room layouts.
4. **Replace** the simple placer with the ML model.

### Phase 3: Build Steps 5-8 — 2 weeks
Detail, validate, generate alternatives, render.

> [!IMPORTANT]
> **Build Step 4 with the simple algorithm FIRST.** Get the full pipeline working end-to-end, then swap in the ML model later. This way you always have a working system.

---

## Summary

| Question | Answer |
|---|---|
| "Is Gemini generating the floor plan image?" | **No.** Gemini understands text and images. It cannot generate valid spatial layouts. |
| "Do we need ML?" | **Not strictly**, but it produces better results. Start with algorithm, add ML later. |
| "How does my data fit in?" | **It's EVERYTHING.** Your 5000 plans are the training data / statistical guide that makes the system work. Without it, you'd just be guessing. |
| "What's Gemini's role going forward?" | Steps 1 (parse), 3 (enrich), maybe help with creative suggestions. Steps 4-8 are algorithm + your trained model. |
| "Should I train a big model?" | **No.** A small model trained on your specific data will outperform a giant general-purpose model for this task. |
