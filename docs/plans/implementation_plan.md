# PlanGen Full-Scale Pipeline — Steps 2 through 8

## Goal

Build the **complete floor plan generation engine** — from the matcher all the way to rendered output. This takes the user's parsed requirements (Step 1 ✅ already built) and produces professional-quality architectural floor plans.

We are going **full-scale Approach B (Hybrid ML + Algorithmic)** — no shortcuts, maximum quality.

---

## Your Question: Is the Matcher Necessary?

> [!IMPORTANT]
> **YES — but the current matcher needs a complete rebuild.** Here's why:

The current [matcher.py](file:///c:/Users/Welcome/Desktop/PlanGen/modules/step2_match/matcher.py) is a **skeleton placeholder**. It:
- Tries to load `indian_standards.json` and `vastu_rules.json` from `sources/` — **those files don't exist there**
- Does simple dictionary lookups — no actual "matching" against your 5000 real plans
- Returns hardcoded fallback values (`min_width=8.0, min_length=8.0, target_area=64.0`)
- **Completely ignores** your most valuable data: `rooms_extracted.json` (3.3 GB), `normalized_extraction.json` (396 MB), `zone_patterns.json`, `zone_patterns_features.json`

**The matcher is the bridge between "what the user wants" and "what real architects actually build."** Without it, the generator would be guessing. With a proper matcher, it draws on 5000 real plans to inform every decision.

**What we're rebuilding it into:**
A **semantic retrieval engine** that finds the 10-20 most similar real plans from your dataset, extracts their statistical DNA (room sizes, adjacencies, zone placements, circulation quality), and packages it into a rich KnowledgeBundle that the enricher and generator can use.

---

## The Full Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STEP 1: PARSE  [DONE ✅]                    │
│  User text/image → Gemini → BuildingRequirements JSON               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 2: SMART MATCHER  [THIS PLAN - Phase 1]           │
│                                                                     │
│  BuildingRequirements → SemanticMatcher → KnowledgeBundle           │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │ Plan Indexer     │   │ Feature Encoder  │   │ Similarity      │  │
│  │ (offline, once)  │──▶│ (plot+rooms →    │──▶│ Search Engine   │  │
│  │                  │   │  feature vector) │   │ (top-K plans)   │  │
│  └─────────────────┘   └──────────────────┘   └────────┬────────┘  │
│                                                         │           │
│  ┌──────────────────────────────────────────────────────▼────────┐  │
│  │ Statistical Aggregator                                        │  │
│  │ • Room size distributions from matched plans                  │  │
│  │ • Adjacency graphs from matched plans                         │  │
│  │ • Zone placement probabilities from matched plans             │  │
│  │ • Circulation quality metrics from matched plans              │  │
│  │ • Indian Standards (NBC) + Vastu rules overlay                │  │
│  └──────────────────────────────────────────────────────┬────────┘  │
│                                                         │           │
│                                              KnowledgeBundle        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STEP 3: ENRICH  [THIS PLAN - Phase 2]                 │
│                                                                     │
│  BuildingRequirements + KnowledgeBundle → Gemini → EnrichedPlan     │
│                                                                     │
│  Fills ALL gaps: room sizes, floor distribution, adjacency prefs,   │
│  bathroom attachments, implicit rooms (passage, utility), setbacks, │
│  Vastu directional assignments, corridor planning                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│         STEP 4: GENERATE  [THIS PLAN - Phase 3]  ⭐ THE BIG ONE     │
│                                                                     │
│  EnrichedPlan → LayoutEngine → RoomLayout (coordinates)             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Stage A: Graph Construction                                 │   │
│  │  Room requirements → Weighted adjacency graph                │   │
│  │  Nodes = rooms, Edges = adjacency weights from data          │   │
│  └─────────────────────────────┬────────────────────────────────┘   │
│                                │                                    │
│  ┌─────────────────────────────▼────────────────────────────────┐   │
│  │  Stage B: GNN Spatial Embedding                              │   │
│  │  Graph → GNN → Room position predictions (x,y,w,h)          │   │
│  │  Trained on 5000 real plans from rooms_extracted.json        │   │
│  │  Model: Graph Attention Network (GAT) + Coordinate Decoder   │   │
│  └─────────────────────────────┬────────────────────────────────┘   │
│                                │                                    │
│  ┌─────────────────────────────▼────────────────────────────────┐   │
│  │  Stage C: Constraint-Based Refinement                        │   │
│  │  Raw positions → Snap to grid → Fix overlaps → Align walls   │   │
│  │  → Fit within plot boundary → Enforce min sizes              │   │
│  │  Algorithm: Quadratic Programming + BSP partitioning         │   │
│  └─────────────────────────────┬────────────────────────────────┘   │
│                                │                                    │
│                          RoomLayout JSON                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STEP 5: DETAIL  [THIS PLAN - Phase 4]                 │
│                                                                     │
│  RoomLayout → DetailEngine → DetailedPlan                           │
│                                                                     │
│  Adds: Wall thickness (150mm external, 100mm internal)              │
│        Door placement (type, width, swing direction)                │
│        Window placement (size based on room type + ventilation)     │
│        Room labels, dimensions, area annotations                    │
│        Optional furniture placement                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               STEP 6: VALIDATE  [THIS PLAN - Phase 4]               │
│                                                                     │
│  DetailedPlan → ConstraintValidator → ValidationReport              │
│                                                                     │
│  Hard rules (fail → re-generate):                                   │
│  • Min room sizes (NBC), plot boundary, setbacks, overlap-free      │
│  • Ventilation: every habitable room has exterior wall access       │
│                                                                     │
│  Soft rules (warn → suggest fix):                                   │
│  • Vastu compliance scoring (using vastuRules1.json)                │
│  • Circulation quality (entry→kitchen ≤ 2 hops)                     │
│  • Room flow (kitchen near dining, bedrooms away from noise)        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│           STEP 7: ALTERNATIVES  [THIS PLAN - Phase 5]               │
│                                                                     │
│  Re-run Step 4 with different seeds / zone preferences              │
│  → 2-3 validated alternative layouts                                │
│  Score and rank by: circulation quality + Vastu + space efficiency   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STEP 8: RENDER  [THIS PLAN - Phase 5]                  │
│                                                                     │
│  DetailedPlan → SVGRenderer → Architectural Drawing                 │
│                                                                     │
│  Output: SVG/PNG/DXF with thick outer walls, thin inner walls,      │
│  door arcs, window marks, dimensions, room names, area in sqft,     │
│  hatching, symbols — professional architect quality                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Changes

### Phase 1: Data Preprocessing + Smart Matcher (Step 2 Rebuild)

This phase transforms your raw datasets into training-ready formats and rebuilds the matcher into a real retrieval engine.

---

#### Data Preprocessing Pipeline

##### [NEW] modules/data_prep/__init__.py
##### [NEW] modules/data_prep/plan_indexer.py

Offline script (run once) that processes your massive datasets into an optimized search index:

1. **Reads** `normalized_extraction.json` (396 MB, 4989 plans) — each plan has: spaces (rooms with polygons, types, dimensions), walls, doors, windows, furniture, stairs
2. **Reads** `zone_patterns_features.json` (1.8 MB, 5000 plans) — each plan has: aspect ratio, depth/lateral spread, doors per room, compartmentalization, BHK type, zone balance
3. **For each plan, computes a feature vector** (~30-50 dimensions):
   - Plot aspect ratio (width/height)
   - Room count, BHK category
   - Room type composition (one-hot: has_kitchen, has_dining, has_balcony, etc.)
   - Zone balance features (% front, % middle, % back)
   - Compartmentalization score
   - Depth spread, lateral spread
   - Total area, average room area
4. **Builds a FAISS index** for fast approximate nearest neighbor search
5. **Saves** the index + plan metadata to `extracted data/plan_index/`

**Why FAISS?** Your 5000 plans need sub-millisecond lookup. FAISS (Facebook AI Similarity Search) does this. For 5000 vectors, even brute-force cosine similarity is fine, but FAISS gives us room to scale.

##### [NEW] modules/data_prep/training_data_builder.py

Converts `normalized_extraction.json` into training samples for the GNN:

1. **For each plan** (4989 plans):
   - Extract the room adjacency graph (which rooms share walls/doors)
   - Extract room coordinates (centroid x, y as % of plot width/height)
   - Extract room dimensions (width, height as % of plot)
   - Extract room type as categorical feature
   - Record plot dimensions and aspect ratio
2. **Output format**: One `.pt` (PyTorch) file per plan, or a single HDF5/pickle with all samples
3. **Data augmentation**: Mirror horizontally, mirror vertically, rotate 90° → 4x training data (4989 → ~20K samples)
4. **Train/val/test split**: 80/10/10

##### [NEW] modules/data_prep/graph_builder.py

Utilities for constructing room adjacency graphs from normalized plan data:
- Determines room adjacency from shared wall segments or door connections
- Builds `torch_geometric.data.Data` objects (nodes = rooms, edges = adjacencies)
- Node features: room type (one-hot), target area ratio, zone position
- Edge features: connection type (door vs wall-only), shared wall length

---

#### Smart Matcher (Step 2 Rebuild)

##### [DELETE] modules/step2_match/matcher.py
##### [NEW] modules/step2_match/semantic_matcher.py

Complete replacement of the placeholder matcher:

```python
class SemanticMatcher:
    """
    Production-grade semantic plan retrieval engine.
    
    Given user requirements, finds the K most similar real plans
    from CubiCasa5K and extracts their statistical DNA into a
    rich KnowledgeBundle.
    """
    
    def __init__(self, index_dir: Path):
        self.index = faiss.read_index(index_dir / "plan_vectors.index")
        self.plan_metadata = load_json(index_dir / "plan_metadata.json")
        self.indian_standards = load_indian_standards()
        self.vastu_rules = load_vastu_rules()
    
    def fetch_patterns(self, reqs: BuildingRequirements) -> KnowledgeBundle:
        # 1. Encode user requirements into same feature space
        query_vector = self._encode_requirements(reqs)
        
        # 2. Find top-K similar plans (K=15)
        distances, indices = self.index.search(query_vector, k=15)
        matched_plans = [self.plan_metadata[i] for i in indices[0]]
        
        # 3. Load full plan data for matched plans
        matched_plan_data = self._load_matched_plans(matched_plans)
        
        # 4. Aggregate statistics from matched plans
        bundle = self._aggregate_statistics(reqs, matched_plan_data)
        
        # 5. Overlay Indian standards + Vastu
        bundle.standards_applied = self.indian_standards
        if reqs.vastu_compliant:
            bundle.vastu_rules_applied = self.vastu_rules
        
        return bundle
```

**Key improvement over current matcher**: Instead of looking up hardcoded averages, this finds the 15 most similar real plans to what the user wants and extracts statistics *from those specific plans*. A 3BHK on a 30×40 plot gets data from similar 3BHK plans on similar-sized plots — not global averages.

##### [NEW] modules/step2_match/indian_standards.py

Hardcoded Indian National Building Code rules (from the NBC PDF you have):
- Minimum room sizes per type
- Minimum ceiling heights
- Setback requirements by plot size
- Ventilation requirements (1/6th of floor area)
- Staircase minimum dimensions
- Fire safety clearances

##### [MODIFY] models.py

Expand `KnowledgeBundle` to hold richer matched plan data:

```python
class MatchedPlanSummary(BaseModel):
    """Summary of a matched reference plan."""
    plan_key: str
    similarity_score: float
    bhk: str
    room_count: int
    aspect_ratio: float
    zone_balance: Dict[str, float]

class KnowledgeBundle(BaseModel):
    # ... existing fields ...
    matched_plans: List[MatchedPlanSummary] = []
    room_size_distributions: Dict[str, RoomSizeDistribution] = {}
    adjacency_graph_template: Optional[Dict] = None
    zone_placement_probabilities: Dict[str, Dict[str, float]] = {}
    circulation_quality_benchmarks: Dict[str, float] = {}
```

---

### Phase 2: Enricher (Step 3)

##### [NEW] modules/step3_enrich/__init__.py
##### [NEW] modules/step3_enrich/enricher.py

Takes `BuildingRequirements` + `KnowledgeBundle` and produces a fully-specified `EnrichedPlan`:

1. **Gap Detection**: What did the user NOT specify?
2. **Statistical Filling**: For each gap, use KnowledgeBundle data:
   - Room sizes → from matched plan distributions (median ± 10%)
   - Floor distribution → from typical patterns in matched plans
   - Adjacency preferences → weighted graph from matched plan adjacencies
   - Missing implicit rooms → add passage/corridor if compartmentalization > 0.6, utility room if kitchen exists
3. **Gemini Reasoning Layer**: For ambiguous decisions, use Gemini Flash:
   - "Given a 30×40 north-facing plot with 3BHK, should the staircase go on the east or west side?" → Gemini reasons with Vastu context
4. **Vastu Overlay** (if enabled):
   - Map rooms to Vastu grid zones from `vastuRules1.json`
   - Tag each room with directional preferences (hard vs soft constraints)
5. **Output**: `EnrichedPlan` — everything needed for Step 4

##### [NEW] modules/step3_enrich/models.py

```python
class EnrichedRoom(BaseModel):
    """Fully specified room for the generator."""
    room_id: str
    room_type: str
    target_width: float  # in ft
    target_height: float  # in ft
    min_width: float
    min_height: float
    target_area: float
    preferred_floor: int
    preferred_zone: str  # "front", "middle", "back"
    preferred_side: str  # "left", "center", "right"
    vastu_direction: Optional[str]  # e.g., "south_west" for master bedroom
    vastu_constraint_type: str  # "hard", "soft", "none"
    attached_rooms: List[str]  # e.g., ["bathroom"] for master bedroom
    adjacency_preferences: Dict[str, float]  # room_type → weight

class EnrichedPlan(BaseModel):
    """Complete specification ready for the generator."""
    plot_width: float
    plot_height: float
    setbacks: Setbacks
    rooms: List[EnrichedRoom]
    floors: int
    entrance_side: str
    adjacency_graph: Dict[str, Dict[str, float]]
    vastu_enabled: bool
    vastu_grid: Optional[Dict]  # 9x9 Paramasayika grid mapping
```

##### [NEW] docs/prompts/step3_enricher_system.md

System prompt for Gemini's reasoning in the enrichment step — architectural decision-making persona.

---

### Phase 3: Layout Generator (Step 4) ⭐ THE CORE

This is the heart of the system. **Two-tier architecture**: GNN for initial placement, algorithm for refinement.

---

#### GNN Model Architecture

##### [NEW] modules/step4_generate/__init__.py
##### [NEW] modules/step4_generate/gnn_model.py

**Graph Attention Network (GAT) with Coordinate Decoder**

```
Input: Room adjacency graph
  ├── Node features per room (dim ~24):
  │   ├── Room type (one-hot, 15 categories)
  │   ├── Target area ratio (area / plot_area)
  │   ├── Zone preference (3-dim: front/mid/back probability)
  │   ├── Side preference (3-dim: left/center/right probability)
  │   └── Floor assignment (1-dim)
  │
  ├── Edge features (dim ~4):
  │   ├── Adjacency weight (from data)
  │   ├── Connection type (door=1, wall=0.5, none=0)
  │   └── Same-floor flag
  │
  └── Global features (dim ~6):
      ├── Plot aspect ratio
      ├── Number of rooms
      ├── Number of floors
      └── Vastu flag
      
Architecture:
  Input → 3× GAT layers (hidden=128, 8 attention heads)
        → Global attention pooling
        → Coordinate Decoder MLP
        → Output: per-room (x, y, w, h) as ratios of plot dimensions

Loss Function:
  L = L_coord + λ₁·L_overlap + λ₂·L_boundary + λ₃·L_adjacency + λ₄·L_area
  
  Where:
  - L_coord: MSE between predicted and real room coordinates
  - L_overlap: Penalty for overlapping rooms (IoU-based)
  - L_boundary: Penalty for rooms outside plot boundary
  - L_adjacency: Penalty when adjacent rooms are placed far apart
  - L_area: Penalty for rooms smaller than minimum size
```

**Why GAT specifically?**
- Graph Attention Networks learn *which* neighbor rooms matter most when deciding where to place a room
- The attention mechanism naturally captures that "kitchen placement depends more on dining room than on bedroom 3"
- 3 layers = each room can consider up to 3-hop neighbors in the graph
- 8 attention heads = learns 8 different types of spatial relationships

##### [NEW] modules/step4_generate/trainer.py

Training pipeline:
1. Load preprocessed training data from Phase 1
2. Train GAT model with custom loss function
3. Learning rate: 1e-3 with cosine annealing
4. Batch size: 32
5. Epochs: 200 (with early stopping, patience=20)
6. Validation metric: mean overlap penalty + boundary violation rate
7. Save best model checkpoint to `models/checkpoints/`

**Hardware**: CPU-trainable (5000 small graphs, small model). GPU optional but not required.

##### [NEW] modules/step4_generate/layout_engine.py

The main layout generation orchestrator:

```python
class LayoutEngine:
    """
    Generates room layouts from enriched plans.
    
    Architecture: GNN prediction → Constraint refinement → Validation
    """
    
    def generate(self, plan: EnrichedPlan) -> RoomLayout:
        # Stage A: Build adjacency graph from enriched plan
        graph = self._build_room_graph(plan)
        
        # Stage B: GNN predicts initial room positions
        raw_layout = self.gnn_model.predict(graph, plan.plot_width, plan.plot_height)
        
        # Stage C: Algorithmic refinement
        refined_layout = self.refiner.refine(
            raw_layout, 
            plan.plot_width, 
            plan.plot_height,
            plan.setbacks,
            constraints=plan.get_constraints()
        )
        
        return refined_layout
```

##### [NEW] modules/step4_generate/constraint_refiner.py

Algorithmic post-processing that takes GNN's rough predictions and produces a valid layout:

1. **Grid Snapping**: Round room positions to nearest 6-inch grid (common in Indian construction)
2. **Overlap Resolution**: If rooms overlap, use force-directed push to separate them while maintaining adjacency
3. **Wall Alignment**: Align room edges that are within 6 inches of each other (architectural wall sharing)
4. **Boundary Enforcement**: Clamp rooms within plot boundary minus setbacks
5. **Minimum Size Enforcement**: If a room got squeezed below NBC minimum, expand it (pushing neighbors)
6. **Space Filling**: If there's leftover space, expand adjacent rooms proportionally

**Algorithm**: Iterative constraint satisfaction with simulated annealing for difficult cases.

##### [NEW] modules/step4_generate/models.py

```python
class RoomPlacement(BaseModel):
    """A single room's position in the layout."""
    room_id: str
    room_type: str
    x: float  # left edge, in ft from plot origin
    y: float  # top edge, in ft from plot origin
    width: float  # in ft
    height: float  # in ft
    floor: int
    
class RoomLayout(BaseModel):
    """Complete spatial layout of all rooms."""
    plot_width: float
    plot_height: float
    rooms: List[RoomPlacement]
    walls: List[WallSegment]
    generation_seed: int
    quality_score: float
```

---

### Phase 4: Detail Engine + Validator (Steps 5 & 6)

##### [NEW] modules/step5_detail/__init__.py
##### [NEW] modules/step5_detail/detail_engine.py

Takes `RoomLayout` and adds architectural details:

1. **Wall Generation**:
   - External walls: 150mm (6") thick
   - Internal walls: 100mm (4") thick
   - Shared walls between rooms become single wall segments
   - T-junctions and L-junctions handled properly

2. **Door Placement** (data-driven from `learned_patterns.json`):
   - Use `door_ratio` per room-pair to determine door positions
   - Main entrance: on entrance side, centered on the entry room
   - Internal doors: placed on shared walls, offset from corners
   - Door widths: Main=1000mm, Bedroom=900mm, Bathroom=750mm, Utility=600mm
   - Swing direction: away from smaller room, into larger room

3. **Window Placement**:
   - Every habitable room MUST have at least one window on an exterior wall
   - Window size: 1/6th of room floor area (NBC ventilation rule)
   - Placement: centered on exterior wall segments, height 900mm from floor
   - Kitchen: additional window for cross-ventilation if possible

4. **Furniture** (optional, toggle):
   - Kitchen: counter along longest wall, sink, stove
   - Bedroom: bed centered on wall opposite door, wardrobe
   - Bathroom: toilet, shower/bathtub, sink
   - Living room: sofa set facing entrance

##### [NEW] modules/step6_validate/__init__.py
##### [NEW] modules/step6_validate/constraint_validator.py

Multi-layer validation engine:

```python
class ConstraintValidator:
    def validate(self, plan: DetailedPlan) -> ValidationReport:
        hard_violations = []
        soft_warnings = []
        
        # Hard rules (must pass)
        hard_violations += self._check_room_sizes(plan)      # NBC minimums
        hard_violations += self._check_plot_boundary(plan)     # All rooms inside
        hard_violations += self._check_setbacks(plan)          # Building line
        hard_violations += self._check_overlaps(plan)          # No room overlaps
        hard_violations += self._check_ventilation(plan)       # Exterior wall access
        hard_violations += self._check_staircase(plan)         # Aligned across floors
        
        # Soft rules (warn but allow)
        soft_warnings += self._check_vastu(plan)               # Vastu scoring
        soft_warnings += self._check_circulation(plan)         # Path efficiency
        soft_warnings += self._check_room_flow(plan)           # Kitchen near dining, etc.
        soft_warnings += self._check_wasted_space(plan)        # Corridor % < 15%
        
        return ValidationReport(
            is_valid=len(hard_violations) == 0,
            hard_violations=hard_violations,
            soft_warnings=soft_warnings,
            vastu_score=self._compute_vastu_score(plan),
            circulation_score=self._compute_circulation_score(plan),
            space_efficiency=self._compute_efficiency(plan),
        )
```

##### [NEW] modules/step6_validate/vastu_scorer.py

Uses your `vastuRules1.json` (the 9×9 Paramasayika Mandala grid) to score room placements:
- Maps each room's position to the nearest Vastu pada (grid cell)
- Checks against ideal_usage and hard_blocks for each energy field
- Scores entrance position against the 32 outer perimeter gates
- Returns a 0-100 Vastu compliance score

---

### Phase 5: Alternatives + Renderer (Steps 7 & 8)

##### [NEW] modules/step7_alternatives/__init__.py
##### [NEW] modules/step7_alternatives/alternatives_generator.py

Generates 2-3 layout alternatives:
1. **Seed Variation**: Re-run Step 4 with different random seeds → different initial GNN predictions → different final layouts
2. **Zone Swapping**: Swap front/back zone assignments for key rooms (e.g., kitchen in front vs back)
3. **Mirror Plans**: Horizontal mirror of the primary plan
4. **Scoring**: Each alternative is validated (Step 6), scored on: Vastu score + circulation quality + space efficiency + adjacency satisfaction
5. **Ranking**: Present best plan as primary, others as alternatives

##### [NEW] modules/step8_render/__init__.py
##### [NEW] modules/step8_render/svg_renderer.py

Professional-quality SVG renderer:

```python
class ArchitecturalRenderer:
    """
    Renders DetailedPlan as a professional architectural drawing.
    
    Output style matches hand-drawn Indian architect plans:
    - Thick outer walls (3px), thin inner walls (1.5px)
    - Door arcs with swing direction
    - Window marks (parallel lines)
    - Room labels (centered, room type + area in sqft)
    - Dimension lines with measurements
    - North arrow
    - Scale bar
    - Title block (plot info, date, scale)
    """
    
    def render_svg(self, plan: DetailedPlan) -> str: ...
    def render_png(self, plan: DetailedPlan, dpi: int = 150) -> bytes: ...
    def render_dxf(self, plan: DetailedPlan) -> str: ...  # AutoCAD format
```

---

### Phase 6: Pipeline Integration

##### [MODIFY] main.py

Update the orchestrator to chain all steps:

```python
def run_pipeline(user_prompt, image_path=None):
    # Step 1: Parse (existing)
    reqs = step1_parse(user_prompt, image_path)
    
    # Step 2: Match (rebuilt)
    bundle = semantic_matcher.fetch_patterns(reqs)
    
    # Step 3: Enrich (new)
    enriched = enricher.enrich(reqs, bundle)
    
    # Step 4: Generate (new - GNN + algorithm)
    layout = layout_engine.generate(enriched)
    
    # Step 5: Detail (new)
    detailed = detail_engine.add_details(layout, enriched)
    
    # Step 6: Validate (new)
    report = validator.validate(detailed)
    if not report.is_valid:
        # Re-generate with adjusted constraints
        layout = layout_engine.generate(enriched, constraints=report.fixes)
        detailed = detail_engine.add_details(layout, enriched)
    
    # Step 7: Alternatives (new)
    alternatives = alt_generator.generate(enriched, count=3)
    
    # Step 8: Render (new)
    primary_svg = renderer.render_svg(detailed)
    alt_svgs = [renderer.render_svg(alt) for alt in alternatives]
    
    return PipelineResult(primary=detailed, alternatives=alternatives, svgs=...)
```

---

## Model Choices — The High-End Stack

| Component | Model/Algorithm | Why This Over Alternatives |
|---|---|---|
| **Text Understanding** (Step 1) | Gemini 2.5 Flash | Already working. Best cost/quality for NLP. |
| **Image Analysis** (Step 1) | Gemini 2.5 Flash (Vision) | Already working. Multimodal. |
| **Plan Retrieval** (Step 2) | FAISS + Custom Feature Encoder | Sub-ms lookup on 5000 plans. Scales to 100K+. |
| **Enrichment Reasoning** (Step 3) | Gemini 2.5 Flash | Architectural reasoning for gap-filling. |
| **Layout Generation** (Step 4) | Graph Attention Network (GAT) | Learns spatial relationships from your data. Superior to MLP (doesn't understand graph structure) and GCN (fixed aggregation weights). GAT's attention mechanism learns which neighbors matter most. |
| **Constraint Refinement** (Step 4) | Simulated Annealing + QP | Handles non-convex constraint spaces. Better than greedy (gets stuck in local minima). |
| **Validation** (Step 6) | Rule engine + Vastu scorer | Deterministic, explainable, auditable. |
| **Rendering** (Step 8) | SVG + Cairo | Vector output, scalable, professional quality. |

> [!IMPORTANT]
> **Why GAT over simpler options?**
> - **Random Forest/MLP**: Can predict room positions individually but ignores relationships between rooms. A bedroom's position depends on WHERE the bathroom is → you need graph reasoning.
> - **GCN (Graph Convolutional Network)**: Treats all neighbors equally. But kitchen's position depends MORE on dining room than on bedroom 3. GAT learns these importance weights via attention.
> - **Conditional VAE**: Great for sampling diverse layouts but harder to control. GAT + constraint refinement gives better controllability.
> - **Diffusion Models**: State-of-the-art quality but needs 10x more training data and GPU-hours. Overkill for 5000 plans. GAT trains in ~1 hour on CPU.

---

## Dependencies to Install

```
# Core ML
torch>=2.0
torch-geometric>=2.4
faiss-cpu>=1.7  (or faiss-gpu if GPU available)

# Data processing
h5py
pandas
numpy
scipy

# Rendering
svgwrite
cairosvg  (for PNG export)
ezdxf     (for DXF export)

# Existing (already installed)
google-genai
pydantic
python-dotenv
```

---

## Open Questions

> [!IMPORTANT]
> **Multi-floor handling**: Your WF.md says up to G+2 (3 floors). Should the GNN handle all floors simultaneously (one big graph with floor-assignment labels), or should we generate each floor independently with staircase position as a shared constraint? Independent per-floor is simpler and each floor gets the full model attention, but the simultaneous approach ensures better consistency.

> [!IMPORTANT]
> **Training compute**: The GAT model trains on CPU in ~1-2 hours with 5000 plans. Do you have a GPU available? If yes, we can also experiment with a larger model variant. If CPU-only, the current architecture is sized perfectly for that.

> [!IMPORTANT]
> **Data split**: CubiCasa5K is a Finnish dataset. Your zone patterns and learned patterns are computed from it. The plans are apartment-style (not specifically Indian residential). For Step 3 (Enrich), we'll overlay Indian standards. But the GNN will learn spatial layouts from Finnish plans. This is actually fine — room adjacency patterns (kitchen near dining, bathroom near bedroom) are universal. The Indian-specific aspects (Vastu, setbacks, typical BHK configurations) are handled by the enricher and validator, not the GNN. Does this approach sound right to you?

---

## Execution Order

| Phase | What | Duration | Prereq |
|---|---|---|---|
| **Phase 1** | Data preprocessing + Plan indexer + FAISS index | 2-3 days | None |
| **Phase 1b** | Smart Matcher rebuild (Step 2) | 1-2 days | Phase 1 |
| **Phase 2** | Enricher (Step 3) | 2-3 days | Phase 1b |
| **Phase 3a** | Training data builder + Graph builder | 2-3 days | Phase 1 |
| **Phase 3b** | GNN model + Training | 3-4 days | Phase 3a |
| **Phase 3c** | Layout engine + Constraint refiner | 3-4 days | Phase 3b |
| **Phase 4** | Detail engine + Validator | 3-4 days | Phase 3c |
| **Phase 5** | Alternatives + Renderer | 2-3 days | Phase 4 |
| **Phase 6** | Pipeline integration + End-to-end testing | 2-3 days | All |

**Total: ~3-4 weeks of intensive work**

---

## Verification Plan

### Automated Tests

```bash
# Phase 1: Data preprocessing
python -m modules.data_prep.plan_indexer --verify   # Builds index, prints stats
python -m modules.data_prep.training_data_builder --verify  # Builds training data, prints stats

# Phase 1b: Smart Matcher
python -m modules.step2_match.semantic_matcher --test "3BHK 30x40 north-facing"

# Phase 2: Enricher
python -m modules.step3_enrich.enricher --test

# Phase 3: GNN Training
python -m modules.step4_generate.trainer --epochs 5 --validate  # Quick smoke test
python -m modules.step4_generate.trainer --epochs 200  # Full training run

# Phase 4: Detail + Validate
python -m modules.step5_detail.detail_engine --test
python -m modules.step6_validate.constraint_validator --test

# Phase 5: Render
python -m modules.step8_render.svg_renderer --test  # Renders sample plan

# Full pipeline
python main.py "3BHK house on 30x40 north-facing plot with Vastu"
```

### Quality Metrics
- **Overlap rate**: < 1% of generated plans should have any room overlaps after refinement
- **Boundary violation**: 0% — no room extends beyond plot boundary
- **Ventilation compliance**: 100% — every habitable room has exterior wall access
- **Space efficiency**: > 85% of buildable area used (< 15% corridor/passage)
- **Vastu score**: > 70/100 when Vastu is enabled
- **Circulation quality**: Entry→kitchen ≤ 2.5 hops average (your data shows 1.34 avg)
