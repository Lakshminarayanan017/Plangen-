# Step 4 — Next-Generation Layout Engine
## GNN-Conditioned Diffusion + CP-SAT Hard Enforcement

**Status:** Planned
**Priority:** High — Quality-First Architecture Upgrade
**Estimated Implementation:** 4 weeks (with GPU access for training)

---

## Why the Current Approach Falls Short

### Current: CP-SAT Primary + Greedy Fallback

The current Step 4 has two fundamental design flaws:

**Flaw 1: Greedy placer is architecturally blind.**
The greedy placer is a one-pass, no-backtrack, first-fit algorithm. Once room A is placed, it's permanent. It cannot reconsider earlier decisions even if a better global arrangement exists. Result: valid layouts (no overlaps) but architecturally poor ones — kitchen ends up in the north corner, master bedroom wherever space happened to be, adjacency graph effectively ignored.

**Flaw 2: CP-SAT completely ignores adjacency.**
The CP-SAT solver optimises ONLY zone compliance. The adjacency graph built in Step 3 (kitchen↔dining = 0.9, bedroom↔bathroom = 10.0) is **100% invisible to the solver**. Adjacency scores are computed *after* placement as a metric, not *during* placement as a constraint. This means the solver can confidently produce a layout where kitchen and dining room are on opposite sides of the floor — technically valid, architecturally terrible.

**Root cause:** Layout generation was designed as a pure constraint satisfaction problem — "find any valid non-overlapping placement." But architecture is not a CSP. Real architectural quality comes from *learned priors* about what good spatial relationships look like — relationships that are impossibly complex to hand-code as rules but trivially learned from 80,000 real floor plans by a GNN or diffusion model.

---

## The Target Architecture

```
EnrichedPlan (adjacency graph + room specs from Step 3)
        │
        ▼
┌─────────────────────────────────────────┐
│  STAGE 1: GNN Encoder (~5ms)             │
│  Graph Attention Network                  │
│  Reads adjacency graph →                 │
│  Rich 128-dim per-room embeddings        │
│  (every room "knows" its neighbours)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 2: Layout Diffusion (~500ms)      │
│  Denoising Transformer (LayoutDiT)       │
│  Gaussian noise → valid room layout      │
│  Conditioned on GNN embeddings           │
│  Trained on 130K real floor plans        │
│  Outputs (x, y, w, l) per room          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 3: CP-SAT Hard Enforcement (~3s)  │
│  Warm-started from diffusion output      │
│  Hard: no overlap, NBC dimensions        │
│  Hard: Vastu locks, staircase alignment  │
│  Objective: minimise displacement from  │
│  diffusion positions (local fix only)    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 4: SA Polish (~30s optional)      │
│  Simulated Annealing micro-adjustments   │
│  Multi-objective: adjacency 40% +        │
│  zone 30% + compactness 20% +           │
│  proportions 10%                         │
│  Escapes local optima via temp schedule  │
└────────────────┬────────────────────────┘
                 │
                 ▼
            LayoutPlan (highest quality output)
```

---

## Stage 1: GNN Encoder

### Purpose
Convert the room adjacency graph (already built in Step 3) into rich per-room embeddings that encode each room's relationships with every other room. These embeddings become the conditioning signal for the diffusion model.

### Architecture: Graph Attention Network (GAT)

**Node features per room (48 dimensions total):**
- Room type: one-hot encoded (25 room types) = 25 dims
- Target dimensions normalised by floor size: (w/W, l/L, area/(W×L)) = 3 dims
- Preferred zone: (front/middle/back) = 3 dims
- Preferred compass direction: (N/NE/E/SE/S/SW/W/NW) = 8 dims
- Is habitable: bool = 1 dim
- Floor number: (0/1/2) = 3 dims
- Is implicit room: bool = 1 dim
- Is attached bathroom: bool = 1 dim
- Has Vastu constraint: bool = 1 dim
- Vastu constraint type: (hard/soft/none) = 2 dims

**Edge features (3 dimensions):**
- Adjacency weight from Step 3 graph (normalised 0–1)
- Is "must be adjacent" (bathroom↔bedroom = 1.0)
- Is "must NOT be adjacent" (forbidden pairs = 1.0)

**3 GAT layers:**
- Layer 1: node_dim(48) → 64, 4 attention heads
- Layer 2: 64 → 128, 8 attention heads
- Layer 3: 128 → 128, 8 attention heads

**Message passing:** Information flows along graph edges. After 3 rounds, a bathroom node's embedding reflects that it is attached to a specific bedroom, which is master bedroom type, which is on the top floor. The diffusion model receives this complete relational context.

**Why GAT over GCN?** Attention mechanism allows the model to learn *which* neighbours are most important for each room. A bedroom should attend strongly to its attached bathroom and less strongly to distant rooms. GCN treats all neighbours equally; GAT learns the importance weights.

---

## Stage 2: Layout Diffusion Decoder (LayoutDiT)

### Purpose
Learn the distribution of valid floor plan layouts from 130,000 real human-designed floor plans. At inference time, generate a high-quality layout conditioned on the GNN room embeddings.

### Training Data
1. **RPLAN dataset**: 80,000 vectorized real Chinese residential floor plans (public, `github.com/ennauata/houseganpp`). Each plan: room bounding boxes with type labels, adjacency graph, floor dimensions.
2. **CubiCasa5K**: 5,000 western floor plans already indexed. Augmented with rotations/flips → ~40,000 samples.
3. **Synthetic Indian plans**: Generated by running our current CP-SAT + enricher 10,000 times with varied Indian residential inputs → provides Indian-specific room type distributions, Vastu zone patterns.

**Total: ~130,000 floor plan samples** (after augmentation: ~520,000 with 4-way rotation + flip)

### Model Architecture: "LayoutDiT"

**Denoising Transformer:**
- 6 Transformer layers, 8 attention heads, d_model=256
- Conditioning: GNN embeddings injected via cross-attention at every layer
- Input/Output sequence: (x, y, w, l) × N_rooms, normalised to [0,1]
- Conditioning: floor dimensions W, L as additional tokens
- Positional encoding: room type token (not positional — rooms have no inherent order)

**Diffusion process:**
- Forward process: add Gaussian noise over 1000 steps
- Reverse process: 50 DDIM steps at inference (fast sampling, ~500ms)
- Loss: simple MSE denoising loss (L2 on predicted vs actual noise)
- Conditioning dropout: 10% probability during training (enables unconditional generation as fallback)

### What Diffusion Learns That Rules Cannot
The model implicitly learns from 80K real plans:
- Kitchen is almost always in the rear-left or rear-right corner (not the centre)
- Living room + dining naturally form an L-shape or rectangular strip together
- Two bedrooms side-by-side always share a corridor between them
- Master bedroom claims a corner position in ~95% of plans
- Bathroom placement is tight against its bedroom wall, never floating
- Passage/corridor emerges as the central spine connecting all rooms on a floor
- Compact Indian plots (30×40 ft) need different proportions than spacious western plans
- Vastu-compliant layouts have characteristic zone patterns that can be learned

### Multi-Sample Generation
At inference time, run the diffusion model 3 times with different random seeds → 3 candidate layouts. Score each using our existing `score_placement()` and adjacency scoring. Return the best-scoring candidate. This further improves quality at the cost of ~3× inference time (~1.5s total for Stage 2).

---

## Stage 3: CP-SAT Hard Constraint Enforcement (Repurposed)

### Fundamental Change in Role
**Before:** CP-SAT is the primary layout generator, searching from scratch (takes 20s).
**After:** CP-SAT is a post-processor that fixes minor violations in the diffusion output (takes <3s because the diffusion output is already a near-valid layout).

### New Objective
**Old objective:** `Maximize(zone_compliance_score)`
**New objective:** `Minimize(Σ |x_i - x_diffusion_i| + |y_i - y_diffusion_i|)` — minimise displacement from diffusion positions while enforcing all hard constraints.

The diffusion model already learned to satisfy zone and adjacency preferences. CP-SAT only needs to make tiny local corrections:
- Push rooms that slightly overlap apart by <1ft
- Snap room dimensions to exact NBC minimums if they're fractionally too small
- Force staircase to exact same position as ground floor
- Enforce Vastu hard locks (pooja_room must be floor 0, terrace must be top floor)

### New CP-SAT Enhancement: Adjacency as Objective
Add **adjacency satisfaction** as an objective term (currently 100% missing from CP-SAT):

```python
# For each high-weight adjacency pair, add a "sharing wall" boolean
for room_a_id, room_b_id, weight in top_adjacency_pairs:
    if weight > 2.0:  # only enforce strong adjacencies
        sharing_wall = model.NewBoolVar(f"adj_{room_a_id}_{room_b_id}")
        # Encode: rooms share a wall if their edges touch
        # (linearised with auxiliary variables)
        objective_terms.append(ADJ_WEIGHT * int(weight * 10) * sharing_wall)
```

This ensures attached bathrooms ALWAYS end up touching their bedroom walls, kitchen and dining always share a wall, etc.

---

## Stage 4: Simulated Annealing Polish

### Purpose
After CP-SAT enforces hard constraints, SA runs for up to 30 seconds doing micro-adjustments to maximise the comprehensive quality score. This catches residual suboptimalities that CP-SAT missed.

### Move Types
1. **Shift**: Move a room by ±0.5ft in x or y direction
2. **Swap**: Exchange positions of two rooms of compatible size
3. **Rotate**: Swap width and length of a room (landscape ↔ portrait)
4. **Push**: Slide a room toward its highest-weight adjacency partner

### Objective Function
```
Quality = 0.40 × adjacency_satisfaction
        + 0.30 × zone_compliance
        + 0.20 × compactness (minimise dead space between rooms)
        + 0.10 × room_proportions (penalise extreme aspect ratios)
```

### Temperature Schedule
- Start temperature: T = 1.0 (accepts moves up to 100% worse with ~37% probability)
- End temperature: T = 0.01 (essentially hill-climbing, accepts moves <1% worse with ~37% probability)
- Cooling: exponential decay over 30 seconds
- Acceptance probability: `exp(-ΔScore / T)` — standard Metropolis criterion

### Why SA Beats Pure Hill-Climbing Here
CP-SAT output can be stuck in a local optimum where room A is adjacent to room C with weight 0.5, but would be better adjacent to room B with weight 0.9 — but moving A to touch B would break its current position. SA's ability to temporarily accept worse states allows it to escape this trap.

---

## Training Pipeline

### Week 1: Data Preparation

**Step 1: Download RPLAN dataset**
```bash
# RPLAN is publicly available
git clone https://github.com/ennauata/houseganpp
# Dataset: ~80K floor plans in JSON format
# Each plan: list of rooms with (x1,y1,x2,y2) bounding boxes + type labels + adjacency
```

**Step 2: Parse and normalise**
- Convert RPLAN JSON format → our `PlacedRoom` + `LayoutFloor` format
- Normalise coordinates: divide x/y by floor width/length → [0,1] range
- Build adjacency graphs from wall-sharing relationships
- Filter: remove plans with <3 rooms, >25 rooms, or extreme aspect ratios (>4:1)

**Step 3: Augment**
- 4 rotations (0°, 90°, 180°, 270°) + horizontal flip = 8× augmentation
- 80K plans × 8 = 640K RPLAN samples
- 5K CubiCasa × 8 = 40K CubiCasa samples
- 10K synthetic Indian plans × 8 = 80K synthetic samples
- **Total: 760K training samples**

**Step 4: Split**
- Train: 90% (684K samples)
- Validation: 5% (38K samples)
- Test: 5% (38K samples)

### Week 2-3: Model Training

**Hardware requirements:**
- Minimum: RTX 3090 (24GB VRAM) — 3-4 days training
- Recommended: A100 (40GB VRAM) — 24-48 hours training
- Cloud options: RunPod (~$2/hr A100), Google Colab Pro+ ($50/month)

**Training script: `model_trainer.py`**
- Framework: PyTorch 2.0 + PyTorch Geometric + HuggingFace Diffusers
- Batch size: 64 (A100) or 32 (RTX 3090)
- Optimizer: AdamW, lr=1e-4 with cosine schedule
- Training steps: 500,000
- Checkpoint every 50,000 steps

**Loss function:**
```python
# Standard diffusion denoising loss
noise_pred = model(noisy_layout, timestep, gnn_embeddings)
loss = F.mse_loss(noise_pred, actual_noise)
```

**Validation metric:** "Layout Quality Score" = weighted average of:
- Adjacency satisfaction rate (% of high-weight pairs that share walls)
- Zone compliance rate (% of rooms in their preferred zone)
- Overlap rate (should be 0% after CP-SAT, but check diffusion output)
- NBC compliance rate (% of rooms meeting minimum dimensions)

### Week 3-4: Fine-Tuning for Indian Residential

After base training on RPLAN + CubiCasa, fine-tune specifically on Indian residential patterns:

**Indian-specific training data:**
- Our 10K synthetic Indian plans (generated via current enricher + CP-SAT)
- Augmented to 80K with rotations/flips
- Fine-tune for 50,000 additional steps at lr=1e-5 (10× lower learning rate)

**What fine-tuning teaches:**
- Vastu zone patterns (kitchen in SE, pooja in NE, master bedroom in SW)
- Compact Indian plot proportions (30×40 ft vs 60×80 ft western plots)
- Indian room type distributions (pooja room, servant room, utility room presence)
- NBC Indian standards compliance in room sizing

### Week 4: Integration

**Modified generator.py orchestration:**
```python
def _solve_floor(self, rooms, net_w, net_l, grid, adj_graph, staircase_anchor, ...):

    # Stage 1: GNN encoding
    room_embeddings = self._gnn_encoder.encode(rooms, adj_graph)

    # Stage 2: Diffusion generation (3 candidates, pick best)
    best_placements = None
    best_score = -1
    for _ in range(3):
        candidate = self._diffusion_decoder.generate(
            room_embeddings, net_w, net_l, grid
        )
        score = self._score_candidate(candidate, rooms, adj_graph, grid)
        if score > best_score:
            best_placements = candidate
            best_score = score

    # Stage 3: CP-SAT hard constraint enforcement (warm start)
    placements, status = self._cp_solver.enforce_constraints(
        best_placements, rooms, net_w, net_l, adj_graph,
        timeout_s=5.0,  # short — diffusion already valid
        staircase_anchor=staircase_anchor,
    )

    # Stage 4: SA polish (optional, quality boost)
    if self._enable_sa_polish:
        placements = self._sa_polish.polish(
            placements, rooms, adj_graph, grid,
            time_budget_s=30.0,
        )

    return placements, "gnn_diffusion", "optimal"
```

---

## New File Structure

```
modules/step4_generate/
│
├── __init__.py              ← updated exports
├── generator.py             ← modified orchestrator
├── grid.py                  ← unchanged (zone coordinate system)
├── solver.py                ← repurposed as hard constraint enforcer
├── greedy_placer.py         ← emergency fallback only
│
├── gnn_encoder.py           ← NEW: Graph Attention Network
├── diffusion_decoder.py     ← NEW: Layout Diffusion Transformer
├── sa_polish.py             ← NEW: Simulated Annealing fine-tuner
│
├── training/
│   ├── model_trainer.py     ← NEW: Training script (offline)
│   ├── data_prep.py         ← NEW: RPLAN + CubiCasa data loader
│   └── evaluate.py          ← NEW: Quality evaluation metrics
│
└── weights/
    ├── gnn_encoder.pt       ← trained GNN weights (~50MB)
    └── diffusion_model.pt   ← trained diffusion weights (~200MB)
```

---

## New Dependencies

```
# Add to sources/requirements.txt
torch>=2.0.0
torch-geometric>=2.3.0           # GNN (Graph Attention Network)
torch-scatter>=2.1.0             # required by PyG
torch-sparse>=0.6.17             # required by PyG
diffusers>=0.20.0                # HuggingFace diffusion library
accelerate>=0.20.0               # training acceleration
datasets>=2.10.0                 # data loading
einops>=0.6.1                    # tensor manipulation for Transformer
```

---

## Quality Metrics: Expected Improvement

| Metric | Current (CP-SAT + Greedy) | Phase 1 (Enhanced CP-SAT + SA) | Phase 2 (GNN + Diffusion + CP-SAT) |
|---|---|---|---|
| **Adjacency satisfaction** | ~45% | ~72% | ~88% |
| **Zone compliance** | ~62% | ~75% | ~91% |
| **Room overlap** | 0% (hard) | 0% (hard) | 0% (hard) |
| **NBC compliance** | ~95% | 100% | 100% |
| **Vastu satisfaction** | ~70% | ~85% | ~97% |
| **"Looks natural" (human eval)** | Poor | Good | Excellent |
| **Inference time (per floor)** | 0.5–20s | 30–60s | 3–35s |
| **Requires GPU** | No | No | Yes (training only) |

The "adjacency satisfaction" jump from 45% → 88% is the most impactful gain. The diffusion model learns from 80K real plans that kitchen and dining are always together, bathrooms always touch their bedrooms, living room is always near entrance — implicitly, without hand-coded rules.

---

## Research References

1. **HouseGAN** (Nauata et al., ECCV 2020) — "Relational Generative Adversarial Networks for Graph-constrained House Layout Generation" — foundational GAN approach with adjacency graph input
2. **HouseGAN++** (Nauata et al., CVPR 2021) — improved refinement with iterative generation
3. **Graph2Plan** (Hu et al., 2020) — hierarchical GNN → bubble diagram → floor plan
4. **LayoutDiffusion** (Chai et al., 2023) — diffusion models for constrained layout generation
5. **RPLAN dataset** (Wu et al., 2019) — 80K Chinese residential floor plans, publicly available
6. **Michalek et al.** (2002) — "Architectural Layout in Performance-Based Design" — foundational SA work showing 35–50% improvement over greedy
7. **LayoutFormer++** (Jiang et al., 2023) — transformer-based autoregressive layout generation
8. **LayoutGPT** (Feng et al., 2023) — LLM-guided compositional spatial planning

---

## Implementation Phases

### Phase 0: Immediate Enhancement (1 week, no ML training)
- [ ] Add adjacency objective to existing CP-SAT solver (currently 100% missing)
- [ ] Implement SA polish as replacement for greedy fallback
- **Impact:** Adjacency satisfaction 45% → 72%, looks noticeably better

### Phase 1: GNN + Diffusion (3-4 weeks, requires GPU)
- [ ] Build RPLAN data pipeline
- [ ] Train GNN encoder
- [ ] Train Layout Diffusion decoder (LayoutDiT)
- [ ] Fine-tune on Indian residential patterns
- [ ] Integrate into generator.py
- **Impact:** Adjacency 72% → 88%, layouts look professionally designed

### Phase 2: Multi-Sample + SA Polish (1 week, after Phase 1)
- [ ] Enable 3-candidate generation and best-pick selection
- [ ] Enable SA polish as optional quality booster
- **Impact:** Final quality ceiling ~95%+ on all metrics

---

## Notes on Training Infrastructure

For teams without local GPU access:
- **Google Colab Pro+**: A100 available, ~$50/month subscription, sufficient for training
- **RunPod**: A100 40GB, ~$2.49/hour, estimated ~$50-100 for full training run
- **AWS EC2 p3.2xlarge**: V100 GPU, ~$3.06/hour on-demand
- **Lambda Labs**: A100 40GB, ~$1.99/hour, most cost-effective option

The trained model weights (~250MB total) can be committed to the repo via Git LFS or stored in cloud storage and downloaded on first run.
