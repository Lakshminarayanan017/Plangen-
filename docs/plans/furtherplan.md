Architecture & Sequencing Document

PlanGen Build Sequence
LATTICE 1.5" cell
EXT WALL 9" / 6 cells
INT WALL 4.5" / 3 cells
Status
Engine complete · Product thin
Tests
259 / 260 passing
Harness mean
69.50 · 28/28 briefs
Placer gate
FAILED −6.92
Revised
2026-09-05
Everything below is ordered by dependency, not preference. The engine underneath PlanGen is genuinely strong — a partition-first carver on an integer lattice where overlaps and gaps are unrepresentable rather than merely checked for. The gaps are almost entirely above it: what reaches the engine, and what the user can do once it answers.

Baseline
What is measured, today
Verified
2026-09-05
Every figure here was run against the working tree, not read from a document. This is the ruler the rest of the plan is measured against.

Subsystem	State	Measurement
Wall-graph engine	Solid	28/28 briefs plan · mean 69.50
Multi-floor + stairs	Solid	stair overlap 100% · walls 100%
Learned critic	Solid	AUC 0.912 · rank 91.4 vs 63.2
Irregular plots	Weak	trapezoid brief scores 5.23
Small plots	Floor	dead below ~450 sqft buildable
Vastu in geometry	Absent	0 refs in step4_generate/
Room shapes	Rect only	guillotine split, by construction
Program reasoning	Absent	scales, never reconsiders
Editing / iteration	Absent	0 of 13 endpoints
Trained placer	Gate failed	68.57 vs 75.49 · fallback 41.8%
Persistence	In-memory	no DB · restart loses all runs
API tests / CI	None	0 endpoint tests · no pipeline
The headline gap
Vastu is computed in full by step 3 — 22 rules, per-room compass directions, plot-relative zones — and then discarded at api/engine_bridge.py, which passes only (name, rtype, target_sqft, zone) where zone is a hardcoded public/service/private lookup. RoomSpec has no direction field to carry it.

On a single-floor plan, Vastu currently changes nothing about where rooms go.

Model
Is the placer design sound? Partly — and the fix is not more epochs
Tier 2
proposer
The engineering is good: masked decoding makes invalid layouts unsamplable, the NumPy inference path matches torch to 7e-7, and two subtle bugs (confidence ceiling, checkpoint-selection drift) were already found and fixed. The formulation has four specific weaknesses, and they explain the failed gate better than undertraining does.

W1
The decoder is spatially blind
Each step conditions on prev_cell and prev_size embeddings of the immediately previous room, plus causal attention over prior room tokens. It never sees an occupancy raster. For a packing task that is like filling a suitcase from a written list instead of looking at it — the legality mask corrects the output, but the model was never trained to anticipate the constraint.

Symptom: mean confidence 0.355 against τ = 0.35 · fallback rate 41.8%

W2
The objective imitates the wrong distribution
Loss is cross-entropy on Finnish room centroids. The engine score is used only for checkpoint selection, never in the gradient. With this domain gap, better imitation means more Finnish — which is why the regressions cluster on narrow deep plots, the archetype where Indian convention diverges most.

20×45 −25.68 · 30×50 −25.68 · 30×60 −32.54 · 50×40 −21.17

W3
The head is mismatched to what the carver consumes
The model emits a 1024-way cell choice, but hub_carver reduces that seed to band membership and within-band rank. Meanwhile it receives no signal about desired proportion — which is why a generous program yields slab rooms.

Observed on 60×40: Bath 1 — 8'3" × 20'10.5"

W4
It cannot see the things that will decide quality
No Vastu direction input, no plot-scale regime, no entrance-relative program intent. If Vastu enters the geometry while the model stays blind to it, the model and the rules will pull against each other.

Decides where the GPU budget goes
Room shape is not a model problem. The transformer's target is (seed cell, size class) derived from polygon centroid and area — it never learns shape, and never will under this contract. Switching CubiCasa → RPLAN would put polygonal rooms through the same prep step and reduce them to centroids again. L-shaped rooms require a carver change (GridPlan.split() demands rectangular faces), with no ML involved.

Redesign — five targeted changes, not a rewrite
Keep the GAT + boundary-CNN + AR skeleton, the masked decoding, the NumPy inference path and the lattice contract. Change the state, the objective, and the head.

R1
Spatial state conditioning
Recompute a small conv encoding of the live state at every decode step — footprint mask, entrance edge, cells claimed so far, distance-to-nearest-placed-room — and cross-attend to it. This converts a sequence model into an actual spatial policy, and it is the precondition for R2c working at all.

R2
Three-stage training: imitate → distil → reinforce
a. Pretrain on CubiCasa as today, for general spatial competence — rooms don't collide, wet rooms cluster, public rooms sit forward. Domain-agnostic priors only.

b. Distil on self-generated Indian data: sweep briefs across the real Indian plot distribution, run the existing algorithmic engine at k=16, score with the 33 rules plus the learned critic, keep the top candidates. That is an unlimited supply of Indian-scored (brief → placement) pairs, available today.

c. Reinforce against reward = soft score + critic, so the model can exceed the hand-tuned proposer rather than merely match it.

R3
Aspect and band-hint heads
Add a coarse aspect-class head and a long-axis orientation head so the carver can honour proportion, and a band-index auxiliary head that predicts literally what the carver consumes. Cheap losses, high alignment, direct hit on the slab-room failure.

R4
Vastu and scale-regime conditioning
Per-room: preferred compass direction (9-way) and constraint hardness. Global: plot area bucket, aspect, and a compact / normal / spacious regime token, so the model learns that 500 sqft and 2500 sqft want different strategies rather than the same one scaled.

R5
Capacity by measurement, not upfront
7.36M is already generous against 4,430 real samples — growing it now buys overfitting. After R2b the effective corpus is unlimited, so capacity becomes worth spending. Hold at ~7–10M for stage a, allow ~25–30M for b/c only if the distillation curve says it is capacity-bound.

Why this works without Indian data
There is no Indian floor-plan corpus available — but there is an Indian-tuned reward function already built and validated: 33 reviewer rules encoding NBC minimums, circulation, openness gradient and the stated quality bar, plus a critic at AUC 0.912. When you cannot get the dataset, the reward is the dataset. R2b and R2c are how the domain gap gets crossed.

Tracks
Six workstreams, three of them parallel
A – F
Letters, not numbers — these run concurrently. The phases that follow are numbered because those are dependency-ordered.

Track	Owns	Blocks
A · Differentiator	Vastu into geometry, program synthesis, scale reasoning	D (reward must be frozen first)
B · Product loop	K-candidate choice, regenerate, prompt edits, manual edits, versioning	—
C · Conversation	Tool-calling agent replacing the state machine	—
D · ML	Placer v3: redesign, pretrain, distil, RL, gate	blocked by A
E · Geometry	Non-rectangular rooms, merge primitive, better irregular-plot use	forces re-gate of D
F · Platform	Persistence, accounts, CI, API tests, observability, exports	—
Sequence
Eight phases, in dependency order
0 → 7
00
Foundation repairs
Track F
Small, unglamorous, and it makes every later change safe to attempt.

Install ijson — clears the one failing test
Add data/nbc_plot_regulations.json — clears the one DEGRADED check
First API tests with TestClient — currently zero exist
CI pipeline running the 260 tests on push
Correct the stale README frontend section and prune phantom deps (svgwrite, the retired diffusion note)
Fix floor_coverage_pct, hardcoded to 100.0 — report real room fill (82–92%)
Risk low
Unblocks everything
01
Vastu into the geometry
Track A · differentiator
The feature being advertised and not shipped. No ML, no new data, and it must land before any proposer is gated — the gate is measured in soft score, and adding a rule family changes what that score means.

Add direction and vastu_zone to RoomSpec; have the bridge pass what step 3 already computes
New VAS-* soft rule family in the reviewer, with per-rule geometric evidence
Directional bias in PriorProposer._depth_base so seeds land in the right quadrant
Promote pooja room and kitchen from soft to hard once soft behaviour has been observed — a one-line change the rule loader already anticipates
Surface per-room compliance in the output: which rules were met, which were traded away, and why
Risk low
Blocks phase 04
Value highest
02
Program synthesis — the "thinking" layer
Track A
Today the program is decided from BHK statistics and then only scaled — never reconsidered. On 500 sqft it shrinks seven rooms until they hit NBC minimums, then refuses. On 2000 sqft it inflates the same seven by 2.5×. It never says "this plot wants a 2BHK" or "you have room for a study here."

A layer between step 3 and step 4 that reads plot area, aspect and shape and decides room count, which rooms, and target proportions
Scale regimes with genuinely different strategies: compact (<600 sqft), normal, spacious (>1800 sqft)
Raise the ~450 sqft floor by relaxing door-swing and circulation strategy at compact scale, rather than failing on DOR-001
Explain every change to the user in their terms — this is what makes it advice rather than rendering
Risk medium
Fixes both size extremes
03
Freeze reward v1
Gate
A checkpoint, not a build. Once Vastu rules and program synthesis are in, the reviewer plus critic is the training signal for everything downstream. Tag it, snapshot the harness baseline, and treat later reward changes as requiring a re-gate.

Deliverable tagged reward + baseline
04
Placer v3 — redesign, distil, reinforce
Track D
R1–R5 above. Runs on your GPU time, parallel to tracks B, C and F.

4a Model redesign in code — state encoder, new heads, Vastu/regime conditioning
4b Self-play corpus generator against frozen reward v1
4c Colab notebook: Drive checkpointing, atomic writes, full resume, frozen eval key
4d Stage a — imitation pretrain on CubiCasa
4e Stage b — distillation on Indian self-play data
4f Stage c — RL fine-tune against reward v1
4g Merge gate. Believe the result either way.
Risk high
Blocked by 01, 03
Parallel to B, C, F
05
The product loop — choose, regenerate, edit
Track B · parallel
Start immediately; nothing blocks it. The engine already produces six ranked candidates and the bridge throws five away.

Show K candidates. Mostly plumbing — and it starts filling critic/preferences.jsonl, which is built, live, and empty because the UI has never offered a choice
Regenerate. Roughly twenty lines: _seed_from(run_id) already makes a new run explore a different candidate set deterministically
Prompt edits. Re-run with added constraints — far easier than mutating a finished plan, and it covers most real requests
Manual edits. Expose lattice mutations with verify() after each; the substrate was designed for exactly this
Version tree. Editing implies history — needs phase 07 persistence to be durable
Risk medium
Side effect critic training data
06
Conversation as an agent, not a state machine
Track C · parallel
It feels dumb because the LLM only writes prose. Validation is a hardcoded three-field checklist; the question loop walks a static field list one item at a time, capped at eight turns; the model is gemini-2.0-flash with thinking disabled, chosen for free-tier quota. Nothing in the loop reasons about the brief.

Give a strong model the live BuildingRequirements state plus tools — set_field, check_feasibility, propose_program, run_pipeline — and let it decide what to ask
Wire it to phase 02, so it can say "three bedrooms on 20×45 will be tight — shall I put the third upstairs?"
Keep the deterministic extractor as the fallback it already is; keep the circuit breaker
Risk low
Pairs with phase 02
07
Platform and non-rectangular rooms
Tracks E + F
Two heavy items that belong after the product shape is settled.

Persistence. Sessions and run status are plain in-memory dicts; a restart loses every run. Needs a real store, accounts, and object storage for artifacts before anyone but you uses it
Non-rectangular rooms. A merge primitive or region-growing to replace pure guillotine splitting. Genuinely hard, touches every rule that calls face_is_rect, and forces a re-gate of the placer — which is exactly why it comes last, not first
Better irregular-plot use: today the building is only the largest inscribed rectangle, which is why the trapezoid brief scores 5.23
Risk high
Invalidates D's gate
Gates
Decisions that should be made by measurement
Do not
decide by
preference
Question	Decide by	Ship if
Is more pretraining worth it?	Was validation engine-score still climbing at epoch 24?	still climbing → continue; flat → skip to R2b
Does placer v3 replace the prior?	Existing merge gate, unchanged	mean improves, no brief −5, fidelity ≥ 0.80
Does distillation need more capacity?	Train/val gap on self-play corpus	underfit → grow to 25–30M; else hold
Do Vastu rules go hard?	Feasibility loss across the golden briefs	no brief loses its plan
Does the critic reflect taste?	Agreement against logged user picks	critic top-1 > soft-score top-1
Known risks
Self-play collapse. Distilling the engine's own output can amplify its blind spots. Mitigate by keeping CubiCasa pretraining in the mix, sampling across the full plot distribution rather than the easy middle, and holding out briefs the generator never saw.
Reward hacking in stage c. RL against a rule-based reward will find the rules' seams. Mitigate with the critic as a second opinion, a KL penalty against the distilled policy, and visual review of the top plans every run — the same discipline the engine already uses.
Vastu versus feasibility. On tight plots Vastu and NBC will genuinely conflict. The system must state the trade-off rather than silently pick — the enricher already has the audit pattern for this.
Non-rectangular rooms as scope creep. It is a real project with real value, and it is not a prerequisite for anything else in this document. Schedule it honestly or defer it honestly.