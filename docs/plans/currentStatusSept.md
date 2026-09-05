PlanGen — Status Report

Verified today against the working tree (259/260 tests green, live engine run, diagnostics, harness + gate baselines).

The short version

Your memory is correct: the request path is 100% algorithmic today. The transformer is trained but sitting on the bench — it failed its merge gate on 30 July and was never wired in. Of your five expectations, one is genuinely done, two are partly done, and two don't exist yet.

What you expected	Status
Multi-floor	✅ Done, and it's the strongest part
Plan generator for Indian builders	🟡 Engine excellent, "Indian" is thinner than it looks
Vastu as a highlight feature	🔴 Computed, then discarded before the geometry
Prompt edits / manual edits to refine	❌ Not built. Zero endpoints.
Algorithm + transformer integrated	🟡 Integration is one line — the model isn't good enough yet
1. Vastu — the finding that matters most

This is not what you think it is, and it's the biggest gap against your stated goal.

Vastu loads and runs: 22 room rules from enricher_rules.json, the VastuMapper maps compass zones to plot-relative zones, the enricher assigns every room a preferred_direction (SW, NE…) and preferred_zone.

Then api/engine_bridge.py:140 throws it away:

engine_type, zone = _TYPE_MAP.get(rtype, (rtype, "private"))
specs.append(RoomSpec(name=name, rtype=engine_type,
                      target_sqft=..., zone=zone))

zone there is a hardcoded public/service/private lookup — not the Vastu zone. room.preferred_direction, room.preferred_zone and room.vastu are never read. And RoomSpec has no direction field to put them in:

RoomSpec : ['name', 'rtype', 'target_sqft', 'zone', 'floor']

A grep for vastu across all of modules/step4_generate/ returns 0 matches. None of the 33 reviewer rules mention a compass direction. The engine's only orientation input is entrance_side.

What Vastu actually does today: floor assignment (ground_floor_only / top_floor_preferred), conflict warnings, and a vastu_direction_assignments block in the output JSON. On a single-floor plan, _assign_floors returns early and Vastu changes nothing about where rooms go. On multi-floor it can only change which floor a room lands on — never its position on that floor.

Also by deliberate design, every rule is emitted at "medium" priority = soft (rule_loader.py:330) — the docstring says promoting pooja_room/kitchen to "high" is a one-line change, once soft behaviour has been observed. It never was, because it never reached the geometry.

The good news: this is very tractable and needs no ML. Three changes — add direction/vastu_zone to RoomSpec, have the bridge pass them, add a VAS-* soft rule family plus a directional bias in PriorProposer._depth_base — and Vastu becomes real. The rule registry and soft-score machinery are already built to absorb exactly this.

2. Editing — not started

13 API endpoints; none of them edit anything. No regenerate, revise, modify, or apply_edit function exists anywhere in api/, modules/, or ml/. The frontend POSTs exactly twice: /parse/text and /pipeline/run. One brief in, one plan out, done.

Two things make this much cheaper than it sounds:

Regenerate is nearly free. _seed_from(run_id) already means a new run_id explores a different candidate set — deterministic per run, different across runs. That's a ~20-line endpoint.
The engine already produces 6 ranked candidates and the bridge keeps only the best. Showing the user all 6 is mostly plumbing.

And there's a compounding win hiding there: critic/preferences.py is fully built to log which candidate a user picks, and critic.train already reports agreement against it. The log is live and empty — because the UI has never offered a choice. Building "here are 6, pick one" delivers your feature and starts generating the only training signal that teaches the critic actual taste rather than damage-detection.

Prompt-based edits and manual (drag-a-wall) edits are real work on top of that — manual editing especially, since it means exposing lattice-level mutations with verify() after each one. But the substrate is built for it: every mutation in GridPlan is already followed by an invariant check, so edits can't silently corrupt a plan.

3. Multi-floor — done properly

This one fully meets your expectation. Floors are planned bottom-up, each conditioned on the one below. The staircase is a reservation (an exact cell rect isolated before any room is placed) and is frozen in settle — the memory notes both were necessary: seeding alone measured 0–1% footprint overlap, reservation without freezing drifted to 23–59%. Five vertical rules (VRT-001…005) reject a floor that doesn't stack. Measured at the M7 gate: stair overlap 100%, wall alignment 100%. Upper floors correctly have no front door — you arrive on the staircase.

4. Algorithm + transformer — the integration isn't the problem

The architecture you want to build already exists. Orchestrator(proposer=...) takes the proposer as a constructor argument, and Tier2Placer is a drop-in implementing the identical propose(request, variant) contract, with confidence-gated fallback to PriorProposer. Wiring the transformer in is one line.

What's missing is a model that earns the slot. The gate ran 30 July:

VERDICT: FAILED
  mean best score improves      FAIL   75.49 -> 68.57  (-6.92)
  no brief regresses > 5 pts    FAIL   9 of 18 briefs regressed
                                       worst: 30x60_N_3bhk -32.54
                                              20x45_S_3bhk -25.68
                                              30x50_S_3bhk -25.68
  mean fidelity >= 0.8          pass   0.843
  no brief loses its plan       pass   18 -> 18
  fallback_rate 0.418           (fell back on 42% of proposals)

Two briefs improved (one by +24.4). Nine regressed. The docs say re-run at 60–70 epochs; this was ~24.

Before you buy more GPU time, though — the model was trained on 4,430 CubiCasa5K plans, which are Finnish apartments. Your own audit noted it: sauna rooms, 22.9% of rooms unlabelled and dropped. You'd be training a network to propose Finnish apartment layouts and then asking it to beat a proposer whose priors are hand-tuned to Indian residential convention. The prior proposer isn't winning by luck — it encodes hub-first circulation, wet-room affinity and zone depths that the corpus doesn't contain.

That's a strategic call only you can make, but "train it longer" is likely the weakest of your available moves right now.

5. The chat agent — you're right, and here's the structural reason

It isn't that Gemini is dumb. It's that the LLM is only writing prose; it isn't doing any of the reasoning.

Validation is a hardcoded if/else checklist of three Tier-1 fields (parser.py:149). The LLM is handed the verdict and asked to phrase it nicely.
The question loop walks a static field list, one field at a time, capped at 8 turns (interactive_gatherer.py).
The fast path bypasses the LLM entirely for short answers — good for quota, but it means most turns have no intelligence in them at all.
Model is gemini-2.0-flash with thinking disabled, chosen for free-tier quota, not quality.
A 600s circuit breaker drops everything to static replies on the first quota error.

So the system is a rigid state machine wearing an LLM as a voice. It can never ask a good architectural follow-up ("you've got 3 bedrooms on a 20×45 — that's tight, want the third upstairs?") because nothing in the loop is reasoning about the brief.

The fix is to invert it: give a strong model the current BuildingRequirements state plus tools (set_field, check_feasibility, run_pipeline) and let it decide what's missing and what's worth asking. Keep the deterministic extractor as the fallback it already is.

Other things worth knowing
Diagnostics: DEGRADED, one warning — data/nbc_plot_regulations.json is missing, so FAR/coverage falls back to enricher_rules.json defaults.
One failing test — tests/ml/test_training_prep can't import ijson. Training-only dep, not a code fault. pip install ijson clears it.
README is stale — it describes the frontend as Vite + React + TS + Tailwind + R3F with a dist/ build. It's plain HTML/CSS/JS now (commit 5d4d553).
Harness baseline is from 30 July and predates the Aug-7 consolidation. Mean 69.5 across 28 briefs, all 28 produce a plan. The weak tail is irregular plots: trapezoid_45x40 scores 5.23, chamfer_35x45 17.65.
The learned critic is genuinely good — held-out AUC 0.912, ranks a clean plan above a damaged sibling 91.4% of the time vs 63.2% for rules alone. It's live and ranking at critic_weight=0.4.
CP-SAT is off by measurement, not caution: 73.22 off vs 70.80 always, with one brief losing its plan entirely.
What I'd do, in order
Vastu into the geometry. Highest value against your headline feature, no ML, no new data. Days, not weeks.
Show 6 candidates + regenerate. Cheap, immediately makes the product feel responsive to dissatisfaction, and starts filling the preference log that the critic needs.
Rebuild the conversation as a tool-calling agent. Fixes the thing that most shapes a user's first impression.
Then editing proper — prompt edits first (re-run with constraints is much easier than mutating a finished plan), manual editing after.
Revisit the transformer last, and when you do, ask whether the corpus problem is worth solving before the epoch problem.

The engine underneath all this is in good shape — the partition-first substrate makes overlaps and gaps unrepresentable, and it's carrying real geometry with stairs, DXF, irregular plots and vertical stacking. The gap is almost entirely above it, in what reaches the engine and what the user can do after it answers.