# PlanGen — Companion Presentation Deck

*16 slides for a 10–12 minute paper presentation plus Q&A. Each slide gives a title, terse bullets, a visual instruction, and speaker notes written as spoken sentences. Timing target: roughly 40 seconds per slide, with slides 5, 7, and 11 given extra weight.*

---

## Slide 1 — Title

**PlanGen: Neuro-Symbolic Residential Floor-Plan Generation**
*Constrained Discrete Seed-Cell Placement with Masked Decoding*

- [Author Name], [Department]
- [Institution], [City, Country]
- [Date]
- Paper track: generative geometry / neuro-symbolic systems

**Visual.** Full-bleed background showing one generated plan, rendered clean in SVG with room labels and wall linework, dimmed to 25% opacity behind the title block. No logo clutter.

**Speaker notes.** Good morning. This talk is about generating residential floor plans, and about a single reformulation that made the problem tractable for us. The short version is that we stopped asking a neural network to draw rooms, and started asking it only to decide where rooms begin. Everything else in the system follows from that one decision.

---

## Slide 2 — The Problem in One Image

- Input: a plain-language brief plus a plot
- Output: an architect-usable plan
- Walls, doors, circulation, code compliance
- Indian context: Vastu, NBC minimums, local programs

**Visual.** Two panels side by side with a bold arrow between them. Left: a hand-lettered brief, for example "40x30 plot, north entrance, 3 bedrooms, puja room, covered parking." Right: the finished SVG plan. Below the arrow, in small caps, the label "the contract."

**Speaker notes.** Here is the contract we are trying to satisfy. On the left, a brief that a homeowner could write. On the right, a drawing an architect could work from. Note the target context: Indian residential, which means Vastu orientation preferences, National Building Code minimums, and room programs that Western and Chinese trained systems simply do not encode.

---

## Slide 3 — Why This Is Hard

- A floor plan is a partition of a polygon
- Rooms must be disjoint and must exhaust the interior
- Not a preference; a definition
- Violate it and you have no plan at all

**Visual.** A polygon with rooms tiling it perfectly, labelled "partition." Beside it, the same polygon with two rooms overlapping in hatched red and a white gap in the corner, labelled "not a floor plan." Below both, the set notation for disjointness and union.

**Speaker notes.** The difficulty is structural. A floor plan is a partition: the rooms must be pairwise disjoint, and together they must exhaust the buildable interior. This is not a quality preference you can trade against aesthetics. A layout with overlapping rooms and a gap in the corner is not a poor floor plan. It is not a floor plan.

---

## Slide 4 — What We Tried First

- v1: GNN encoder plus autoregressive transformer
- 512-dim, 12 layers, Mixture-of-Gaussians heads
- 46 epochs on RPLAN, 80,000 plans
- Regressed continuous boxes: cx, cy, w, h

**Visual.** A clean block diagram of the v1 architecture, drawn confidently and at full contrast, with the Mixture-of-Gaussians output head highlighted. Everything looks correct. No failure indication anywhere on this slide.

**Speaker notes.** Our first system was the conventional answer. A graph encoder for the room program, an autoregressive transformer decoder, mixture density heads emitting continuous room boxes, trained for forty-six epochs on eighty thousand plans. Nothing here is exotic. This is what the literature would suggest, and it is what we built.

---

## Slide 5 — v1 Post-Mortem

- Wrong task formulation: about 60%
- Corrupted supervision: about 30%, 13.4% overlap in ground truth
- No boundary conditioning: structural
- Invisible failure: NLL fell to -33, no layout metric
- Data quantity: about 10%

**Visual.** The attributed-cause table, rendered large, one row revealed at a time. The 13.4% cell and the "-33" cell called out in a contrasting colour. Bottom banner in large type: "the failure was measured, not guessed."

**Speaker notes.** It did not work, so we instrumented it. Sixty percent of the failure was the task formulation itself. Thirty percent was the data: our own ground truth contained thirteen point four percent room-pair overlap, and the rooms did not tile their own plots. There was no polygon mask, so irregular plots were impossible. And critically, we only tracked negative log-likelihood, which fell smoothly to minus thirty-three while the output was garbage. The failure was measured, not guessed, and that is why we knew exactly what to change.

---

## Slide 6 — The Key Insight

- Repair operates after sampling, on a sampled point
- It cannot remove a defect the space admits
- Overlap is representational, not a capacity problem
- So change the representation, not the model size

**Visual.** Left: a sampled overlapping layout with arrows showing push-apart repair, and a new gap opening as a side effect, marked with a small red cross. Right: the same brief with the caption "the representation admits this configuration." Bottom line: "you cannot repair what the space permits."

**Speaker notes.** The tempting fix is post-hoc repair: push the rooms apart, snap them to walls. That fails, and the reason is worth stating precisely. Repair takes a sampled point and moves it, but the model's distribution still puts probability mass on invalid configurations, because the output space contains them. This is a representational limitation, not a capacity limitation. More parameters and more data move the wrong dial.

---

## Slide 7 — Reformulation

- Before: continuous boxes, four real numbers per room
- After: one of 1024 discrete seed cells
- Plus one of 40 area classes, 25 sq ft each
- Placement is learned; geometry is algorithmic

**Visual.** A before/after diagram. Left: a room drawn with four floating dimension arrows over a continuous plane, tinted grey. Right: a 32 by 32 grid with a single cell highlighted, plus a small strip of 40 size buckets with one selected. A dividing line between them labelled "the reformulation."

**Speaker notes.** So we changed what the model outputs. Instead of four continuous numbers per room, the model picks one cell out of a thirty-two by thirty-two grid, and one size class out of forty area buckets. That is the entire learned decision. Geometry is no longer the model's job. This decouples placement, which is learned, from geometry, which is algorithmic, and it is the crux of the contribution.

---

## Slide 8 — System Architecture

- Parse, Match, Enrich into a typed EngineRequest
- Propose, Realize, Settle, Connect, Validate, Repair, Rank
- Orchestrator runs best-of-K with bounded repair
- Only two tiers are learned; both have statistical twins

**Visual.** The six-tier pipeline diagram from Fig. 1, with the two learned tiers, Propose and Rank, in a distinct colour and every deterministic tier in neutral grey. Axiom in large type across the bottom: "ML decides, algorithms guarantee."

**Speaker notes.** Here is the whole system. Three front-end stages parse the brief and enrich the program into a typed contract. Then the orchestrator runs a best-of-K loop: propose, realize, settle, connect, validate, repair if needed, and rank. Notice the colouring. Only two of these tiers are learned. Every correctness property is produced by a deterministic tier. The axiom is: ML decides, algorithms guarantee.

---

## Slide 9 — The Ownership Grid

- Geometry lives on a 1.5-inch cell grid
- Every cell has exactly one owner
- Overlap is not checked; it is inexpressible
- Unowned cells make coverage countable

**Visual.** A zoomed grid where each cell is tinted with its owning room's colour, with the boundary between two rooms magnified in a callout to show single ownership cell by cell. Caption beneath: "one cell, one owner."

**Speaker notes.** Underneath the carver is a cell-ownership grid at inch-and-a-half resolution, and every cell has exactly one owner. This is the strongest guarantee in the paper. We do not check for overlap and we do not repair it, because two rooms cannot own the same cell. Coverage becomes arithmetic as well: unowned interior cells are simply countable, so incomplete coverage is detectable and localisable.

---

## Slide 10 — The Placement Model

- GATv2 program encoder: 3 layers, dim 128, 4 heads
- CNN boundary encoder: 2-channel 64x64, 64 tokens
- Causal decoder: 6 layers, d=256, 8 heads
- 7.36 M parameters, free-tier GPU trainable
- Two heads: 1024-way cell, 40-way size

**Visual.** An annotated block diagram: graph on the left feeding GATv2; plot raster on the left-bottom feeding the CNN; both plus a global token concatenating into a memory block; decoder on the right with two output heads drawn explicitly. Parameter count in a badge in the corner.

**Speaker notes.** The model itself is modest: seven point three six million parameters, sized so we can retrain it in a single donated GPU session. A GATv2 encoder over the room program, because the informative structure there is relational. A convolutional encoder over a two-channel raster of the plot, which is exactly the boundary conditioning v1 lacked. And a six-layer causal decoder that cross-attends to both, emitting one cell and one size class per room.

---

## Slide 11 — Masked Decoding

- Before each draw, set illegal cell logits to minus infinity
- Illegal: outside boundary, claimed, zone-forbidden
- Legality recomputed after every placement
- Sample at temperature over the legal remainder only

**Visual.** A 32 by 32 grid with three greyed-out regions annotated: outside the polygon, inside the claimed radius of two already-placed rooms, and a zone-forbidden band. The remaining legal cells glow, with one being sampled. Punchline in large type: "invalid layouts are unsamplable."

**Speaker notes.** This is the mechanism the paper is named for. Before we sample each room's cell, we set the logits to minus infinity for every cell that is outside the boundary, inside the claimed radius of an already-placed room, or forbidden by zone rules. We then sample at temperature over what remains. The consequence is the punchline: invalid layouts are not improbable, they are unsamplable, and that holds even for an undertrained checkpoint at high temperature.

---

## Slide 12 — Data Pipeline and the Five Gates

- CubiCasa5K: 4,989 plans, per-room polygons
- Five hard gates; a failure aborts with a named reason
- Yield 4,430 plans, 89%, all gates pass
- Raw pass had 5 collisions in 32,574 seeds

**Visual.** A funnel from 4,989 down to 4,430 with the five gates as labelled rings, each carrying its measured value. Off to the side, a small inset showing a seed being nudged one cell to a free position. Bottom banner: "fix the data, don't lower the bar."

**Speaker notes.** We prepared CubiCasa5K through five hard gates, where a failing gate aborts preparation with a named reason. There are no silent passes. Eighty-nine percent yield, all five gates green. The seed-collision gate is the one worth telling you about: a raw pass had five collisions out of thirty-two thousand five hundred seventy-four. We could have relaxed the gate. Instead we added a nudge step and reached a true zero. Fix the data, do not lower the bar. Corrupted supervision is thirty percent of why v1 failed.

---

## Slide 13 — Training That Measures the Right Thing

- v1's lesson: the loss measured the wrong thing
- Checkpoint selection by engine metrics, never by loss
- Proposals run through the real engine, 50 held-out boundaries
- Keep the checkpoint with the best mean soft score

**Visual.** Two training curves side by side. Left: a smooth descending NLL curve labelled "v1: looks like success." Right: the same descending curve overlaid with a flat or erratic engine-score curve, labelled "what we now measure." An arrow marks the selected checkpoint on the engine curve, not the loss curve.

**Speaker notes.** The methodological fix is here. In v1 the loss fell smoothly while the output was worthless, so now we never select on loss. After each evaluation epoch we take the model's proposals, run them through the real engine, realize, settle, connect, validate, on fifty held-out boundaries, and we keep the checkpoint with the best mean soft score. The training signal and the deliverable are now the same quantity.

---

## Slide 14 — The Merge Gate

- 1. Beats the statistical PriorProposer on mean best score
- 2. Regresses no single brief by more than 5 points
- 3. Proposal fidelity at least 0.8 after realization
- All three, or it does not ship

**Visual.** Three gate icons in a row, each with its criterion and a status chip currently reading "pending." Behind them, a branch diagram showing the trained model held on a side branch, not merged into main. Bottom line: "a model that loses here is the safety system working."

**Speaker notes.** Every learned tier has a statistical twin, and it ships only if it beats that twin on our harness. Three criteria: it must win on mean score, it must not regress any single brief by more than five points, and proposal fidelity must be at least zero point eight. That third one is a validity condition, not a quality target: if the realizer mutes the model, the A/B measures the realizer and tells us nothing. A model that loses this gate is the safety system working, not a project failure.

---

## Slide 15 — Results and Status

- Engine complete and tested; baseline harness mean about 75.5
- Placer implemented, 7.36 M params, NumPy path verified to 7e-7
- 114 of 114 tests pass; training in progress on Colab
- Not shipped: tuned weights, learned critic, the placer itself
- Weight tuning: 18 briefs cannot identify 20 parameters, 28% cap

**Visual.** A two-column status board. Left column headed "Shipping," listing the algorithmic engine and its baseline. Right column headed "Not shipping, and why," listing the tuned weights with "data-sufficiency ceiling," the critic with "not built," and the placer with "merge gate pending." Honest, unembellished typography.

**Speaker notes.** Here is the honest state. The algorithmic engine is complete, tested, and scores about seventy-five point five on the harness. The placer is fully implemented, its torch-free NumPy path agrees with PyTorch to seven times ten to the minus seven, and a hundred and fourteen of a hundred and fourteen tests pass. Training is running now. And one result is plainly negative: eighteen labelled briefs cannot identify twenty free weights, agreement caps at twenty-eight percent, so we did not ship the tuned weights. The deliverable there is the infrastructure, not the numbers.

---

## Slide 16 — Conclusion and Future Work

- Do not train a model to stop violating a constraint
- Build a representation where violation is inexpressible
- Next: Indian fine-tune, irregular rooms, learned critic
- Multi-floor realization; contracts already carry floor context

**Visual.** The central lesson set in large type, occupying the top half of the slide alone. Lower half: a compact roadmap strip with four milestones, the first one, "Indian fine-tune," highlighted as next.

**Speaker notes.** To close. v1 did not fail because it was too small or because eighty thousand plans were too few. It failed because freely regressed rectangles cannot express a partition. So the lesson we would offer is this: in domains with hard structural constraints, the productive question is not how to train a model to stop violating the constraint, but how to build a representation in which violation is inexpressible. Next up: an Indian fine-tune, irregular rooms, and a learned critic. Thank you.

---

# 90-Second Elevator Version

*Spoken prose, roughly 230 words.*

Floor-plan generation looks like an image or sequence generation problem, and almost everyone treats it that way: condition a network on a room program, regress a box per room. We built exactly that system first. It failed, and we instrumented the failure rather than guessing at it. Sixty percent of the cause was the task formulation itself, because a floor plan is a partition of a polygon, and a set of freely regressed rectangles cannot represent a partition. Overlap there is a property of the output space, not of the fit, which means no amount of training removes it and no amount of post-hoc repair fixes it.

So we changed the representation. Our model no longer draws rooms. For each room, it picks one discrete cell out of a thirty-two by thirty-two grid, and one size class out of forty area buckets. Before every draw, we mask out every cell that is outside the plot, already claimed, or forbidden by zone rules, so invalid layouts are not improbable, they are unsamplable. A deterministic carver on a single-owner cell grid then produces the exact geometry, which makes overlap inexpressible.

Two principles govern the whole engine. ML decides, algorithms guarantee. And every learned tier has a statistical twin it must beat on our quality harness before it ships. That second principle has already stopped us from shipping one component, and it may yet stop us from shipping the model itself.

---

# Q&A Preparation Annex

**1. Why not just use a diffusion model?**
A diffusion model over room boxes or over a layout raster has the same structural property as our v1 system: its sample space contains invalid partitions, so validity is at best highly probable. Guidance and classifier-free conditioning shape the density but do not remove support. Our objection is not to a particular generative family; it is to any formulation in which an invalid configuration is representable at sampling time. A diffusion model could plausibly replace our decoder as a proposal distribution over discrete seed cells, and the masking and the carver would still supply the guarantee.

**2. Isn't masking just hand-engineering?**
It is, deliberately, and the paper's position is that this is where hand-engineering belongs. The mask encodes what is geometrically or statutorily illegal, which is knowledge we possess exactly and do not need to learn approximately. What we do not hand-engineer is which legal cell is a good cell, and that is precisely the judgment the model supplies. The closest antecedent is grammar-constrained decoding for language models, where the grammar is hand-written and nobody regards the language model as thereby diminished.

**3. Why train on Finnish data for Indian homes?**
Because it is the corpus with per-room polygons, an open licence, and the geometric fidelity our seed extraction requires. We do not claim the domain transfers; we say so in the paper, we quantify what differs, and the corpus literally contains sauna rooms. The plan is to learn placement regularities that are largely geometric, meaning circulation-centred organisation and plausible adjacency, then fine-tune on Indian data with owned Indian program priors. The merge gate exists so that if the transfer fails, we find out by measurement rather than by shipping it.

**4. What if the model never beats the baseline?**
Then the model does not ship, and the paper still stands. Three things survive that outcome independently of the placer: the reformulation and its guarantee, which are properties of the representation and not of any trained weights; the algorithmic engine, which produces valid plans at a harness mean of about seventy-five point five today; and the protocol itself. We would then treat it as a domain-gap result, run the Indian fine-tune, and re-run the gate. A gate that never rejects anything is not a gate.

**5. How do you know the soft score reflects real design quality?**
We do not, fully, and that is the honest answer. It is a hand-weighted proxy. We attempted to validate and correct it against human preference in Phase A, and the study returned a negative result: eighteen labelled briefs cannot identify twenty free weights, and top-1 agreement caps at twenty-eight percent. So we did not ship the tuned weights. The correct remediation is more preference labels and a learned critic that scales with them, which is Tier 5. Meanwhile, the hard metrics, overlap, coverage, and code compliance, do not depend on the soft score at all.

**6. Why 32 by 32 and not finer?**
Because the seed grid is a placement vocabulary, not the geometry resolution. Geometry lives on a separate inch-and-a-half ownership grid, so a coarse seed grid costs no precision in the output. A thousand and twenty-four classes is already a large softmax for the data we have, and finer discretisation would fragment supervision across near-identical cells while adding nothing the carver cannot recover. If we later find that placement decisions are resolution-limited rather than data-limited, the grid is a single constant.

**7. Isn't the carver doing all the real work?**
The carver does all the geometric work, and that is the design, not an accident. But the carver is a deterministic function of the seeds and size classes it receives; give it poor seeds and it produces a valid, badly organised home. The learned component decides which of roughly a thousand legal positions each room takes, and that decision determines circulation, zoning, and daylight. Our merge gate measures exactly this claim by comparing the learned proposer against a statistical proposer with the same carver behind both.

**8. How does this scale to multi-floor?**
The contracts have carried floor context since the first version, before any multi-floor realization existed, specifically so that the field would not have to be retrofitted. The program encoder already embeds a floor index per node. What is not built is the realization logic for cross-floor coupling, meaning stair placement, wet-stack alignment, and structural continuity between floors. Those are hard constraints across floors, and our expectation is that they enter as additional mask terms on the seed distribution, which is the same mechanism used within a floor.

**9. What is your actual novelty claim?**
Not the transformer, not graph attention, and not masked decoding in the abstract, all of which are established. The claim is the formulation: that residential layout generation should be posed as constrained discrete seed placement with algorithmic geometry, that masking over a spatial vocabulary recomputed from evolving occupancy yields a hard partition guarantee, and that this is supported by a measured post-mortem of the alternative rather than by assertion. Secondarily, we claim the deployment protocol, meaning engine-metric checkpoint selection and the statistical-twin merge gate, as a reusable methodology.

**10. What would falsify your central claim?**
Two results would. First, an end-to-end continuous-regression system that demonstrably attains exact zero overlap and complete coverage across a diverse brief distribution without a post-hoc projection step; that would show the representational argument is not binding in practice. Second, and more accessible, an ablation in which our own masking is disabled and the resulting layouts show no meaningful degradation in validity; that would indicate the guarantee was being supplied by the carver and the data distribution rather than by the mask. We regard the second as a worthwhile experiment to run and report.
