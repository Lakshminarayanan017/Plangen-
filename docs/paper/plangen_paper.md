# PlanGen: Neuro-Symbolic Residential Floor-Plan Generation via Constrained Discrete Seed-Cell Placement

### Masked Decoding with Guaranteed-Partition Carving

**[Author Name]**
*[Department]*
*[Institution]*
[City, Country]
[email]

---

## Abstract

Residential floor-plan design is a labour-intensive, expertise-gated stage of housing delivery, and automating it would compress a design loop that is repeated for every plot, every client, and every revision. The prevailing paradigm, end-to-end neural generation that regresses continuous room boxes, cannot satisfy the defining property of the artefact it produces: a floor plan is a partition of a polygon, and an unconstrained product of continuous box distributions places non-zero probability mass on overlapping and non-covering configurations. This is a representational limitation rather than a capacity or data limitation, and post-hoc geometric repair cannot remove a defect that the output space admits. We reformulate the task from continuous regression to constrained discrete classification: for each room in a canonical order, the model predicts a seed cell on a boundary-clipped 32 x 32 grid and one of 40 discrete area classes, while geometry is delegated to a deterministic carver. We present a six-tier neuro-symbolic pipeline in which a graph attention network v2 (GATv2) program encoder, a convolutional neural network (CNN) boundary encoder, and a causal transformer decoder with masked decoding propose placements, and algorithmic tiers realize, settle, connect, validate, and rank them. We report a quantified post-mortem of a prior end-to-end system, including 13.4% room-pair overlap in its own supervision; a five-gate data pipeline yielding 4,430 of 4,989 plans with a zero seed-collision rate over 32,574 seeds; a torch-free NumPy inference path agreeing with the PyTorch reference to 7e-7; a statistical baseline harness mean soft score of approximately 75.5; and a merge-gate protocol that has so far withheld one tuned component from deployment. The transferable principles are that machine learning decides while algorithms guarantee, and that every learned tier must outperform a statistical twin before it ships.

**Index Terms** — architectural layout synthesis, constrained generation, graph attention networks, masked autoregressive decoding, neuro-symbolic systems, spatial reasoning, transformer models

---

## I. INTRODUCTION

This section establishes the economic setting of residential layout synthesis, states the structural flaw in the prevailing generative paradigm, presents the measured failure that motivated our redesign, and enumerates our contributions.

Residential layout design sits on the critical path of housing delivery. A plot, a budget, and a family's room program must be converted into a partition of buildable area that satisfies statutory minimums, admits daylight and ventilation, keeps circulation short, and reads as a coherent home rather than a packing of boxes. The work is expertise-gated and iterative: each revision to the brief, the plot, or the budget propagates through the entire layout. Automated synthesis is therefore economically attractive, and the attraction grows in markets where the volume of small individual builds far exceeds the supply of architects available to serve them.

The Indian residential context is such a market, and it is underserved by existing systems. Layouts there are shaped by Vastu orientation preferences that constrain which functions may occupy which compass sectors, by National Building Code (NBC) minimum room dimensions that differ from Western and Chinese equivalents, and by room programs with locally specific elements such as a puja space, a separate wet utility area, and covered parking integrated into the plot rather than a detached garage. Systems trained on Western or Chinese corpora encode neither the code constraints nor the program priors of this domain. The gap is not merely one of aesthetics; a layout that violates an NBC minimum is not a stylistic mismatch but an unbuildable drawing.

The dominant computational approach to this problem is end-to-end neural generation, in which a network conditioned on a room program and a plot emits, for each room, a continuous box $(c_x, c_y, w, h)$, typically through a regression head or a mixture density head. The approach is appealing because it is uniform and differentiable end to end. It is also structurally unable to produce the artefact it targets. A floor plan is a partition: the room regions must be pairwise disjoint and must jointly exhaust the buildable interior. A set of freely regressed rectangles is a subset of the plane with no such property. Overlap in such systems is not an artefact of insufficient training, an under-parameterised model, or a small corpus; it is a property of the output space, which contains overlapping configurations and assigns them positive probability. It follows that post-hoc geometric repair, whether push-apart relaxation, wall snapping, or non-maximum suppression over boxes, cannot resolve the defect. Repair operates on samples after the fact and can only trade one violation for another, because the representation itself never encoded the constraint that repair is attempting to restore.

We did not adopt this position a priori. We adopted it after instrumenting the failure of our own first system. Version 1 (v1) of PlanGen was a conventional graph neural network (GNN) encoder coupled to an autoregressive transformer decoder, with a model dimension of 512, 12 layers, and Mixture-of-Gaussians (MoG) output heads over continuous box parameters, trained for 46 epochs on the RPLAN corpus. It failed to produce usable layouts, and the post-mortem attributed the failure to four causes with measured evidence, summarised in Table I. Approximately 60% of the failure was attributed to the wrong task formulation, that is, to continuous box regression. Approximately 30% was attributed to corrupted supervision: the bounding-box training cache contained 13.4% room-pair overlap in the ground truth itself, and the ratio of summed room area to plot area had a p10 to p90 range of 0.79 to 1.12, meaning the reference layouts did not tile their own plots. The absence of boundary conditioning was structural rather than fractional: the global token carried only plot width and height, with no polygon mask, so irregular plots were not representable at all. Finally, the failure was methodologically invisible: only MoG negative log-likelihood (NLL) was tracked, it fell smoothly to -33, and because no layout-quality metric was computed, degenerate output was indistinguishable from progress. Each of these findings determined a specific element of the redesign, and we present the post-mortem as this paper's motivating experiment rather than as an appendix.

Our approach follows directly from that diagnosis. We reformulate placement as constrained discrete classification. For each room, in a canonical generation order, the model predicts a seed cell drawn from a boundary-clipped $32 \times 32 = 1024$ cell grid and a size class drawn from 40 area buckets of 25 sq ft each, with the top bucket absorbing areas of 1000 sq ft and above. Before sampling at each step, the cell logits are masked to $-\infty$ wherever placement would be geometrically or semantically illegal, so the support of the sampler contains no invalid configuration. Geometry is then produced by a deterministic hub-first carver operating on a cell-ownership grid in which each cell has exactly one owner, which makes overlap structurally impossible. Around this core we build a six-tier pipeline that parses briefs, retrieves statistical priors, enriches programs with adjacency, Vastu, and NBC requirements, proposes placements, realizes and settles geometry, connects circulation, validates against a rules catalog, repairs within a bounded number of rounds, and ranks candidates. Two principles govern the design: machine learning decides, algorithms guarantee; and every learned tier has a statistical twin that it must outperform before it ships.

Our contributions are the following.

- **We reformulate residential floor-plan generation from continuous box regression to constrained discrete seed-cell classification**, decoupling learned placement from algorithmic geometry, and we show that this reformulation, rather than additional capacity or data, is what removes the overlap and coverage failure modes measured in our v1 system (Section IV).
- **We introduce masked decoding over the seed-cell distribution as a structural correctness guarantee**, in which cells outside the boundary, inside the claimed radius of a placed room, or forbidden by zone and adjacency rules are removed from the sampler's support, so that invalid layouts are not improbable but unsamplable, and we pair it with a confidence threshold that falls back to a statistical proposer (Section VI).
- **We present a gate-driven data pipeline** that converts 4,989 CubiCasa5K plans into 4,430 prepared training records at an 89% yield under five hard gates that abort preparation with a named reason, and we document the decision to eliminate 5 residual seed collisions out of 32,574 by correcting the data rather than relaxing the gate (Section VII).
- **We establish engine-metric checkpoint selection and a three-criterion merge gate as a deployment protocol**, under which checkpoints are chosen by mean soft score obtained by running proposals through the real engine on held-out boundaries, and a learned tier is wired into production only if it beats its statistical twin, regresses no brief by more than 5 points, and preserves proposal fidelity of at least 0.8 (Sections VII and VIII).
- **We report a candid negative result on soft-score weight identifiability**: soft score is exactly linear in its weights, which reduces tuning to convex pairwise learning-to-rank, but 18 labelled briefs cannot identify 20 free parameters, top-1 cross-validated human-pick agreement caps at 28%, and the tuned weights were consequently not shipped (Section VIII).

The remainder of the paper is organised as follows. Section II positions the work against learning-based, constraint-conditioned, procedural, and neuro-symbolic prior art; Section III formalises the problem and the reformulation; Section IV describes the six-tier system; Section V details the learned placement model; Section VI covers the data pipeline and training methodology; Section VII reports evaluation, results, and current status; Section VIII discusses engineering and reproducibility; and Sections IX and X give limitations, future work, and conclusions.

---

## II. RELATED WORK

This section positions PlanGen against four bodies of work and states, for each, the specific respect in which our formulation differs.

### A. Learning-Based Floor-Plan Synthesis

A substantial line of work generates floor plans directly with neural models, either as rasters over a room-label image or as vector primitives emitted by a sequence model [CITATION NEEDED: raster-based floor-plan generation with generative adversarial or diffusion models], [CITATION NEEDED: sequence models emitting vector room primitives]. Raster formulations obtain coverage almost by construction, since every pixel receives a label, but they surrender the crisp vector geometry that downstream architectural use requires, and label noise at room boundaries becomes wall ambiguity. Vector formulations recover geometry but reintroduce the partition problem, because nothing in a set of emitted boxes enforces disjointness. The recurring failure signature reported across this literature, and reproduced in our own measured v1 post-mortem, is a combination of residual overlap and incomplete coverage that persists after training converges [CITATION NEEDED: reported overlap or coverage metrics in learned layout generation]. Unlike these approaches, PlanGen never asks a learned component to emit geometry at all; the network emits a discrete seed and a size class, and the partition is constructed algorithmically from them.

### B. Graph-Conditioned and Constraint-Conditioned Layout Generation

A second line conditions generation on a bubble diagram or adjacency graph, so that the desired topology of the home is supplied explicitly and the model is left to realise it geometrically [CITATION NEEDED: graph-conditioned floor-plan generation]. Related work in general 2D layout synthesis conditions on relational or alignment constraints and penalises violations in the loss [CITATION NEEDED: constraint-conditioned 2D layout generation with differentiable penalties]. Conditioning of this kind is soft: a violated adjacency raises the loss but remains in the model's support, and at sampling time nothing prevents it. Soft conditioning is therefore appropriate for preferences and inappropriate for statutory minimums or for disjointness, which admit no gradient trade-off. Unlike these approaches, PlanGen partitions its requirements explicitly into hard constraints, which are enforced by construction through masking and through the ownership grid, and soft objectives, which are scored and used only for ranking.

### C. Procedural, Optimisation-Based, and Space-Partitioning Methods

A third line predates the learned methods and remains the source of the strongest guarantees. Squarified treemaps, guillotine and slicing-tree partitions, and rectangular dualisation produce exact partitions of a rectangle by construction, and simulated annealing or other stochastic search can optimise such partitions against weighted objectives [CITATION NEEDED: squarified treemap layout], [CITATION NEEDED: guillotine or slicing-tree floor-plan partitioning], [CITATION NEEDED: simulated annealing for architectural layout optimisation]. These methods guarantee what learned methods struggle to guarantee, but they carry no design judgment: the objective function must encode every preference a designer holds implicitly, and hand-weighted objectives generalise poorly across briefs. Unlike these approaches, PlanGen retains an exact partitioning algorithm as the geometry producer but replaces hand-tuned search over placements with a learned proposal distribution, so that design judgment is acquired from data while the guarantee remains algorithmic.

### D. Neuro-Symbolic Systems and Constrained Decoding

The closest methodological antecedent to our central mechanism comes from language and program synthesis rather than from architecture. Grammar-constrained and mask-based decoding restricts a language model's next-token distribution to the tokens permitted by a formal grammar, a type system, or a schema, so that generated output is syntactically valid by construction rather than valid with high probability [CITATION NEEDED: grammar-constrained decoding for language models], [CITATION NEEDED: constrained decoding for semantic parsing or program synthesis]. Broader neuro-symbolic systems similarly delegate satisfaction of hard structure to a symbolic component while retaining a learned scoring or proposal component [CITATION NEEDED: neuro-symbolic architectures combining learned proposal with symbolic verification]. Unlike these approaches, which mask over a discrete token vocabulary defined by a grammar, PlanGen masks over a discrete spatial vocabulary defined by geometry, where legality is recomputed after every placement from the evolving occupancy state, and the resulting guarantee is geometric disjointness rather than syntactic well-formedness.

---

## III. PROBLEM FORMULATION

This section fixes the notation used for the remainder of the paper, states the partition constraint formally, shows why unconstrained regression cannot satisfy it, and defines the reformulated learning task.

### A. Inputs and Outputs

An instance is a tuple $(P, R, e, C)$. The plot polygon $P \subset \mathbb{R}^2$ is a simple polygon, obtained either from textual dimensions such as `40x30`, in which case it is a rectangle, or from an image of the plot boundary, in which case it is an arbitrary simple polygon. The room program is an ordered set $R = \{r_1, \dots, r_n\}$ where each $r_i = (\tau_i, a_i, z_i)$ carries a room type $\tau_i$ drawn from the engine vocabulary, a target area $a_i$, and a zone label $z_i$ such as public, private, or service. The entrance side $e$ identifies the polygon edge at which the plot is approached. The context $C$ carries multi-floor flags, Vastu orientation preferences, and the applicable NBC constraints.

The output is a floor plan $F = (\Omega, W, D, \Gamma)$, in which $\Omega = \{\Omega_1, \dots, \Omega_n\}$ assigns a region to each room, $W$ is the wall set, $D$ is the set of doors and openings, and $\Gamma$ is the circulation structure. The regions must form a partition of the buildable interior:

$$
\Omega_i \cap \Omega_j = \emptyset \quad \forall i \neq j,
\qquad
\bigcup_{i=1}^{n} \Omega_i = \operatorname{int}(P).
\tag{1}
$$

Equation (1) is the defining property of the artefact. A candidate that violates either clause is not a low-quality floor plan; it is not a floor plan.

### B. Why Regression Fails

Consider the prevailing formulation, in which a model defines a distribution over box parameters $b_i = (c_{x,i}, c_{y,i}, w_i, h_i)$ and factorises the layout as $p(b_1, \dots, b_n \mid P, R) = \prod_i p(b_i \mid b_{<i}, P, R)$ with each factor supported on $\mathbb{R}^4$ or on a continuous subset of it, as with a mixture density head. Let $\mathcal{V} \subset \mathbb{R}^{4n}$ denote the set of configurations violating (1). Because $\mathcal{V}$ has positive Lebesgue measure in $\mathbb{R}^{4n}$, and because each factor is absolutely continuous with support covering a neighbourhood of any attainable configuration, it follows that $p(\mathcal{V}) > 0$ for every setting of the parameters. Training can reduce $p(\mathcal{V})$; no amount of training can make it zero, because the constraint is not expressible in the output space. Two consequences follow. First, overlap is a representational limitation rather than a capacity limitation, so scaling the model or the corpus addresses the wrong axis. Second, post-hoc geometric repair is a palliative: it maps a sampled point in $\mathcal{V}$ to some nearby point outside it, but the mapping is not the distribution the model learned, so repaired samples carry neither the model's design judgment nor a coverage guarantee. Our v1 measurements are consistent with this analysis, and are aggravated by the fact that the supervision itself lay partly in $\mathcal{V}$, with 13.4% room-pair overlap in the ground-truth cache.

### C. The Reformulated Learning Task

We restrict the learned component to a finite decision space. Let $G$ be a $32 \times 32$ grid over the axis-aligned bounding box of $P$, clipped to the interior of $P$, and index its cells $1, \dots, 1024$. Let $S = \{1, \dots, 40\}$ index area buckets of 25 sq ft each, with bucket 40 absorbing all areas of 1000 sq ft and above. For each room $r_t$ in a canonical generation order $t = 1, \dots, n$, the model predicts

$$
c_t \in \{1, \dots, 1024\},
\qquad
s_t \in S,
\tag{2}
$$

where $c_t$ is a seed cell and $s_t$ is a size class. The joint model factorises autoregressively as $p(c_{1:n}, s_{1:n} \mid P, R, e, C) = \prod_t p(c_t, s_t \mid c_{<t}, s_{<t}, P, R, e, C)$, and each factor is a pair of categorical distributions rather than a continuous density. A deterministic carver $\mathcal{C}$ then maps the seed and size sequence to geometry, $\Omega = \mathcal{C}(c_{1:n}, s_{1:n}, P)$, with (1) satisfied by construction. The decoupling thesis is therefore explicit: *placement is learned; geometry is algorithmic.*

### D. Hard Constraints Satisfied by Construction

The following are enforced structurally rather than through loss penalties, and a candidate violating any of them is rejected regardless of its soft score: pairwise non-overlap of room regions; full coverage of the buildable interior; NBC minimum room dimensions; and reachability of every room from the entrance through the circulation structure $\Gamma$. Preferences that admit trade-offs, including Vastu orientation alignment, adjacency satisfaction, aspect-ratio quality, and daylight exposure, are treated as soft objectives and enter only through the ranking score defined in Section IV-C.

---

## IV. SYSTEM ARCHITECTURE

This section presents the six-tier pipeline, the orchestrator that drives it, the ownership grid that carries the central guarantee, and the rules catalog that defines acceptance. Fig. 1 gives the overall structure.

```
                         Brief (+ optional plot image)
                                      |
                                      v
   Step 1  PARSE      natural language / image -> RoomProgram      [LLM + rules]
   Step 2  MATCH      priors from ~5K extracted plans              [statistical]
   Step 3  ENRICH     adjacency graph, Vastu, NBC minimums         [rules]
                                      |
                                      v
                        EngineRequest  (typed contract)
                                      |
                                      v
        ORCHESTRATOR  -- per candidate, repeated K times:
          |
          +-- T2  PROPOSE    seed cells + size classes   [ML placer <-> prior fallback]
          +-- T3  REALIZE    hub-first carver -> geometry            [algorithmic]
          +-- T4  SETTLE     squeeze-and-settle area calibration     [algorithmic]
          +--     CONNECT    openness gradient + reachability        [algorithmic]
          +--     VALIDATE   hard-rule flags + soft score            [rules]
          +--    [REPAIR -> re-validate]  (<= 2-3 rounds)            [algorithmic]
          +--     RANK       critic or soft score        [ML critic <-> soft score]
                                      |
                                      v
                              Renderer (SVG)
```

**Fig. 1.** The PlanGen pipeline. Tiers marked `[algorithmic]` or `[rules]` are deterministic and carry the correctness guarantees; only PROPOSE and RANK are learned, and each has a statistical twin behind a typed contract, so substituting a trained model for its fallback is a single constructor argument. The `EngineRequest` boundary separates brief interpretation from layout synthesis.

### A. Stages

**Parse (Step 1).** Input: a natural-language brief and optionally an image of the plot boundary. Mechanism: a large language model extracts entities, and rule-based normalisation resolves dimensions, counts, and room names against the engine vocabulary; an interactive gatherer requests missing fields. Output: a structured `RoomProgram` and a plot polygon. Learned in part, through the language model; the normalisation is algorithmic.

**Match (Step 2).** Input: the parsed program. Mechanism: retrieval of statistical priors, including type frequencies, area distributions, and adjacency tendencies, from approximately 5,000 extracted plans. Output: prior-augmented program fields, including target areas where the brief left them unspecified. Statistical.

**Enrich (Step 3).** Input: the prior-augmented program. Mechanism: construction of the desired adjacency graph, application of Vastu orientation preferences to zone assignments, and imposition of NBC minimum dimensions per room type. Output: an `EngineRequest`, the typed contract on which the orchestrator operates. Algorithmic.

**Propose (Tier 2).** Input: `EngineRequest`, including the boundary raster and the program graph. Mechanism: the learned placement model of Section V under masked decoding, or the statistical `PriorProposer` when confidence falls below threshold. Output: a seed cell and size class per room, that is, the pair $(c_{1:n}, s_{1:n})$ of (2). Learned, with an algorithmic twin.

**Realize (Tier 3).** Input: seeds and size classes. Mechanism: a hub-first carver that establishes circulation hubs and then expands room regions by guillotine splits on the ownership grid, honouring size classes as area targets. Output: exact rectilinear geometry satisfying (1). Algorithmic.

**Settle (Tier 4).** Input: realized geometry. Mechanism: squeeze-and-settle area calibration, which redistributes cell ownership at shared boundaries to bring realized areas towards their targets without breaking the partition. Output: area-calibrated geometry. Algorithmic.

**Connect.** Input: settled geometry. Mechanism: derivation of doors and openings along an openness gradient, in which public-to-service transitions are opened and private rooms are given doors, followed by a reachability pass from the entrance. Output: the door and opening set $D$ and circulation $\Gamma$. Algorithmic.

**Validate.** Input: the complete candidate plan. Mechanism: evaluation of the rules catalog of Section IV-C. Output: a verdict object carrying hard-violation flags and a soft score. Rule-based.

**Repair.** Input: a verdict with hard violations. Mechanism: targeted local edits addressing named violations, followed by re-validation, bounded to two or three rounds so that failure is surfaced rather than masked by unbounded iteration. Output: an amended candidate or an explicit failure. Algorithmic.

**Rank.** Input: the surviving candidates from the best-of-K loop. Mechanism: the soft score, or a learned critic once one exists. Output: the selected plan, passed to the SVG renderer. Statistical at present, learned in future.

### B. The Orchestrator

The orchestrator executes the propose-through-rank sequence $K$ times per request, retaining the best-scoring candidate that carries no hard violation. Three properties matter. First, $K$-sampling converts the diversity of the proposal distribution into an ensemble, so a single unlucky sample does not determine the output. Second, repair is bounded at two to three rounds, which is a deliberate choice: an unbounded repair loop can hide a systematic proposal defect behind expensive local patching, whereas a bounded loop turns that defect into a visible failure. Third, every tier reports per-tier failure telemetry, so a drop in output quality is attributable to the stage that produced it. This is robustness engineering, but it is also the instrumentation that made the v1 diagnosis possible and that makes the merge gate of Section VI-C measurable.

### C. The Cell-Ownership Grid

Geometry is represented on a cell-ownership grid with 1.5-inch cells, in which every cell has exactly one owner. Non-overlap is therefore not enforced, checked, or repaired; it is inexpressible, because two rooms cannot simultaneously own a cell. This is the paper's strongest guarantee and the direct structural answer to the v1 failure of Table I. Coverage is similarly a matter of accounting: unowned interior cells are enumerable, so incomplete coverage is a detectable and localisable condition rather than a diffuse quality problem. The representation is also forward compatible. The grid natively supports rectilinear and L-shaped regions even though the current carver emits rectangles through guillotine splits, and because the learned model places only seeds, richer room shapes can be introduced by changing the carver alone, without retraining the placement model.

### D. The Rules Catalog

Acceptance is defined by a modular catalog covering `structure`, `circulation`, `doors`, `openings`, `light`, `walls`, `freespace`, and `nbc`. Each rule is declared hard or soft. Hard rules produce violation flags that disqualify a candidate outright; soft rules contribute weighted terms to a score in the range 0 to 100 and above, which is used only for ranking. A representative hard rule is STR-003, which requires that any parking space have an exterior vehicle gate, that is, a boundary segment on the plot perimeter wide enough for a vehicle. STR-003 illustrates why hard rules cannot be expressed as penalties: a parking bay unreachable by a vehicle is not a slightly worse parking bay. The validator returns a single verdict object carrying both the flags and the score, so that ranking never has the opportunity to trade a hard violation against soft quality.

### E. Design Principles

The architecture is governed by four axioms, stated here in the form used throughout the codebase and the remainder of this paper.

- **ML decides, algorithms guarantee.** No learned tier ever emits final coordinates. Correctness properties, meaning non-overlap, coverage, and code compliance, are always produced algorithmically.
- **Every ML tier has a statistical twin.** Each learned component is paired with a statistical implementation behind the same typed contract, serving simultaneously as a guarantee-preserving fallback and as the A/B baseline.
- **An ML tier ships only if it beats its fallback.** Deployment is conditional on measured superiority on the golden harness, formalised as the merge gate of Section VI-C.
- **Typed contracts isolate every tier.** Stages communicate through explicit contract objects, which permits substitution, per-tier telemetry, and independent testing.

---

## V. THE LEARNED PLACEMENT MODEL

This section specifies the Tier-2 placement model in sufficient detail for reimplementation, and derives its central guarantee.

### A. Overview and Parameter Budget

The model comprises approximately 7.36 M parameters and is deliberately sized to train on free-tier GPU allocations. The budget is a design requirement rather than a concession: a model that cannot be retrained by its maintainers within a single donated GPU session cannot be iterated on, and iteration is what closed the v1 diagnosis loop. The architecture is an encoder-decoder in which two encoders produce a shared memory, a causal transformer decodes one room per step, and two classification heads emit the discrete decisions of (2). Table II summarises the configuration.

### B. Program Encoder

The room program is encoded as a graph rather than a sequence. Nodes correspond to rooms and carry learned embeddings of type $\tau_i$, zone $z_i$, and floor index; edges encode desired adjacencies produced by the Enrich stage. Three GATv2 layers with node dimension 128 and 4 attention heads, using ELU activations with residual connections and LayerNorm, produce one token per room. Graph attention is preferred to a flat sequence encoder because the program's informative structure is relational and permutation-sensitive only through the adjacency it declares: two briefs with identical room lists but different adjacency requirements are different design problems, and a sequence encoder would have to infer that distinction from ordering rather than receive it directly. Attention over declared edges also lets a room's representation depend on the rooms it must neighbour, which is precisely the conditioning the decoder needs when it places that room.

### C. Boundary Encoder

The plot is rasterised into a two-channel $64 \times 64$ image, in which channel 0 is the interior footprint mask and channel 1 marks the entrance edge. A four-block strided CNN reduces this to an $8 \times 8$ spatial grid, yielding 64 boundary memory tokens. This encoder is the capability v1 lacked. In v1 the only geometric conditioning was plot width and height carried in a global token, which is sufficient to describe a rectangle and insufficient to describe anything else; irregular plots were therefore not representable, and the model could not learn that a corner of the interior was unavailable. With a mask channel, an arbitrary simple polygon is conditioned on directly, and the entrance channel supplies the orientation reference from which circulation and public-zone placement are learned.

### D. Global Context Token

A four-element vector $[\log W, \log H, \log(W/H), n/\kappa]$, containing log plot width, log height, log aspect ratio, and normalised room count, is projected and appended as one additional memory token. Log scaling keeps the dynamic range of plot dimensions comparable across the corpus, and the room-count term gives the decoder immediate access to the length of the sequence it is about to generate.

### E. Decoder

The decoder is a causal autoregressive transformer with 6 layers, $d_{\text{model}} = 256$, 8 attention heads, feed-forward dimension 1024, pre-norm residual blocks, and dropout 0.1. It cross-attends to the concatenated memory $[\,\text{program} \,\|\, \text{boundary} \,\|\, \text{global}\,]$, so every placement decision has simultaneous access to programmatic structure, plot geometry, and global scale. At step $t$ it emits two logit vectors: a 1024-way cell head over $c_t$ and a 40-way size head over $s_t$. Training is teacher-forced, with each step conditioned on the previous room's ground-truth placement, and a START token seeds step 0. The canonical generation order is fixed, so the model learns a consistent placement policy rather than an order-marginalised one.

### F. Masked Decoding

Masked decoding is the signature mechanism of the system. Let $O_{t-1}$ denote the occupancy state after $t-1$ placements and let $\ell_t \in \mathbb{R}^{1024}$ be the raw cell logits at step $t$. Define the legal set

$$
\mathcal{L}_t = \{\, c : c \in \operatorname{int}(P) \,\wedge\, c \notin \mathcal{B}(O_{t-1}) \,\wedge\, c \in \mathcal{Z}(r_t) \,\},
\tag{3}
$$

where $\operatorname{int}(P)$ restricts to cells inside the boundary mask, $\mathcal{B}(O_{t-1})$ is the union of claimed radii around already-placed seeds, and $\mathcal{Z}(r_t)$ is the set of cells permitted for room $r_t$ by zone and adjacency rules. The masked logits are

$$
\tilde{\ell}_t[c] =
\begin{cases}
\ell_t[c], & c \in \mathcal{L}_t, \\
-\infty, & \text{otherwise,}
\end{cases}
\tag{4}
$$

and sampling proceeds over $\operatorname{softmax}(\tilde{\ell}_t / T)$ at temperature $T$.

**Proposition 1.** *Under (3) and (4), the support of the sampler at every step contains no geometrically invalid placement, and therefore no sampled sequence $c_{1:n}$ induces a configuration violating the disjointness clause of (1).*

The proof is immediate: a cell assigned $-\infty$ receives zero probability after the softmax, so illegal cells are never drawn, and $\mathcal{L}_t$ is recomputed from $O_{t-1}$ after each placement, so legality is maintained inductively. The consequence is the property v1 never had. Invalid layouts are not merely improbable; they are unsamplable. This distinction is the reason the guarantee survives distribution shift, an undertrained checkpoint, or a high sampling temperature, none of which can move probability mass into a region the mask has removed.

Temperature sampling over the legal remainder is what produces the $K$ diverse proposals the orchestrator consumes. Diversity is obtained without risking validity, because every sample in the ensemble is drawn from a support that has already been intersected with the constraint set. The model additionally emits a confidence signal, defined as the mean top-1 post-mask probability across steps. When confidence falls below $\tau = 0.35$, the proposal is discarded and the statistical `PriorProposer` is invoked. Because the fallback satisfies the same typed contract, this is a guarantee-preserving degradation: the system becomes less clever, not less correct.

### G. Torch-Free Production Inference

The entire forward pass is reimplemented in pure NumPy and SciPy, covering linear layers, LayerNorm, GELU computed through `erf`, im2col-based 2D convolution, multi-head attention, and GATv2 segment-softmax. A unit test asserts agreement between the NumPy and PyTorch logits below a tolerance of 1e-4; the measured maximum discrepancy is 7e-7. The deployment consequence is that the production server loads a `.npz` weight archive and never imports PyTorch, which becomes a training-only dependency. We report this as a contribution to deployability rather than as an implementation note. Removing a heavyweight framework from the serving path reduces the container footprint, eliminates a class of version-coupling failures between training and serving, and makes the inference path auditable in a single readable module; the parity test is what converts those benefits from a claim into a measured property.

---

## VI. DATA PIPELINE AND TRAINING METHODOLOGY

This section describes how supervision is constructed and gated, and how training is instrumented so that the invisible-failure mode of v1 cannot recur.

### A. Data Pipeline

The corpus is CubiCasa5K, comprising 4,989 plans with per-room polygons under a CC BY-NC licence. RPLAN, at 80,000 plans, is identified as a planned second source and is not yet prepared. Preparation proceeds through six stages.

- `schema_audit` dumps the real record shape rather than trusting the dataset documentation, establishing what fields exist and with what types before any parsing logic is written.
- `vocab` maps CubiCasa room types onto the engine vocabulary, dropping undefined, outdoor, and sauna categories.
- `geometry` supplies dependency-free shoelace area and centroid computation together with a NumPy polygon rasteriser.
- `prep_cubicasa` performs the per-plan conversion: room polygons yield an interior footprint bounding box and a $64 \times 64$ boundary mask; each room yields a seed cell and a size class; adjacency edges are derived from shared polygon edges; and the entrance side is identified.
- `gates` applies the five hard quality gates described below.
- `render_samples` emits an HTML artefact for direct visual inspection of prepared records.

The five gates, reported in Table III, are hard: a failing gate aborts preparation with a named reason, and there are no silent passes. Gate 1 requires a seed-collision rate of zero, measured as 0 of 32,574 seeds. Gate 2 requires every seed to lie inside its own boundary mask, measured as zero out-of-boundary seeds. Gate 3 is a coverage sanity check requiring the sum of size-class areas to fall within $[0.30, 1.25]$ times the plot area, which catches both systematic under-segmentation and scale errors. Gate 4 emits a distribution report of room-count, type, and size histograms, whose purpose is domain-gap visibility rather than pass or fail filtering. Gate 5 freezes a 10% validation split by plan identifier and versions it to a manifest, so that evaluation across training runs and across machines refers to the same held-out set. The yield is 4,989 plans in and 4,430 prepared plans out, an 89% pass rate, with skips attributable to plans whose rooms are entirely undefined. All five gates pass.

The seed-collision gate warrants specific attention because of what was done when it failed. A raw preparation pass produced 5 seed collisions out of 32,574. The available responses were to relax the gate to a tolerance of a few collisions, or to correct the data. The pipeline adopts the latter: a collision-nudge step moves the seed of the later-generation room to the nearest free in-boundary cell, after which the measured rate is a true zero. The principle is stated plainly because it is transferable: *fix the data, do not lower the bar.* The alternative would have introduced, at the very first opportunity, exactly the class of supervision defect that Table I attributes 30% of the v1 failure to.

Domain-gap findings are reported as method rather than as caveat, because knowing the shape of the gap is what makes the merge gate of Section VI-C meaningful. CubiCasa5K is Finnish and literally contains sauna rooms, a room type with no counterpart in the Indian target domain. A share of 22.9% of rooms carry the label "undefined" and are dropped. Per-room foot-scale metadata is present in 99.7% of records, so real dimensions are recoverable rather than inferred. Finally, framing every plan on the interior footprint bounding box rather than the raw plan bounding box, which includes the yard, lifted mean mask fill from 0.29 to 0.74; the raw framing had been spending most of the raster resolution on empty exterior.

### B. Training Methodology

The governing lesson from v1 is that its loss measured the wrong thing, and each training choice is derived from that lesson.

- **Loss.** Cross-entropy on the seed cell with Gaussian spatial label smoothing at $\sigma = 1$ cell, so that a placement one cell from the reference receives partial credit, plus cross-entropy on the size class. Spatial smoothing matters because the discretisation is arbitrary at cell boundaries: an unsmoothed target treats a one-cell miss and a whole-plot miss as equally wrong.
- **Augmentation.** The dihedral group of order 8, comprising four rotations and a reflection, is applied with consistent remapping of the boundary mask, the seed cells, and the entrance side. This multiplies effective corpus size eightfold and enforces the invariance that a layout's quality does not depend on the compass framing of the raster.
- **Curriculum.** Programs of at most 6 rooms are presented first, followed by the full distribution. Short sequences establish the placement policy before the model must maintain a long occupancy state.
- **Checkpoint selection by engine metrics, never by loss.** After each evaluation epoch, the model's proposals are run through the real engine, that is, realize, settle, connect, and validate, on 50 held-out boundaries, and the checkpoint retained is the one with the best mean soft score. This is the direct, mechanical correction for the invisible-failure row of Table I: in v1, NLL fell smoothly to -33 while output quality was never measured, so the training signal and the deliverable were decoupled. Under engine-metric selection they are the same quantity.
- **Resumable, portable training.** A single `checkpoint.pt`, containing model weights, optimiser state, epoch counter, best-so-far record, and training log, is written atomically after every epoch, which permits continuation across interrupted free-tier Colab sessions. The production `.npz` is re-exported whenever a new best checkpoint is found, so the deployable artefact never lags the selected one.

### C. The Merge Gate

Deployment of the learned placer is conditional on three criteria, evaluated on the golden harness, and stated here as a non-negotiable ship criterion.

1. The learned placer **outperforms** the statistical `PriorProposer` on mean best score.
2. It regresses **no single brief by more than 5 points**, so that an average gain cannot conceal a catastrophic case.
3. It maintains **proposal fidelity of at least 0.8**, measured as seed displacement and size-class drift between proposal and realization.

The third criterion deserves emphasis because it is easy to omit. If the realizer substantially mutates the model's proposal, then the A/B comparison measures the realizer rather than the model, and a favourable result would be uninterpretable. Fidelity is therefore a validity condition on the experiment, not merely a quality target. Until all three criteria are met, the trained model remains on a branch or behind the confidence fallback of Section V-F, and the shipped system continues to use the statistical proposer. Table IV records the current status of each criterion.

---

## VII. EVALUATION, RESULTS, AND STATUS

This section states the evaluation protocol, reports the statistical baseline, presents a negative result on weight tuning in full, and gives the honest current status of every component.

### A. Evaluation Protocol

Evaluation uses a golden harness comprising a fixed set of representative briefs, each pairing a plot with a room program, scored by the validator of Section IV-D. [TO VERIFY: the exact number of briefs in the golden harness and their composition by plot size and program length.] The protocol is hard metrics first. Before any soft comparison is considered, a candidate must exhibit zero overlap, 100% coverage, and NBC compliance; a plan violating a hard rule is a failure regardless of its soft score, and is not eligible to contribute to a mean. Only among hard-valid candidates is the soft score, in the range 0 to 100 and above, used for ranking and for A/B comparison. On this harness, the statistical baseline achieves a mean soft score of approximately 75.5. This figure is the reference against which criterion 1 of the merge gate is evaluated.

### B. Human-Preference Study and a Negative Result

Phase A of the evaluation programme addressed the soft-score weights, of which there are approximately 25, through a learning-to-rank study against human preference judgments. The analytical starting point is that the soft score is exactly linear in its weights,

$$
\text{score} = 100 + \mathbf{w} \cdot \mathbf{g},
\tag{5}
$$

where $\mathbf{g}$ is the vector of rule-term values for a candidate. Linearity has an immediate methodological consequence: tuning reduces to convex pairwise learning-to-rank over preference pairs, requiring neither a GPU nor a derivative-free optimiser such as CMA-ES. The objective was set as human-pick agreement rather than raw score, because maximising the score with respect to its own weights is a degenerate objective that admits unbounded solutions with no relation to design quality.

The result was negative, and we report it without hedging. With 18 labelled briefs and 20 free weights, the system is underdetermined: top-1 cross-validated agreement with human picks caps at 28%. This is a data-sufficiency ceiling rather than a modelling failure, and no choice of regulariser or optimiser recovers identifiability that the label budget does not supply. [TO VERIFY: the relationship between the approximately 25 soft-score weights in the catalog and the 20 free parameters entering the tuning study, that is, which weights were held fixed.] The tuned weights were consequently **not shipped**, and the shipped system retains its hand-set weights. The deliverable of Phase A is the tested tuning infrastructure, which becomes useful the moment the label budget is increased. Reporting this matters because the failure mode it guards against is common and quiet: a tuned weight vector that improves the metric it was fitted to, is shipped on that basis, and degrades the artefact for every user. The measurement is what distinguishes the two outcomes, and publishing the measurement is what makes the distinction available to others.

### C. Current Status

The status below is stated in the present tense and describes what exists at the time of writing.

The algorithmic engine is complete and tested. It produces valid, overlap-free, code-respecting plans, and attains a baseline golden-harness mean soft score of approximately 75.5.

The Tier-2 placement model is fully implemented at approximately 7.36 M parameters. Its NumPy inference path is verified against the PyTorch reference to a tolerance of 1e-4, with a measured maximum discrepancy of 7e-7; 114 of 114 tests in the remastered package pass; and drop-in fallback to the statistical proposer is verified.

Training is in progress on Colab GPU, resumable across sessions through the atomic checkpoint described in Section VI-B. [TO VERIFY: epochs completed, wall-clock training time, and current best engine-metric score for the in-progress run.] The merge-gate A/B has not yet been run, and it is what will decide whether the learned placer ships.

The learned critic of Tier 5 is not built. The tuned soft-score weights are not shipped. RPLAN and ResPlan preparation is not started.

### D. Stated Expectation

Because the training corpus is Finnish and the target domain is Indian, we anticipate that the first trained model may lose to the Indian statistical fallback on the golden harness. We state this expectation before the experiment is run, and we regard the outcome as informative in either direction. If the model wins, the reformulation is validated end to end. If it loses, the merge gate has prevented a regression from reaching users, which is the safety system working as designed rather than a project failure; the indicated next step is then an Indian fine-tune with owned Indian program priors, as described in Section IX. The value of the protocol is precisely that it makes the second outcome survivable and legible instead of invisible.

### E. Tables

**TABLE I.** v1 Post-Mortem: Attributed Root Causes with Measured Evidence

| Root cause | Measured evidence | Attributed contribution |
|---|---|---|
| Wrong task formulation, regressing continuous $(c_x, c_y, w, h)$ boxes | Free boxes cannot represent a partition; overlap is a property of the output space, not of the fit | ~60% |
| Corrupted supervision | Bounding-box training cache exhibited 13.4% room-pair overlap in the ground truth; room-area to plot-area ratio p10–p90 = 0.79–1.12, so reference rooms did not tile their plots | ~30% |
| No boundary conditioning | Global token carried plot width and height only, with no polygon mask; irregular plots were not representable | Structural |
| Invisible failure | Only MoG NLL was tracked, falling smoothly to -33; no layout-quality metric was computed, so degenerate output was indistinguishable from progress | Methodological |
| Data quantity | 80,000 RPLAN plans are ample for a discrete formulation | ~10% |

**TABLE II.** Tier-2 Placement Model Configuration (approximately 7.36 M parameters total)

| Component | Architecture | Key hyperparameters | Parameter share |
|---|---|---|---|
| Program encoder | GATv2 graph attention over the room program | 3 layers, node dim 128, 4 heads, ELU, residual, LayerNorm; one token per room | [TO VERIFY: per-component parameter share] |
| Boundary encoder | Strided CNN over a 2-channel $64 \times 64$ raster | 4 blocks; channel 0 interior mask, channel 1 entrance edge; output $8 \times 8 = 64$ tokens | [TO VERIFY: per-component parameter share] |
| Global context | Linear projection to one memory token | $[\log W, \log H, \log(W/H), n/\kappa]$ | [TO VERIFY: per-component parameter share] |
| Decoder | Causal autoregressive transformer, cross-attending to concatenated memory | 6 layers, $d_{\text{model}}=256$, 8 heads, FF 1024, pre-norm, dropout 0.1 | [TO VERIFY: per-component parameter share] |
| Output heads | Two linear classification heads per step | Cell head 1024-way; size head 40-way | [TO VERIFY: per-component parameter share] |

**TABLE III.** Data-Pipeline Gates on CubiCasa5K

| Gate | Criterion | Measured value | Status |
|---|---|---|---|
| 1. Seed collision | Collision rate = 0% | 0 / 32,574 seeds, after collision-nudge | Pass |
| 2. Seed containment | Every seed inside its boundary mask | 0 out-of-boundary seeds | Pass |
| 3. Coverage sanity | $\sum$ size-class areas within $[0.30, 1.25] \times$ plot area | Within bounds for all prepared plans | Pass |
| 4. Distribution report | Room-count, type, and size histograms emitted for domain-gap visibility | Report generated | Pass |
| 5. Frozen validation split | 10% by plan identifier, versioned to a manifest | Split frozen and manifested | Pass |
| Yield | Prepared plans / source plans | 4,430 / 4,989 = 89%; skips are all-undefined plans | Pass |

**TABLE IV.** Merge-Gate Criteria and Current Status

| # | Criterion | Threshold | Status |
|---|---|---|---|
| 1 | Mean best score versus statistical `PriorProposer` | Strictly greater | Not yet evaluated; training in progress |
| 2 | Worst-case per-brief regression | $\leq$ 5 points | Not yet evaluated |
| 3 | Proposal fidelity after realization | $\geq$ 0.8 | Not yet evaluated |
| — | Deployment decision | All three must pass | Learned placer not shipped; statistical proposer in production |

---

## VIII. ENGINEERING AND REPRODUCIBILITY

This section argues that the deployment and reproduction properties of the system are research contributions rather than chores, because each one is what converts a claim in an earlier section into a checkable fact.

Typed contracts isolate every tier, and multi-floor context is carried in those contracts from the first version onward, before multi-floor realization exists, specifically to prevent the contract rot that occurs when a cross-cutting field is retrofitted after every consumer has been written without it. The contract boundary is also what makes the statistical-twin principle operational: substituting a learned proposer for its fallback changes one constructor argument and nothing else, which is the precondition for a controlled A/B comparison.

Determinism and structural guarantees rest on the 1.5-inch cell-ownership grid, where single ownership makes overlap inexpressible and unowned-cell enumeration makes coverage checkable. Production inference is torch-free through the verified NumPy path, whose 7e-7 agreement with the PyTorch reference is asserted by a unit test rather than measured once and reported. Portable atomic checkpointing allows training to proceed across interrupted free-tier GPU sessions without partial-write corruption, which is what makes a long training run reproducible by a reader who has no dedicated hardware. Gate-driven data preparation guarantees no silent passes: each gate aborts with a named reason, so a corpus that does not meet the bar cannot be trained on by accident, which is the specific accident that Table I attributes 30% of the v1 failure to.

The implementation comprises approximately 78 Python modules. Unit-test coverage spans model output shapes; mask legality, verifying that no illegal cell survives (4); overfit-a-batch sanity, verifying that the model can drive training loss to near zero on a single batch and therefore that the optimisation path is sound; NumPy and PyTorch numerical equality; device agnosticism; augmentation correctness, verifying that dihedral transforms remap mask, seeds, and entrance side consistently; and the polygon mathematics of data preparation. The remastered package reports 114 of 114 tests passing. Each of these tests exists because it makes a specific claim in this paper falsifiable by execution rather than by inspection.

---

## IX. LIMITATIONS AND FUTURE WORK

This section states the limitations of the present system before turning to planned work.

### A. Limitations

The most significant limitation is the domain gap. The training corpus is Finnish and the announced second source is Chinese, while the target domain is Indian; the corpus contains room types with no Indian counterpart, and the program priors it encodes are not those of the target market. Second, the current carver emits rectangular rooms only. The ownership grid supports rectilinear and L-shaped regions, but non-rectangular realization is out of scope for the present system. Third, the Tier-5 learned critic is not built, so ranking currently relies entirely on the hand-weighted soft score. Fourth, evaluation is concentrated on a fixed golden harness, which is a small, curated instrument; it is well suited to detecting regressions and poorly suited to establishing generalisation across the full space of briefs. Fifth, soft-score weight tuning is blocked by a data-sufficiency ceiling, as reported in Section VII-B, and the shipped weights remain hand-set.

### B. Future Work

In priority order, the following are planned.

- **An Indian fine-tune** of the placement model, using owned Indian program priors, which is the direct remediation for the domain gap and the indicated next step should the first model fail the merge gate.
- **Irregular rooms** of five and six sides. This is enabled without retraining, because the seed-cell contract makes the carver the sole owner of shape; the model's output space is unchanged when the carver's expressiveness increases.
- **A Tier-5 learned critic**, structured as a two-tower ranker combining a CNN over the rendered plan with a GNN over the layout graph, trained on human pairwise preference. The critic addresses the ranking limitation directly and, unlike weight tuning, scales with additional preference labels rather than saturating against a fixed parameterisation.
- **A Tier-1 zone planner**, which would produce coarse zone assignments before placement and supply a stronger $\mathcal{Z}(r_t)$ term in the mask of (3).
- **Optional reinforcement-learning fine-tuning** of the proposer against a reward combining the validator verdict and the learned critic, which is deferred until the critic exists, since the reward would otherwise be the hand-weighted score the system already optimises indirectly.
- **Additional data sources**, specifically RPLAN at 80,000 plans and ResPlan at approximately 17,000 vector plans, prepared through the same five gates.
- **Multi-floor realization**, covering stair placement, wet-stack alignment, and structural continuity between floors. This is a first-class requirement rather than an extension, and the typed contracts already carry floor context in anticipation of it.

---

## X. CONCLUSION

We have presented PlanGen, a neuro-symbolic engine for residential floor-plan generation built on a single reformulation: placement is a constrained discrete classification problem and geometry is an algorithmic one. The model predicts a seed cell from a boundary-clipped $32 \times 32$ grid and a size class from 40 area buckets, and masked decoding removes every geometrically or semantically illegal cell from the sampler's support before each draw, so that invalid layouts are unsamplable rather than merely improbable. A deterministic hub-first carver on a single-owner cell grid converts seeds into exact geometry, which makes overlap inexpressible and coverage checkable. The system is governed by two principles: machine learning decides and algorithms guarantee, and every learned tier must beat a statistical twin on a quality harness before it ships. The second principle has already withheld one component, the tuned soft-score weights, on the basis of a measured data-sufficiency ceiling, and it may yet withhold the placement model itself given the Finnish-to-Indian domain gap.

The transferable lesson concerns how hard structural constraints should be approached. Our v1 system did not fail because it was too small, or because 80,000 plans were too few; it failed because a set of freely regressed rectangles cannot express a partition, and because its supervision, its conditioning, and its metrics all shared that blind spot. In domains with hard structural constraints, the productive question is therefore not how to train a model to stop violating the constraint, but how to build a representation in which violation is inexpressible.

---

## REFERENCES

*The following entries are placeholders. No author names, titles, venues, or years have been fabricated; each entry records the claim requiring support.*

[1] [CITATION NEEDED: raster-based floor-plan generation with generative adversarial or diffusion models]

[2] [CITATION NEEDED: sequence models emitting vector room primitives for floor-plan synthesis]

[3] [CITATION NEEDED: reported overlap or coverage metrics in learned layout generation]

[4] [CITATION NEEDED: graph-conditioned floor-plan generation from bubble diagrams or adjacency graphs]

[5] [CITATION NEEDED: constraint-conditioned 2D layout generation with differentiable penalties]

[6] [CITATION NEEDED: squarified treemap layout algorithm]

[7] [CITATION NEEDED: guillotine or slicing-tree partitioning for floor-plan layout]

[8] [CITATION NEEDED: simulated annealing for architectural layout optimisation]

[9] [CITATION NEEDED: grammar-constrained decoding for language models]

[10] [CITATION NEEDED: constrained decoding for semantic parsing or program synthesis]

[11] [CITATION NEEDED: neuro-symbolic architectures combining learned proposal with symbolic verification]

[12] [CITATION NEEDED: GATv2 graph attention networks]

[13] [CITATION NEEDED: transformer architecture]

[14] [CITATION NEEDED: CubiCasa5K dataset]

[15] [CITATION NEEDED: RPLAN dataset]

[16] [CITATION NEEDED: ResPlan vector floor-plan dataset]

[17] [CITATION NEEDED: National Building Code of India, room dimension minimums]

[18] [CITATION NEEDED: CMA-ES derivative-free optimisation, referenced as the method not required here]

---

## APPENDIX A — OUTSTANDING VERIFICATION CHECKLIST

Facts required by the paper that are absent from the source document, collected for resolution before submission.

1. `[TO VERIFY]` The exact number of briefs in the golden harness and their composition by plot size and program length. (Section VII-A)
2. `[TO VERIFY]` The relationship between the approximately 25 soft-score weights in the rules catalog and the 20 free parameters entering the Phase A tuning study, that is, which weights were held fixed. (Section VII-B)
3. `[TO VERIFY]` Epochs completed, wall-clock training time, and current best engine-metric score for the in-progress training run. (Section VII-C)
4. `[TO VERIFY]` Per-component parameter share within the 7.36 M total, for Table II.
5. `[TO VERIFY]` The value of $K$ in the orchestrator's best-of-K loop. (Section IV-B)
6. `[TO VERIFY]` The definition of the claimed radius $\mathcal{B}(\cdot)$ used in the mask of (3), in cell units and its dependence on size class.
7. `[TO VERIFY]` The size of the extracted-plan corpus used by the Match stage, stated as approximately 5,000, and its relationship to CubiCasa5K.
8. All `[CITATION NEEDED]` markers listed in the reference section above.

---

## APPENDIX B — SELF-CHECK

- [x] **Every numeric claim traces to the source document.** All figures (13.4%, 0.79–1.12, -33, 46 epochs, 512-dim, 12 layers, 7.36 M, 32x32, 1024, 40 buckets, 25 sq ft, 1000 sq ft, 64x64, 8x8, 6 layers, 256, 8 heads, 1024 FF, dropout 0.1, tau = 0.35, 1e-4, 7e-7, 1.5-inch, 4,989, 4,430, 89%, 0/32,574, 5 collisions, 22.9%, 99.7%, 0.29 to 0.74, [0.30, 1.25], 10% split, sigma = 1, x8 augmentation, <= 6 rooms, 50 boundaries, 5 points, 0.8 fidelity, 75.5, 18 briefs, 20 weights, 28%, 114/114, ~78 modules, 80,000 RPLAN, 17,000 ResPlan, ~5,000 extracted plans) appear in `paper_presentation_details.md`.
- [x] **No fabricated citation, baseline, or experimental result appears.** All references are `[CITATION NEEDED]` placeholders with no invented authors, titles, venues, or years. No comparison to any external system is claimed.
- [x] **All `[TO VERIFY]` and `[CITATION NEEDED]` markers are collected** in Appendix A and the reference list.
- [x] **The abstract executes all six moves** (context, gap, insight, method, evidence, significance) and expands GATv2, CNN, and NLL-adjacent acronyms at first use within the abstract; it contains no citations and no bullets.
- [x] **Exactly five contributions are listed**, each with a forward reference to Section IV, VI, VII, or VIII.
- [x] **Every figure and table is cited in the body before it appears.** Fig. 1 is referenced in Section IV's opening; Tables I–IV are referenced in Sections I, V, VI, and VII respectively.
- [x] **Unfinished work is described in the present tense, never as completed.** Training is "in progress"; the critic "is not built"; the merge-gate A/B "has not yet been run"; tuned weights "are not shipped"; RPLAN/ResPlan prep "is not started".
- [x] **The negative results appear in the introduction** (contribution 5), **the evaluation** (Section VII-B and VII-D), **and the conclusion** (Section X).
- [x] **No prohibited marketing vocabulary appears.** Checked for: revolutionary, game-changing, cutting-edge, state-of-the-art, seamless, powerful, robustly handles.
- [x] **Notation introduced in Section III is used consistently thereafter**: $P$, $R$, $r_i = (\tau_i, a_i, z_i)$, $e$, $C$, $\Omega$, $c_t$, $s_t$, $\mathcal{L}_t$, $O_{t-1}$, $\mathcal{B}$, $\mathcal{Z}$ recur in Sections IV–IX.

**Deviations from the specification, stated explicitly:**

1. **The body exceeds the stated word target, and the two specifications conflict.** The per-section budgets sum to roughly 7,300 words at their minima, above the stated 5,500–7,000 total. The measured body, Sections I through X excluding table rows, is **7,827 words**. Sections were written at or near the lower bound of their individual ranges; reaching 7,000 overall would require cutting most sections below their stated minima. The section budgets were prioritised. If the total is the binding constraint, the recommended cuts are Section II-C and II-D (roughly 200 words), the stage-by-stage detail in Section IV-A (roughly 250 words), and Section VIII (roughly 200 words), none of which carry a numeric claim or a contribution.
2. Section numbering follows the paper's own IEEE sequence (I–X). The prompt's lettered sections A–L map onto them as: A to the front matter, B to I, C to II, D to III, E to IV, F to V, G to VI, H to VII, I to VIII, J to IX, K to X.
3. Em-dashes are avoided throughout in favour of commas, semicolons, and full stops, per the prohibition on using them as a substitute for sentence structure.
