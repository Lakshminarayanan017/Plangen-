# PlanGen Upgrade Proposal – Detailed Explanation & Technical Review

## Part 1 – Plain-English Explanation

### What is this proposal trying to fix?

Your current PlanGen engine follows a deterministic pipeline:

```text
Input
   │
   ▼
Proposer
   │
   ▼
Geometry Generator
   │
   ▼
Optimizer
   │
   ▼
Doors
   │
   ▼
Validator
   │
   ▼
Best Layout
```

Today almost every stage is algorithmic.

This means the engine is very good at:

- No overlapping rooms
- Proper connectivity
- Correct room sizes
- Legal circulation

However, layouts can still feel merely "okay" instead of architecturally pleasing.

For example:

```text
Kitchen      Bedroom

      Hall

Dining
```

Nothing is technically wrong.

But a human architect immediately asks:

> "Why is the kitchen so far from the dining room?"

That is not a geometry problem.

It is a judgment problem.

---

## The Biggest Idea

The proposal divides every task into two categories.

### 1. Correctness

Examples:

- Room overlap?
- Bathroom accessible?
- Minimum width satisfied?
- Area correct?

These have objectively correct answers.

Algorithms are excellent at these.

ML should not replace them.

---

### 2. Judgment

Examples:

Should the kitchen go here?

Option A

Kitchen

or

Option B

Kitchen

Both satisfy every rule.

Which one feels architecturally better?

There is no mathematical answer.

This is exactly where ML excels.

---

## Why PlanGen v1 Failed

The original system directly predicted geometry.

Example:

Bedroom

x = 4
y = 10
w = 18
h = 15

If the model predicted overlapping rooms, the entire house became invalid.

The neural network became responsible for correctness.

That is fundamentally unsafe.

---

## The New Idea

Never let ML generate geometry.

Instead ML predicts intent.

Example:

Bedroom

→ Top-right corner
→ Medium size

That's it.

Then the deterministic geometry engine creates the actual room.

This means ML can no longer break geometry.

---

## The Current Proposer

Today the proposer uses manually selected constants.

Example:

```python
_ZONE_DEPTH = {
    "public": 0.85,
    "private": 0.25,
    "service": 0.60
}
```

Those numbers represent human assumptions.

Instead, train on thousands of real floor plans so the proposer learns natural room relationships such as:

- Kitchen near dining
- Utility near kitchen
- Bathroom near bedrooms

---

## Seed Placement

The proposal suggests dividing the plot into a 32×32 grid.

Instead of predicting room coordinates, the model predicts:

Bedroom
→ Cell (18,9)
→ Large

Kitchen
→ Cell (3,7)
→ Medium

Hall
→ Cell (14,15)
→ Huge

Then the deterministic engine creates legal geometry.

---

## Why This Is Safer

A seed cell cannot overlap another room.

It is only an intention.

The geometry engine later guarantees:

- No overlap
- Proper wall alignment
- Legal corridors
- Valid room shapes

---

## Why a Transformer?

The output is simply a sequence:

Hall → Cell 23 → Large

Kitchen → Cell 6 → Medium

Bedroom → Cell 17 → Large

Bathroom → Cell 21 → Small

Transformers are excellent sequence models.

---

## Masked Decoding

Suppose the transformer predicts:

Bedroom → Cell 90

If Cell 90 lies outside the plot:

Probability(Cell90) = 0

Similarly:

- Already occupied → masked
- Doesn't fit → masked
- Illegal → masked

Therefore invalid outputs become impossible.

---

## Why Use a GNN?

A GNN naturally represents room relationships.

Example graph:

Bedroom
   |
Bathroom
   |
Hall
   |
Kitchen
   |
Dining

The GNN captures adjacency while the transformer predicts the sequence.

---

## Why Not Diffusion?

Diffusion predicts continuous coordinates.

Those coordinates must later be:

- Snapped
- Repaired
- De-overlapped

This reintroduces the exact problems of v1.

Since PlanGen predicts discrete cells, transformers are a better fit.

---

## Why Not GANs?

GANs suffer from:

- Mode collapse
- Training instability
- Weak conditioning

Modern layout generation has largely moved beyond them.

---

## The Critic

Suppose the engine generates 1000 layouts.

Today it selects the layout with the highest heuristic score.

A learned critic instead asks:

> "Does this look like something an architect would design?"

This captures qualities difficult to express through rules.

---

## CNN + GNN Critic

CNN learns visual properties:

- Symmetry
- Corridor shapes
- Proportions

GNN learns relational properties:

- Adjacency
- Functional flow
- Room relationships

Together they provide a much richer quality estimate.

---

## CMA-ES

Today many scoring weights are hand chosen.

For example:

Area = 90

Centrality = 6

Distance = 15

CMA-ES automatically searches for better values using human evaluation as feedback.

No neural network is required.

---

## Reinforcement Learning

Only after everything else works.

Reward =

Validator Score

+

Critic Score

The proposer gradually improves by maximizing this reward.

---

## Suggested Roadmap

Phase A

- Optimize scoring weights
- Increase candidate count

Phase B

- Train AR proposer

Phase C

- Train learned critic

Phase D

- Reinforcement learning fine tuning

---

# Part 2 – My Technical Opinion

Overall, I think this is a very strong architectural proposal.

The biggest strength is that it does **not** attempt to replace algorithms with AI.

Instead it separates:

- AI for judgment
- Algorithms for correctness

That is exactly the balance modern hybrid AI systems aim for.

## What I Strongly Agree With

- Separating judgment from correctness.
- Using ML only for proposing and ranking.
- Keeping deterministic geometry generation.
- Adding a learned critic to improve selection.
- Starting with CMA-ES/Bayesian optimization before training deep models.

## Where I Would Be Cautious

### 1. Data Quality

The proposer can only be as good as the training data.

Poor annotations or biased layouts will affect performance.

### 2. 32×32 Representation

A 32×32 grid is a reasonable design choice but should be experimentally validated.

### 3. Critic Supervision

Real-vs-generated training is useful, but human pairwise rankings will likely provide the strongest supervision.

### 4. Transformer Choice

Transformers are a natural fit because the output is a sequence of discrete tokens.

However, architecture decisions should always remain evidence-driven.

## Overall Assessment

Architecture: ★★★★★

Risk Management: ★★★★★

Practicality: ★★★★☆

Expected Quality Improvement: High

---

The long-term vision becomes:

```text
             AI
      (Architectural Intuition)
               │
               ▼
         Seed Placement
               │
               ▼
  Deterministic Geometry Engine
               │
               ▼
 Deterministic Optimization
               │
               ▼
        AI Quality Critic
               │
               ▼
        Best Final Layout
```

Rather than asking AI to produce an entire building, this design lets AI make the subjective decisions while deterministic algorithms guarantee correctness.

That is, in my opinion, the strongest aspect of the proposed PlanGen upgrade.
