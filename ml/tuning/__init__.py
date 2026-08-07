"""
tuning — Phase A: learn the engine's candidate *selector* from human taste.

The engine already generates K valid candidates per brief; which one it
shows as "best" is decided by the 20 soft-score weights in EngineConfig.
Those weights do NOT change geometry — they only rank the fixed candidate
set. So tuning them to "maximize score" is degenerate (drives every weight
to zero). The only sound objective is: make the engine's argmax agree with
the plan a human would pick. That makes Phase A a preference-learning /
learning-to-rank problem over the weight vector.

Because soft_score is exactly affine in the weights for a fixed plan
(`soft_score = 100 + w · g`, see tuning.features), ranking candidates by
human preference is a CONVEX logistic learning-to-rank fit — solved with
scipy (already a dependency), no CMA-ES / no GPU. CMA-ES from the roadmap
is reserved for the two knobs that actually move geometry
(settle_aspect_weight / settle_aspect_limit), out of scope for this module.

Pipeline (see tuning.cli):
    export  → run the engine over the golden briefs, collect every kept
              candidate with its feature vector g and a rendered SVG
    label   → a self-contained HTML page; the human picks the best plan
              per brief; exports labels.json
    tune    → convex pairwise logistic fit with leave-one-brief-out CV to
              pick the regularization strength; writes tuned_config.json
    verify  → re-rank the same candidate sets base-vs-tuned, report the
              honest held-out selection-agreement improvement

The candidate corpus + preference labels produced here are the same data
foundation the Phase C learned critic will train and be evaluated on.
"""
