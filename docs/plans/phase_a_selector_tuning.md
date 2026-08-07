# Phase A — Learning the Candidate Selector from Human Taste

## Context

The remastered engine generates K valid candidates per brief and shows the
one with the highest `soft_score`. That score is a weighted sum of ~20 hand-
tuned penalties/bonuses in `EngineConfig`. Those weights are the engine's
*taste* — and they were guessed, not learned. This is the first, cheapest
lever in the [ML upgrade roadmap](../../plangen_remastered): improve *which*
candidate gets picked, without touching generation or geometry.

**The decisive fact.** The 20 soft weights do **not** change geometry. They
only re-rank a fixed candidate set (generation — proposer variants, carve
retries, dedup, top-up — is entirely weight-independent). Two consequences:

1. "Tune weights to maximize score" is degenerate — it drives every penalty
   weight to zero so every plan scores 100. The **only** sound objective is:
   make the engine's `argmax` agree with the plan a **human** would pick.
   Phase A is therefore preference learning / learning-to-rank, and it needs
   human labels by construction.
2. The candidate corpus + human labels built here are the **same** data
   foundation the Phase C learned critic trains and is evaluated on. Phase A
   is not throwaway; it is C's substrate and the baseline C must beat.

## The math that makes it cheap

For a **fixed** plan, `soft_score(w) = 100 + w · g`, where `g` is a signed
20-vector of the (transformed) per-rule measures — exact, because every
weight enters the score as one linear multiplier. So ranking candidates by
human preference is a **convex logistic learning-to-rank** problem solved by
`scipy` (already installed) — no CMA-ES, no GPU. CMA-ES from the roadmap is
reserved for the two knobs that actually move geometry
(`settle_aspect_weight/limit`), which are out of scope here.

`g` is recovered **without editing the engine**: re-score the fixed plan
under probe weights (one weight = 1000, rest 0) and read `g_k =
(score-100)/1000`. Verified to reproduce the validator to within its 2-dp
rounding on every real candidate (`tests/test_tuning.py`).

> Subtlety respected in code: `FSP-001` is registered `"hard"` but adds a
> `w_freespace` *soft* penalty when `fsp001_hard=False`, so the extractor
> runs the full rule set, not only soft-severity rules.

## What was built — `plangen_remastered/tuning/`

| module | role |
|---|---|
| `features.py` | exact `g` extraction + `reconstruct_score`; `WEIGHT_KEYS` order |
| `corpus.py` | run any `EngineConfig` over the golden briefs → every valid candidate with `g`, breakdown, SVG; JSON round-trip |
| `labeling.py` | `Preferences` type + a self-contained (no-network) HTML page: pick the best plan per brief, mark any "bad", export `labels.json` |
| `tuner.py` | convex pairwise logistic fit (`w ≥ 0`, relative-L2 pull toward defaults), **leave-one-brief-out CV** to choose the regularizer; honest top-1 / pairwise agreement metrics |
| `verify.py` | re-rank base-vs-tuned on the fixed corpus; side-by-side SVGs of every brief whose pick changed; before/after agreement |
| `cli.py` | `export` → `tune` → `verify` |
| `tests/test_tuning.py` | 12 tests: score reconstruction on the real engine (default + perturbed weights) + tuner recovers a known `w_true` from synthetic labels + JSON/UI smoke |

Zero engine files were modified.

## How to run

```bash
cd plangen_remastered
python -m tuning.cli export          # builds results/corpus.json + labeling.html
# open tuning/results/labeling.html, pick the best plan per brief,
# "Export labels.json" → save into tuning/results/labels.json
python -m tuning.cli tune            # → results/tuned_config.json + tune_report.json
python -m tuning.cli verify          # → results/verify.html
```

## Verification / success criteria

The honest, reported number is **cross-validated top-1 selection agreement**
(does the tuned `argmax` match the held-out human pick), never raw score.
Phase A succeeds if `tuned_top1_cv > base_top1` on real labels; if it does
not, the defaults already rank as well as the labels justify and the tuned
weights ship only if `verify.html` shows they help. Artifacts:
`tuned_config.json` (the deliverable — a drop-in `EngineConfig`),
`tune_report.json` (metrics + λ sweep), `verify.html` (the plans that
swapped).

## Open risks (validate, don't assume — per the roadmap caveats)

- **Label scarcity.** 18 briefs is a small preference set for 20 weights.
  Mitigated by relative-L2 regularization toward defaults + CV-chosen λ;
  raise `--k` or add briefs if CV stays noisy.
- **Candidate diversity.** If the K candidates per brief look alike, the
  preference signal is weak — a symptom that the *proposer* (Phase B), not
  the selector, is the bottleneck. The corpus makes this visible.
- **Linear ceiling.** A weighted sum of hand-designed features cannot
  capture gestalt; that is exactly what the Phase C critic is for. Phase A
  sets its baseline.

## Not yet wired into the live app

`tuned_config.json` is produced but **not** auto-loaded by
`api/engine_bridge.py` / the `Orchestrator` default. Wiring it in is a
deliberate follow-up once real labels validate it on `verify.html`.
