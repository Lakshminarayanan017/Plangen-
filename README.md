# PlanGen

Architecture plan generator — natural-language brief in, carved multi-floor
floor plans (SVG + DXF) out, tuned to Indian residential practice (NBC
minimums, Vastu, setbacks/FAR).

## Pipeline

The whole system is one numbered pipeline. Each step is a package under
`modules/`, and each consumes the previous step's output:

| Step | Package | In → Out |
|------|---------|----------|
| 1 | `modules/step1_parse` | user text/image → `BuildingRequirements` |
| 2 | `modules/step2_match` | requirements → `KnowledgeBundle` (stats from real plans) |
| 3 | `modules/step3_enrich` | bundle → `EnrichedPlan` (sized, zoned, floored rooms) |
| 4 | `modules/step4_generate` | enriched plan → carved geometry, SVG, DXF |

`api/server.py` runs steps 1–3 and hands off to `api/engine_bridge.py`, which
adapts the `EnrichedPlan` into an `EngineRequest` and drives step 4.

## Layout

```
models.py            Pydantic schemas shared across every step
api/                 FastAPI server + the bridge into the engine
frontend/            static UI (index.html, app.js, styles.css)
modules/
  step1_parse/       parser, image analyzer, interactive gatherer, offline parser
  step2_match/       semantic matcher, feature encoder, stats aggregator, IS/NBC standards
  step3_enrich/      enricher, room resolver, vastu mapper
  step4_generate/    the wall-graph partition engine  (see its own README)
    core/            lattice units, GridPlan, polygon ops
    carve/           hub-first carving, squeeze-and-settle, stairs
    engine/          orchestrator, contracts, rules/, cpsat/, multifloor
    critic/          learned critic that ranks candidates
    render/          SVG + DXF export
    demos/           runnable visual demos
    room_budget.py   generation order + area fractions (used by step 3)
sources/             config the code reads: enricher_rules.json, prompts/, key rotator
data/                small distilled stats the RUNTIME reads (zone_patterns.json …)
ml/                  training-time only — never imported by the request path
  tier2_placer/      Tier-2 seed proposer: model, train, export, inference
  training/          CubiCasa5K → training samples
  tuning/            CMA-ES tuning of selector weights
  harness/           golden briefs + A/B harness
  data/              bulk corpora (gitignored)
tests/
  pipeline/          steps 1–3 + the engine bridge
  engine/            step 4 (the engine itself)
  ml/                ml/ packages
docs/                PRD, plans, paper, engine architecture notes
output/              generated artifacts (gitignored)
unwanted/            retired code + dead bulk data, kept out of the way (gitignored)
```

## Running

Everything runs from the project root as a package — nothing injects
`sys.path`.

```bash
python -m uvicorn api.server:app --reload      # the app
python -m unittest discover -s tests -t .      # all 280 tests
python -m modules.step4_generate.demos.demo_engine   # engine demo → output/
python -m ml.harness.run_harness               # quality harness
```

## The two data directories

They are easy to confuse, so:

- **`data/`** — small, curated, **tracked**. Read at request time
  (`engine/priors.py` loads `zone_patterns.json` from here).
- **`ml/data/`** — gigabytes of raw corpora, **gitignored**, only ever touched
  by `ml/training/`. Nothing in the request path reads it.
