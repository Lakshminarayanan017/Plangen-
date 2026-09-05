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
frontend/            the UI — plain HTML/CSS/JS, no build step
  index.html         landing page
  chatPage.html      the brief -> plan conversation
  script.js          session, parse loop, pipeline polling, sheet rendering
  assets/            backgrounds and landmark elevations
modules/
  step1_parse/       parser, image analyzer, interactive gatherer, offline parser
  step2_match/       semantic matcher, feature encoder, stats aggregator, IS/NBC standards
  step3_enrich/      enricher, room resolver, vastu mapper
  step4_generate/    the wall-graph partition engine  (see its own README)
    core/            lattice units, GridPlan, polygon ops
    carve/           hub-first carving, squeeze-and-settle, stairs
    engine/          orchestrator, contracts, rules/, cpsat/, multifloor
      vastu/         compass frame, the 81-pada mandala, compliance report
    critic/          learned critic that ranks candidates
    render/          SVG + DXF export
    demos/           runnable visual demos
    room_budget.py   generation order + area fractions (used by step 3)
sources/             config the code reads: enricher_rules.json, prompts/, key rotator
data/                small distilled stats the RUNTIME reads (zone_patterns.json …)
ml/                  training-time only — never imported by the request path
  placer_v3/         Tier-2 seed proposer v3: state-conditioned model, three
                     -stage training (pretrain / distil / RL), self-play corpus
  tier2_placer/      v2 proposer, kept as the A/B baseline for the merge gate
  training/          CubiCasa5K → training samples
  tuning/            CMA-ES tuning of selector weights
  harness/           golden briefs + A/B harness
  data/              bulk corpora (gitignored)
tests/
  pipeline/          steps 1–3 + the engine bridge
  engine/            step 4 (the engine itself)
  ml/                ml/ packages
docs/                PRD, plans, paper, engine architecture notes — the ONLY
                     home for planning documents
output/              generated artifacts (gitignored)
unwanted/            gitignored holding pen, never imported by anything
  retired/           code no longer reachable from any entry point
  ui_handoff/        raw design source the shipped frontend/ was built from
  legacy_step4/      the retired v1 autoregressive engine
  archives/          old exports; extracted_data/ raw dumps
```

## Running

Everything runs from the project root as a package — nothing injects
`sys.path`.

```bash
# run the app — serves the UI and the API on one port, no build step
python -m uvicorn api.server:app --reload      # http://127.0.0.1:8000

python -m unittest discover -s tests -t .      # the full suite
python -m modules.data_prep.plan_indexer       # (re)build data/plan_index
python -m modules.diagnostics                  # what is real vs on fallbacks
python -m ml.harness.diagnose --worst 5        # why a brief scores what it does
python -m modules.step4_generate.demos.demo_engine   # engine demo → output/
python -m ml.harness.run_harness               # quality harness
```

The UI is served from the same origin as the API, so editing a file under
`frontend/` and reloading the browser is the whole frontend workflow.

## Is it actually working?

Every knowledge source in this system fails soft, so a silent startup used to
be indistinguishable from a healthy one. It now says:

```bash
python -m modules.diagnostics      # or GET /api/v1/diagnostics
```

`ok` = working · `degraded` = running on a fallback · `missing` = a claimed
capability is absent. The server prints this at boot and the UI shows it in
the brief's title block.

## Program synthesis

The pipeline decides **what to build** before it decides where things go.
Until phase 02 the program came from BHK statistics and was then only
*scaled*: inflated by one uniform factor on a large plot — measured at 5.9x,
a 45 sqft bathroom realised at 266 — and compressed toward NBC minimums on a
small one.

`modules/step3_enrich/program_synth.py` reads the plot and decides:

- **compact** (under 600 sqft buildable) — sheds rooms an earlier stage
  *inferred*, never one the user asked for by name, and says which
- **normal** — leaves a fitting program alone
- **spacious** (over 1800 sqft) — adds a dining room, pooja room, utility or
  study rather than inflating the rooms already there

Every room carries a size ceiling, so surplus goes to rooms that can use it.
What is still left over shrinks the **building**, not the rooms: a 2BHK on a
large plot is a 2BHK house standing in a garden, and the open ground is
reported rather than absorbed. A small remainder becomes a courtyard — which
is also what the Brahmasthan asks for, so the two requirements agree.

The capacity curve is a measurement, not a guess: the engine was run over 14
plot sizes x 7 program shapes and the largest program still scoring above 60
was recorded. Note what it rules out — the reviewer's soft score always
prefers *fewer* rooms (a 1BHK scores 100 on a 4,200 sqft plot), so selecting
a program by score would recommend a studio for a mansion.

## Vastu

Vastu is a first-class reviewed constraint, not a flag on the output. Three
things make it work, and all three are required:

1. **A compass frame.** `EngineRequest.north_side` says which grid edge faces
   geographic north. Without it the engine cannot tell north-east from
   top-right, and every Vastu rule is unexpressible — which is the state it
   was in until phase 01.
2. **Per-room directions.** Step 3 assigns each room a preferred and
   prohibited compass sector from `sources/enricher_rules.json`; the bridge
   now carries them onto `RoomSpec` instead of dropping them.
3. **Rules and a bias.** Eleven `VAS-*` reviewer rules score the carved plan,
   and `EngineConfig.vastu_bias` pulls the proposer's seeds toward each
   room's sector so the engine tries rather than only measuring.

Beyond room placement, `data/vastuRules1.json` (an 81-pada Paramasayika
Mandala) drives the 32-gate entrance check, the Brahmasthan protection, the
six marma diagonals, and the global structural modifiers.

```bash
python -m modules.diagnostics        # reports mandala health and any defects
```

Every run returns a per-room compliance scorecard — what was asked, what was
achieved, and what would have been better — under `vastu` in the run result.
A non-Vastu request is scored exactly as it was before any of this existed;
`tests/engine/test_vastu.py::TestNonVastuUnaffected` enforces that.

## The frozen reward

Stage (b) of the placer imitates plans the reviewer chose, and stage (c)
optimises the reviewer's score directly. Both only mean something if the
reward holds still — retune a weight or add a rule mid-training and every
number from before that moment quietly stops being comparable.

`modules/step4_generate/engine/reward_v1.json` is the snapshot: a hash over
the rule registry, every scoring weight, and the critic's identity, plus the
golden-harness numbers it was measured against.

```bash
python -m modules.step4_generate.engine.reward            # check for drift
python -m modules.step4_generate.engine.reward --freeze --with-baseline
```

Generation knobs (`settle_sweeps`, `cpsat_mode`, `vastu_bias`) are recorded
but deliberately NOT hashed: they change which plans get made, not how one is
judged, so a corpus stays valid across them. `selfplay`, `train` and
`rl_finetune` all check the reward on startup, stamp the fingerprint into
what they write, and take `--strict-reward` to refuse to run on drift.

## The two data directories

They are easy to confuse, so:

- **`data/`** — small, curated, **tracked**. Read at request time
  (`engine/priors.py` loads `zone_patterns.json` from here).
- **`ml/data/`** — gigabytes of raw corpora, **gitignored**, only ever touched
  by `ml/training/`. Nothing in the request path reads it.
