"""
make_notebook.py — generate the Colab training notebook.

The notebook is GENERATED rather than hand-maintained so it cannot drift from
the CLI it drives: every cell shells out to the same `python -m ml.placer_v3.*`
entry points this repo tests, so anything green here is green there.

    python -m ml.placer_v3.make_notebook          # -> colab_train_v3.ipynb
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

OUT_PATH = os.path.join(os.path.dirname(__file__), "colab_train_v3.ipynb")


def md(text: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> Dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def cells() -> List[Dict]:
    return [
        md(r"""
# PlanGen — Placer v3 training

Three stages, one notebook. **Every cell is safe to re-run** — training
resumes from the last Drive checkpoint, so a Colab disconnect costs you the
current epoch and nothing else.

| Stage | What it does | Why |
|---|---|---|
| **a** · pretrain | imitate CubiCasa5K | general spatial competence — rooms don't collide, wet rooms cluster |
| **b** · distil | imitate the engine's own best plans | crosses the Finnish→Indian domain gap |
| **c** · RL | optimise the reviewer's reward directly | lets the model *exceed* the hand-tuned proposer |

### What v3 changed, and why it matters here

- **R1 — the decoder can see the board.** v2 conditioned on the previous
  room's `(cell, size)` embeddings only, so it packed blind and the legality
  mask corrected it afterwards. Its confidence sat at **0.355** against a tau
  of **0.35** and it fell back on **41.8%** of proposals. Watch the `conf`
  column: it should climb well clear of 0.35.
- **R2 — the reward is the dataset.** There is no Indian floor-plan corpus,
  but there is an Indian-tuned reward (33 reviewer rules + a critic at AUC
  0.912). Stage b manufactures training data from it.
- **R3/R4 — aspect, orientation, band, Vastu and scale-regime** are inputs
  and outputs now, not things the carver has to guess.

### The reward is frozen

Stages (b) and (c) are graded by the reviewer + critic, hashed as
`reward_v1.json` (fingerprint `f1e8f7ebbe912ae2`, 44 rules, 39 weights).
Every training entry point checks it on startup and stamps it into what it
writes, so a corpus built under one reward can never be silently trained
against another. Pass `--strict-reward` to refuse to run on drift.

### Safety measures baked in

Atomic checkpoint writes · full optimizer/scheduler/AMP/RNG state · frozen
**eval key** (the bug that froze v2's best checkpoint at epoch 2) · verified
loads with fallback to the previous checkpoint · rotation so Drive can't
fill · append-only fsynced JSONL log.
"""),

        md("## 1 · GPU and environment"),
        code(r"""
!nvidia-smi || echo "NO GPU — set Runtime ▸ Change runtime type ▸ GPU"
import torch, platform
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} · {p.total_memory/1e9:.1f} GB")
print("python", platform.python_version())
"""),

        md("""
## 2 · Mount Drive

Everything durable lives under `DRIVE_ROOT`: checkpoints, logs, the self-play
corpus. Nothing important is written to Colab's local disk, which vanishes.
"""),
        code(r"""
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/plangen_v3'
os.makedirs(DRIVE_ROOT, exist_ok=True)

RUN_A  = f'{DRIVE_ROOT}/stage_a'
RUN_B  = f'{DRIVE_ROOT}/stage_b'
RUN_C  = f'{DRIVE_ROOT}/stage_c'
CORPUS = f'{DRIVE_ROOT}/corpus'
for d in (RUN_A, RUN_B, RUN_C, CORPUS):
    os.makedirs(d, exist_ok=True)

import shutil
free = shutil.disk_usage(DRIVE_ROOT).free / 1e9
print(f'Drive free: {free:.1f} GB')
if free < 4:
    print('  WARNING: a checkpoint is ~110 MB and the manager keeps '
          'last-3 + best + milestones. Under 4 GB, lower --milestone-every.')
"""),

        md("""
## 3 · Get the code

Either clone your repo, or upload a zip of the project root. The notebook
only needs `models.py`, `modules/`, `ml/`, `sources/` and `data/`.
"""),
        code(r"""
# ── OPTION A: clone (recommended — the repo has a remote) ─────────────
# !git clone https://github.com/Lakshminarayanan017/Plangen-.git /content/PlanGen

# ── OPTION B: upload a zip of the project root ────────────────────────
# from google.colab import files; files.upload()      # -> PlanGen.zip
# !unzip -q -o PlanGen.zip -d /content/

PROJECT = '/content/PlanGen'
import os, sys
assert os.path.isdir(PROJECT), f'{PROJECT} not found — use option A or B'
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
print(os.getcwd())
!ls
"""),

        md("## 4 · Dependencies"),
        code(r"""
!pip -q install numpy scipy pydantic python-dotenv ortools ijson
# torch is preinstalled on Colab; only install if the import above failed
import torch; print('torch ok', torch.__version__)
"""),

        md("""
## 5 · Verify the engine works here

The engine is the reward function for stages b and c, so it has to be healthy
before any training is meaningful. This runs the real test suite for the
engine and prints the subsystem self-check.
"""),
        code(r"""
!python -m modules.diagnostics
print()
!python -m unittest discover -s tests/engine -t . 2>&1 | tail -5
"""),

        md("""
## 6 · Data check

Stage **a** needs `ml/training/prepared/` (samples.jsonl + masks.npy). If you
did not upload it, regenerate it from `ml/data/normalized_extraction.json`.
"""),
        code(r"""
import os, json
PREP = 'ml/training/prepared'

# samples.jsonl (4 MB) and masks.npy (18 MB) are GITIGNORED, so a clone does
# NOT bring them. manifest.json IS tracked and carries the frozen val split,
# so it must match the data you upload.
missing = [f for f in ('samples.jsonl', 'masks.npy')
           if not os.path.exists(f'{PREP}/{f}')]
if missing:
    print(f'MISSING (gitignored, upload them): {missing}')
    print('Run the cell below, or copy them from Drive.')
else:
    man = json.load(open(f'{PREP}/manifest.json'))
    print(f"prepared: {man['n_prepared']} samples · {man['n_val']} val split")
"""),

        code(r"""
# Upload the two gitignored files (22 MB total) if the check above flagged them.
# Faster alternative: put them in Drive once and copy from there each session.
import os, shutil
PREP = 'ml/training/prepared'
DRIVE_PREP = f'{DRIVE_ROOT}/prepared'

if os.path.exists(f'{DRIVE_PREP}/samples.jsonl'):
    os.makedirs(PREP, exist_ok=True)
    for f in ('samples.jsonl', 'masks.npy'):
        shutil.copy(f'{DRIVE_PREP}/{f}', f'{PREP}/{f}')
    print('copied from Drive')
else:
    from google.colab import files
    print('Select ml/training/prepared/samples.jsonl and masks.npy')
    up = files.upload()
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(DRIVE_PREP, exist_ok=True)
    for name in up:
        shutil.move(name, f'{PREP}/{name}')
        shutil.copy(f'{PREP}/{name}', f'{DRIVE_PREP}/{name}')  # cache for next time
    print('uploaded and cached to Drive')
"""),

        md("""
## 7 · Dry run — validate the whole loop before spending GPU hours

20 items, 2 epochs, 4 eval briefs. If this passes, the loop, the checkpoint
manager and the engine evaluator all work. It takes a couple of minutes.
"""),
        code(r"""
!python -m ml.placer_v3.train --stage a --dry-run --out /content/dry_run
"""),

        md("""
## 8 · Stage (a) — imitation pretrain

**Re-run this cell after any disconnect.** It resumes from
`ckpt_last.pt` on Drive with optimizer, scheduler, AMP scaler and RNG state
intact, so the curve continues rather than restarting.

Columns to watch:
- **ENGINE** — mean soft score of plans carved from the model's proposals.
  This is what selects checkpoints, *never* the loss.
- **conf** — the 3×3 confidence. v2 plateaued at 0.355; this needs to clear
  0.35 comfortably or the deployed arm will just fall back.
- **asp / band** — stay 0.0000 here by design: CubiCasa has no carved
  geometry, so those heads are masked until stage b.
"""),
        code(r"""
EPOCHS_A = 70

# DECIDE --eval-n, --eval-seed AND --eval-k NOW AND NEVER CHANGE THEM.
# Together they form the eval key; changing one mid-run re-baselines
# best_score, which is exactly what froze v2's best checkpoint at epoch 2.
#
# The eval runs the REAL carver and is CPU-bound: ~5.9s per brief at k=2.
# n=24 every 3 epochs is ~2.3 min per eval, under an hour across 70
# epochs. (n=48 at k=4 every epoch would have been ~11.7 hours of eval.)
!python -m ml.placer_v3.train \
    --stage a \
    --out "$RUN_A" \
    --epochs {EPOCHS_A} \
    --preset small \
    --items-per-epoch 3000 \
    --lr 3e-4 \
    --accum 8 \
    --eval-n 24 \n    --eval-k 2 \n    --eval-every 3 \n    --milestone-every 10
"""),

        md("### Progress — is it still climbing?"),
        code(r"""
import json, os
import matplotlib.pyplot as plt

def curve(run_dir, title):
    path = f'{run_dir}/train_log.jsonl'
    if not os.path.exists(path):
        print(f'no log at {path}'); return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ep     = [r['epoch'] for r in rows]
    engine = [r.get('engine_score', 0) for r in rows]
    loss   = [r['train']['total'] for r in rows if 'train' in r]
    conf   = [r.get('model_confidence', 0) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(ep, engine, marker='o', ms=3); ax[0].set_title(f'{title} — ENGINE score')
    ax[0].axhline(75.49, ls='--', c='crimson', label='PriorProposer 75.49')
    ax[0].legend(); ax[0].set_xlabel('epoch')
    if loss:
        ax[1].plot(ep[:len(loss)], loss, marker='o', ms=3, c='seagreen')
    ax[1].set_title('train loss'); ax[1].set_xlabel('epoch')
    ax[2].plot(ep, conf, marker='o', ms=3, c='darkorange')
    ax[2].axhline(0.35, ls='--', c='crimson', label='tau 0.35')
    ax[2].set_title('confidence (3x3 mass)'); ax[2].legend(); ax[2].set_xlabel('epoch')
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    best = max(rows, key=lambda r: r.get('engine_score', 0))
    tail = [r.get('engine_score', 0) for r in rows[-10:]]
    print(f"best {best.get('engine_score')} @ epoch {best['epoch']}")
    if len(tail) >= 6:
        first, second = tail[:len(tail)//2], tail[len(tail)//2:]
        delta = sum(second)/len(second) - sum(first)/len(first)
        verdict = 'STILL CLIMBING — keep going' if delta > 0.5 else \
                  'FLAT — more epochs will not help; move to stage b'
        print(f'last-10 trend: {delta:+.2f}  ->  {verdict}')

curve(RUN_A, 'stage a')
"""),

        md("""
## 9 · Stage (b) — self-play corpus

This is the domain-gap crossing. It sweeps briefs across the real Indian plot
distribution, runs the **algorithmic engine** at k=16 per brief, scores every
candidate with the 33 reviewer rules blended with the learned critic, and
keeps the top 2.

Cost is CPU-bound (the carver), not GPU — Colab's 2 vCPUs make it slow here.

> **Run this on your own machine instead, in parallel with stage (a).**
> ```
> python -m ml.placer_v3.selfplay --briefs 4000 --k 16 --workers 8 >     --out ml/placer_v3/corpus
> ```
> Then upload `selfplay.jsonl` + `manifest.json` to `CORPUS` on Drive.
> Nothing about it needs a GPU, and it costs you no Colab time at all.

Fully resumable either way — re-run and it skips briefs already in the
corpus. The manifest records the reward fingerprint the corpus was graded by,
so a later reward change is detectable rather than silent.
"""),
        code(r"""
!python -m ml.placer_v3.selfplay \
    --briefs 4000 \
    --k 16 \
    --keep 2 \
    --min-reward 55 \
    --workers 2 \
    --out "$CORPUS"
"""),

        code(r"""
# corpus health — reward spread and program variety
import json, collections
recs = [json.loads(l) for l in open(f'{CORPUS}/selfplay.jsonl') if l.strip()]
print(f'{len(recs)} records from {len({r["brief"] for r in recs})} briefs')
rw = sorted(r['reward'] for r in recs)
if rw:
    print(f'reward  p10 {rw[len(rw)//10]:.1f}  median {rw[len(rw)//2]:.1f}  '
          f'p90 {rw[9*len(rw)//10]:.1f}  max {rw[-1]:.1f}')
print('rooms/plan :', collections.Counter(len(r['rooms']) for r in recs).most_common(6))
print('aspect     :', dict(sorted(collections.Counter(
    rm['aspect_class'] for r in recs for rm in r['rooms']).items())))
print('\naspect class 5 is the slab-shaped room the reviewer penalises.')
print('A large share there means the corpus is teaching a flaw — lower')
print('--keep or raise --min-reward and regenerate.')
"""),

        md("""
### Stage (b) — distillation training

Warm-starts from stage (a)'s best checkpoint. The aspect / orientation / band
heads start learning here, because these records carry real carved geometry.
"""),
        code(r"""
EPOCHS_B = 45
!python -m ml.placer_v3.train \
    --stage b \
    --out "$RUN_B" \
    --init "$RUN_A/ckpt_best.pt" \
    --corpus "$CORPUS" \
    --epochs {EPOCHS_B} \
    --preset small \
    --lr 1.5e-4 \
    --items-per-epoch 3000 \
    --reward-weighting \
    --eval-n 24 \n    --eval-k 2 \n    --eval-every 3 \n    --milestone-every 10
"""),
        code("curve(RUN_B, 'stage b')"),

        md("""
## 10 · Stage (c) — RL against the reviewer's reward

GRPO-shaped: sample a group of k layouts per brief, carve and score each one
for real, and push toward the ones that scored well — with a KL anchor to the
stage-(b) policy so it cannot drift into gaming the rules.

Throughput is bounded by the **carver**, not the GPU. Start small and read
the numbers before scaling up.

Watch for reward hacking: if `reward` climbs while `ENGINE` stalls or falls,
the policy is exploiting the reward's seams. Raise `--beta-kl`.
"""),
        code(r"""
!python -m ml.placer_v3.rl_finetune \
    --init "$RUN_B/ckpt_best.pt" \
    --out "$RUN_C" \
    --epochs 30 \
    --briefs-per-epoch 64 \
    --group 6 \
    --lr 2e-5 \
    --beta-kl 0.02 \
    --eval-n 24 \n    --eval-k 2
"""),
        code("curve(RUN_C, 'stage c (RL)')"),

        md("""
## 11 · Export and parity check

Production is torch-free — the server runs a pure-NumPy forward pass. This
exports the weights and asserts the two agree to <1e-4. v2's equivalent check
came out at 7e-7; anything worse means the export dropped a tensor.
"""),
        code(r"""
BEST = f'{RUN_C}/ckpt_best.pt'
import os
if not os.path.exists(BEST):
    BEST = f'{RUN_B}/ckpt_best.pt'
if not os.path.exists(BEST):
    BEST = f'{RUN_A}/ckpt_best.pt'
print('exporting', BEST)

!python -m ml.placer_v3.export --checkpoint "$BEST" --out "$DRIVE_ROOT/placer_v3.npz" --verify
"""),

        md("""
## 12 · The merge gate — does v3 replace the prior proposer?

The gate is unchanged from v2, deliberately: same briefs, same seeds, same
engine config, so the only variable is who proposes the seeds.

```
PASS iff  mean best score improves
     and  no single brief regresses by more than 5 points
     and  mean fidelity >= 0.80
     and  no brief loses its plan
```

v2 scored **68.57 against the prior proposer's 75.49** and failed on nine
briefs. Believe this result either way — that is what the gate is for.
"""),
        code(r"""
!python -m ml.placer_v3.merge_gate \
    --weights "$DRIVE_ROOT/placer_v3.npz" \
    --json "$DRIVE_ROOT/gate_v3.json" \
    --html "$DRIVE_ROOT/gate_v3.html"
"""),

        md("""
## Troubleshooting

**Disconnected mid-epoch.** Re-run the training cell. It resumes from
`ckpt_last.pt` with optimizer, scheduler, AMP and RNG state. You lose the
current epoch only.

**"EVAL SET CHANGED" on resume.** You changed `--eval-n`, `--eval-seed`
or `--eval-k` — all three form the eval key.
`best_score` is reset because scores from different eval sets are not
comparable — this is the exact bug that froze v2's best checkpoint at epoch
2. Either change it back, or accept the re-baseline.

**Checkpoint won't load.** The manager automatically falls back to the next
newest that does. If all fail, delete `ckpt_last.pt` and it will resume from
the newest milestone.

**Drive full.** Lower `--milestone-every` (or set it to 0) and `--keep-last`.
The manager prunes history automatically below 1.5 GB free, but prevention is
cheaper.

**OOM.** Lower `--items-per-epoch` first (it does not change the maths, only
the epoch length), then `--accum`. Use `--preset small`. Batch size is
effectively 1 plan with gradient accumulation, so OOM here means the model is
too big for the GPU, not the data.

**ENGINE score flat while loss falls.** Working as intended as a *warning*:
the model is getting better at imitating and no better at producing good
plans. That is the stage-(a) ceiling, and it is precisely why stages (b) and
(c) exist.
"""),
    ]


def build() -> Dict:
    return {
        "cells": cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1)
    n = len(build()["cells"])
    print(f"wrote {OUT_PATH} ({n} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
