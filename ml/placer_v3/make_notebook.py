"""
make_notebook.py — generate the Colab training notebook.

The notebook is GENERATED rather than hand-maintained so it cannot drift from
the CLI it drives: every cell shells out to the same `python -m ml.placer_v3.*`
entry points this repo tests, so anything green here is green there.

    python -m ml.placer_v3.make_notebook          # -> colab_train_v3.ipynb

WHY EVERY SHELL COMMAND IS BUILT IN PYTHON. The first version wrote them as
backslash-continued shell lines, and three of the four training cells shipped
with a literal two-character `\\n` where a newline was meant. Bash reads that
as the bare word `n`, so argparse died on "unrecognized arguments: n" — every
training cell in the notebook was broken. Commands are now assembled as a
single-line Python string and run with `!{CMD}`: there is no continuation
character to get wrong, and the cell prints exactly what it is about to run.

WHAT THE CLONE SUPPLIES. The repo carries every training input:

    ml/training/prepared/samples.jsonl   4.3 MB   stage (a)
    ml/training/prepared/masks.npy        18 MB   stage (a)
    ml/placer_v3/corpus/selfplay.jsonl    13 MB   stage (b)

so `git clone` is the whole data step. The corpus is tracked deliberately —
it costs hours of carver time across many cores to rebuild and Colab has two
vCPUs, which is the wrong machine for it.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

OUT_PATH = os.path.join(os.path.dirname(__file__), "colab_train_v3.ipynb")

REPO_URL = "https://github.com/Lakshminarayanan017/Plangen-.git"
REPO_BRANCH = "main"


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

Three stages, one notebook. **Run the cells in order.** Every cell is safe to
re-run — training resumes from the last Drive checkpoint, so a Colab
disconnect costs you the current epoch and nothing else.

| Stage | What it does | Why |
|---|---|---|
| **a** · pretrain | imitate CubiCasa5K | general spatial competence — rooms don't collide, wet rooms cluster |
| **b** · distil | imitate the engine's own best plans | crosses the Finnish→Indian domain gap |
| **c** · RL | optimise the reviewer's reward directly | lets the model *exceed* the hand-tuned proposer |

### Everything comes from the clone

Cell 3 clones the repo, and the repo carries all three training inputs:

| File | Size | Used by |
|---|---|---|
| `ml/training/prepared/samples.jsonl` | 4.3 MB | stage (a) |
| `ml/training/prepared/masks.npy` | 18 MB | stage (a) |
| `ml/placer_v3/corpus/selfplay.jsonl` | 13 MB | stage (b) |

There is **nothing to upload**. The self-play corpus was built on a real
machine (hours of carver time across many cores) and committed, because Colab
gives you two vCPUs and that is the wrong hardware to rebuild it on.

### What v3 changed, and why it matters here

- **R1 — the decoder can see the board.** v2 conditioned on the previous
  room's `(cell, size)` embeddings only, so it packed blind and the legality
  mask corrected it afterwards. Its confidence sat at **0.355** against a tau
  of **0.35** and it fell back on **41.8%** of proposals. Watch the `conf`
  column: it should climb well clear of 0.35.
- **R2 — the reward is the dataset.** There is no Indian floor-plan corpus,
  but there is an Indian-tuned reward (44 reviewer rules + a critic at AUC
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

        md("""
## 1 · GPU and environment

Set **Runtime ▸ Change runtime type ▸ T4 GPU** before running anything. If
this prints no GPU, stages (a) and (b) will still run but roughly 20× slower.
"""),
        code(r"""
!nvidia-smi || echo "NO GPU — set Runtime > Change runtime type > GPU"
import torch, platform
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} - {p.total_memory/1e9:.1f} GB")
print("python", platform.python_version())
"""),

        md("""
## 2 · Mount Drive

Checkpoints and logs live under `DRIVE_ROOT`. Nothing durable is written to
Colab's local disk, which vanishes when the session ends.

The *code and data* deliberately do **not** live here — they come from the
clone in the next cell, so you are always training against what is actually
committed rather than a stale copy someone uploaded months ago.
"""),
        code(r"""
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/plangen_v3'
os.makedirs(DRIVE_ROOT, exist_ok=True)

RUN_A = f'{DRIVE_ROOT}/stage_a'
RUN_B = f'{DRIVE_ROOT}/stage_b'
RUN_C = f'{DRIVE_ROOT}/stage_c'
for d in (RUN_A, RUN_B, RUN_C):
    os.makedirs(d, exist_ok=True)

import shutil
free = shutil.disk_usage(DRIVE_ROOT).free / 1e9
print(f'Drive free: {free:.1f} GB')
if free < 4:
    print('  WARNING: a checkpoint is ~110 MB and the manager keeps '
          'last-3 + best + milestones. Under 4 GB, lower --milestone-every.')
"""),

        md(f"""
## 3 · Clone the repo

Public repo, so no token is needed. Re-running this cell is safe: if the
clone is already there it fast-forwards to the latest `{REPO_BRANCH}` instead
of failing on a non-empty directory.

`--depth 1` keeps it quick — you need the current tree, not the history.
"""),
        code(rf"""
import os, sys

REPO    = '{REPO_URL}'
BRANCH  = '{REPO_BRANCH}'
PROJECT = '/content/PlanGen'

if os.path.isdir(os.path.join(PROJECT, '.git')):
    print('already cloned - fetching the latest', BRANCH)
    !git -C {{PROJECT}} fetch --depth 1 origin {{BRANCH}} && git -C {{PROJECT}} reset --hard origin/{{BRANCH}}
else:
    !git clone --depth 1 --branch {{BRANCH}} {{REPO}} {{PROJECT}}

assert os.path.isdir(PROJECT), f'clone failed - {{PROJECT}} does not exist'
os.chdir(PROJECT)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

print()
print('cwd:', os.getcwd())
!git log --oneline -1
"""),

        md("""
### What the clone brought

Every training input, checked before a single GPU-hour is spent. If any line
below says MISSING, stop here — the later cells will fail in less obvious
ways than this one does.
"""),
        code(r"""
import os, json

CORPUS = 'ml/placer_v3/corpus'          # stage (b), from the clone
PREP   = 'ml/training/prepared'         # stage (a), from the clone

WANT = [
    (f'{PREP}/samples.jsonl',   'stage (a) CubiCasa samples'),
    (f'{PREP}/masks.npy',       'stage (a) legality masks'),
    (f'{PREP}/manifest.json',   'stage (a) frozen val split'),
    (f'{CORPUS}/selfplay.jsonl', 'stage (b) self-play corpus'),
    (f'{CORPUS}/manifest.json',  'stage (b) corpus provenance'),
    ('modules/step4_generate/engine/reward_v1.json', 'frozen reward'),
]

ok = True
for path, what in WANT:
    if os.path.exists(path):
        print(f'  {os.path.getsize(path)/1e6:8.2f} MB  {path}')
    else:
        ok = False
        print(f'   MISSING            {path}   <- {what}')

if ok:
    man = json.load(open(f'{PREP}/manifest.json'))
    print(f"\nstage (a): {man['n_prepared']} samples, {man['n_val']} held out")
    cman = json.load(open(f'{CORPUS}/manifest.json'))
    print(f"stage (b): {cman['records']} records from {cman['briefs_swept']} "
          f"briefs (k={cman['k']}, keep={cman['keep']}, "
          f"min_reward={cman['min_reward']})")
    print(f"           graded by reward {cman.get('reward_fingerprint', '?')}")
else:
    print('\nSomething is missing from the clone. Re-run cell 3, and check '
          'the file is actually committed (it may be gitignored).')
"""),

        md("""
## 4 · Dependencies

`torch` is preinstalled on Colab. The rest is what the engine needs — the
engine is the reward function, so it has to import cleanly here.
"""),
        code(r"""
!pip -q install numpy scipy pydantic python-dotenv ortools ijson
import torch; print('torch ok', torch.__version__)
"""),

        md("""
## 5 · Verify the engine works here

The engine grades stages (b) and (c), so it must be healthy before any
training is meaningful. This runs the real engine test suite and the
subsystem self-check.
"""),
        code(r"""
!python -m modules.diagnostics
print()
# grep for the verdict rather than tail: some tests legitimately print
# alarming-looking text (the reward-drift detector proves itself by
# triggering), and `tail` was cutting off the summary line that says OK.
!python -m unittest discover -s tests/engine -t . 2>&1 | grep -E "^Ran |^OK|^FAILED|^ERROR:|^FAIL:"
"""),

        md("""
## 6 · Dry run — validate the whole loop before spending GPU hours

20 items, 2 epochs, 4 eval briefs. If this passes, the training loop, the
checkpoint manager and the engine evaluator all work. A couple of minutes.
"""),
        code(r"""
!python -m ml.placer_v3.train --stage a --dry-run --out /content/dry_run
"""),

        md("""
## 7 · Stage (a) — imitation pretrain

**Re-run this cell after any disconnect.** It resumes from `ckpt_last.pt` on
Drive with optimizer, scheduler, AMP scaler and RNG state intact, so the
curve continues rather than restarting.

Columns to watch:
- **ENGINE** — mean soft score of plans carved from the model's proposals.
  This is what selects checkpoints, *never* the loss.
- **conf** — the 3×3 confidence. v2 plateaued at 0.355; this needs to clear
  0.35 comfortably or the deployed arm will just fall back to the prior.
- **asp / band** — stay 0.0000 here by design: CubiCasa has no carved
  geometry, so those heads are masked until stage (b).

> **Decide `--eval-n`, `--eval-seed` and `--eval-k` now and never change
> them.** Together they form the eval key; changing one mid-run re-baselines
> `best_score`, which is exactly what froze v2's best checkpoint at epoch 2.
> The eval runs the real carver and is CPU-bound (~5.9 s per brief at k=2),
> so n=24 every 3 epochs is ~2.3 min per eval — under an hour across 70
> epochs. n=48 at k=4 every epoch would have been ~11.7 hours of *eval*.
"""),
        code(r"""
EPOCHS_A = 70

CMD = (
    'python -m ml.placer_v3.train'
    ' --stage a'
    f' --out "{RUN_A}"'
    f' --epochs {EPOCHS_A}'
    ' --preset small'
    ' --items-per-epoch 3000'
    ' --lr 3e-4'
    ' --accum 8'
    ' --eval-n 24'
    ' --eval-k 2'
    ' --eval-every 3'
    ' --milestone-every 10'
)
print(CMD, '\n' + '-' * 72)
!{CMD}
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
    if not rows:
        print('log is empty'); return
    ep     = [r['epoch'] for r in rows]
    engine = [r.get('engine_score', 0) for r in rows]
    loss   = [r['train']['total'] for r in rows if 'train' in r]
    conf   = [r.get('model_confidence', 0) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(ep, engine, marker='o', ms=3); ax[0].set_title(f'{title} - ENGINE score')
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
        verdict = 'STILL CLIMBING - keep going' if delta > 0.5 else \
                  'FLAT - more epochs will not help; move to stage b'
        print(f'last-10 trend: {delta:+.2f}  ->  {verdict}')

curve(RUN_A, 'stage a')
"""),

        md("""
## 8 · Stage (b) — distillation on the self-play corpus

This is the domain-gap crossing. There is no Indian floor-plan dataset, but
there **is** an Indian-tuned reward, so the corpus was manufactured from it:
sweep briefs across the real Indian plot distribution, run the algorithmic
engine at k=16 per brief, score every candidate with the 44 reviewer rules
blended with the learned critic, keep the top 2.

**The corpus is already in the clone** — cell 3 brought it and the check
after it printed the record count. You do not need to build one.

Warm-starts from stage (a)'s best checkpoint. The aspect / orientation / band
heads start learning here, because these records carry real carved geometry.
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
print()
print('aspect class 5 is the slab-shaped room the reviewer penalises.')
print('A large share there means the corpus is teaching a flaw - rebuild it')
print('with a lower --keep or a higher --min-reward.')
"""),
        code(r"""
EPOCHS_B = 45

CMD = (
    'python -m ml.placer_v3.train'
    ' --stage b'
    f' --out "{RUN_B}"'
    f' --init "{RUN_A}/ckpt_best.pt"'
    f' --corpus "{CORPUS}"'
    f' --epochs {EPOCHS_B}'
    ' --preset small'
    ' --lr 1.5e-4'
    ' --items-per-epoch 3000'
    ' --reward-weighting'
    ' --eval-n 24'
    ' --eval-k 2'
    ' --eval-every 3'
    ' --milestone-every 10'
)
print(CMD, '\n' + '-' * 72)
!{CMD}
"""),
        code("curve(RUN_B, 'stage b')"),

        md("""
## 9 · Stage (c) — RL against the reviewer's reward

GRPO-shaped: sample a group of k layouts per brief, carve and score each one
for real, and push toward the ones that scored well — with a KL anchor to the
stage-(b) policy so it cannot drift into gaming the rules.

Throughput is bounded by the **carver**, not the GPU. Start small and read
the numbers before scaling up.

Watch for reward hacking: if `reward` climbs while `ENGINE` stalls or falls,
the policy is exploiting the reward's seams. Raise `--beta-kl`.
"""),
        code(r"""
CMD = (
    'python -m ml.placer_v3.rl_finetune'
    f' --init "{RUN_B}/ckpt_best.pt"'
    f' --out "{RUN_C}"'
    ' --epochs 30'
    ' --briefs-per-epoch 64'
    ' --group 6'
    ' --lr 2e-5'
    ' --beta-kl 0.02'
    ' --eval-n 24'
    ' --eval-k 2'
)
print(CMD, '\n' + '-' * 72)
!{CMD}
"""),
        code("curve(RUN_C, 'stage c (RL)')"),

        md("""
## 10 · Export and parity check

Production is torch-free — the server runs a pure-NumPy forward pass. This
exports the weights and asserts the two agree to <1e-4. v2's equivalent check
came out at 7e-7; anything worse means the export dropped a tensor.

Picks the furthest stage that actually finished, so it works even if you
stopped after (a) or (b).
"""),
        code(r"""
import os

BEST = next((p for p in (f'{RUN_C}/ckpt_best.pt',
                         f'{RUN_B}/ckpt_best.pt',
                         f'{RUN_A}/ckpt_best.pt') if os.path.exists(p)), None)
assert BEST, 'no checkpoint yet - run at least stage (a)'
print('exporting', BEST)

CMD = (
    'python -m ml.placer_v3.export'
    f' --checkpoint "{BEST}"'
    f' --out "{DRIVE_ROOT}/placer_v3.npz"'
    ' --verify'
)
print(CMD, '\n' + '-' * 72)
!{CMD}
"""),

        md("""
## 11 · The merge gate — does v3 replace the prior proposer?

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
CMD = (
    'python -m ml.placer_v3.merge_gate'
    f' --weights "{DRIVE_ROOT}/placer_v3.npz"'
    f' --json "{DRIVE_ROOT}/gate_v3.json"'
    f' --html "{DRIVE_ROOT}/gate_v3.html"'
)
print(CMD, '\n' + '-' * 72)
!{CMD}
"""),

        md("""
## Appendix — rebuilding the self-play corpus

Skip this unless the reward changed (the check in cell 3 prints the
fingerprint the corpus was graded by) or the health cell above looks wrong.

It is CPU-bound on the carver, not the GPU, and Colab's two vCPUs make it
slow. Build it on a real machine and commit the result instead:

```
python -m ml.placer_v3.selfplay --briefs 4000 --k 16 --workers 8 \
    --out ml/placer_v3/corpus
git add ml/placer_v3/corpus && git commit -m "rebuild self-play corpus" && git push
```

Fully resumable either way — re-run and it skips briefs already present.
"""),
        code(r"""
# Only if you really need to rebuild in Colab. Writes to Drive, because a
# multi-hour run must not die with the session.
REBUILD_CORPUS = False

if REBUILD_CORPUS:
    CORPUS = f'{DRIVE_ROOT}/corpus'
    os.makedirs(CORPUS, exist_ok=True)
    CMD = (
        'python -m ml.placer_v3.selfplay'
        ' --briefs 4000'
        ' --k 16'
        ' --keep 2'
        ' --min-reward 55'
        ' --workers 2'
        f' --out "{CORPUS}"'
    )
    print(CMD, '\n' + '-' * 72)
    !{CMD}
else:
    print(f'using the corpus from the clone: {CORPUS}')
"""),

        md("""
## Troubleshooting

**Disconnected mid-epoch.** Re-run the training cell. It resumes from
`ckpt_last.pt` with optimizer, scheduler, AMP and RNG state. You lose the
current epoch only.

**Session restarted — do I re-run everything?** Re-run cells 1–5 (they are
quick), then the training cell you were on. The clone and the pip installs
are gone with the container; Drive checkpoints are not.

**"EVAL SET CHANGED" on resume.** You changed `--eval-n`, `--eval-seed` or
`--eval-k` — all three form the eval key. `best_score` is reset because
scores from different eval sets are not comparable. This is the exact bug
that froze v2's best checkpoint at epoch 2. Either change it back, or accept
the re-baseline.

**A file is MISSING in the cell-3 check.** It is not committed. Check it
against `.gitignore` on your machine — `git status` stays silent about
ignored files, so a file can look pushed when it never left your laptop.

**Checkpoint won't load.** The manager automatically falls back to the next
newest that does. If all fail, delete `ckpt_last.pt` and it resumes from the
newest milestone.

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
    nb = build()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {OUT_PATH} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
