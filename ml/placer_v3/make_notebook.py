"""
make_notebook.py — generate the Colab training notebook.

The notebook is GENERATED rather than hand-maintained so it cannot drift from
the CLI it drives: every cell shells out to the same `python -m ml.placer_v3.*`
entry points this repo tests, so anything green here is green there.

    python -m ml.placer_v3.make_notebook          # -> colab_train_v3.ipynb

WHY COMMANDS ARE BUILT IN PYTHON. The first version wrote them as
backslash-continued shell lines, and three of the four training cells shipped
with a literal two-character `\\n` where a newline was meant. Bash reads that
as the bare word `n`, so argparse died on "unrecognized arguments: n" — every
training cell was broken. Commands are now assembled by one helper, printed
before they run, and there is no continuation character left to get wrong.

WHAT THE CLONE SUPPLIES. The repo carries every training input:

    ml/training/prepared/samples.jsonl   4.1 MB   stage (a)
    ml/training/prepared/masks.npy      17.3 MB   stage (a)
    ml/placer_v3/corpus/selfplay.jsonl  13.2 MB   stage (b)

so `git clone` is the whole data step — nothing to upload, nothing to keep in
Drive. The corpus is tracked deliberately: it costs hours of carver time
across many cores and Colab gives you two vCPUs.

RESUMING AND CONTINUING. `train.py` loops
`range(state.epoch + 1, args.epochs + 1)`, so BOTH behaviours fall out of one
number: re-running after a disconnect resumes where it stopped, and raising
`--epochs` continues an already-finished run. The notebook exposes that
directly rather than wrapping it.

BEST WEIGHTS EVERY EPOCH. `ckpt_best.pt` is only rewritten when an engine
eval produces a new high score, and evals run every `--eval-every` epochs —
so best-per-epoch means `--eval-every 1`. Measured cost at n=24 k=2: 142.9 s
per eval, i.e. 2.78 h across 70 epochs against 0.95 h at every-3. That is the
real price and the config cell states it. `--eval-every` is NOT part of the
eval key (`stage:k:n:seed`), so changing it never re-baselines `best_score`.
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
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def cells() -> List[Dict]:
    return [
        md(r"""
# PlanGen — Placer v3 training

Three stages, one notebook, everything pulled from GitHub. **Run the cells in
order.** Every cell is safe to re-run.

| Stage | What it does | Why |
|---|---|---|
| **a** · pretrain | imitate CubiCasa5K | general spatial competence — rooms don't collide, wet rooms cluster |
| **b** · distil | imitate the engine's own best plans | crosses the Finnish→Indian domain gap |
| **c** · RL | optimise the reviewer's reward directly | lets the model *exceed* the hand-tuned proposer |

### Nothing to upload

Section 3 clones the repo and that is the entire data step:

| File | Size | Used by |
|---|---|---|
| `ml/training/prepared/samples.jsonl` | 4.1 MB | stage (a) |
| `ml/training/prepared/masks.npy` | 17.3 MB | stage (a) |
| `ml/placer_v3/corpus/selfplay.jsonl` | 13.2 MB | stage (b) |

The self-play corpus is committed on purpose — it costs hours of carver time
across many cores, and Colab gives you two vCPUs.

### Safety measures

- **Best weights after every epoch** — `EVAL_EVERY = 1`, so `ckpt_best.pt` is
  re-examined each epoch rather than every third.
- **Full resume** — `ckpt_last.pt` is written every epoch with optimizer,
  scheduler, AMP scaler and RNG state. A disconnect costs the current epoch.
- **Continue any time** — raise the epoch total and re-run the same cell.
  Section 8 does this in one line.
- Atomic write-fsync-rename · verified loads with fallback to the previous
  checkpoint · rotation so Drive cannot fill · append-only fsynced JSONL log ·
  frozen eval key (the bug that froze v2's best checkpoint at epoch 2).

### The reward is frozen

Stages (b) and (c) are graded by the reviewer + critic, hashed as
`reward_v1.json` (fingerprint `f1e8f7ebbe912ae2`, 44 rules, 39 weights). Every
entry point checks it on startup and stamps it into what it writes, so a
corpus built under one reward can never be silently trained against another.
"""),

        md("""
## 1 · GPU and environment

Set **Runtime ▸ Change runtime type ▸ T4 GPU** first. Without a GPU the
training steps still run, just far slower.
"""),
        code(r"""
!nvidia-smi || echo "NO GPU - set Runtime > Change runtime type > GPU"
import torch, platform
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} - {p.total_memory/1e9:.1f} GB")
print("python", platform.python_version())
"""),

        md("""
## 2 · Mount Drive

Checkpoints and logs live under `DRIVE_ROOT` and survive the session. Code and
data deliberately do **not** — they come from the clone, so you are always
training against what is actually committed.
"""),
        code(r"""
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/plangen_v3'
RUN_A = f'{DRIVE_ROOT}/stage_a'
RUN_B = f'{DRIVE_ROOT}/stage_b'
RUN_C = f'{DRIVE_ROOT}/stage_c'
for d in (DRIVE_ROOT, RUN_A, RUN_B, RUN_C):
    os.makedirs(d, exist_ok=True)

import shutil
free = shutil.disk_usage(DRIVE_ROOT).free / 1e9
print(f'Drive free: {free:.1f} GB')
if free < 4:
    print('  WARNING: a checkpoint is ~105 MB and the manager keeps '
          'last-3 + best + milestones. Under 4 GB, raise MILESTONE_EVERY.')
"""),

        md(f"""
## 3 · Clone the repo

Public repo, so no token is needed. Re-running is safe: if the clone is
already there it resets to the latest `{REPO_BRANCH}` instead of failing on a
non-empty directory. `--depth 1` keeps it quick — you need the current tree,
not the history.
"""),
        code(rf"""
import os, sys

REPO    = '{REPO_URL}'
BRANCH  = '{REPO_BRANCH}'
PROJECT = '/content/PlanGen'

if os.path.isdir(os.path.join(PROJECT, '.git')):
    print('already cloned - resetting to the latest', BRANCH)
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

Checked before a single GPU-hour is spent. If anything says MISSING, stop —
the later cells would fail in much less obvious ways than this one does.
"""),
        code(r"""
import os, json

CORPUS = 'ml/placer_v3/corpus'          # stage (b), from the clone
PREP   = 'ml/training/prepared'         # stage (a), from the clone

WANT = [
    (f'{PREP}/samples.jsonl',    'stage (a) CubiCasa samples'),
    (f'{PREP}/masks.npy',        'stage (a) legality masks'),
    (f'{PREP}/manifest.json',    'stage (a) frozen val split'),
    (f'{CORPUS}/selfplay.jsonl', 'stage (b) self-play corpus'),
    (f'{CORPUS}/manifest.json',  'stage (b) corpus provenance'),
    ('ml/placer_v3/train.py',    'stages (a) and (b)'),
    ('ml/placer_v3/rl_finetune.py', 'stage (c)'),
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
    print('\nSomething is missing from the clone. git says nothing about '
          'IGNORED files, so a file can look pushed when it never left your '
          'machine - check it against .gitignore.')
"""),

        md("""
## 4 · Dependencies

`torch` ships with Colab. The rest is what the engine needs, and the engine is
the reward function for stages (b) and (c), so it must import cleanly here.
"""),
        code(r"""
!pip -q install numpy scipy pydantic python-dotenv ortools ijson
import torch; print('torch ok', torch.__version__)
"""),

        md("""
## 5 · Verify the engine works here

The engine grades stages (b) and (c). If it is unhealthy, training is
measuring nothing.
"""),
        code(r"""
!python -m modules.diagnostics
print()
# grep for the verdict rather than tail: some tests legitimately print
# alarming-looking text (the reward-drift detector proves itself by
# triggering), and `tail` was cutting off the summary line that says OK.
!python -m unittest discover -s tests/engine -t . 2>&1 | grep -E "^Ran |^OK|^FAILED|^ERROR:|^FAIL:"
"""),

        md(r"""
## 6 · Configuration — set once, then leave alone

Everything the training cells use lives here, so there is one place to change
and one place to look.

### The eval key — decide now, never change

`--eval-n`, `--eval-seed` and `--eval-k` together identify *what is being
measured*. Change one mid-run and `best_score` is reset, because scores from
different eval sets are not comparable. This is the exact bug that froze v2's
best checkpoint at epoch 2, so the manager now shouts about it instead of
silently carrying on.

`EVAL_EVERY` is **not** part of that key — you can change it freely.

### What best-every-epoch costs

`ckpt_best.pt` is only rewritten when an engine eval finds a new high score,
so `EVAL_EVERY = 1` is what makes "best weights after every epoch" true. The
eval runs the real carver on CPU and is the slow part. Measured at n=24, k=2:

| `EVAL_EVERY` | per eval | across 70 epochs | best updated |
|---|---|---|---|
| **1** (default here) | 143 s | **2.8 h** | every epoch |
| 3 | 143 s | 0.95 h | every 3rd epoch |

If 2.8 h of eval is too much, drop `EVAL_N` to 16 **before you start** rather
than raising `EVAL_EVERY` — you keep per-epoch best and pay about a third
less. Just never change it once a run is under way.
"""),
        code(r"""
# ── the eval key: FIX THESE THREE NOW ────────────────────────────────
EVAL_N    = 24        # briefs per eval
EVAL_K    = 2         # candidates per brief
EVAL_SEED = 900000    # which briefs

# ── safe to change any time ──────────────────────────────────────────
EVAL_EVERY     = 1    # 1 = re-examine best weights EVERY epoch
MILESTONE_EVERY = 10  # keep a permanent ckpt_epochN.pt this often
KEEP_LAST      = 3    # rolling snapshots kept besides best/milestones
PRESET         = 'small'      # 9.11M params; 'large' is 25.95M
ITEMS_PER_EPOCH = 3000

# ── how far each stage runs (raise and re-run to continue) ───────────
TOTAL_EPOCHS_A = 70
TOTAL_EPOCHS_B = 45
TOTAL_EPOCHS_C = 30

print(f'eval key -> stage:k{EVAL_K}:n={EVAL_N}:seed={EVAL_SEED}')
print(f'best weights re-examined every {EVAL_EVERY} epoch(s)')
if MILESTONE_EVERY == 1:
    print('WARNING: MILESTONE_EVERY=1 keeps EVERY epoch permanently '
          '(~105 MB each). 70 epochs is ~7.3 GB and milestones are exempt '
          'from pruning.')
"""),

        md("""
### The training helper

One function builds every command, so the flags cannot drift between the
stage cells and the continue cells. It prints what it runs before running it.
"""),
        code(r"""
import json, os

def train_stage(stage, total_epochs, out, *, init=None, corpus=None,
                lr=3e-4, extra=''):
    '''Run (or continue) a training stage up to `total_epochs`.

    train.py loops `range(state.epoch + 1, total_epochs + 1)`, so this both
    RESUMES an interrupted run and CONTINUES a finished one - the only thing
    that decides which is whether `total_epochs` is above where it stopped.
    '''
    cmd = (f'python -m ml.placer_v3.train --stage {stage} --out "{out}"'
           f' --epochs {total_epochs} --preset {PRESET}'
           f' --items-per-epoch {ITEMS_PER_EPOCH} --lr {lr}'
           f' --eval-n {EVAL_N} --eval-k {EVAL_K} --eval-seed {EVAL_SEED}'
           f' --eval-every {EVAL_EVERY} --milestone-every {MILESTONE_EVERY}'
           f' --keep-last {KEEP_LAST}')
    if init:
        cmd += f' --init "{init}"'
    if corpus:
        cmd += f' --corpus "{corpus}"'
    if extra:
        cmd += ' ' + extra
    print(cmd)
    print('-' * 78)
    get_ipython().system(cmd)


def status(run_dir, label=''):
    '''Where a run got to, and what is on Drive for it.'''
    log = f'{run_dir}/train_log.jsonl'
    print(f'== {label or run_dir}')
    if os.path.exists(log):
        rows = [json.loads(l) for l in open(log) if l.strip()]
        if rows:
            last = rows[-1]
            scored = [r for r in rows if r.get('engine_score') is not None]
            best = max(scored, key=lambda r: r['engine_score'], default=None)
            print(f'   last epoch {last["epoch"]}')
            if best:
                print(f'   best ENGINE {best["engine_score"]:.2f} '
                      f'@ epoch {best["epoch"]}')
    else:
        print('   not started')
    if os.path.isdir(run_dir):
        for f in sorted(os.listdir(run_dir)):
            if f.endswith('.pt'):
                mb = os.path.getsize(f'{run_dir}/{f}') / 1e6
                print(f'   {mb:7.1f} MB  {f}')
    print()

print('helpers ready')
"""),

        md("""
## 7 · Dry run — validate the whole loop before spending GPU hours

20 items, 2 epochs, 4 eval briefs. If this passes, the training loop, the
checkpoint manager and the engine evaluator all work. A couple of minutes.
"""),
        code(r"""
!python -m ml.placer_v3.train --stage a --dry-run --out /content/dry_run
"""),

        md("""
## 8 · Stage (a) — imitation pretrain

**Re-run this cell after any disconnect.** It resumes from `ckpt_last.pt` on
Drive with optimizer, scheduler, AMP scaler and RNG state intact, so the
curve continues rather than restarting.

Columns to watch:
- **ENGINE** — mean soft score of plans carved from the model's proposals.
  This is what selects checkpoints, *never* the loss.
- **conf** — the 3×3 confidence. v2 plateaued at 0.355 against a tau of 0.35
  and fell back on 41.8% of proposals; this needs to clear 0.35 comfortably.
- **asp / band** — stay 0.0000 here **by design**: CubiCasa has no carved
  geometry, so those heads are masked until stage (b).
"""),
        code(r"""
train_stage('a', TOTAL_EPOCHS_A, RUN_A, lr=3e-4, extra='--accum 8')
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
    ax[0].plot(ep, engine, marker='o', ms=3)
    ax[0].set_title(f'{title} - ENGINE score')
    ax[0].axhline(75.49, ls='--', c='crimson', label='PriorProposer 75.49')
    ax[0].legend(); ax[0].set_xlabel('epoch')
    if loss:
        ax[1].plot(ep[:len(loss)], loss, marker='o', ms=3, c='seagreen')
    ax[1].set_title('train loss'); ax[1].set_xlabel('epoch')
    ax[2].plot(ep, conf, marker='o', ms=3, c='darkorange')
    ax[2].axhline(0.35, ls='--', c='crimson', label='tau 0.35')
    ax[2].set_title('confidence (3x3 mass)'); ax[2].legend()
    ax[2].set_xlabel('epoch')
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.show()

    scored = [r for r in rows if r.get('engine_score') is not None]
    if scored:
        best = max(scored, key=lambda r: r['engine_score'])
        print(f"best {best['engine_score']:.2f} @ epoch {best['epoch']}")
    tail = [r.get('engine_score', 0) for r in rows[-10:]]
    if len(tail) >= 6:
        first, second = tail[:len(tail)//2], tail[len(tail)//2:]
        delta = sum(second)/len(second) - sum(first)/len(first)
        verdict = 'STILL CLIMBING - keep going' if delta > 0.5 else \
                  'FLAT - more epochs will not help; move on'
        print(f'last-10 trend: {delta:+.2f}  ->  {verdict}')

curve(RUN_A, 'stage a')
status(RUN_A, 'stage a')
"""),

        md("""
### Continue stage (a) for more epochs

The loop runs `range(last_epoch + 1, total + 1)`, so continuing is just a
bigger number. `MORE = 20` picks up wherever it stopped and runs twenty more
epochs — the optimizer, scheduler and RNG all carry over, so the curve is
identical to having asked for that many in the first place.

Nothing happens if the total is already reached; it prints the summary and
exits. Safe to run twice.
"""),
        code(r"""
# OPT-IN. Left at 0 so the notebook is safe to run straight through:
# on a first pass stage (a) has just finished, and continuing would
# silently tack on more epochs nobody asked for.
MORE = 0        # <- set to 20 (or whatever) to continue, then re-run

log = f'{RUN_A}/train_log.jsonl'
done = 0
if os.path.exists(log):
    rows = [json.loads(l) for l in open(log) if l.strip()]
    done = rows[-1]['epoch'] if rows else 0

if not MORE:
    print(f'stage (a) is at epoch {done}. Set MORE above to continue.')
else:
    TOTAL_EPOCHS_A = done + MORE
    print(f'stage (a) at epoch {done} -> continuing to {TOTAL_EPOCHS_A}')
    train_stage('a', TOTAL_EPOCHS_A, RUN_A, lr=3e-4, extra='--accum 8')
"""),

        md("""
## 9 · Stage (b) — distillation on the self-play corpus

This is the domain-gap crossing. There is no Indian floor-plan dataset, but
there **is** an Indian-tuned reward, so the corpus was manufactured from it:
sweep briefs across the real Indian plot distribution, run the algorithmic
engine at k=16 per brief, score every candidate with the 44 reviewer rules
blended with the learned critic, keep the top 2.

The corpus came with the clone. Warm-starts from stage (a)'s best checkpoint,
and the aspect / orientation / band heads start learning here because these
records carry real carved geometry — expect `asp` and `band` to leave 0.0000.
"""),
        code(r"""
# corpus health - reward spread and program variety
import json, collections
recs = [json.loads(l) for l in open(f'{CORPUS}/selfplay.jsonl') if l.strip()]
print(f'{len(recs)} records from {len({r["brief"] for r in recs})} briefs')
rw = sorted(r['reward'] for r in recs)
if rw:
    print(f'reward  p10 {rw[len(rw)//10]:.1f}  median {rw[len(rw)//2]:.1f}  '
          f'p90 {rw[9*len(rw)//10]:.1f}  max {rw[-1]:.1f}')
print('rooms/plan :', collections.Counter(
    len(r['rooms']) for r in recs).most_common(6))
print('aspect     :', dict(sorted(collections.Counter(
    rm['aspect_class'] for r in recs for rm in r['rooms']).items())))
print()
print('aspect class 5 is the slab-shaped room the reviewer penalises.')
print('A large share there means the corpus is teaching a flaw - rebuild it')
print('with a lower --keep or a higher --min-reward (appendix).')
"""),
        code(r"""
# train.py skips a --init it cannot find WITHOUT complaining (train.py:355),
# so a missing stage (a) would quietly train this from random weights and
# you would only notice from the curve, hours later.
INIT_A = f'{RUN_A}/ckpt_best.pt'
assert os.path.exists(INIT_A), (
    'stage (a) has not produced a best checkpoint yet - run section 8 first. '
    'Stage (b) warm-starts from it; without it this trains from scratch and '
    'the three-stage design buys you nothing.')

train_stage('b', TOTAL_EPOCHS_B, RUN_B,
            init=INIT_A, corpus=CORPUS,
            lr=1.5e-4, extra='--reward-weighting')
"""),
        code(r"""
curve(RUN_B, 'stage b')
status(RUN_B, 'stage b')
"""),

        md("""
### Continue stage (b) for more epochs

Same mechanism as stage (a).
"""),
        code(r"""
# OPT-IN. Left at 0 so the notebook is safe to run straight through:
# on a first pass stage (b) has just finished, and continuing would
# silently tack on more epochs nobody asked for.
MORE = 0        # <- set to 15 (or whatever) to continue, then re-run

log = f'{RUN_B}/train_log.jsonl'
done = 0
if os.path.exists(log):
    rows = [json.loads(l) for l in open(log) if l.strip()]
    done = rows[-1]['epoch'] if rows else 0

if not MORE:
    print(f'stage (b) is at epoch {done}. Set MORE above to continue.')
else:
    TOTAL_EPOCHS_B = done + MORE
    print(f'stage (b) at epoch {done} -> continuing to {TOTAL_EPOCHS_B}')
    # --init only applies at epoch 0, so this continues stage (b)'s own
    # weights rather than restarting from stage (a)
    train_stage('b', TOTAL_EPOCHS_B, RUN_B,
                init=f'{RUN_A}/ckpt_best.pt', corpus=CORPUS,
                lr=1.5e-4, extra='--reward-weighting')
"""),

        md("""
## 10 · Stage (c) — RL against the reviewer's reward

GRPO-shaped: sample a group of k layouts per brief, carve and score each one
for real, and push toward the ones that scored well — with a KL anchor to the
stage-(b) policy so it cannot drift into gaming the rules.

Throughput is bounded by the **carver**, not the GPU.

Watch for reward hacking: if `reward` climbs while `ENGINE` stalls or falls,
the policy is exploiting the reward's seams. Raise `--beta-kl`.

`rl_finetune` resumes the same way, so re-running after a disconnect (or with
a larger `TOTAL_EPOCHS_C`) continues rather than restarting.
"""),
        code(r"""
assert os.path.exists(f'{RUN_B}/ckpt_best.pt'), (
    'stage (b) has not produced a best checkpoint yet - run section 9 first.')

CMD = (
    'python -m ml.placer_v3.rl_finetune'
    f' --init "{RUN_B}/ckpt_best.pt"'
    f' --out "{RUN_C}"'
    f' --epochs {TOTAL_EPOCHS_C}'
    ' --briefs-per-epoch 64'
    ' --group 6'
    ' --lr 2e-5'
    ' --beta-kl 0.02'
    f' --eval-n {EVAL_N}'
    f' --eval-k {EVAL_K}'
    f' --eval-seed {EVAL_SEED}'
)
print(CMD)
print('-' * 78)
!{CMD}
"""),
        code(r"""
curve(RUN_C, 'stage c (RL)')
status(RUN_C, 'stage c')
"""),

        md("""
## 11 · What is saved — the safety inventory

Everything durable, in one view. Run this any time you want to know what you
would still have if the session died right now.
"""),
        code(r"""
import shutil
for run, name in ((RUN_A, 'stage a'), (RUN_B, 'stage b'), (RUN_C, 'stage c')):
    status(run, name)
free = shutil.disk_usage(DRIVE_ROOT).free / 1e9
print(f'Drive free: {free:.1f} GB')
print()
print('ckpt_last.pt  - written EVERY epoch, full resume state')
print('ckpt_best.pt  - written when an eval sets a new high ENGINE score')
print(f'ckpt_epochN   - permanent, every {MILESTONE_EVERY} epochs, never pruned')
"""),

        md("""
## 12 · Export and parity check

Production is torch-free — the server runs a pure-NumPy forward pass. This
exports the weights and asserts the two agree to <1e-4. v2's equivalent check
came out at 7e-7; anything worse means the export dropped a tensor.

Picks the furthest stage that actually finished, so it works even if you
stopped after (a) or (b).
"""),
        code(r"""
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
print(CMD)
print('-' * 78)
!{CMD}
"""),

        md("""
## 13 · The merge gate — does v3 replace the prior proposer?

Unchanged from v2, deliberately: same briefs, same seeds, same engine config,
so the only variable is who proposes the seeds.

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
print(CMD)
print('-' * 78)
!{CMD}
"""),

        md("""
## Appendix — rebuilding the self-play corpus

Skip this unless the reward fingerprint changed or the health cell looked
wrong. It is CPU-bound on the carver and Colab's two vCPUs are the wrong
hardware. Build it on a real machine and commit the result:

```
python -m ml.placer_v3.selfplay --briefs 4000 --k 16 --workers 8 --out ml/placer_v3/corpus
git add ml/placer_v3/corpus && git commit -m "rebuild self-play corpus" && git push
```

Resumable either way — re-run and it skips briefs already present.
"""),
        code(r"""
REBUILD_CORPUS = False

if REBUILD_CORPUS:
    CORPUS = f'{DRIVE_ROOT}/corpus'      # Drive: a multi-hour run must not
    os.makedirs(CORPUS, exist_ok=True)   # die with the session
    CMD = ('python -m ml.placer_v3.selfplay --briefs 4000 --k 16 --keep 2'
           f' --min-reward 55 --workers 2 --out "{CORPUS}"')
    print(CMD)
    print('-' * 78)
    get_ipython().system(CMD)
else:
    print(f'using the corpus from the clone: {CORPUS}')
"""),

        md("""
## Troubleshooting

**Disconnected mid-epoch.** Re-run the stage cell. It resumes from
`ckpt_last.pt` with optimizer, scheduler, AMP and RNG state. You lose the
current epoch only.

**Session restarted.** Re-run sections 1–6 (quick), then the stage you were
on. The clone and the pip installs die with the container; Drive does not.

**I want more epochs.** Use the continue cell after the stage, or raise
`TOTAL_EPOCHS_*` in section 6 and re-run the stage cell. Same thing.

**"EVAL SET CHANGED" on resume.** You changed `EVAL_N`, `EVAL_SEED` or
`EVAL_K` — the three that form the eval key. `best_score` is reset because
scores from different eval sets are not comparable. Change it back, or accept
the re-baseline. `EVAL_EVERY` is not part of the key and is safe to change.

**Training is slow and it is the eval.** `EVAL_EVERY = 1` spends 143 s per
epoch on the real carver to keep best-weights current. Lowering `EVAL_N`
helps but changes the eval key, so only do it before a run starts.

**A file is MISSING in the section-3 check.** It is not committed. `git
status` says nothing about ignored files, so a file can look pushed when it
never left your machine — check it against `.gitignore`.

**Checkpoint won't load.** The manager falls back to the next newest that
does. If all fail, delete `ckpt_last.pt` and it resumes from the newest
milestone.

**Drive full.** Raise `MILESTONE_EVERY` and lower `KEEP_LAST`. Milestones are
exempt from pruning, so `MILESTONE_EVERY = 1` keeps every epoch forever —
~105 MB each, ~7.3 GB across 70 epochs.

**OOM.** Lower `ITEMS_PER_EPOCH` first (it changes epoch length, not the
maths), then `--accum`. Keep `PRESET = 'small'`. Batch size is effectively one
plan with gradient accumulation, so OOM means the model is too big for the
GPU, not the data.

**ENGINE flat while loss falls.** Working as intended as a *warning*: the
model is getting better at imitating and no better at producing good plans.
That is the stage-(a) ceiling, and precisely why stages (b) and (c) exist.
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
