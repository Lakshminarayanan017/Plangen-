"""
train.py — supervised training, stages (a) and (b).

    stage a   imitation pretrain on CubiCasa5K   -> general spatial competence
    stage b   distillation on self-play corpus   -> the Indian domain

Stage (c), RL against the engine reward, lives in `rl_finetune.py` because it
has a different inner loop (it runs the carver to get a gradient signal).

Three decisions worth stating, because each one is a v2 lesson:

  CHECKPOINTS ARE SELECTED BY THE ENGINE METRIC, NEVER BY LOSS. The loss is
  agreement with a Finnish centroid; the metric is whether the carved plan
  is good. Those come apart, and only one of them is the product.

  THE EVAL SET IS FROZEN AND KEYED. Changing --eval-n mid-run moved v2's
  mean by ~12 points for no modelling reason and froze its best checkpoint
  at epoch 2. The key is recorded in every checkpoint and a mismatch
  re-baselines loudly instead of silently.

  CELL LOSS IS SMOOTHED, SIZE/ASPECT/ORIENTATION/BAND ARE NOT. A seed one
  cell away is very nearly right; a bathroom labelled as a bedroom is not.

Run:
    python -m ml.placer_v3.train --stage a --epochs 60 --out runs/stage_a
    python -m ml.placer_v3.train --stage b --epochs 40 \
        --init runs/stage_a/ckpt_best.pt --corpus ml/placer_v3/corpus
    python -m ml.placer_v3.train --stage a --dry-run     # 20 items, 2 epochs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ml.placer_v3.checkpoint import CheckpointManager, TrainState, eval_key_for
from ml.placer_v3.config import (
    IGNORE_INDEX, PRESET_LARGE, PlacerV3Config, STATE_GRID,
)
from ml.placer_v3.dataset import PreparedDataset, SelfPlayDataset, describe
from ml.placer_v3.model.placer_net import PlacerNetV3

# sigma for the Gaussian spread of the cell target, in cells
CELL_SIGMA = 1.0


# ── losses ──────────────────────────────────────────────────────────────────

def gaussian_cell_targets(target_cell: torch.Tensor, grid: int = STATE_GRID,
                          sigma: float = CELL_SIGMA) -> torch.Tensor:
    """(N, grid*grid) soft targets: a Gaussian bump centred on the true cell.

    Being one cell out is nearly right, and a hard one-hot would punish it as
    hard as being across the plot. This is also why confidence is measured as
    3x3 mass rather than top-1 — a perfectly fitted model puts only ~0.159 on
    its best single cell BY DESIGN (see decode.PERFECT_TOP1).
    """
    device = target_cell.device
    rows = torch.arange(grid, device=device).view(1, grid, 1)
    cols = torch.arange(grid, device=device).view(1, 1, grid)
    tr = (target_cell // grid).view(-1, 1, 1)
    tc = (target_cell % grid).view(-1, 1, 1)
    d2 = (rows - tr) ** 2 + (cols - tc) ** 2
    w = torch.exp(-d2.float() / (2.0 * sigma ** 2))
    w = w.view(target_cell.size(0), -1)
    return w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _masked_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that skips IGNORE_INDEX rows and returns 0 if all rows
    are ignored — CubiCasa supplies no aspect/orientation/band targets, and
    a NaN there would poison the whole step."""
    valid = target != IGNORE_INDEX
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def compute_loss(out: Dict[str, torch.Tensor], arrays: Dict[str, torch.Tensor],
                 cfg: PlacerV3Config) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_p = F.log_softmax(out["cell"], dim=-1)
    soft = gaussian_cell_targets(arrays["target_cell"])
    cell = -(soft * log_p).sum(dim=-1).mean()

    size = F.cross_entropy(out["size"], arrays["target_size"])
    aspect = _masked_ce(out["aspect"], arrays["target_aspect"])
    orient = _masked_ce(out["orientation"], arrays["target_orientation"])
    band = _masked_ce(out["band"], arrays["target_band"])

    total = (cfg.w_cell * cell + cfg.w_size * size + cfg.w_aspect * aspect
             + cfg.w_orientation * orient + cfg.w_band * band)
    return total, {
        "cell": float(cell), "size": float(size), "aspect": float(aspect),
        "orientation": float(orient), "band": float(band),
        "total": float(total),
    }


# ── device plumbing ─────────────────────────────────────────────────────────

_LONG = ("type_ids", "zone_ids", "floor_ids", "vastu_dir_ids",
         "vastu_strength_ids", "edge_index", "target_cell", "target_size",
         "target_aspect", "target_orientation", "target_band")
_FLOAT = ("boundary", "global", "state")


def to_device(arrays: Dict, device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in _LONG:
        if key in arrays:
            out[key] = torch.as_tensor(np.asarray(arrays[key]),
                                       dtype=torch.long, device=device)
    for key in _FLOAT:
        if key in arrays:
            out[key] = torch.as_tensor(np.asarray(arrays[key]),
                                       dtype=torch.float32, device=device)
    out["n_rooms"] = arrays["n_rooms"]
    if "reward" in arrays:
        out["reward"] = arrays["reward"]
    return out


# ── engine evaluation (the metric that decides checkpoints) ────────────────

def engine_eval(net, n_briefs: int, seed: int, device: str,
                tau: float = 0.0) -> Dict[str, float]:
    """Run the model as a proposer through the real orchestrator.

    tau=0 means NEVER fall back: this measures the network, not the
    network-plus-safety-net. The deployed arm (with the real tau) is what the
    merge gate judges; this is what training should steer by, because a model
    that improves only by falling back has not improved.
    """
    from modules.step4_generate.engine.contracts import EngineConfig
    from modules.step4_generate.engine.orchestrator import Orchestrator
    from ml.placer_v3.proposer import PlacerV3Proposer
    from ml.placer_v3.selfplay import brief_at

    proposer = PlacerV3Proposer(net, tau=tau, device=device)
    orch = Orchestrator(config=EngineConfig(), proposer=proposer)

    scores, fidelities, planned = [], [], 0
    for i in range(n_briefs):
        request = brief_at(seed + i)
        request.k = 4
        try:
            result = orch.generate(request)
        except Exception:
            scores.append(0.0)
            continue
        if result.best:
            planned += 1
            scores.append(result.best.verdict.soft_score)
            fidelities.append(result.best.fidelity or 0.0)
        else:
            scores.append(0.0)
    return {
        "engine_score": round(float(np.mean(scores)) if scores else 0.0, 3),
        "fidelity": round(float(np.mean(fidelities)) if fidelities else 0.0,
                          3),
        "plan_rate": round(planned / max(n_briefs, 1), 3),
        "model_confidence": round(proposer.mean_confidence, 3),
    }


# ── training ────────────────────────────────────────────────────────────────

def build_datasets(args) -> Tuple[object, object]:
    if args.stage == "a":
        return (PreparedDataset(args.prepared, "train", augment=True,
                                max_rooms=args.max_rooms),
                PreparedDataset(args.prepared, "val", augment=False,
                                max_rooms=args.max_rooms))
    return (SelfPlayDataset(args.corpus, "train", augment=True,
                            min_reward=args.min_reward,
                            max_rooms=args.max_rooms),
            SelfPlayDataset(args.corpus, "val", augment=False,
                            min_reward=args.min_reward,
                            max_rooms=args.max_rooms))


def sample_weight(arrays: Dict, args) -> float:
    """Stage (b) weights each plan by how much the reviewer liked it, so the
    model leans toward the plans the Indian reward actually preferred rather
    than treating every kept candidate as equally exemplary."""
    if args.stage != "b" or not args.reward_weighting:
        return 1.0
    r = float(arrays.get("reward", 0.0))
    return float(np.clip((r - args.reward_baseline) / args.reward_scale + 1.0,
                         0.4, 1.8))


def run_epoch(net, ds, optimizer, scaler, args, device, train: bool,
              order: List[int]) -> Dict[str, float]:
    net.train(train)
    totals: Dict[str, float] = {}
    n_seen = 0
    accum = max(1, args.accum)

    for step, idx in enumerate(order):
        arrays = to_device(ds[idx], device)
        if arrays["n_rooms"] < 2:
            continue
        weight = sample_weight(ds[idx], args)

        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                            enabled=args.amp and "cuda" in device):
            out = net(arrays)
            loss, parts = compute_loss(out, arrays, net.cfg)
            loss = loss * weight

        if train:
            scaled = loss / accum
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            if (step + 1) % accum == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(),
                                               args.clip_grad)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n_seen += 1

    return {k: round(v / max(n_seen, 1), 4) for k, v in totals.items()}


def make_scheduler(optimizer, total_steps: int, warmup: int):
    """Linear warmup then cosine decay to 5% of peak. Warmup matters here:
    the state encoder starts from noise and a cold high LR drives the cell
    head to a uniform distribution it takes several epochs to leave."""
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def main(argv=None) -> int:
    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()

    p = argparse.ArgumentParser(prog="ml.placer_v3.train")
    p.add_argument("--stage", choices=["a", "b"], default="a")
    p.add_argument("--out", default="runs/placer_v3")
    p.add_argument("--prepared", default=None)
    p.add_argument("--corpus", default=os.path.join(
        os.path.dirname(__file__), "corpus"))
    p.add_argument("--init", default=None,
                   help="checkpoint to warm-start from (stage b uses a's best)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--items-per-epoch", type=int, default=3000)
    p.add_argument("--max-rooms", type=int, default=None)
    p.add_argument("--preset", choices=["small", "large"], default="small")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--eval-n", type=int, default=48)
    p.add_argument("--eval-seed", type=int, default=900000)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--min-reward", type=float, default=0.0)
    p.add_argument("--reward-weighting", action="store_true", default=True)
    p.add_argument("--reward-baseline", type=float, default=65.0)
    p.add_argument("--reward-scale", type=float, default=15.0)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--milestone-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--strict-reward", action="store_true",
                   help="refuse to train if the reward has drifted from the "
                        "frozen snapshot")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.prepared is None:
        from ml.training.paths import PREPARED_DIR
        args.prepared = PREPARED_DIR
    if args.dry_run:
        args.epochs, args.items_per_epoch, args.eval_n = 2, 20, 4
        args.warmup_steps = 5
        print(">>> DRY RUN — 20 items, 2 epochs, 4 eval briefs\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Stage (b) imitates plans the reward chose and stage (c) optimises it
    # directly, so a drifted reward silently invalidates both.
    from modules.step4_generate.engine import reward as reward_mod
    reward_drift = reward_mod.require(strict=args.strict_reward)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PRESET_LARGE if args.preset == "large" else PlacerV3Config()
    net = PlacerNetV3(cfg).to(device)
    print(f"device {device} · preset {args.preset} · "
          f"{net.num_params() / 1e6:.2f}M params")

    train_ds, val_ds = build_datasets(args)
    print(" ", describe(train_ds))
    print(" ", describe(val_ds))
    if len(train_ds) == 0:
        raise SystemExit("empty training set — check --prepared / --corpus")

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, min(args.items_per_epoch, len(train_ds))
                          // max(1, args.accum))
    scheduler = make_scheduler(optimizer, steps_per_epoch * args.epochs,
                               args.warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    manager = CheckpointManager(args.out, keep_last=args.keep_last,
                                milestone_every=args.milestone_every)
    key = eval_key_for(f"selfplay_briefs_stage{args.stage}", args.eval_n,
                       args.eval_seed)
    state = manager.resume(model=net, optimizer=optimizer,
                           scheduler=scheduler, scaler=scaler, eval_key=key)
    state.stage = f"stage_{args.stage}"

    if state.epoch == 0 and args.init and os.path.exists(args.init):
        payload = torch.load(args.init, map_location=device,
                             weights_only=False)
        missing, unexpected = net.load_state_dict(payload["model"],
                                                  strict=False)
        print(f"  warm-started from {args.init} "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    rng = random.Random(args.seed)
    print(f"\n{'ep':>4} {'loss':>8} {'cell':>7} {'size':>7} {'asp':>7} "
          f"{'band':>7} {'ENGINE':>8} {'fid':>6} {'plan%':>6} {'conf':>6} "
          f"{'lr':>9} {'min':>6}")
    print("-" * 96)

    for epoch in range(state.epoch + 1, args.epochs + 1):
        t0 = time.perf_counter()
        order = list(range(len(train_ds)))
        rng.shuffle(order)
        order = order[:args.items_per_epoch]

        train_parts = run_epoch(net, train_ds, optimizer, scaler, args,
                                device, True, order)
        scheduler.step()

        metrics = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                metrics = engine_eval(net, args.eval_n, args.eval_seed,
                                      device)

        score = metrics.get("engine_score", float("-inf"))
        is_best = score > state.best_score
        if is_best:
            state.best_score, state.best_epoch = score, epoch

        state.epoch = epoch
        row = {"epoch": epoch, "stage": state.stage,
               "lr": round(scheduler.get_last_lr()[0], 8),
               "minutes": round((time.perf_counter() - t0) / 60.0, 2),
               "train": train_parts, **metrics, "is_best": bool(is_best)}
        state.history.append(row)
        manager.append_log(row)
        manager.save(model=net, optimizer=optimizer, scheduler=scheduler,
                     scaler=scaler, state=state,
                     config={**cfg.to_dict(),
                             "reward_fingerprint": reward_mod.fingerprint(),
                             "reward_matches_frozen": reward_drift.matches},
                     is_best=is_best)

        print(f"{epoch:>4} {train_parts['total']:>8.4f} "
              f"{train_parts['cell']:>7.4f} {train_parts['size']:>7.4f} "
              f"{train_parts['aspect']:>7.4f} {train_parts['band']:>7.4f} "
              f"{score if score > -1e9 else 0:>8.2f} "
              f"{metrics.get('fidelity', 0):>6.3f} "
              f"{metrics.get('plan_rate', 0):>6.2f} "
              f"{metrics.get('model_confidence', 0):>6.3f} "
              f"{scheduler.get_last_lr()[0]:>9.2e} "
              f"{row['minutes']:>6.1f}" + ("  <- best" if is_best else ""))

    print("-" * 96)
    print(f"best engine score {state.best_score:.2f} at epoch "
          f"{state.best_epoch}\nbest weights -> {manager.best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
