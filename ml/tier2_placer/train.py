"""
train.py — train the Tier-2 placer; select checkpoints by ENGINE metrics.

The v1 lesson: loss went down while the plans were garbage, because loss
measured the wrong thing. So the best checkpoint here is the one whose
sampled proposals score best when run through the REAL engine
(realize→settle→connect→review), never the one with the lowest loss. Loss
is logged; it decides nothing.

Loss = cross-entropy on the seed cell with Gaussian spatial label smoothing
(σ=1 cell — a near-miss gets partial credit) + cross-entropy on the size
class. Curriculum: small plans first. Data is ×8 augmented (dataset.py).

Runs on CPU (dev sanity, tiny subsets) or Colab GPU (full). Resumable;
saves the best .pt and exports .npz (production NumPy weights).

    python -m tier2_placer.train --epochs 40 --batch 16 --eval-every 2
    python -m tier2_placer.train --smoke        # 1 epoch, few plans (CI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

try:                                    # live progress bar (Colab has it)
    from tqdm.auto import tqdm
except Exception:                       # fallback: plain text heartbeat
    tqdm = None
import torch.nn.functional as F

from modules.step4_generate.engine.contracts import SEED_GRID, EngineConfig, EngineRequest, RoomSpec
from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.tier2_placer.config import PlacerConfig
from ml.tier2_placer.dataset import PreparedDataset
from ml.tier2_placer.model.placer_net import PlacerNet
from ml.tier2_placer.tier2_placer import Tier2Placer
from ml.tier2_placer.torch_infer import TorchPlacer

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


# ── loss ─────────────────────────────────────────────────────────────────────

def gaussian_soft_targets(target_cells: torch.Tensor, sigma: float = 1.0,
                          grid: int = SEED_GRID) -> torch.Tensor:
    """(N,) cell indices -> (N, grid*grid) Gaussian soft labels around each
    target cell (σ in cells). Near-misses get partial credit."""
    n = target_cells.size(0)
    dev = target_cells.device                       # build the grid where the
    rows = (target_cells // grid).float()           # targets already live
    cols = (target_cells % grid).float()
    gr = torch.arange(grid, dtype=torch.float32, device=dev).view(1, grid, 1)
    gc = torch.arange(grid, dtype=torch.float32, device=dev).view(1, 1, grid)
    d2 = (gr - rows.view(n, 1, 1)) ** 2 + (gc - cols.view(n, 1, 1)) ** 2
    soft = torch.exp(-d2 / (2 * sigma * sigma)).view(n, grid * grid)
    return soft / soft.sum(dim=1, keepdim=True)


def plan_loss(net: PlacerNet, arrays: Dict[str, torch.Tensor],
              sigma: float) -> torch.Tensor:
    cell_logits, size_logits = net(arrays)
    soft = gaussian_soft_targets(arrays["target_cell"], sigma)
    cell_loss = -(soft * F.log_softmax(cell_logits, dim=1)).sum(dim=1).mean()
    size_loss = F.cross_entropy(size_logits, arrays["target_size"])
    return cell_loss + size_loss


# ── engine-metric evaluation (the checkpoint selector) ───────────────────────

def sample_to_request(sample: Dict) -> EngineRequest:
    """A prepared/augmented sample -> an EngineRequest the engine can build."""
    rooms, seen = [], {}
    for r in sample["rooms"]:
        base = r["rtype"].replace("_", " ").title()
        seen[base] = seen.get(base, 0) + 1
        name = base if seen[base] == 1 else f"{base} {seen[base]}"
        rooms.append(RoomSpec(name=name, rtype=r["rtype"],
                              target_sqft=max(20.0, r["area_sqft"]),
                              zone=r["zone"]))
    return EngineRequest(
        plot_w_ft=int(round(np.clip(sample["plot_w_ft"], 12, 200))),
        plot_h_ft=int(round(np.clip(sample["plot_h_ft"], 12, 200))),
        entrance_side=sample["entrance_side"], rooms=rooms, k=3, seed=7)


def engine_score(net: PlacerNet, samples: List[Dict], n: int) -> Dict:
    """Mean best soft score + validity of the MODEL's proposals over n val
    plans (no fallback — we are measuring the model itself).

    The engine logs a `program tight for plot` warning whenever CubiCasa's
    (often small) plots squeeze the program — expected here and not
    actionable, so we mute the engine logger for the duration of the eval
    only; it stays live everywhere else."""
    placer = Tier2Placer(TorchPlacer(net), tau=-1.0)   # never fall back
    orch = Orchestrator(proposer=placer, config=EngineConfig())
    eng_log = logging.getLogger("PlanGen.Engine")
    prev_level = eng_log.level
    eng_log.setLevel(logging.ERROR)                     # hush eval-only noise
    scores, valid = [], 0
    used = 0
    try:
        for s in samples[:n]:
            try:
                req = sample_to_request(s)
                res = orch.generate(req)
            except Exception:
                continue
            used += 1
            if res.best:
                valid += 1
                scores.append(res.best.verdict.soft_score)
    finally:
        eng_log.setLevel(prev_level)                    # always restore
    return {
        "mean_soft_score": round(float(np.mean(scores)), 2) if scores else 0.0,
        "validity_rate": round(valid / used, 3) if used else 0.0,
        "n_eval": used,
    }


# ── training ─────────────────────────────────────────────────────────────────

def _to_torch(arrays: Dict) -> Dict:
    out = {k: torch.as_tensor(v) for k, v in arrays.items()
           if isinstance(v, np.ndarray)}
    out["n_rooms"] = int(arrays["n_rooms"])
    return out


def _atomic_save(obj: Dict, path: str) -> None:
    """Write then rename, so a disconnect mid-write can't corrupt the file
    the next Colab account depends on."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


CHECKPOINT_NAME = "checkpoint.pt"     # the ONE portable file, carried between
#                                       Colab accounts to resume training


def eval_key_for(eval_n: int, n_val: int) -> str:
    """Identity of the evaluation set a `best_score` was measured on.

    Engine scores are only comparable within one eval set: raising --eval-n
    (or changing the val split) changes the mix of plans and therefore the
    achievable mean, typically by MORE than the training gains being chased.
    Without this key a run that widens its eval mid-training freezes
    `best_model` forever at whatever epoch happened to sit on the easier
    set — the checkpoint keeps saving, but never updates the weights it
    calls best. Recorded in the checkpoint and re-baselined on mismatch.
    """
    return f"eval_n={min(eval_n, n_val)}|n_val={n_val}"


def train(cfg: PlacerConfig, data_dir: str, epochs: int, batch: int,
          lr: float, sigma: float, eval_every: int, eval_n: int,
          curriculum_epochs: int, out_dir: str, device: str,
          smoke: bool = False, heartbeat: int = 2000,
          resume: bool = False) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    net = PlacerNet(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    from ml.tier2_placer.dataset import load_prepared
    from ml.tier2_placer.export import export_npz
    all_samples, _, manifest = load_prepared(data_dir)
    val_samples = [all_samples[i] for i in manifest.get("val_indices", [])]

    log: List[Dict] = []
    best_score = -1.0
    best_state = None
    start_epoch = 0
    ckpt_path = os.path.join(out_dir, CHECKPOINT_NAME)
    npz_path = os.path.join(out_dir, "placer.npz")
    eval_key = eval_key_for(eval_n if not smoke else 4, len(val_samples))

    # ── resume: reload model + optimizer + progress from the portable file ──
    if resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("next_epoch", 0))
        best_score = float(ck.get("best_score", -1.0))
        best_state = ck.get("best_model")
        log = ck.get("log", [])
        print(f"  RESUMED from {ckpt_path}: continuing at epoch "
              f"{start_epoch}, best score so far {best_score}", flush=True)

        # The stored best_score is only meaningful on the eval set that
        # produced it. If this run evaluates differently, re-measure the
        # stored best weights HERE so the comparison that gates every future
        # checkpoint is apples-to-apples (see eval_key_for).
        stored_key = ck.get("eval_key")
        if best_state is not None and stored_key != eval_key:
            probe = PlacerNet(cfg).to(dev)
            probe.load_state_dict(best_state)
            probe.eval()
            rebased = engine_score(probe, val_samples,
                                   eval_n if not smoke else 4)
            print(f"  EVAL SET CHANGED "
                  f"({stored_key or 'unrecorded'} -> {eval_key}): "
                  f"re-baselined best_score {best_score} -> "
                  f"{rebased['mean_soft_score']} by re-scoring the stored "
                  f"best weights on the new set.", flush=True)
            log.append({"epoch": start_epoch, "event": "rebaseline",
                        "from_key": stored_key, "to_key": eval_key,
                        "old_best": best_score,
                        "new_best": rebased["mean_soft_score"]})
            best_score = rebased["mean_soft_score"]
            del probe
    elif resume:
        print(f"  --resume set but no {ckpt_path} found; starting fresh.",
              flush=True)

    def save_checkpoint(next_epoch: int) -> None:
        _atomic_save({"model": net.state_dict(), "opt": opt.state_dict(),
                      "next_epoch": next_epoch, "best_score": best_score,
                      "best_model": best_state, "log": log,
                      "eval_key": eval_key,
                      "config": cfg.to_dict()}, ckpt_path)

    for epoch in range(start_epoch, epochs):
        net.train()
        # curriculum: small plans first
        max_rooms = 6 if epoch < curriculum_epochs else cfg.max_rooms
        ds = PreparedDataset(data_dir, split="train",
                             augment=not smoke, max_rooms=max_rooms)
        order = np.random.default_rng(epoch).permutation(len(ds))
        if smoke:
            order = order[:12]

        total = len(order)
        t_epoch = time.time()
        # live progress: tqdm bar if available, else a periodic text line
        use_bar = tqdm is not None and not smoke
        bar = tqdm(total=total, desc=f"epoch {epoch}", unit="step",
                   dynamic_ncols=True, leave=True) if use_bar else None
        if not use_bar:
            print(f"  epoch {epoch:>3}  starting  ({total} steps, "
                  f"max_rooms={max_rooms}) ...", flush=True)

        opt.zero_grad()
        running, n_seen = 0.0, 0
        for step, idx in enumerate(order):
            arrays = _to_torch(ds[int(idx)])
            arrays = {k: (v.to(dev) if torch.is_tensor(v) else v)
                      for k, v in arrays.items()}
            if arrays["n_rooms"] < 2:
                if bar is not None:
                    bar.update(1)
                continue
            loss = plan_loss(net, arrays, sigma) / batch
            loss.backward()
            running += loss.item() * batch
            n_seen += 1
            if (step + 1) % batch == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if bar is not None:
                bar.update(1)
                if (step + 1) % 50 == 0:
                    bar.set_postfix(loss=f"{running / max(n_seen, 1):.3f}")
            elif not smoke and (step + 1) % heartbeat == 0:
                el = time.time() - t_epoch
                rate = (step + 1) / max(el, 1e-6)
                eta = (total - step - 1) / max(rate, 1e-6)
                print(f"      step {step+1:>6}/{total}  "
                      f"loss {running / max(n_seen, 1):.3f}  "
                      f"{rate:.0f} it/s  eta {eta/60:.1f} min", flush=True)
        opt.step(); opt.zero_grad()
        if bar is not None:
            bar.close()
        mean_loss = running / max(1, n_seen)

        entry = {"epoch": epoch, "loss": round(mean_loss, 4),
                 "max_rooms": max_rooms}
        if (epoch % eval_every == 0) or epoch == epochs - 1:
            net.eval()
            metrics = engine_score(net, val_samples, eval_n if not smoke else 4)
            entry.update(metrics)
            entry["eval_key"] = eval_key
            if metrics["mean_soft_score"] > best_score:
                best_score = metrics["mean_soft_score"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                entry["saved_best"] = True
                # keep placer.npz on Drive always == best-so-far
                export_npz({"state_dict": best_state, "config": cfg.to_dict()},
                           npz_path)
        log.append(entry)
        # persist the portable checkpoint AFTER every epoch (atomic)
        save_checkpoint(next_epoch=epoch + 1)
        print(f"  epoch {epoch:>3}  loss {entry['loss']:<8} "
              f"score {entry.get('mean_soft_score','-')}  "
              f"valid {entry.get('validity_rate','-')}"
              f"{'  <-- best' if entry.get('saved_best') else ''}"
              f"  [checkpoint saved]", flush=True)

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump({"log": log, "best_score": best_score,
                   "eval_key": eval_key, "config": cfg.to_dict()}, f, indent=2)
    # final export of the best weights (falls back to last if never evaluated)
    final_state = best_state or {k: v.detach().cpu().clone()
                                 for k, v in net.state_dict().items()}
    export_npz({"state_dict": final_state, "config": cfg.to_dict()}, npz_path)
    return {"best_score": best_score, "checkpoint": ckpt_path,
            "npz": npz_path, "log": log}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tier2_placer.train")
    from ml.training.paths import PREPARED_DIR
    p.add_argument("--data-dir", default=PREPARED_DIR)
    p.add_argument("--out-dir", default=WEIGHTS_DIR)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--curriculum-epochs", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="1-2 epochs, few plans — a fast CI/dev sanity run")
    p.add_argument("--heartbeat", type=int, default=2000,
                   help="text progress every N steps (fallback if no tqdm)")
    p.add_argument("--resume", action="store_true",
                   help="continue from <out-dir>/checkpoint.pt if present "
                        "(carry that file between Colab accounts)")
    args = p.parse_args(argv)

    cfg = PlacerConfig()
    print(f"training placer ({PlacerNet(cfg).num_params():,} params) on "
          f"{args.device} …")
    t0 = time.time()
    out = train(cfg, args.data_dir,
                epochs=1 if args.smoke else args.epochs,
                batch=args.batch, lr=args.lr, sigma=args.sigma,
                eval_every=1 if args.smoke else args.eval_every,
                eval_n=args.eval_n,
                curriculum_epochs=0 if args.smoke else args.curriculum_epochs,
                out_dir=args.out_dir, device=args.device, smoke=args.smoke,
                heartbeat=args.heartbeat, resume=args.resume)
    print(f"done in {time.time()-t0:.0f}s  best engine score "
          f"{out['best_score']}  -> {out['npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
