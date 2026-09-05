"""
checkpoint.py — training you can lose the machine in the middle of.

Colab disconnects. It disconnects at hour 11 of a 12-hour run, it disconnects
while a 118 MB file is half-written to Drive, and it disconnects on the epoch
that finally beat the baseline. Every measure here exists because one of
those costs a training run:

  ATOMIC WRITES     write to .tmp, fsync, os.replace. A kill mid-write leaves
                    the PREVIOUS checkpoint intact instead of a truncated
                    file that loads as garbage.
  FULL STATE        model + optimizer + scheduler + AMP scaler + epoch + step
                    + best score + RNG state for python/numpy/torch/cuda.
                    Restoring weights alone silently restarts the LR schedule
                    and reshuffles the data — the run continues, and the
                    curve is no longer the curve you were reading.
  EVAL KEY          every checkpoint records WHICH eval set produced its
                    score. v2 raised --eval-n mid-run, the mean fell for that
                    reason alone, and `best_model` froze at epoch 2 for the
                    rest of training. A key mismatch now re-baselines loudly.
  VERIFIED LOADS    a checkpoint is torch.load-ed before it is trusted; a
                    corrupt newest falls back to the newest that does load.
  ROTATION          keep last N + best + every Kth milestone. At 100+ MB a
                    checkpoint, an unbounded history fills a 15 GB Drive in
                    about 130 epochs and the run dies on write.
  APPEND-ONLY LOG   one JSON line per epoch, flushed and fsynced, so the
                    history survives even when the checkpoint does not.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

CHECKPOINT_VERSION = 3


@dataclass
class TrainState:
    """Everything needed to continue as if nothing happened."""
    epoch: int = 0
    global_step: int = 0
    best_score: float = float("-inf")
    best_epoch: int = -1
    eval_key: str = ""
    stage: str = "pretrain"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainState":
        fields = set(cls().__dict__)
        return cls(**{k: v for k, v in d.items() if k in fields})


def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu()
                            if hasattr(state["torch"], "cpu")
                            else state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    except Exception as exc:                     # never fatal
        print(f"  [ckpt] RNG restore skipped ({exc})")


def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    """Write, flush, fsync, then rename. os.replace is atomic on POSIX and on
    NTFS, so the destination is either the old file or the complete new one —
    never a half-written blob."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return float("inf")


class CheckpointManager:
    """Owns one training run's directory on Drive."""

    def __init__(self, root: str, *, keep_last: int = 3,
                 milestone_every: int = 10, min_free_gb: float = 1.5):
        self.root = root
        self.keep_last = keep_last
        self.milestone_every = milestone_every
        self.min_free_gb = min_free_gb
        os.makedirs(root, exist_ok=True)
        self.log_path = os.path.join(root, "train_log.jsonl")

    # ── paths ───────────────────────────────────────────────────────────
    def _epoch_path(self, epoch: int) -> str:
        return os.path.join(self.root, f"ckpt_epoch{epoch:04d}.pt")

    @property
    def last_path(self) -> str:
        return os.path.join(self.root, "ckpt_last.pt")

    @property
    def best_path(self) -> str:
        return os.path.join(self.root, "ckpt_best.pt")

    # ── save ────────────────────────────────────────────────────────────
    def save(self, *, model, optimizer, scheduler, scaler,
             state: TrainState, config: Dict[str, Any],
             is_best: bool) -> None:
        payload = {
            "version": CHECKPOINT_VERSION,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "state": state.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "rng": _rng_state(),
        }

        free = free_gb(self.root)
        if free < self.min_free_gb:
            # Drive is nearly full: drop rotation history first, and if that
            # is not enough keep only ckpt_last so the run survives.
            print(f"  [ckpt] only {free:.1f} GB free — pruning history")
            self._prune(keep_last=1, force=True)

        self._write_with_retry(payload, self.last_path)
        if is_best:
            self._write_with_retry(payload, self.best_path)
        if self.milestone_every and state.epoch % self.milestone_every == 0:
            self._write_with_retry(payload, self._epoch_path(state.epoch))
        self._prune(self.keep_last)

    def _write_with_retry(self, payload: Dict[str, Any], path: str,
                          attempts: int = 3) -> None:
        """Drive's FUSE mount throws transient IO errors under load. Three
        tries with backoff turns a lost epoch into a two-second pause."""
        for i in range(attempts):
            try:
                _atomic_torch_save(payload, path)
                return
            except Exception as exc:
                print(f"  [ckpt] write failed ({exc}); retry {i + 1}"
                      f"/{attempts}")
                time.sleep(2.0 * (i + 1))
        print(f"  [ckpt] GAVE UP writing {path} — training continues, "
              f"but this epoch is not recoverable")

    def _prune(self, keep_last: int, force: bool = False) -> None:
        snaps = sorted(
            f for f in os.listdir(self.root)
            if f.startswith("ckpt_epoch") and f.endswith(".pt"))
        if not force and self.milestone_every:
            snaps = [f for f in snaps
                     if int(f[len("ckpt_epoch"):-3]) % self.milestone_every]
        for name in snaps[:-keep_last] if keep_last else snaps:
            try:
                os.remove(os.path.join(self.root, name))
            except OSError:
                pass

    # ── load ────────────────────────────────────────────────────────────
    def _candidates(self) -> List[str]:
        out = [self.last_path]
        out += sorted(
            (os.path.join(self.root, f) for f in os.listdir(self.root)
             if f.startswith("ckpt_epoch") and f.endswith(".pt")),
            reverse=True)
        out.append(self.best_path)
        return [p for p in out if os.path.exists(p)]

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Newest checkpoint that actually loads. A truncated newest is
        skipped with a warning rather than crashing the resume."""
        for path in self._candidates():
            try:
                payload = torch.load(path, map_location="cpu",
                                     weights_only=False)
            except Exception as exc:
                print(f"  [ckpt] {os.path.basename(path)} unreadable "
                      f"({exc}) — trying the one before it")
                continue
            if payload.get("version") != CHECKPOINT_VERSION:
                print(f"  [ckpt] {os.path.basename(path)} is version "
                      f"{payload.get('version')}, expected "
                      f"{CHECKPOINT_VERSION} — skipping")
                continue
            payload["_path"] = path
            return payload
        return None

    def resume(self, *, model, optimizer=None, scheduler=None, scaler=None,
               eval_key: str = "") -> TrainState:
        """Restore everything, or return a fresh state if there is nothing
        to restore. Always safe to call."""
        payload = self.load_latest()
        if payload is None:
            print("  [ckpt] no checkpoint found — starting from scratch")
            return TrainState(eval_key=eval_key)

        model.load_state_dict(payload["model"])
        for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"),
                         (scaler, "scaler")):
            if obj is not None and payload.get(key):
                try:
                    obj.load_state_dict(payload[key])
                except Exception as exc:
                    print(f"  [ckpt] {key} not restored ({exc}) — "
                          f"continuing with a fresh {key}")
        _restore_rng(payload.get("rng"))

        state = TrainState.from_dict(payload["state"])
        print(f"  [ckpt] resumed {os.path.basename(payload['_path'])} "
              f"@ epoch {state.epoch}, best {state.best_score:.3f} "
              f"(epoch {state.best_epoch})")

        if eval_key and state.eval_key and eval_key != state.eval_key:
            # This is the v2 bug, caught instead of suffered.
            print(f"  [ckpt] EVAL SET CHANGED\n"
                  f"         was: {state.eval_key}\n"
                  f"         now: {eval_key}\n"
                  f"         best_score reset — scores across different eval "
                  f"sets are not comparable, and comparing them is what froze "
                  f"v2's best checkpoint at epoch 2.")
            state.best_score = float("-inf")
            state.best_epoch = -1
        state.eval_key = eval_key or state.eval_key
        return state

    # ── log ─────────────────────────────────────────────────────────────
    def append_log(self, row: Dict[str, Any]) -> None:
        """One JSON line per epoch, fsynced. The cheapest thing in the run
        and the last thing you still have when everything else is gone."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as exc:
            print(f"  [ckpt] log append failed ({exc})")

    def read_log(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        rows = []
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows


def eval_key_for(name: str, n_items: int, seed: int) -> str:
    """A stable identity for an eval set. Any change to WHAT is being
    measured changes this string, and the manager refuses to compare across
    it."""
    return f"{name}:n={n_items}:seed={seed}"
