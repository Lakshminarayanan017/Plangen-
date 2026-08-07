"""
export.py — torch checkpoint -> NumPy .npz weights (production has no torch).

Saves every parameter array by its state_dict name, plus the config JSON, so
tier2_placer.numpy_infer can rebuild the exact forward pass without torch.
numpy_infer is unit-tested for logit equality (< 1e-4) with the torch model.

Also the CLI that converts a training checkpoint (the portable
`checkpoint.pt` carried between Colab accounts) into deployable weights:

    python -m tier2_placer.export --checkpoint ../sources/checkpoint.pt \\
        --which both --out-dir tier2_placer/weights

`--which last` takes the most recent epoch, `best` the engine-metric best,
`both` writes placer_last.npz + placer_best.npz and reports which epoch each
came from. torch is needed only here (a dev/train dependency), never to run
the exported weights.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def export_npz(ckpt: Dict, path: str) -> str:
    """`ckpt` = {"state_dict": {name: tensor}, "config": {...}}."""
    state = ckpt["state_dict"]
    arrays = {}
    for name, tensor in state.items():
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") \
            else np.asarray(tensor)
        arrays[name] = arr
    arrays["__config__"] = np.frombuffer(
        json.dumps(ckpt.get("config", {})).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)
    return path


def load_npz(path: str):
    """Return (weights: {name: ndarray}, config: dict)."""
    data = np.load(path, allow_pickle=False)
    weights, config = {}, {}
    for name in data.files:
        if name == "__config__":
            config = json.loads(bytes(data[name]).decode("utf-8"))
        else:
            weights[name] = data[name]
    return weights, config


# ── checkpoint -> deployable weights (CLI) ───────────────────────────────────

def checkpoint_summary(ckpt: Dict) -> Dict:
    """Human-readable state of a training checkpoint, without loading torch
    tensors into anything but memory. Reports the epoch each stored state
    corresponds to, so an exported .npz is never anonymous."""
    log: List[Dict] = ckpt.get("log", []) or []
    scored = [e for e in log if "mean_soft_score" in e]
    best_epoch = None
    if scored:
        best = max(scored, key=lambda e: e["mean_soft_score"])
        if abs(best["mean_soft_score"] - float(ckpt.get("best_score", -1))) < 1e-6:
            best_epoch = best["epoch"]
    return {
        "epochs_done": int(ckpt.get("next_epoch", len(log))),
        "best_score": float(ckpt.get("best_score", -1.0)),
        "best_epoch": best_epoch,
        "has_best_model": ckpt.get("best_model") is not None,
        "eval_key": ckpt.get("eval_key"),
        "last_loss": scored[-1].get("loss") if log else None,
        "evaluated_epochs": [e["epoch"] for e in scored],
    }


_STATE_KEY = {"last": "model", "best": "best_model"}


def export_from_checkpoint(ckpt_path: str, out_dir: str,
                           which: str = "both") -> List[Tuple[str, str]]:
    """Convert a training checkpoint to .npz weight files.

    Returns [(which, path), ...]. Requires torch (dev/train only)."""
    import torch                                     # dev-only dependency

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config") or {}
    if not config:
        raise ValueError(f"{ckpt_path} has no 'config' — cannot export "
                         f"weights whose architecture is unknown")
    wanted = ["last", "best"] if which == "both" else [which]
    os.makedirs(out_dir, exist_ok=True)

    written: List[Tuple[str, str]] = []
    for kind in wanted:
        state = ckpt.get(_STATE_KEY[kind])
        if state is None:
            if which == "both":
                continue                              # nothing scored yet
            raise ValueError(f"checkpoint has no {_STATE_KEY[kind]!r} state")
        path = os.path.join(out_dir, f"placer_{kind}.npz")
        export_npz({"state_dict": state, "config": config}, path)
        written.append((kind, path))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tier2_placer.export")
    p.add_argument("--checkpoint", required=True,
                   help="training checkpoint .pt (model/best_model/config)")
    p.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "weights"))
    p.add_argument("--which", default="both",
                   choices=["last", "best", "both"])
    args = p.parse_args(argv)

    import torch
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    info = checkpoint_summary(ckpt)
    print(f"checkpoint : {os.path.abspath(args.checkpoint)}")
    print(f"  epochs done      : {info['epochs_done']}")
    print(f"  best engine score: {info['best_score']} "
          f"(epoch {info['best_epoch'] if info['best_epoch'] is not None else '?'})")
    print(f"  eval key         : {info['eval_key'] or '(not recorded)'}")

    for kind, path in export_from_checkpoint(
            args.checkpoint, args.out_dir, args.which):
        size_mb = os.path.getsize(path) / 1e6
        print(f"  wrote {kind:<5} -> {path}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
