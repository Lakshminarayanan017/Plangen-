"""
export.py — checkpoint -> portable .npz weights.

Production is meant to stay torch-free: the server loads NumPy arrays and
runs a hand-written forward pass, the way v2 did (verified at 7e-7 against
torch). This module owns the first half of that — dumping every tensor plus
the config and the provenance needed to rebuild the model.

    python -m ml.placer_v3.export --checkpoint runs/stage_c/ckpt_best.pt \
        --out weights/placer_v3.npz --verify

`--verify` currently round-trips through torch: it reloads the npz into a
fresh PlacerNetV3 and asserts the logits match the checkpoint's model to
<1e-5. That catches the failure this step actually has — a dropped or
mis-shaped tensor — but it does NOT yet prove a NumPy forward agrees, because
`numpy_infer.py` for v3 is not written. Until it is, the merge gate and the
engine run the torch model. Do not claim a torch-free server path on the
strength of this check.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from ml.placer_v3.config import PRESET_LARGE, PlacerV3Config
from ml.placer_v3.model.placer_net import PlacerNetV3


def export_npz(checkpoint: str, out_path: str) -> Dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("config") or PlacerV3Config().to_dict()
    state = payload["model"]

    arrays = {k: v.detach().cpu().numpy() for k, v in state.items()}
    meta = {
        "config": cfg_dict,
        "train_state": payload.get("state", {}),
        "saved_at": payload.get("saved_at", ""),
        "source_checkpoint": os.path.abspath(checkpoint),
        "n_tensors": len(arrays),
        "n_params": int(sum(a.size for a in arrays.values())),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    np.savez_compressed(out_path, __meta__=json.dumps(meta), **arrays)

    sidecar = os.path.splitext(out_path)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def verify(out_path: str, checkpoint: str, tol: float = 1e-5) -> float:
    """Rebuild from the npz and compare logits against the checkpoint.

    Catches dropped tensors, shape drift and dtype surprises — the things
    that actually go wrong in an export — on a fixed synthetic plan.
    """
    blob = np.load(out_path, allow_pickle=False)
    meta = json.loads(str(blob["__meta__"]))
    cfg = PlacerV3Config.from_dict(meta["config"])

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reference = PlacerNetV3(cfg)
    reference.load_state_dict(payload["model"])
    reference.eval()

    restored = PlacerNetV3(cfg)
    restored.load_state_dict(
        {k: torch.from_numpy(blob[k]) for k in blob.files
         if k != "__meta__"})
    restored.eval()

    from ml.placer_v3.state import build_state_stack
    rng = np.random.default_rng(0)
    n = 8
    cells = rng.integers(0, cfg.cell_count, n)
    sizes = rng.integers(1, cfg.size_count + 1, n)
    zones = rng.integers(1, 4, n)
    boundary = np.ones((2, cfg.boundary_grid, cfg.boundary_grid), np.float32)
    arrays = {
        "type_ids": torch.randint(1, cfg.n_room_types, (n,)),
        "zone_ids": torch.as_tensor(zones),
        "floor_ids": torch.zeros(n, dtype=torch.long),
        "vastu_dir_ids": torch.randint(0, cfg.n_vastu_dirs, (n,)),
        "vastu_strength_ids": torch.randint(0, cfg.n_vastu_strengths, (n,)),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "boundary": torch.as_tensor(boundary),
        "global": torch.zeros(cfg.global_dim),
        "state": torch.as_tensor(
            build_state_stack(boundary[0], "S", cells, sizes, zones)),
        "target_cell": torch.as_tensor(cells),
        "target_size": torch.as_tensor(sizes - 1),
    }
    with torch.no_grad():
        a = reference(arrays)
        b = restored(arrays)
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    if worst > tol:
        raise SystemExit(
            f"EXPORT MISMATCH: max |delta| {worst:.2e} > {tol:.0e} — the npz "
            f"does not reproduce the checkpoint, do not ship it")
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ml.placer_v3.export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args(argv)

    meta = export_npz(args.checkpoint, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"exported {meta['n_tensors']} tensors · "
          f"{meta['n_params'] / 1e6:.2f}M params · {size_mb:.1f} MB")
    print(f"  -> {os.path.abspath(args.out)}")
    best = (meta.get("train_state") or {}).get("best_score")
    if best is not None:
        print(f"  best engine score at export: {best}")

    if args.verify:
        worst = verify(args.out, args.checkpoint, args.tol)
        print(f"  verify OK — max |delta| {worst:.2e} (tol {args.tol:.0e})")
        print("  NOTE: this proves the npz reproduces the TORCH model. A "
              "torch-free\n        server path needs ml/placer_v3/"
              "numpy_infer.py, which is not written yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
