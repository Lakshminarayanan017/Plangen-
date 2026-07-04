#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/reexport_weights.py
=========================
Re-export the existing PyTorch checkpoint (.pt) into fresh .npz files
using the key names that GNNEncoderNumpy and LayoutTransformerNumpy
actually read at inference time.

Run this ONCE after training (or after updating the code) — it takes the
checkpoint you've already trained and writes the flat-key .npz files.
No retraining needed.

Usage
-----
    source planenv/bin/activate
    python3 tools/reexport_weights.py            # re-export + quick check
    python3 tools/reexport_weights.py --verify   # also runs a numpy smoke test

What it does
------------
1. Loads  modules/step4_generate/weights/ar_checkpoint_latest.pt
2. Restores GNNEncoder + LayoutTransformer from the checkpoint state_dicts
3. Calls .save_numpy_weights() on each to write flat-key .npz files
4. Overwrites  gnn_encoder.npz  and  ar_transformer.npz  in the weights dir
5. Optionally runs a numpy forward pass (smoke test) to confirm correctness

Expected checkpoint keys
------------------------
  {
      "gnn":       <GNNEncoder state_dict>,
      "ar_model":  <LayoutTransformer state_dict>,
      "optimiser": <AdamW state_dict>,
      "epoch":     int,
      "best_val_loss": float,
  }

This format is written by modules/step4_generate/training/model_trainer.py.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

WEIGHTS_DIR     = os.path.join(_ROOT, "modules", "step4_generate", "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "ar_checkpoint_latest.pt")
GNN_NPZ_PATH    = os.path.join(WEIGHTS_DIR, "gnn_encoder.npz")
AR_NPZ_PATH     = os.path.join(WEIGHTS_DIR, "ar_transformer.npz")


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def reexport(checkpoint_path: str = CHECKPOINT_PATH, verify: bool = False) -> None:
    torch = _try_import_torch()
    if torch is None:
        print("ERROR: PyTorch is not installed in this environment.")
        print("       Run:  pip install torch  (or use the planenv venv)")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at:\n  {checkpoint_path}")
        print()
        print("Train the model first:")
        print("  python3 modules/step4_generate/training/model_trainer.py \\")
        print("          --cache_dir modules/step4_generate/weights/cache \\")
        print("          --out_dir   modules/step4_generate/weights \\")
        print("          --epochs 50")
        sys.exit(1)

    print("=" * 60)
    print("  PlanGen — AR Transformer Weight Re-export")
    print("=" * 60)
    print(f"  Checkpoint : {os.path.relpath(checkpoint_path, _ROOT)}")
    print()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("  Loading checkpoint …")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint keys : {list(ckpt.keys())}")
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    print(f"  Epoch           : {epoch}")
    print(f"  Best val loss   : {val_loss}")

    gnn_sd = ckpt.get("gnn")
    ar_sd  = ckpt.get("ar_model")

    if gnn_sd is None:
        print("  WARNING: 'gnn' key missing — skipping GNN export")
    if ar_sd is None:
        print("  WARNING: 'ar_model' key missing — skipping AR transformer export")

    if gnn_sd is None and ar_sd is None:
        print("\nERROR: Checkpoint has neither 'gnn' nor 'ar_model' weights.")
        print("       Expected format: {'gnn': ..., 'ar_model': ..., ...}")
        print()
        print("  If you have an old 'diff' key (diffusion checkpoint), retraining")
        print("  is required — the old weights are incompatible with the new AR model.")
        sys.exit(1)

    # ── Re-export GNN encoder ─────────────────────────────────────────────────
    if gnn_sd is not None:
        print("\n  [1/2] Re-exporting GNN encoder …")
        from modules.step4_generate.gnn_encoder import GNNEncoder
        gnn = GNNEncoder(dropout=0.0)
        try:
            gnn.load_state_dict(gnn_sd)
            print("       state_dict loaded (strict)")
        except RuntimeError as e:
            print(f"  WARNING: strict load failed ({e})")
            print("           Retrying with strict=False …")
            gnn.load_state_dict(gnn_sd, strict=False)

        gnn.save_numpy_weights(GNN_NPZ_PATH)
        size_kb = os.path.getsize(GNN_NPZ_PATH) / 1024
        print(f"  ✓  Saved  {os.path.relpath(GNN_NPZ_PATH, _ROOT)}  ({size_kb:.1f} KB)")

        import numpy as np
        keys = list(np.load(GNN_NPZ_PATH).keys())
        print(f"     Keys ({len(keys)}): {keys[:8]}{'…' if len(keys) > 8 else ''}")

    # ── Re-export AR transformer ──────────────────────────────────────────────
    if ar_sd is not None:
        print("\n  [2/2] Re-exporting AR Transformer …")
        from modules.step4_generate.autoregressive_transformer import LayoutTransformer
        ar_model = LayoutTransformer()
        try:
            ar_model.load_state_dict(ar_sd)
            print("       state_dict loaded (strict)")
        except RuntimeError as e:
            print(f"  WARNING: strict load failed ({e})")
            print("           Retrying with strict=False …")
            ar_model.load_state_dict(ar_sd, strict=False)

        ar_model.save_numpy_weights(AR_NPZ_PATH)
        size_kb = os.path.getsize(AR_NPZ_PATH) / 1024
        print(f"  ✓  Saved  {os.path.relpath(AR_NPZ_PATH, _ROOT)}  ({size_kb:.1f} KB)")

        import numpy as np
        keys = list(np.load(AR_NPZ_PATH).keys())
        print(f"     Keys ({len(keys)}): {keys[:8]}{'…' if len(keys) > 8 else ''}")

    # ── Optional smoke test ───────────────────────────────────────────────────
    if verify:
        print("\n  [Verify] Running numpy inference smoke test …")
        import numpy as np
        from modules.step4_generate.gnn_encoder import GNNEncoderNumpy
        from modules.step4_generate.autoregressive_transformer import (
            LayoutTransformerNumpy, LayoutTokenizer,
        )

        rng = np.random.default_rng(42)
        N = 6
        node_feat = rng.random((N, 24)).astype("float32")
        edge_idx  = np.array([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=np.int64)
        edge_feat = rng.random((6, 7)).astype("float32")

        # GNN
        gnn_np = GNNEncoderNumpy(GNN_NPZ_PATH)
        node_emb, global_emb = gnn_np.forward(node_feat, edge_idx, edge_feat)
        print(f"  GNN output      : node_emb={node_emb.shape}, global={global_emb.shape}")
        assert node_emb.shape   == (N, 256), f"Wrong GNN node shape: {node_emb.shape}"
        assert global_emb.shape == (1, 256), f"Wrong GNN global shape: {global_emb.shape}"
        print("  ✓  GNN numpy forward pass OK")

        # AR Transformer
        ar_np = LayoutTransformerNumpy(AR_NPZ_PATH)
        tok   = LayoutTokenizer(net_w_ft=30.0, net_l_ft=40.0)
        forced_types = [0, 1, 2, 3, 4, 5]   # living_room … bathroom
        boxes_norm, type_ids = ar_np.generate(
            gnn_node_emb    = node_emb,
            gnn_global_emb  = global_emb,
            tokenizer       = tok,
            n_rooms         = N,
            type_ids_forced = forced_types,
            temperature     = 0.8,
            seed            = 42,
        )
        print(f"  AR output       : boxes={boxes_norm.shape}, types={type_ids.shape}")
        assert boxes_norm.shape == (N, 4), f"Wrong AR boxes shape: {boxes_norm.shape}"
        assert type_ids.shape   == (N,),   f"Wrong AR types shape: {type_ids.shape}"
        assert (boxes_norm >= 0).all() and (boxes_norm <= 1).all(), \
            f"AR boxes out of [0,1]: min={boxes_norm.min():.3f} max={boxes_norm.max():.3f}"
        print("  ✓  AR Transformer numpy forward pass OK")
        print("  ✓  Smoke test passed — weights are loadable and produce valid output")

    print()
    print("=" * 60)
    print("  ✅  Re-export complete!")
    print()
    print("  Next steps:")
    print("  1.  Run a quick inference check:")
    print("         python3 tools/infer_diffusion.py --preset 3bhk_villa --both")
    print()
    print("  2.  If results still look poor, retrain for more epochs:")
    print("         python3 modules/step4_generate/training/model_trainer.py \\")
    print("                 --epochs 100 --cache_dir modules/step4_generate/weights/cache")
    print("      Then re-run this script to update the .npz files.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-export PlanGen AR checkpoint to .npz (GNN + AR Transformer)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a numpy inference smoke test after export",
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help=f"Path to .pt checkpoint (default: ar_checkpoint_latest.pt)",
    )
    args = parser.parse_args()
    reexport(checkpoint_path=args.checkpoint, verify=args.verify)
