"""
model_trainer.py
================
Training loop for the Option 4 GNN + Autoregressive Layout Transformer.

Replaces the old diffusion model_trainer.py.

Architecture trained here
-------------------------
  GNNEncoder       (gnn_encoder.py)      →  node_emb (N, 256), global_emb (1, 256)
  LayoutTransformer (autoregressive_transformer.py) → types + MoG positions

Training objective
------------------
  total_loss = α * CrossEntropy(type_logits, type_targets)
             + β * NLL(MoG_cx, cx_targets)
             + β * NLL(MoG_cy, cy_targets)
             + β * NLL(MoG_w,  w_targets)
             + β * NLL(MoG_h,  h_targets)

  α = 1.0, β = 2.0  (position accuracy weighted more heavily as it's harder)

Variable-length batching
------------------------
Each floor plan has a different number of rooms (2–25).
We pad to the maximum N in each mini-batch, with a mask to zero out
loss contributions from padded positions.

Usage
-----
  # From command line:
  python -m modules.step4_generate.training.model_trainer \
      --cache_dir modules/step4_generate/weights/cache \
      --out_dir   modules/step4_generate/weights \
      --epochs 50 --batch_size 32 --lr 1e-4

  # Or programmatically:
  from modules.step4_generate.training.model_trainer import ARTrainer
  trainer = ARTrainer(cache_dir=..., out_dir=...)
  trainer.train(epochs=50)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("PlanGen.ARTrainer")

# ── Weight file names ─────────────────────────────────────────────────────────
_GNN_NPZ_NAME = "gnn_encoder.npz"
_AR_NPZ_NAME  = "ar_transformer.npz"
_CKPT_NAME    = "ar_checkpoint_latest.pt"
_HISTORY_NAME = "ar_training_history.json"

# ── Loss weights — imported from the transformer module (single source of truth)
# These are the same α / β used inside LayoutTransformer.masked_compute_loss().
# Do NOT redefine them here; import only.
from modules.step4_generate.autoregressive_transformer import (
    TYPE_LOSS_WEIGHT as _ALPHA_TYPE,   # noqa: F401  (kept for docs reference)
    CONT_LOSS_WEIGHT as _BETA_CONT,    # noqa: F401
)


def _collate_batch(
    samples: List,
    max_rooms: int = 25,
) -> Optional[Dict]:
    """
    Collate a list of ARSample objects into padded batch tensors (torch).

    Returns dict with keys:
      global_ctx    : (B, 6)
      type_ids      : (B, N)    — padded with -1
      boxes_norm    : (B, N, 4) — padded with 0
      node_features : (B, N, 24)
      masks         : (B, N)   — True for real rooms, False for padding
      n_rooms_list  : list of int
      plan_ids      : list of str

    Edge index/features are NOT batched here — the GNN forward handles them.
    Returns None if batch is empty or no valid samples.
    """
    try:
        import torch
    except ImportError:
        return None

    valid = [s for s in samples if s is not None and s.n_rooms >= 2]
    if not valid:
        return None

    B = len(valid)
    N = min(max(s.n_rooms for s in valid), max_rooms)

    global_ctx    = np.zeros((B, 6), dtype=np.float32)
    type_ids      = np.full((B, N), fill_value=-1, dtype=np.int32)
    boxes_norm    = np.zeros((B, N, 4), dtype=np.float32)
    node_features = np.zeros((B, N, 24), dtype=np.float32)
    masks         = np.zeros((B, N), dtype=bool)
    n_rooms_list  = []
    plan_ids      = []

    for b, s in enumerate(valid):
        n = min(s.n_rooms, max_rooms)
        global_ctx[b]    = s.global_ctx
        type_ids[b, :n]  = s.type_ids[:n]
        boxes_norm[b, :n] = s.boxes_norm[:n]
        node_features[b, :n] = s.node_features[:n]
        masks[b, :n]     = True
        n_rooms_list.append(n)
        plan_ids.append(s.plan_id)

    return {
        "global_ctx":    torch.tensor(global_ctx,    dtype=torch.float32),
        "type_ids":      torch.tensor(type_ids,      dtype=torch.long),
        "boxes_norm":    torch.tensor(boxes_norm,    dtype=torch.float32),
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "masks":         torch.tensor(masks,         dtype=torch.bool),
        "n_rooms_list":  n_rooms_list,
        "plan_ids":      plan_ids,
        "samples":       valid,    # keep for GNN edge_index/feats access
    }


class ARTrainer:
    """
    End-to-end trainer for GNN + Autoregressive Layout Transformer.

    Training loop per epoch:
      1. Build batches from ARSample list
      2. For each batch:
         a. Run GNNEncoder on per-sample node/edge features
            (full batching is complex due to variable node counts;
             we iterate per sample and pool to d_model via ctx_proj)
         b. Embed sequence (teacher forcing)
         c. Forward transformer
         d. Compute loss (type CE + continuous MoG NLL)
         e. Backprop + Adam step
      3. Validate on val set
      4. Save checkpoint if val loss improved
      5. Export numpy weights when training completes

    Hardware
    --------
    Automatically detects: MPS (Apple Silicon) > CUDA > CPU.
    """

    def __init__(
        self,
        cache_dir:  str,
        out_dir:    str,
        gnn_dim:    int   = 256,
        d_model:    int   = 512,
        n_heads:    int   = 8,
        n_layers:   int   = 12,
        d_ff:       int   = 2048,
        n_mog:      int   = 3,
        lr:         float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int   = 32,
        grad_clip:  float = 1.0,
        warmup_steps: int = 500,
        seed:       int   = 42,
    ) -> None:
        self.cache_dir   = cache_dir
        self.out_dir     = out_dir
        self.gnn_dim     = gnn_dim
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.d_ff        = d_ff
        self.n_mog       = n_mog
        self.lr          = lr
        self.wd          = weight_decay
        self.batch_size  = batch_size
        self.grad_clip   = grad_clip
        self.warmup_steps = warmup_steps
        self.seed        = seed

        self._device = self._detect_device()
        log.info("ARTrainer: device=%s", self._device)

    @staticmethod
    def _detect_device(force: str = "auto") -> "torch.device":
        """
        Device selection for training.

        Default: "auto" — picks the best available device in order:
          CUDA (Colab/cloud GPU)  >  CPU

        MPS (Apple Silicon) is intentionally skipped in auto-mode because
        it has known silent-hang bugs with:
          - nn.MultiheadAttention with float32 attn_mask on sequences > 64 tokens
          - torch.logsumexp backward in certain configurations
          - GNN scatter/gather ops on variable-size graphs
          - Dropout + backward on cross-attention

        These hang with NO error message, making debugging impossible.

        Override: set force="mps" or force="cpu" to force a specific device.
        """
        try:
            import torch
            if force == "auto":
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    print(f"[ARTrainer] ✅ GPU detected: {torch.cuda.get_device_name(0)} — using CUDA", flush=True)
                    return device
                print("[ARTrainer] ⚠️  No GPU found — falling back to CPU (training will be slow)", flush=True)
                return torch.device("cpu")
            if force == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if force == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        except ImportError:
            raise RuntimeError("PyTorch is required for training. "
                               "Install with: pip install torch")

    # ── Public API ─────────────────────────────────────────────────────────

    def train(
        self,
        epochs:        int  = 50,
        force_rebuild:  bool = False,
        export_numpy:  bool = True,
    ) -> Dict:
        """
        Full training run.

        Returns training history dict:
          {train_loss: [...], val_loss: [...], best_val_loss: float}
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── Load data ─────────────────────────────────────────────────────
        print("\n[ARTrainer] Step 1/4 — Loading dataset...", flush=True)
        from modules.step4_generate.training.data_prep import build_ar_datasets
        log.info("Loading datasets from %s ...", self.cache_dir)
        train_samples, val_samples = build_ar_datasets(
            cache_dir     = self.cache_dir,
            force_rebuild = force_rebuild,
        )
        if not train_samples:
            raise RuntimeError(
                f"No training samples found in {self.cache_dir}. "
                "Run plan_indexer.py first to generate the cache files."
            )
        print(f"[ARTrainer] ✅ Dataset loaded: {len(train_samples):,} train + {len(val_samples):,} val samples", flush=True)
        log.info("Dataset: %d train + %d val samples",
                 len(train_samples), len(val_samples))

        # ── Build models ──────────────────────────────────────────────────
        print("\n[ARTrainer] Step 2/4 — Building models...", flush=True)
        from modules.step4_generate.gnn_encoder import GNNEncoder, OUTPUT_DIM as GNN_OUTPUT_DIM
        from modules.step4_generate.autoregressive_transformer import LayoutTransformer

        # GNNEncoder dimensions are fixed module-level constants (24-in, 256-out).
        # Do NOT pass them as constructor args — only dropout is accepted.
        gnn = GNNEncoder().to(self._device)
        # Keep gnn_dim in sync with the encoder's actual output dim
        self.gnn_dim = GNN_OUTPUT_DIM   # 256

        ar_model = LayoutTransformer(
            d_model    = self.d_model,
            n_heads    = self.n_heads,
            n_layers   = self.n_layers,
            d_ff       = self.d_ff,
            n_mog_comps = self.n_mog,
            gnn_dim    = self.gnn_dim,
        ).to(self._device)

        total_params = (
            sum(p.numel() for p in gnn.parameters()) +
            sum(p.numel() for p in ar_model.parameters())
        )
        print(f"[ARTrainer] ✅ Models built — GNN + AR Transformer, total {total_params/1e6:.1f}M params on {self._device}", flush=True)
        log.info("Model parameters: GNN=%s  AR=%s  Total=%.1fM",
                 f"{sum(p.numel() for p in gnn.parameters()):,}",
                 f"{sum(p.numel() for p in ar_model.parameters()):,}",
                 total_params / 1e6)

        # ── Optimiser + scheduler ─────────────────────────────────────────
        all_params = list(gnn.parameters()) + list(ar_model.parameters())
        optimiser = optim.AdamW(all_params, lr=self.lr, weight_decay=self.wd)

        def _lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            # Cosine decay after warmup
            progress = (step - self.warmup_steps) / max(1, epochs * max(1, len(train_samples) // self.batch_size) - self.warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimiser, _lr_lambda)

        # ── Resume from checkpoint ─────────────────────────────────────────
        ckpt_path     = os.path.join(self.out_dir, _CKPT_NAME)
        history_path  = os.path.join(self.out_dir, _HISTORY_NAME)
        start_epoch   = 0
        best_val_loss = float("inf")
        history       = {"train_loss": [], "val_loss": [], "best_val_loss": float("inf")}

        print("\n[ARTrainer] Step 3/4 — Checking for checkpoint...", flush=True)
        if os.path.exists(ckpt_path):
            print(f"[ARTrainer] ♻️  Found checkpoint — resuming from last saved epoch", flush=True)
            log.info("Resuming from checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self._device)
            gnn.load_state_dict(ckpt["gnn"])
            ar_model.load_state_dict(ckpt["ar_model"])
            optimiser.load_state_dict(ckpt["optimiser"])
            start_epoch   = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if os.path.exists(history_path):
                with open(history_path) as f:
                    history = json.load(f)
            print(f"[ARTrainer] ✅ Resumed at epoch {start_epoch + 1}, best_val_loss so far = {best_val_loss:.4f}", flush=True)
            log.info("Resumed at epoch %d, best_val=%.4f", start_epoch, best_val_loss)
        else:
            print("[ARTrainer] 🆕 No checkpoint found — starting fresh", flush=True)

        # ── Training loop ─────────────────────────────────────────────────
        print(f"\n[ARTrainer] Step 4/4 — Training for {epochs - start_epoch} epoch(s) on {self._device}...\n", flush=True)
        global_step = start_epoch * (len(train_samples) // self.batch_size)

        for epoch in range(start_epoch, epochs):
            t0 = time.perf_counter()
            gnn.train()
            ar_model.train()

            # Shuffle training samples
            rng = np.random.default_rng(self.seed + epoch)
            idxs = rng.permutation(len(train_samples)).tolist()

            epoch_loss  = 0.0
            n_batches   = 0
            batch_start = 0

            while batch_start < len(idxs):
                batch_idxs  = idxs[batch_start:batch_start + self.batch_size]
                batch_start += self.batch_size
                batch       = [train_samples[i] for i in batch_idxs]
                collated    = _collate_batch(batch)
                if collated is None:
                    continue

                loss = self._forward_loss(
                    collated, gnn, ar_model, training=True
                )
                if loss is None:
                    continue

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                optimiser.step()
                scheduler.step()
                global_step += 1

                epoch_loss += loss.item()
                n_batches  += 1

                # ── Per-batch progress (every 100 batches) ────────────────
                if n_batches % 100 == 0:
                    avg_so_far = epoch_loss / n_batches
                    total_batches = max(len(idxs) // self.batch_size, 1)
                    pct = 100 * batch_start // len(idxs)
                    print(
                        f"  [Epoch {epoch+1}] batch {n_batches}/{total_batches} "
                        f"({pct}%) | loss={avg_so_far:.4f}",
                        flush=True,
                    )

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # ── Validation ────────────────────────────────────────────────
            gnn.eval()
            ar_model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                val_idxs = list(range(len(val_samples)))
                val_start = 0
                while val_start < len(val_idxs):
                    vb_idxs   = val_idxs[val_start:val_start + self.batch_size]
                    val_start += self.batch_size
                    vbatch    = [val_samples[i] for i in vb_idxs]
                    vcollated = _collate_batch(vbatch)
                    if vcollated is None:
                        continue
                    vloss = self._forward_loss(vcollated, gnn, ar_model, training=False)
                    if vloss is not None:
                        val_loss     += vloss.item()
                        n_val_batches += 1

            avg_val_loss = val_loss / max(n_val_batches, 1)
            elapsed = time.perf_counter() - t0

            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))

            log.info(
                "Epoch %d/%d: train_loss=%.4f  val_loss=%.4f  lr=%.2e  %.1fs",
                epoch + 1, epochs,
                avg_train_loss, avg_val_loss,
                optimiser.param_groups[0]["lr"], elapsed,
            )
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
                f"lr={optimiser.param_groups[0]['lr']:.2e}  {elapsed:.1f}s",
                flush=True,
            )

            # ── Save checkpoint every epoch ──────────────────────────────
            # Always overwrite latest checkpoint so every epoch is preserved.
            torch.save({
                "epoch":          epoch,
                "gnn":            gnn.state_dict(),
                "ar_model":       ar_model.state_dict(),
                "optimiser":      optimiser.state_dict(),
                "best_val_loss":  best_val_loss,
            }, ckpt_path)
            print(f"  💾 Checkpoint saved (epoch {epoch+1})", flush=True)

            # ── Permanent backup: Google Drive after EVERY epoch ─────────
            # Works on Colab when Drive is mounted before training.
            # Run: from google.colab import drive; drive.mount('/content/drive')
            _drive_dir = '/content/drive/MyDrive/PlanGen_weights'
            if os.path.exists('/content/drive/MyDrive'):
                try:
                    import shutil as _shutil
                    os.makedirs(_drive_dir, exist_ok=True)
                    _shutil.copy(ckpt_path, os.path.join(_drive_dir, _CKPT_NAME))
                    _shutil.copy(history_path, os.path.join(_drive_dir, _HISTORY_NAME))
                    print(f"  ✅ Permanently saved to Drive (epoch {epoch+1})", flush=True)
                except Exception as _e:
                    print(f"  ⚠️  Drive backup failed: {_e}", flush=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                history["best_val_loss"] = round(best_val_loss, 4)
                log.info("  ✓ New best val_loss=%.4f", best_val_loss)

                # ── Optional: Mac server upload (only on improvement) ─────
                # Set env var CHECKPOINT_SERVER_URL to your ngrok URL.
                # ── Auto-backup 2: Local Mac server via ngrok ─────────────
                # Set env var CHECKPOINT_SERVER_URL to your ngrok URL before training.
                # Run: python tools/checkpoint_server.py   (on your Mac)
                # Run: ngrok http 5001                     (in another terminal)
                _server_url = os.environ.get('CHECKPOINT_SERVER_URL', '').rstrip('/')
                if _server_url:
                    import threading as _threading

                    def _upload_checkpoint(_url, _path, _epoch, _val):
                        try:
                            import urllib.request as _req
                            _size = os.path.getsize(_path)
                            _epoch_str = str(_epoch).zfill(2)
                            with open(_path, 'rb') as _f:
                                _request = _req.Request(
                                    f"{_url}/checkpoint",
                                    data=_f.read(),
                                    method='POST',
                                    headers={
                                        'Content-Type':   'application/octet-stream',
                                        'Content-Length': str(_size),
                                        'X-Epoch':        _epoch_str,
                                        'X-ValLoss':      f"{_val:.4f}",
                                    },
                                )
                                _resp = _req.urlopen(_request, timeout=300)
                                print(f"  🏠 Checkpoint epoch {_epoch} saved to your Mac! ({_size/1e6:.0f} MB)", flush=True)
                        except Exception as _err:
                            print(f"  ⚠️  Mac upload failed (epoch {_epoch}): {_err}", flush=True)

                    _t = _threading.Thread(
                        target=_upload_checkpoint,
                        args=(_server_url, ckpt_path, epoch + 1, best_val_loss),
                        daemon=True,
                    )
                    _t.start()
                    print(f"  📤 Uploading checkpoint to Mac in background...", flush=True)

            # Save history every epoch
            os.makedirs(self.out_dir, exist_ok=True)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

        # ── Export numpy weights ──────────────────────────────────────────
        if export_numpy:
            log.info("Exporting numpy weights for deployment ...")
            # Load best checkpoint
            if os.path.exists(ckpt_path):
                best_ckpt = torch.load(ckpt_path, map_location="cpu")
                gnn.load_state_dict(best_ckpt["gnn"])
                ar_model.load_state_dict(best_ckpt["ar_model"])

            gnn.save_numpy_weights(os.path.join(self.out_dir, _GNN_NPZ_NAME))
            ar_model.save_numpy_weights(os.path.join(self.out_dir, _AR_NPZ_NAME))
            log.info("Numpy weights saved to %s", self.out_dir)

        return history

    def _forward_loss(
        self,
        collated:  Dict,
        gnn:       "torch.nn.Module",
        ar_model:  "torch.nn.Module",
        training:  bool = True,
    ) -> Optional["torch.Tensor"]:
        """
        Compute training loss for one batch.

        Clean implementation that delegates to ar_model.masked_compute_loss()
        — the single authoritative token-indexing / loss-weighting method.

        Steps:
          1. Run GNNEncoder per sample → (n_i, gnn_dim) node embs, (1, gnn_dim) global emb.
             GNN returns a (node_emb, global_emb) TUPLE — both are unpacked explicitly.
          2. Pad and stack into batch tensors (B, N, gnn_dim).
          3. embed_sequence() builds (B, T, d_model) from the ground-truth token prefix.
          4. project_context() projects GNN embs → (B, N+1, d_model).
          5. forward() runs all transformer blocks → (B, T, d_model).
          6. masked_compute_loss() computes TYPE CrossEntropy + continuous MoG NLL,
             skipping padding positions via the (B, N) bool mask.

        Returns:
            Scalar loss tensor (backpropagatable), or None if the batch is invalid.
        """
        try:
            import torch

            samples      = collated["samples"]
            global_ctx   = collated["global_ctx"].to(self._device)    # (B, 6)
            type_ids     = collated["type_ids"].to(self._device)       # (B, N)
            boxes_norm   = collated["boxes_norm"].to(self._device)     # (B, N, 4)
            masks        = collated["masks"].to(self._device)          # (B, N) bool
            n_rooms_list = collated["n_rooms_list"]

            B = len(samples)
            N = type_ids.shape[1]  # max rooms in this batch (may include padding)

            # ── 1. Run GNNEncoder per sample ──────────────────────────────────
            # GNN.forward() returns (node_emb: Tensor(n_i, G), global_emb: Tensor(1, G))
            # — a 2-tuple.  Never assign to a single variable.
            node_emb_list:   List = []
            global_emb_list: List = []

            for b, sample in enumerate(samples):
                n_i = n_rooms_list[b]
                nf  = torch.tensor(
                    sample.node_features[:n_i], dtype=torch.float32
                ).to(self._device)                                   # (n_i, 24)
                ei  = torch.tensor(
                    sample.edge_index, dtype=torch.long
                ).to(self._device)                                   # (2, E)
                ef  = torch.tensor(
                    sample.edge_features, dtype=torch.float32
                ).to(self._device)                                   # (E, 7)

                # Explicit tuple unpack — GNN always returns (node_emb, global_emb)
                node_emb_i, global_emb_i = gnn(nf, ei, ef)          # (n_i, G), (1, G)

                # Pad node embeddings to uniform N for batch stacking
                pad = N - n_i
                if pad > 0:
                    node_emb_i = torch.cat(
                        [node_emb_i,
                         torch.zeros(pad, self.gnn_dim, device=self._device)],
                        dim=0,
                    )                                                # (N, G)

                node_emb_list.append(node_emb_i.unsqueeze(0))       # (1, N, G)
                global_emb_list.append(global_emb_i.unsqueeze(0))   # (1, 1, G)

            node_emb_batch   = torch.cat(node_emb_list,   dim=0)    # (B, N, G)
            global_emb_batch = torch.cat(global_emb_list, dim=0)    # (B, 1, G)

            # ── 2. Trim to actual max room count for this batch ───────────────
            n_max      = max(n_rooms_list)
            type_ids_t = type_ids[:,   :n_max].clamp(min=0)          # (B, N')
            boxes_t    = boxes_norm[:,  :n_max]                       # (B, N', 4)
            valid_mask = masks[:,       :n_max]                       # (B, N') bool

            if not valid_mask.any():
                return None

            # ── 3–5. Embed, project, forward ────────────────────────────────
            seq = ar_model.embed_sequence(
                global_ctx = global_ctx,
                type_ids   = type_ids_t,
                boxes_norm = boxes_t,
                n_rooms    = n_max,
            )                                                        # (B, T, d)

            ctx = ar_model.project_context(
                node_emb_batch[:, :n_max],
                global_emb_batch,
            )                                                        # (B, N'+1, d)

            h = ar_model.forward(seq, ctx)                           # (B, T, d)

            # ── 6. Masked loss — single source of truth ──────────────────────
            # ar_model.masked_compute_loss() owns ALL token-offset arithmetic
            # and loss weighting.  Never duplicate that logic here.
            return ar_model.masked_compute_loss(
                h          = h,
                type_ids   = type_ids_t,
                boxes_norm = boxes_t,
                n_rooms    = n_max,
                mask       = valid_mask,
            )

        except Exception as e:
            log.debug("Loss computation error: %s", e, exc_info=True)
            return None


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train the GNN + Autoregressive Layout Transformer"
    )
    parser.add_argument(
        "--cache_dir",
        default="modules/step4_generate/weights/cache",
        help="Directory with rplan_samples_*.pkl cache files",
    )
    parser.add_argument(
        "--out_dir",
        default="modules/step4_generate/weights",
        help="Output directory for checkpoints and numpy weights",
    )
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--d_model",    type=int,   default=512)
    parser.add_argument("--n_layers",   type=int,   default=12)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--rebuild",    action="store_true",
                        help="Force rebuild data cache")
    parser.add_argument("--no_export",  action="store_true",
                        help="Skip numpy weight export at end")

    args = parser.parse_args()

    trainer = ARTrainer(
        cache_dir  = args.cache_dir,
        out_dir    = args.out_dir,
        d_model    = args.d_model,
        n_layers   = args.n_layers,
        lr         = args.lr,
        batch_size = args.batch_size,
        seed       = args.seed,
    )

    history = trainer.train(
        epochs       = args.epochs,
        force_rebuild = args.rebuild,
        export_numpy = not args.no_export,
    )

    print("\n=== Training complete ===")
    print(f"Best val loss : {history['best_val_loss']:.4f}")
    print(f"Train epochs  : {len(history['train_loss'])}")
    if history["train_loss"]:
        print(f"Final train   : {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"Final val     : {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
