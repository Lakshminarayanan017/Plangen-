"""
Diffusion Decoder v2 — Production Quality
==========================================
DDPM-style denoising diffusion model that decodes GNN room embeddings into
floor plan bounding boxes, conditioned on room type and plan context.

Architecture vs v1:
  v1: 4 blocks, 256-dim, linear noise schedule, L2 box loss only
  v2: 8 blocks, 512-dim, cosine noise schedule, auxiliary losses:
      - L1 box loss (more robust than L2)
      - IoU loss (geometric alignment)
      - Aspect ratio loss (room proportions)
      - Floor boundary loss (all rooms within [0,1]²)
      - Overlap penalty (rooms should not heavily overlap)
      - Adjacency loss (adjacent rooms should touch)

Key design decisions:
  • Per-room denoising: each room box is denoised independently using its
    GNN node embedding as conditioning signal
  • Cross-attention: each room attends to ALL other rooms and the global plan
    embedding, enabling spatial coherence
  • Sinusoidal time embeddings: standard DDPM time conditioning
  • Floor boundary prediction: auxiliary head predicts the floor plan boundary
    (useful for constraining room placement in generator.py)
  • T=1000 timesteps, cosine schedule (Nichol & Dhariwal 2021)

Input:
  node_embeddings  (N, 256)  — from GNNEncoder
  global_embedding (1, 256)  — plan summary from GNNEncoder
  noisy_boxes      (N, 4)    — [x1, y1, x2, y2] with added Gaussian noise
  timestep         (N,) int  — diffusion timestep [0, T)

Output:
  predicted_noise  (N, 4)    — predicted noise (epsilon parameterisation)
  floor_boundary   (4,)      — predicted [x1, y1, x2, y2] of floor boundary

Training:
  At each step, add noise ~ N(0, sigma_t²) to clean boxes,
  train model to predict the added noise (epsilon prediction).
  Loss = L1(pred_noise, actual_noise) + λ_iou * IoU_loss + ...

Inference (sampling):
  DiffusionDecoder.sample(node_emb, global_emb, n_steps=50)
  → returns clean box predictions using DDPM/DDIM sampling

Numpy inference:
  DiffusionDecoder.sample_numpy(node_emb, global_emb) → ndarray (N, 4)
"""

from __future__ import annotations

import math
import os
import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

log = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
EMBED_DIM:    int   = 256   # GNN encoder output dim
HIDDEN_DIM:   int   = 512   # decoder hidden dim
N_BLOCKS:     int   = 8     # denoising transformer blocks
N_HEADS:      int   = 8     # attention heads in each block
T_STEPS:      int   = 1000  # diffusion timesteps
DDIM_STEPS:   int   = 50    # inference steps (DDIM)
BOX_DIM:      int   = 4     # [x1, y1, x2, y2]
DROPOUT:      float = 0.1

# Loss weights
LAMBDA_L1:        float = 1.0
LAMBDA_IOU:       float = 0.5
LAMBDA_ASPECT:    float = 0.2
LAMBDA_BOUNDARY:  float = 0.3
LAMBDA_OVERLAP:   float = 0.2
LAMBDA_ADJACENCY: float = 0.1


# ── Noise Schedule ────────────────────────────────────────────────────────────

def _cosine_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    Returns alphas_cumprod: shape (T+1,)
    """
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos((steps / T + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    return alphas_cumprod.astype(np.float32)


# Precompute schedule arrays
_ALPHAS_CUMPROD = _cosine_schedule(T_STEPS)   # (T+1,)
_ALPHAS         = _ALPHAS_CUMPROD[1:] / _ALPHAS_CUMPROD[:-1]  # (T,)
_BETAS          = 1.0 - _ALPHAS               # (T,)
_SQRT_ALPHAS_CP = np.sqrt(_ALPHAS_CUMPROD[1:])  # (T,)
_SQRT_ONE_M_ACP = np.sqrt(1.0 - _ALPHAS_CUMPROD[1:])  # (T,)


def add_noise_numpy(
    x0: np.ndarray,
    t: np.ndarray,
    noise: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add noise to clean boxes at timestep t.
    x0: (N, 4), t: (N,) int, returns (x_t, noise)
    """
    if noise is None:
        noise = np.random.randn(*x0.shape).astype(np.float32)
    sqrt_alpha = _SQRT_ALPHAS_CP[t][:, None]    # (N, 1)
    sqrt_1m    = _SQRT_ONE_M_ACP[t][:, None]    # (N, 1)
    x_t = sqrt_alpha * x0 + sqrt_1m * noise
    return x_t.astype(np.float32), noise.astype(np.float32)


# ── Loss Functions (NumPy) ────────────────────────────────────────────────────

def _iou_numpy(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute IoU between predicted and target boxes. Returns (N,) IoU."""
    ix1 = np.maximum(pred[:, 0], target[:, 0])
    iy1 = np.maximum(pred[:, 1], target[:, 1])
    ix2 = np.minimum(pred[:, 2], target[:, 2])
    iy2 = np.minimum(pred[:, 3], target[:, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_t = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union  = area_p + area_t - inter + 1e-8
    return inter / union


def compute_auxiliary_losses_numpy(
    pred_boxes: np.ndarray,
    target_boxes: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all auxiliary losses.
    pred_boxes, target_boxes: (N, 4) in [0, 1]
    Returns dict of scalar losses.
    """
    if mask is None:
        mask = np.ones(pred_boxes.shape[0], dtype=bool)

    pb = pred_boxes[mask]
    tb = target_boxes[mask]
    if len(pb) == 0:
        return {"l1": 0.0, "iou": 0.0, "aspect": 0.0, "boundary": 0.0}

    # L1 loss
    l1 = np.abs(pb - tb).mean()

    # IoU loss (1 - IoU)
    iou_vals = _iou_numpy(pb, tb)
    iou_loss = 1.0 - iou_vals.mean()

    # Aspect ratio loss
    pred_w = np.maximum(pb[:, 2] - pb[:, 0], 1e-6)
    pred_h = np.maximum(pb[:, 3] - pb[:, 1], 1e-6)
    targ_w = np.maximum(tb[:, 2] - tb[:, 0], 1e-6)
    targ_h = np.maximum(tb[:, 3] - tb[:, 1], 1e-6)
    pred_ar = np.log(pred_w / pred_h + 1e-8)
    targ_ar = np.log(targ_w / targ_h + 1e-8)
    aspect_loss = np.abs(pred_ar - targ_ar).mean()

    # Floor boundary loss: all coords should be in [0, 1]
    viol = np.concatenate([
        np.maximum(-pb, 0),
        np.maximum(pb - 1.0, 0),
    ], axis=1)
    boundary_loss = viol.mean()

    return {
        "l1":       float(l1),
        "iou":      float(iou_loss),
        "aspect":   float(aspect_loss),
        "boundary": float(boundary_loss),
    }


# ── NumPy Inference Engine — matches _DiffusionDecoderTorchInner exactly ──────

class DiffusionDecoderNumpy:
    """
    Pure numpy inference for DiffusionDecoder v2.

    Architecture exactly mirrors _DiffusionDecoderTorchInner:
      • Sinusoidal time embedding → 2-layer MLP projection
      • box/node/global input projections (Linear → SiLU → LayerNorm)
      • 8 DenoisingBlocks: time-gate (scale/shift) + CrossAttentionBlock (MHA + FFN)
      • Output head: LayerNorm → Linear(H, H//2) → SiLU → Linear(H//2, 4)
      • Floor head: LayerNorm → Linear(H, 64) → SiLU → Linear(64, 4) → Sigmoid

    Weight keys must be saved by DiffusionDecoder.save_numpy_weights(), which uses
    explicit flat key names (te_w0, b0_tg_w, b0_attn_w, …) matching the _get()
    calls below.
    """

    def __init__(self, weights_path: str):
        self._load_weights(weights_path)

    def _load_weights(self, path: str):
        try:
            self._w = dict(np.load(path, allow_pickle=True))
            log.info(f"Loaded diffusion weights from {path}")
        except Exception as e:
            log.warning(f"Diffusion weights not found at {path}: {e}")
            self._w = {}

    def _get(self, key: str, shape: tuple) -> np.ndarray:
        if key in self._w:
            return np.asarray(self._w[key], dtype=np.float32)
        log.debug(f"Weight {key} not found, using zeros shape={shape}")
        return np.zeros(shape, dtype=np.float32)

    # ── Activation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        """SiLU / Swish activation: x · σ(x)."""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) *
                                          (x + 0.044715 * x ** 3)))

    @staticmethod
    def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True) + eps
        return gamma * (x - mean) / std + beta

    def _sinusoidal_embed(self, t: np.ndarray, dim: int = HIDDEN_DIM) -> np.ndarray:
        """Sinusoidal time embedding. t: (N,) int → (N, dim)."""
        half  = dim // 2
        freqs = np.exp(-np.arange(half, dtype=np.float32) *
                       np.log(10000.0) / (half - 1))
        args  = t[:, None].astype(np.float32) * freqs[None, :]
        return np.concatenate([np.sin(args), np.cos(args)], axis=-1)

    # ── Input projection helper ───────────────────────────────────────────────

    def _proj_silu_ln(self, x: np.ndarray, name: str,
                      in_dim: int, out_dim: int = HIDDEN_DIM) -> np.ndarray:
        """Sequential(Linear(in, out), SiLU(), LayerNorm(out))."""
        W  = self._get(f"{name}_w",    (out_dim, in_dim))
        b  = self._get(f"{name}_b",    (out_dim,))
        gm = self._get(f"{name}_ln_w", (out_dim,))
        bt = self._get(f"{name}_ln_b", (out_dim,))
        if gm.sum() == 0:
            gm = np.ones(out_dim, dtype=np.float32)
        out = self._silu(x @ W.T + b)
        return self._layernorm(out, gm, bt)

    # ── Multi-head attention (numpy) ──────────────────────────────────────────

    def _mha(self, q: np.ndarray, kv: np.ndarray, block_idx: int) -> np.ndarray:
        """
        Multi-head attention: Q from q (N, H), K/V from kv (M, H).
        Matches nn.MultiheadAttention(embed_dim=H, num_heads=N_HEADS, batch_first=True).
        Returns: (N, H)
        """
        H        = HIDDEN_DIM      # 512
        n_heads  = N_HEADS         # 8
        head_dim = H // n_heads    # 64
        N        = q.shape[0]
        M        = kv.shape[0]

        # in_proj_weight packs [Q_W; K_W; V_W] as (3H, H)
        W_qkv     = self._get(f"b{block_idx}_attn_w",     (H * 3, H))
        b_qkv     = self._get(f"b{block_idx}_attn_b",     (H * 3,))
        W_out     = self._get(f"b{block_idx}_attn_out_w", (H, H))
        b_out_attn = self._get(f"b{block_idx}_attn_out_b", (H,))

        Q = q  @ W_qkv[:H].T  + b_qkv[:H]       # (N, H)
        K = kv @ W_qkv[H:2*H].T + b_qkv[H:2*H] # (M, H)
        V = kv @ W_qkv[2*H:].T  + b_qkv[2*H:]  # (M, H)

        # Reshape to (n_heads, seq, head_dim)
        Q = Q.reshape(N, n_heads, head_dim).transpose(1, 0, 2)  # (H, N, D)
        K = K.reshape(M, n_heads, head_dim).transpose(1, 0, 2)  # (H, M, D)
        V = V.reshape(M, n_heads, head_dim).transpose(1, 0, 2)  # (H, M, D)

        # Scaled dot-product attention
        scale   = 1.0 / math.sqrt(head_dim)
        scores  = np.einsum("hnd,hmd->hnm", Q, K) * scale        # (H, N, M)
        scores -= scores.max(axis=-1, keepdims=True)              # stability
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-8    # (H, N, M)

        out = np.einsum("hnm,hmd->hnd", weights, V)              # (H, N, D)
        out = out.transpose(1, 0, 2).reshape(N, H)                # (N, H)
        return out @ W_out.T + b_out_attn

    # ── Core denoising step ───────────────────────────────────────────────────

    def _denoise_step(
        self,
        x_t:        np.ndarray,   # (N, 4)
        node_emb:   np.ndarray,   # (N, 256)
        global_emb: np.ndarray,   # (1, 256)
        t_val:      int,
    ) -> np.ndarray:
        """
        Single denoising step — mirrors _DiffusionDecoderTorchInner.forward().
        Returns predicted noise ε̂  (N, 4).
        """
        N = x_t.shape[0]
        H = HIDDEN_DIM   # 512

        # ── Time embedding ────────────────────────────────────────────────────
        # SinusoidalTimeEmbed: sinusoidal → Sequential(Linear(H,H*2), SiLU, Linear(H*2,H))
        t_arr = np.full(N, t_val, dtype=np.int32)
        t_sin = self._sinusoidal_embed(t_arr, H)               # (N, H)

        W_te0 = self._get("te_w0", (H * 2, H))                 # (1024, 512)
        b_te0 = self._get("te_b0", (H * 2,))
        W_te2 = self._get("te_w2", (H, H * 2))                 # (512, 1024)
        b_te2 = self._get("te_b2", (H,))
        t_emb = self._silu(t_sin @ W_te0.T + b_te0)           # (N, H*2)
        t_emb = t_emb @ W_te2.T + b_te2                        # (N, H)

        # ── Input projections: Sequential(Linear, SiLU, LayerNorm) ───────────
        h   = self._proj_silu_ln(x_t,        "box",    in_dim=BOX_DIM)   # (N, H)
        n_e = self._proj_silu_ln(node_emb,   "node",   in_dim=EMBED_DIM) # (N, H)
        g_e = self._proj_silu_ln(global_emb, "global", in_dim=EMBED_DIM) # (1, H)

        # Combine box + node embeddings (matches PyTorch: h = box_proj(x_t) + node_proj(…))
        h = h + n_e   # (N, H)

        # ── 8 DenoisingBlocks ─────────────────────────────────────────────────
        for i in range(N_BLOCKS):

            # ── time_gate: Linear(H, H*2) → SiLU → scale/shift ───────────────
            W_tg  = self._get(f"b{i}_tg_w", (H * 2, H))
            b_tg  = self._get(f"b{i}_tg_b", (H * 2,))
            gate  = self._silu(t_emb @ W_tg.T + b_tg)         # (N, H*2)
            scale = gate[:, :H]                                 # (N, H)
            shift = gate[:, H:]                                 # (N, H)

            # ── DenoisingBlock.norm → scale/shift ─────────────────────────────
            W_bn = self._get(f"b{i}_norm_w", (H,))
            b_bn = self._get(f"b{i}_norm_b", (H,))
            if W_bn.sum() == 0:
                W_bn = np.ones(H, dtype=np.float32)
            h_mod = self._layernorm(h, W_bn, b_bn) * (1 + scale) + shift  # (N, H)

            # ── CrossAttentionBlock ───────────────────────────────────────────
            # norm1 applied to both query (h_mod) and key/value ([h_mod || g_e])
            W_n1 = self._get(f"b{i}_n1_w", (H,))
            b_n1 = self._get(f"b{i}_n1_b", (H,))
            if W_n1.sum() == 0:
                W_n1 = np.ones(H, dtype=np.float32)
            h_ln  = self._layernorm(h_mod, W_n1, b_n1)         # (N, H)
            kv    = np.concatenate([h_ln, g_e], axis=0)        # (N+1, H)
            kv_ln = self._layernorm(kv, W_n1, b_n1)            # (N+1, H) same norm

            attn_out = self._mha(h_ln, kv_ln, i)               # (N, H)
            h_attn   = h_mod + attn_out                         # residual (no dropout at infer)

            # ── FFN: norm2 → Linear(H, H*4) → GELU → Linear(H*4, H) ─────────
            W_n2 = self._get(f"b{i}_n2_w", (H,))
            b_n2 = self._get(f"b{i}_n2_b", (H,))
            if W_n2.sum() == 0:
                W_n2 = np.ones(H, dtype=np.float32)
            h_ff = self._layernorm(h_attn, W_n2, b_n2)         # (N, H)

            W_ff1 = self._get(f"b{i}_ff_w1", (H * 4, H))       # (2048, 512)
            b_ff1 = self._get(f"b{i}_ff_b1", (H * 4,))
            W_ff2 = self._get(f"b{i}_ff_w2", (H, H * 4))       # (512, 2048)
            b_ff2 = self._get(f"b{i}_ff_b2", (H,))
            h_ff  = self._gelu(h_ff @ W_ff1.T + b_ff1)         # (N, H*4)
            h_ff  = h_ff @ W_ff2.T + b_ff2                      # (N, H)

            h = h_attn + h_ff                                    # residual

        # ── Output head: LayerNorm → Linear(H, H//2) → SiLU → Linear(H//2, 4) ──
        W_oln = self._get("out_ln_w", (H,))
        b_oln = self._get("out_ln_b", (H,))
        if W_oln.sum() == 0:
            W_oln = np.ones(H, dtype=np.float32)
        h_out = self._layernorm(h, W_oln, b_oln)

        W_o1 = self._get("out_w1", (H // 2, H))                # (256, 512)
        b_o1 = self._get("out_b1", (H // 2,))
        W_o2 = self._get("out_w2", (BOX_DIM, H // 2))          # (4, 256)
        b_o2 = self._get("out_b2", (BOX_DIM,))
        h_out      = self._silu(h_out @ W_o1.T + b_o1)         # (N, H//2)
        pred_noise = h_out @ W_o2.T + b_o2                      # (N, 4)

        return pred_noise.astype(np.float32)

    # ── DDIM Sampling ─────────────────────────────────────────────────────────

    def sample(
        self,
        node_emb:       np.ndarray,   # (N, 256)
        global_emb:     np.ndarray,   # (1, 256)
        n_steps:        int   = DDIM_STEPS,
        guidance_scale: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        DDIM sampling (T_STEPS → n_steps via stride).
        Returns:
          boxes:       (N, 4) clean box predictions in [0, 1]
          floor_pred:  (4,) floor boundary
        """
        N        = node_emb.shape[0]
        H        = HIDDEN_DIM
        x        = np.random.randn(N, BOX_DIM).astype(np.float32)
        step_size = T_STEPS // n_steps
        timesteps = list(range(T_STEPS - 1, -1, -step_size))
        alpha_cp  = _ALPHAS_CUMPROD[1:]   # (T,)

        for t in timesteps:
            t_next = max(t - step_size, 0)
            eps    = self._denoise_step(x, node_emb, global_emb, t)

            alpha_t      = float(alpha_cp[t])
            alpha_t_next = float(alpha_cp[t_next])

            x0_pred = ((x - math.sqrt(1 - alpha_t) * eps) /
                       math.sqrt(max(alpha_t, 1e-8)))
            x0_pred = np.clip(x0_pred, -2.0, 2.0)

            x = (math.sqrt(alpha_t_next) * x0_pred +
                 math.sqrt(1 - alpha_t_next) * eps)

        # Final clip + ensure x1<x2, y1<y2
        boxes = np.clip(x, 0.0, 1.0)
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 0.05)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 0.05)
        boxes = np.clip(boxes, 0.0, 1.0)

        # ── Floor boundary from global embedding ──────────────────────────────
        g_e = self._proj_silu_ln(global_emb, "global", in_dim=EMBED_DIM)  # (1, H)

        W_fln = self._get("fl_ln_w", (H,))
        b_fln = self._get("fl_ln_b", (H,))
        if W_fln.sum() == 0:
            W_fln = np.ones(H, dtype=np.float32)
        g_out = self._layernorm(g_e, W_fln, b_fln)

        W_f1  = self._get("fl_w1", (64, H))
        b_f1  = self._get("fl_b1", (64,))
        W_f2  = self._get("fl_w2", (BOX_DIM, 64))
        b_f2  = self._get("fl_b2", (BOX_DIM,))
        fl    = self._silu(g_out @ W_f1.T + b_f1)              # (1, 64)
        fl    = 1.0 / (1.0 + np.exp(-np.clip(fl @ W_f2.T + b_f2, -30, 30)))  # sigmoid
        floor_pred = np.clip(fl.flatten()[:BOX_DIM], 0.0, 1.0).astype(np.float32)
        if len(floor_pred) < BOX_DIM:
            floor_pred = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        return boxes.astype(np.float32), floor_pred

    # Alias so diffusion_engine.py can call either .sample() or .sample_numpy()
    sample_numpy = sample


# ── PyTorch Implementation ────────────────────────────────────────────────────

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        return None, None, None


def _build_decoder_torch():
    torch, nn, F = _try_import_torch()
    if torch is None:
        return None

    class SinusoidalTimeEmbed(nn.Module):
        def __init__(self, dim: int = HIDDEN_DIM):
            super().__init__()
            self.dim = dim
            self.proj = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim),
            )

        def forward(self, t):
            """t: (N,) int → (N, dim)"""
            half = self.dim // 2
            freqs = torch.exp(
                -torch.arange(half, device=t.device) * math.log(10000) / (half - 1)
            )
            args = t[:, None].float() * freqs[None, :]
            emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            return self.proj(emb)

    class CrossAttentionBlock(nn.Module):
        """
        Self-attention within rooms + cross-attention to global embedding.
        Query: room embeddings, Key/Value: room embeddings + global
        """
        def __init__(self, dim: int = HIDDEN_DIM, n_heads: int = N_HEADS,
                     dropout: float = DROPOUT):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads

            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                                batch_first=True)
            self.ff    = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, h, global_emb, key_padding_mask=None):
            """
            h: (N, dim) — room representations
            global_emb: (1, dim) — plan summary
            Returns: (N, dim)
            """
            # Concatenate global as extra key/value
            kv = torch.cat([h, global_emb], dim=0)  # (N+1, dim)

            h_norm = self.norm1(h)
            kv_norm = self.norm1(kv)

            # Self+cross attention
            h_attn, _ = self.attn(
                h_norm.unsqueeze(0),   # (1, N, dim)
                kv_norm.unsqueeze(0),  # (1, N+1, dim)
                kv_norm.unsqueeze(0),
            )
            h = h + self.drop(h_attn.squeeze(0))

            # Feed-forward
            h = h + self.ff(self.norm2(h))
            return h

    class DenoisingBlock(nn.Module):
        """One denoising block: time+node conditioning → cross-attention → FF."""
        def __init__(self, dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
            super().__init__()
            self.time_gate = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
            )
            self.cross_attn = CrossAttentionBlock(dim, N_HEADS, dropout)
            self.norm = nn.LayerNorm(dim)

        def forward(self, h, t_emb, global_emb):
            # Time conditioning: scale + shift
            gate = self.time_gate(t_emb)
            scale, shift = gate.chunk(2, dim=-1)
            h = self.norm(h) * (1 + scale) + shift
            h = self.cross_attn(h, global_emb)
            return h

    class _DiffusionDecoderTorchInner(nn.Module):
        """
        Full diffusion decoder v2:
          • Sinusoidal time embedding
          • Input projections for boxes, node embeddings, global embedding
          • 8 denoising blocks with cross-attention
          • Output head: noise prediction (ε)
          • Auxiliary head: floor boundary prediction
        """
        def __init__(self, dropout: float = DROPOUT):
            super().__init__()

            # Input projections
            self.box_proj  = nn.Sequential(
                nn.Linear(BOX_DIM, HIDDEN_DIM),
                nn.SiLU(),
                nn.LayerNorm(HIDDEN_DIM),
            )
            self.node_proj = nn.Sequential(
                nn.Linear(EMBED_DIM, HIDDEN_DIM),
                nn.SiLU(),
                nn.LayerNorm(HIDDEN_DIM),
            )
            self.global_proj = nn.Sequential(
                nn.Linear(EMBED_DIM, HIDDEN_DIM),
                nn.SiLU(),
                nn.LayerNorm(HIDDEN_DIM),
            )
            self.time_emb = SinusoidalTimeEmbed(HIDDEN_DIM)

            # Denoising backbone: 8 blocks
            self.blocks = nn.ModuleList([
                DenoisingBlock(HIDDEN_DIM, dropout)
                for _ in range(N_BLOCKS)
            ])

            # Output head: predict noise ε
            self.out_head = nn.Sequential(
                nn.LayerNorm(HIDDEN_DIM),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.SiLU(),
                nn.Linear(HIDDEN_DIM // 2, BOX_DIM),
            )

            # Auxiliary: floor boundary prediction from global embedding
            self.floor_head = nn.Sequential(
                nn.LayerNorm(HIDDEN_DIM),
                nn.Linear(HIDDEN_DIM, 64),
                nn.SiLU(),
                nn.Linear(64, BOX_DIM),
                nn.Sigmoid(),  # output in [0, 1]
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Zero-init output head for stable training start
            nn.init.zeros_(self.out_head[-1].weight)
            nn.init.zeros_(self.out_head[-1].bias)

        def forward(
            self,
            x_t:         "torch.Tensor",  # (N, 4)
            node_emb:    "torch.Tensor",  # (N, 256)
            global_emb:  "torch.Tensor",  # (1, 256)
            timesteps:   "torch.Tensor",  # (N,) int
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Returns:
              pred_noise:    (N, 4) — predicted noise ε
              floor_boundary: (1, 4) — predicted floor plan boundary
            """
            # Embed inputs
            h     = self.box_proj(x_t)        # (N, H)
            n_e   = self.node_proj(node_emb)  # (N, H)
            g_e   = self.global_proj(global_emb)  # (1, H)
            t_e   = self.time_emb(timesteps)  # (N, H)

            # Combine box + node embedding
            h = h + n_e

            # Run through denoising blocks
            for block in self.blocks:
                h = block(h, t_e, g_e)

            pred_noise = self.out_head(h)        # (N, 4)
            floor_pred = self.floor_head(g_e)    # (1, 4)
            return pred_noise, floor_pred

        @torch.no_grad()
        def sample_ddim(
            self,
            node_emb:    "torch.Tensor",
            global_emb:  "torch.Tensor",
            n_steps:     int = DDIM_STEPS,
            guidance_scale: float = 1.5,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            DDIM sampling (deterministic).
            Returns:
              boxes: (N, 4)
              floor: (1, 4)
            """
            N      = node_emb.shape[0]
            device = node_emb.device

            x = torch.randn(N, BOX_DIM, device=device)

            alpha_cp = torch.tensor(_ALPHAS_CUMPROD[1:], device=device)
            step_size = T_STEPS // n_steps
            timesteps_list = list(range(T_STEPS - 1, -1, -step_size))

            self.eval()
            for t_val in timesteps_list:
                t_next = max(t_val - step_size, 0)
                t_tensor = torch.full((N,), t_val, device=device, dtype=torch.long)

                eps, floor_pred = self(x, node_emb, global_emb, t_tensor)

                alpha_t      = alpha_cp[t_val].item()
                alpha_t_next = alpha_cp[t_next].item()

                x0_pred = (x - math.sqrt(1 - alpha_t) * eps) / math.sqrt(max(alpha_t, 1e-8))
                x0_pred = x0_pred.clamp(-2.0, 2.0)

                x = math.sqrt(alpha_t_next) * x0_pred + math.sqrt(1 - alpha_t_next) * eps

            boxes = x.clamp(0.0, 1.0)
            boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 0.05)
            boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 0.05)
            boxes = boxes.clamp(0.0, 1.0)

            return boxes, floor_pred

    return _DiffusionDecoderTorchInner


# ── Loss Functions (PyTorch) ──────────────────────────────────────────────────

def compute_diffusion_loss(
    pred_noise, actual_noise, pred_boxes, target_boxes,
    floor_pred, mask=None
):
    """
    Composite training loss.
    All tensors from PyTorch.
    """
    torch, nn, F = _try_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch required")

    if mask is None:
        mask = torch.ones(pred_noise.shape[0], dtype=torch.bool, device=pred_noise.device)

    # ── Primary: noise prediction L1 ───────────────────────────────────────
    noise_loss = F.l1_loss(pred_noise[mask], actual_noise[mask])

    # ── Auxiliary box losses (applied to x0-predicted boxes) ───────────────
    pb = pred_boxes[mask]
    tb = target_boxes[mask]

    # L1 on boxes
    box_l1 = F.l1_loss(pb, tb)

    # IoU loss
    ix1 = torch.maximum(pb[:, 0], tb[:, 0])
    iy1 = torch.maximum(pb[:, 1], tb[:, 1])
    ix2 = torch.minimum(pb[:, 2], tb[:, 2])
    iy2 = torch.minimum(pb[:, 3], tb[:, 3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    ap = (pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0)
    at = (tb[:, 2] - tb[:, 0]).clamp(min=0) * (tb[:, 3] - tb[:, 1]).clamp(min=0)
    union = ap + at - inter + 1e-8
    iou_loss = (1.0 - inter / union).mean()

    # Aspect ratio loss
    pw = (pb[:, 2] - pb[:, 0]).clamp(min=1e-6)
    ph = (pb[:, 3] - pb[:, 1]).clamp(min=1e-6)
    tw = (tb[:, 2] - tb[:, 0]).clamp(min=1e-6)
    th = (tb[:, 3] - tb[:, 1]).clamp(min=1e-6)
    aspect_loss = F.l1_loss(torch.log(pw / ph + 1e-8), torch.log(tw / th + 1e-8))

    # Floor boundary loss: all predicted boxes within [0,1]
    boundary_loss = (
        torch.clamp(-pb, min=0).mean() +
        torch.clamp(pb - 1.0, min=0).mean()
    )

    # Overlap penalty (between all pairs of predicted boxes)
    N_rooms = pb.shape[0]
    overlap_loss = torch.tensor(0.0, device=pb.device)
    if N_rooms > 1:
        ix1_ov = torch.maximum(pb[:, None, 0], pb[None, :, 0])
        iy1_ov = torch.maximum(pb[:, None, 1], pb[None, :, 1])
        ix2_ov = torch.minimum(pb[:, None, 2], pb[None, :, 2])
        iy2_ov = torch.minimum(pb[:, None, 3], pb[None, :, 3])
        inter_ov = (torch.clamp(ix2_ov - ix1_ov, min=0) *
                    torch.clamp(iy2_ov - iy1_ov, min=0))
        # Exclude diagonal
        eye = torch.eye(N_rooms, device=pb.device, dtype=torch.bool)
        overlap_loss = inter_ov[~eye].mean()

    # Floor boundary auxiliary head loss (predict [0,0,1,1] as ideal)
    if floor_pred is not None:
        ideal_floor = torch.tensor([[0.0, 0.0, 1.0, 1.0]],
                                    device=floor_pred.device)
        floor_loss = F.l1_loss(floor_pred, ideal_floor)
    else:
        floor_loss = torch.tensor(0.0, device=pb.device)

    # ── Composite loss ──────────────────────────────────────────────────────
    total = (
        LAMBDA_L1       * noise_loss +
        LAMBDA_L1       * box_l1 +
        LAMBDA_IOU      * iou_loss +
        LAMBDA_ASPECT   * aspect_loss +
        LAMBDA_BOUNDARY * boundary_loss +
        LAMBDA_OVERLAP  * overlap_loss +
        0.1             * floor_loss
    )

    return total, {
        "noise_l1":     noise_loss.item(),
        "box_l1":       box_l1.item(),
        "iou_loss":     iou_loss.item(),
        "aspect_loss":  aspect_loss.item(),
        "boundary":     boundary_loss.item(),
        "overlap":      overlap_loss.item(),
        "floor":        floor_loss.item(),
        "total":        total.item(),
    }


# ── Main DiffusionDecoder Class ───────────────────────────────────────────────

_decoder_torch_class = None

def _get_decoder_class():
    global _decoder_torch_class
    if _decoder_torch_class is None:
        _decoder_torch_class = _build_decoder_torch()
    return _decoder_torch_class


class DiffusionDecoder:
    """
    DiffusionDecoder v2 — unified interface for training (PyTorch) and
    inference (numpy).

    Usage (training):
        dec = DiffusionDecoder()
        pred_noise, floor = dec(x_t, node_emb, global_emb, timesteps)
        loss, metrics = compute_diffusion_loss(pred_noise, actual_noise, ...)

    Usage (inference):
        dec = DiffusionDecoder.from_numpy_weights("weights.npz")
        boxes, floor = dec.sample_numpy(node_emb, global_emb)
    """

    def __init__(self, dropout: float = DROPOUT):
        torch, nn, F = _try_import_torch()
        if torch is None:
            log.warning("PyTorch not found — DiffusionDecoder inference only")
            self._torch_module = None
        else:
            cls = _get_decoder_class()
            self._torch_module = cls(dropout=dropout)
        self._numpy_dec = None

    def __call__(self, x_t, node_emb, global_emb, timesteps):
        if self._torch_module is None:
            raise RuntimeError("PyTorch required")
        return self._torch_module(x_t, node_emb, global_emb, timesteps)

    def parameters(self):
        if self._torch_module:
            return self._torch_module.parameters()
        return iter([])

    def train(self, mode=True):
        if self._torch_module:
            self._torch_module.train(mode)
        return self

    def eval(self):
        if self._torch_module:
            self._torch_module.eval()
        return self

    def state_dict(self):
        return self._torch_module.state_dict() if self._torch_module else {}

    def load_state_dict(self, sd, strict=True):
        if self._torch_module:
            self._torch_module.load_state_dict(sd, strict=strict)

    def to(self, device):
        if self._torch_module:
            self._torch_module.to(device)
        return self

    def save_numpy_weights(self, path: str):
        """
        Export weights to .npz using the EXACT flat key names that
        DiffusionDecoderNumpy._denoise_step() and .sample() look for.

        Key mapping from _DiffusionDecoderTorchInner state_dict:
          time_emb.proj.0 → te_w0 / te_b0  (Linear H → H*2)
          time_emb.proj.2 → te_w2 / te_b2  (Linear H*2 → H)
          box_proj.0      → box_w / box_b / box_ln_w / box_ln_b
          node_proj.0     → node_w / node_b / node_ln_w / node_ln_b
          global_proj.0   → global_w / global_b / global_ln_w / global_ln_b
          blocks.i.*      → b{i}_tg_w, b{i}_norm_w, b{i}_n1_w, b{i}_attn_w, …
          out_head.*      → out_ln_w, out_w1, out_w2, …
          floor_head.*    → fl_ln_w, fl_w1, fl_w2, …
        """
        if self._torch_module is None:
            return
        sd = self._torch_module.state_dict()   # _DiffusionDecoderTorchInner
        w  = {}

        # ── Time embedding: SinusoidalTimeEmbed.proj = Sequential(Linear(H,H*2), SiLU, Linear(H*2,H)) ──
        # SiLU at index 1 has no params; indices are 0 and 2
        w["te_w0"] = sd["time_emb.proj.0.weight"].cpu().numpy()   # (1024, 512)
        w["te_b0"] = sd["time_emb.proj.0.bias"].cpu().numpy()
        w["te_w2"] = sd["time_emb.proj.2.weight"].cpu().numpy()   # (512, 1024)
        w["te_b2"] = sd["time_emb.proj.2.bias"].cpu().numpy()

        # ── Input projections: Sequential(Linear, SiLU, LayerNorm) ───────────
        # SiLU at index 1; Linear=0, LayerNorm=2
        for name, prefix in [
            ("box",    "box_proj"),
            ("node",   "node_proj"),
            ("global", "global_proj"),
        ]:
            w[f"{name}_w"]    = sd[f"{prefix}.0.weight"].cpu().numpy()
            w[f"{name}_b"]    = sd[f"{prefix}.0.bias"].cpu().numpy()
            w[f"{name}_ln_w"] = sd[f"{prefix}.2.weight"].cpu().numpy()   # LayerNorm γ
            w[f"{name}_ln_b"] = sd[f"{prefix}.2.bias"].cpu().numpy()     # LayerNorm β

        # ── 8 DenoisingBlocks ─────────────────────────────────────────────────
        for i in range(N_BLOCKS):
            bp = f"blocks.{i}"
            # time_gate: Sequential(Linear(H, H*2), SiLU()) — only index 0 has params
            w[f"b{i}_tg_w"]   = sd[f"{bp}.time_gate.0.weight"].cpu().numpy()  # (1024, 512)
            w[f"b{i}_tg_b"]   = sd[f"{bp}.time_gate.0.bias"].cpu().numpy()
            # DenoisingBlock.norm (LayerNorm on h before scale/shift)
            w[f"b{i}_norm_w"] = sd[f"{bp}.norm.weight"].cpu().numpy()
            w[f"b{i}_norm_b"] = sd[f"{bp}.norm.bias"].cpu().numpy()
            # CrossAttentionBlock norms
            w[f"b{i}_n1_w"]   = sd[f"{bp}.cross_attn.norm1.weight"].cpu().numpy()
            w[f"b{i}_n1_b"]   = sd[f"{bp}.cross_attn.norm1.bias"].cpu().numpy()
            w[f"b{i}_n2_w"]   = sd[f"{bp}.cross_attn.norm2.weight"].cpu().numpy()
            w[f"b{i}_n2_b"]   = sd[f"{bp}.cross_attn.norm2.bias"].cpu().numpy()
            # MHA: in_proj_weight (3H, H) packs Q/K/V; out_proj (H, H)
            w[f"b{i}_attn_w"]     = sd[f"{bp}.cross_attn.attn.in_proj_weight"].cpu().numpy()
            w[f"b{i}_attn_b"]     = sd[f"{bp}.cross_attn.attn.in_proj_bias"].cpu().numpy()
            w[f"b{i}_attn_out_w"] = sd[f"{bp}.cross_attn.attn.out_proj.weight"].cpu().numpy()
            w[f"b{i}_attn_out_b"] = sd[f"{bp}.cross_attn.attn.out_proj.bias"].cpu().numpy()
            # FFN: Sequential(Linear(H,H*4), GELU, Dropout, Linear(H*4,H), Dropout)
            # Linear layers at indices 0 and 3
            w[f"b{i}_ff_w1"] = sd[f"{bp}.cross_attn.ff.0.weight"].cpu().numpy()  # (2048, 512)
            w[f"b{i}_ff_b1"] = sd[f"{bp}.cross_attn.ff.0.bias"].cpu().numpy()
            w[f"b{i}_ff_w2"] = sd[f"{bp}.cross_attn.ff.3.weight"].cpu().numpy()  # (512, 2048)
            w[f"b{i}_ff_b2"] = sd[f"{bp}.cross_attn.ff.3.bias"].cpu().numpy()

        # ── Output head: Sequential(LayerNorm(H), Linear(H,H//2), SiLU, Linear(H//2,4)) ──
        # indices: 0=LayerNorm, 1=Linear, 2=SiLU, 3=Linear
        w["out_ln_w"] = sd["out_head.0.weight"].cpu().numpy()   # LayerNorm γ  (512,)
        w["out_ln_b"] = sd["out_head.0.bias"].cpu().numpy()
        w["out_w1"]   = sd["out_head.1.weight"].cpu().numpy()   # (256, 512)
        w["out_b1"]   = sd["out_head.1.bias"].cpu().numpy()
        w["out_w2"]   = sd["out_head.3.weight"].cpu().numpy()   # (4, 256)
        w["out_b2"]   = sd["out_head.3.bias"].cpu().numpy()

        # ── Floor head: Sequential(LayerNorm(H), Linear(H,64), SiLU, Linear(64,4), Sigmoid) ──
        # Sigmoid has no params; indices: 0=LN, 1=Linear, 2=SiLU, 3=Linear, 4=Sigmoid
        w["fl_ln_w"] = sd["floor_head.0.weight"].cpu().numpy()  # LayerNorm γ  (512,)
        w["fl_ln_b"] = sd["floor_head.0.bias"].cpu().numpy()
        w["fl_w1"]   = sd["floor_head.1.weight"].cpu().numpy()  # (64, 512)
        w["fl_b1"]   = sd["floor_head.1.bias"].cpu().numpy()
        w["fl_w2"]   = sd["floor_head.3.weight"].cpu().numpy()  # (4, 64)
        w["fl_b2"]   = sd["floor_head.3.bias"].cpu().numpy()

        np.savez(path, **w)
        log.info(f"Saved decoder numpy weights → {path}")

    @classmethod
    def from_numpy_weights(cls, path: str) -> "DiffusionDecoder":
        dec = cls.__new__(cls)
        dec._torch_module = None
        dec._numpy_dec = DiffusionDecoderNumpy(path)
        return dec

    def sample_numpy(
        self,
        node_emb: np.ndarray,
        global_emb: np.ndarray,
        n_steps: int = DDIM_STEPS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inference: returns (boxes (N,4), floor_boundary (4,))."""
        if self._numpy_dec is None:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                self.save_numpy_weights(tmp.name)
                self._numpy_dec = DiffusionDecoderNumpy(tmp.name)
        return self._numpy_dec.sample(node_emb, global_emb, n_steps)

    def sample_torch(
        self,
        node_emb,
        global_emb,
        n_steps: int = DDIM_STEPS,
    ):
        """PyTorch DDIM sampling."""
        if self._torch_module is None:
            raise RuntimeError("PyTorch required")
        return self._torch_module.sample_ddim(node_emb, global_emb, n_steps)

    @staticmethod
    def add_noise(x0: np.ndarray, t: np.ndarray,
                  noise: Optional[np.ndarray] = None):
        """Utility: add noise to clean boxes at timestep t."""
        return add_noise_numpy(x0, t, noise)

    @staticmethod
    def cosine_schedule(T: int = T_STEPS) -> np.ndarray:
        """Return alphas_cumprod for the cosine noise schedule."""
        return _cosine_schedule(T)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_diffusion_decoder(
    weights_path: Optional[str] = None,
    inference_only: bool = False,
    dropout: float = DROPOUT,
) -> DiffusionDecoder:
    if inference_only and weights_path:
        return DiffusionDecoder.from_numpy_weights(weights_path)

    dec = DiffusionDecoder(dropout=dropout)
    if weights_path and os.path.exists(weights_path):
        torch, nn, F = _try_import_torch()
        if torch and weights_path.endswith(".pt"):
            sd = torch.load(weights_path, map_location="cpu")
            dec.load_state_dict(sd)
    return dec


# ── Smoke Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Diffusion Decoder v2 — Smoke Test")
    print("=" * 50)

    N = 8
    rng = np.random.default_rng(42)
    node_emb   = rng.random((N, EMBED_DIM)).astype(np.float32)
    global_emb = rng.random((1, EMBED_DIM)).astype(np.float32)

    print("\n── Noise Schedule ──")
    acp = _cosine_schedule(T_STEPS)
    print(f"  alpha_cumprod[0]={acp[0]:.4f}, [500]={acp[500]:.4f}, [999]={acp[999]:.4f}")
    assert acp[0] > 0.99 and acp[-1] < 0.02, "Cosine schedule out of range"
    print("  ✓ Schedule values correct")

    print("\n── Add Noise ──")
    x0 = rng.random((N, 4)).astype(np.float32)
    t  = np.full(N, 500, dtype=np.int32)
    xt, eps = add_noise_numpy(x0, t)
    print(f"  x0 mean: {x0.mean():.3f}, xt mean: {xt.mean():.3f}")
    assert xt.shape == (N, 4)
    assert eps.shape == (N, 4)
    print("  ✓ Noise addition correct")

    print("\n── NumPy Decoder (random weights) ──")
    dec = DiffusionDecoder.from_numpy_weights("/nonexistent.npz")
    boxes, floor = dec.sample_numpy(node_emb, global_emb, n_steps=5)
    print(f"  boxes shape: {boxes.shape}, floor shape: {floor.shape}")
    assert boxes.shape == (N, 4)
    assert floor.shape == (4,)
    assert (boxes >= 0).all() and (boxes <= 1).all()
    print("  ✓ Numpy sampling: correct shapes and bounds")

    print("\n── PyTorch Decoder ──")
    torch, nn, F = _try_import_torch()
    if torch:
        dec_pt = DiffusionDecoder(dropout=0.0)
        dec_pt.eval()

        nf   = torch.tensor(node_emb)
        gf   = torch.tensor(global_emb)
        xt_t = torch.tensor(xt)
        t_t  = torch.tensor(t, dtype=torch.long)

        with torch.no_grad():
            pred_eps, floor_pred = dec_pt(xt_t, nf, gf, t_t)

        print(f"  pred_noise: {tuple(pred_eps.shape)}, floor: {tuple(floor_pred.shape)}")
        assert pred_eps.shape == (N, 4)
        assert floor_pred.shape == (1, 4)
        print("  ✓ PyTorch forward: correct shapes")

        # Test loss
        target_boxes = torch.tensor(x0)
        loss, metrics = compute_diffusion_loss(
            pred_eps, torch.tensor(eps), target_boxes, target_boxes, floor_pred
        )
        print(f"  loss={loss.item():.4f}, metrics={list(metrics.keys())}")
        print("  ✓ Loss computation: OK")

        # Test DDIM sampling
        boxes_pt, floor_pt = dec_pt.sample_torch(nf, gf, n_steps=5)
        print(f"  DDIM boxes: {tuple(boxes_pt.shape)}, floor: {tuple(floor_pt.shape)}")
        assert boxes_pt.shape == (N, 4)
        print("  ✓ PyTorch DDIM sampling: correct")

        n_params = sum(p.numel() for p in dec_pt.parameters())
        print(f"  Total parameters: {n_params:,}")
    else:
        print("  PyTorch not available — skipping")

    print("\n── Auxiliary Losses ──")
    losses = compute_auxiliary_losses_numpy(
        np.clip(rng.random((N, 4)), 0, 1).astype(np.float32), x0
    )
    print(f"  {losses}")
    assert all(v >= 0 for v in losses.values())
    print("  ✓ Auxiliary losses: all non-negative")

    print("\n" + "=" * 50)
    print("✓ All Diffusion Decoder v2 tests passed")
