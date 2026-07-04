"""
autoregressive_transformer.py
==============================
Option 4 — GNN + Autoregressive Layout Transformer

Architecture overview
---------------------
This module replaces the DDPM diffusion decoder with an autoregressive
causal transformer that generates room positions ONE ROOM AT A TIME, with
each room's (width, height, x, y) conditioned on ALL previously generated
rooms AND the full GNN graph context.

Why this is fundamentally better than the diffusion approach
------------------------------------------------------------
The diffusion model receives pre-set room sizes from a lookup table and
then finds positions.  Sizes are FIXED before placement begins.

The AR transformer generates both sizes AND positions jointly.  When
generating room i, it has already seen rooms 0..i-1 fully placed
(type + width + height + x + y), so it adapts room i's size to whatever
space remains.  If the living room ended up large, the kitchen will shrink
proportionally — exactly as a human architect would reason.

Token sequence format
---------------------
The model consumes and produces a flat token sequence:

  [GLOBAL | R0_type, R0_cx, R0_cy, R0_w, R0_h | R1_type, ... | END]

  GLOBAL token  : sinusoidal + linear encoding of
                    (net_w, net_h, n_rooms, vastu_on, entrance_cos, entrance_sin)
  ROOM tokens   : 5 slots per room — TYPE (categorical 0-15), CX, CY, W, H (continuous)
  END token     : signals end-of-sequence (optional at inference time)

  Maximum rooms = 25  →  max_seq_len = 1 + 25*5 + 1 = 127

Cross-attention to GNN
----------------------
At EVERY transformer layer, each position cross-attends to ALL N GNN node
embeddings (the 256-dim embeddings produced by GNNEncoder).  This means
every token in the sequence has direct access to the complete room
relationship graph, regardless of autoregressive position.

Output heads
------------
- TYPE positions   → Linear(d_model, 16)  then softmax
- CONTINUOUS pos   → MixtureOfGaussiansHead with 3 components per scalar

Training
--------
Teacher forcing: feed the entire ground-truth prefix, predict each next
token in parallel.  Loss = CrossEntropy(type) + NLL(MoG for cx,cy,w,h).

Inference
---------
Pure autoregressive: generate token-by-token.  Supports temperature
scaling and top-p filtering on the TYPE tokens.

Numpy inference
---------------
DiffusionDecoderNumpy is replaced by LayoutTransformerNumpy which loads
the .npz weights and runs the full forward pass in pure numpy for
deployment without PyTorch.

Usage
-----
  from modules.step4_generate.autoregressive_transformer import (
      LayoutTransformer, LayoutTransformerNumpy,
      LayoutTokenizer, ROOM_VOCAB, NUM_ROOM_TYPES,
  )
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Room vocabulary (must match training/data_prep.py ROOM_VOCAB) ─────────────
ROOM_VOCAB: Dict[str, int] = {
    "living_room":    0,
    "master_bedroom": 1,
    "bedroom":        2,
    "kitchen":        3,
    "dining_room":    4,
    "bathroom":       5,
    "balcony":        6,
    "study":          7,
    "storage":        8,
    "outdoor":        9,
    "hallway":       10,
    "garage":        11,
    "laundry":       12,
    "office":        13,
    "utility":       14,
    "undefined":     15,
}
NUM_ROOM_TYPES: int = len(ROOM_VOCAB)  # 16
ROOM_VOCAB_INV: Dict[int, str] = {v: k for k, v in ROOM_VOCAB.items()}

# Canonical room type normalisation (matches training data_prep.py)
_ROOM_NORMALISE: Dict[str, str] = {
    "master_bedroom": "master_bedroom",
    "bedroom":        "bedroom",
    "bedroom_kids":   "bedroom",
    "bedroom_guest":  "bedroom",
    "living_room":    "living_room",
    "drawing_room":   "living_room",
    "kitchen":        "kitchen",
    "dining_room":    "dining_room",
    "bathroom":       "bathroom",
    "toilet":         "bathroom",
    "balcony":        "balcony",
    "study_room":     "study",
    "study":          "study",
    "store_room":     "storage",
    "utility_room":   "utility",
    "pooja_room":     "undefined",
    "foyer":          "hallway",
    "passage":        "hallway",
    "staircase":      "hallway",
    "car_parking":    "garage",
    "servant_room":   "undefined",
    "gym_room":       "undefined",
    "home_theater":   "undefined",
}

# ── Special token IDs ────────────────────────────────────────────────────────
TOKEN_GLOBAL:   int = 0   # role id for the plot-context token
TOKEN_TYPE:     int = 1   # role id for a room-type token
TOKEN_CX:       int = 2   # role id for centre-x token
TOKEN_CY:       int = 3   # role id for centre-y token
TOKEN_W:        int = 4   # role id for width token
TOKEN_H:        int = 5   # role id for height token
NUM_TOKEN_ROLES: int = 6

# ── Sequence limits ──────────────────────────────────────────────────────────
MAX_ROOMS:    int = 25
TOKENS_PER_ROOM: int = 5          # TYPE, CX, CY, W, H
MAX_SEQ_LEN: int = 1 + MAX_ROOMS * TOKENS_PER_ROOM  # 126

# ── Architecture hyper-params ────────────────────────────────────────────────
D_MODEL:      int = 512
N_HEADS:      int = 8
N_LAYERS:     int = 12
D_FF:         int = 2048
N_MOG_COMPS:  int = 3     # Mixture of Gaussians components for continuous tokens
DROPOUT:      float = 0.1
GNN_DIM:      int = 256   # must match GNNEncoder output dim

# ── Training loss weights (single source of truth) ────────────────────────────
# Imported by model_trainer.py so there is never a discrepancy between the
# loss computed during training and the one referenced in docs/metrics.
TYPE_LOSS_WEIGHT: float = 1.0   # α — CrossEntropy weight for room-type tokens
CONT_LOSS_WEIGHT: float = 2.0   # β — MoG NLL weight for continuous (cx,cy,w,h)


# ============================================================================
# Helper: sinusoidal positional / value encoding
# ============================================================================

def _sinusoidal_embed(values: np.ndarray, d: int) -> np.ndarray:
    """
    Encode a 1-D array of scalar values into a (len(values), d) sinusoidal
    embedding matrix.  Each scalar is broadcast into `d` dimensions using
    sine/cosine at exponentially-spaced frequencies — same as the original
    Transformer position encoding but applied to arbitrary scalar values.

    Args:
        values : float32 array of shape (M,)
        d      : embedding dimension (must be even)

    Returns:
        float32 array of shape (M, d)
    """
    M = len(values)
    d = d - (d % 2)                        # ensure even
    div = np.exp(
        np.arange(0, d, 2, dtype=np.float32) *
        -(math.log(10000.0) / d)
    )                                       # (d/2,)
    v = values[:, None].astype(np.float32)  # (M, 1)
    enc = np.zeros((M, d), dtype=np.float32)
    enc[:, 0::2] = np.sin(v * div)
    enc[:, 1::2] = np.cos(v * div)
    return enc


# ============================================================================
# LayoutTokenizer  —  EnrichedRoom list  ↔  token sequence
# ============================================================================

class LayoutTokenizer:
    """
    Converts an ordered list of EnrichedRoom objects (from the enricher) into
    the flat token sequence that the LayoutTransformer consumes, and converts
    the model's continuous predictions back into room coordinates in feet.

    Token sequence (all normalised to [0, 1]):
      Position 0             : GLOBAL  — plot context vector
      Positions 1..5         : Room 0  — [TYPE_ID, CX, CY, W, H]
      Positions 6..10        : Room 1  — [TYPE_ID, CX, CY, W, H]
      ...

    Continuous values (CX, CY, W, H) are normalised by net buildable dimensions:
      cx_norm = cx_ft / net_w_ft
      cy_norm = cy_ft / net_l_ft
      w_norm  = w_ft  / net_w_ft
      h_norm  = h_ft  / net_l_ft

    The TYPE token stores the integer class index (0-15), NOT one-hot.
    """

    def __init__(
        self,
        net_w_ft: float,
        net_l_ft: float,
        entrance_dir: str = "N",
        vastu_on: bool = False,
    ) -> None:
        self.net_w = max(net_w_ft, 1.0)
        self.net_l = max(net_l_ft, 1.0)
        self.entrance_dir = entrance_dir
        self.vastu_on = vastu_on

        # Entrance direction → (cos, sin) for the global token
        _DIR_ANGLE = {
            "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
            "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0,
        }
        angle = math.radians(_DIR_ANGLE.get(entrance_dir.upper(), 0.0))
        self.entrance_cos = math.cos(angle)
        self.entrance_sin = math.sin(angle)

    # ── Normalise room type string → integer ──────────────────────────────
    @staticmethod
    def room_type_to_id(room_type: str) -> int:
        norm = _ROOM_NORMALISE.get(room_type, room_type)
        return ROOM_VOCAB.get(norm, ROOM_VOCAB["undefined"])

    @staticmethod
    def id_to_room_type(type_id: int) -> str:
        return ROOM_VOCAB_INV.get(int(type_id), "undefined")

    # ── Build global context vector (6-dim) ───────────────────────────────
    def global_context(self, n_rooms: int) -> np.ndarray:
        """
        Returns a 6-dim float32 vector representing the plot context.
        This is fed to the linear projection layer as the GLOBAL token input.
        """
        return np.array([
            self.net_w / 100.0,         # normalise by 100 ft typical max
            self.net_l / 100.0,
            min(n_rooms, MAX_ROOMS) / MAX_ROOMS,
            1.0 if self.vastu_on else 0.0,
            self.entrance_cos,
            self.entrance_sin,
        ], dtype=np.float32)

    # ── Decode placed boxes (N, 4) norm → feet ────────────────────────────
    def decode_boxes(self, boxes_norm: np.ndarray) -> np.ndarray:
        """
        Convert normalised (cx, cy, w, h) array (N, 4) to absolute feet.

        Returns (N, 4): [x_ft, y_ft, w_ft, h_ft]  (x,y = bottom-left corner)
        """
        cx_ft = boxes_norm[:, 0] * self.net_w
        cy_ft = boxes_norm[:, 1] * self.net_l
        w_ft  = np.clip(boxes_norm[:, 2] * self.net_w, 0.5, self.net_w)
        h_ft  = np.clip(boxes_norm[:, 3] * self.net_l, 0.5, self.net_l)
        x_ft  = np.clip(cx_ft - w_ft * 0.5, 0.0, self.net_w - w_ft)
        y_ft  = np.clip(cy_ft - h_ft * 0.5, 0.0, self.net_l - h_ft)
        return np.stack([x_ft, y_ft, w_ft, h_ft], axis=1)

    # ── Encode ground-truth rooms → sequence for training ─────────────────
    def encode_rooms_to_sequence(
        self,
        rooms: list,   # list of RoomRecord (from data_prep)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a list of RoomRecord objects to:
          type_ids     : int32  (N,)  — room type class index per room
          boxes_norm   : float32 (N, 4) — normalised [cx, cy, w, h] per room
          token_roles  : int32  (1 + N*5,) — role ID per sequence position

        Used by the training data pipeline.
        """
        N = len(rooms)
        type_ids   = np.zeros(N, dtype=np.int32)
        boxes_norm = np.zeros((N, 4), dtype=np.float32)
        token_roles = np.zeros(1 + N * TOKENS_PER_ROOM, dtype=np.int32)
        token_roles[0] = TOKEN_GLOBAL

        for i, r in enumerate(rooms):
            type_ids[i] = ROOM_VOCAB.get(r.room_type, ROOM_VOCAB["undefined"])
            cx = (r.x1 + r.x2) * 0.5
            cy = (r.y1 + r.y2) * 0.5
            w  = r.x2 - r.x1
            h  = r.y2 - r.y1
            boxes_norm[i] = [cx, cy, w, h]
            base = 1 + i * TOKENS_PER_ROOM
            token_roles[base + 0] = TOKEN_TYPE
            token_roles[base + 1] = TOKEN_CX
            token_roles[base + 2] = TOKEN_CY
            token_roles[base + 3] = TOKEN_W
            token_roles[base + 4] = TOKEN_H

        return type_ids, boxes_norm, token_roles


# ============================================================================
# PyTorch LayoutTransformer  (training + validation)
# ============================================================================

class MixtureOfGaussiansHead:
    """
    Pure-numpy inference helper for the Mixture of Gaussians output head.
    Predicts a distribution over a single continuous scalar using K Gaussians.

    PyTorch training counterpart is implemented inline in LayoutTransformer.
    """

    def __init__(self, weights_npz: np.lib.npyio.NpzFile, prefix: str) -> None:
        self.w = weights_npz[f"{prefix}_w"].astype(np.float32)  # (d_model, 3K)
        self.b = weights_npz[f"{prefix}_b"].astype(np.float32)  # (3K,)
        self.K = self.b.shape[0] // 3

    def forward(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            h : (B, d_model) hidden state
        Returns:
            logits   : (B, K) mixture weights (unnormalised)
            means    : (B, K) Gaussian means
            log_stds : (B, K) Gaussian log standard deviations
        """
        out = h @ self.w.T + self.b      # (B, 3K)
        K = self.K
        logits   = out[:, :K]
        means    = out[:, K:2*K]
        log_stds = np.clip(out[:, 2*K:], -6.0, 2.0)
        return logits, means, log_stds

    def sample(
        self,
        h: np.ndarray,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Draw one sample per batch element from the predicted MoG distribution.

        Args:
            h           : (B, d_model)
            temperature : controls sharpness (lower = more deterministic)
            rng         : numpy random generator (seeded for reproducibility)
        Returns:
            samples : (B,) float32 — one continuous value per batch element
        """
        if rng is None:
            rng = np.random.default_rng()
        logits, means, log_stds = self.forward(h)

        # Softmax over mixture weights with temperature
        logits_t = logits / max(temperature, 1e-6)
        exp_l = np.exp(logits_t - logits_t.max(axis=-1, keepdims=True))
        weights = exp_l / exp_l.sum(axis=-1, keepdims=True)   # (B, K)

        # Sample one component per example
        B = h.shape[0]
        chosen = np.array([
            rng.choice(self.K, p=weights[b])
            for b in range(B)
        ], dtype=np.int32)                                     # (B,)

        means_sel    = means[np.arange(B), chosen]             # (B,)
        log_stds_sel = log_stds[np.arange(B), chosen]         # (B,)
        stds_sel     = np.exp(log_stds_sel)

        noise = rng.standard_normal(B).astype(np.float32)
        return (means_sel + stds_sel * noise * temperature).astype(np.float32)


# ============================================================================
# Numpy LayerNorm, Linear, MultiheadAttention helpers
# ============================================================================

def _np_layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var( axis=-1, keepdims=True)
    return w * (x - mu) / np.sqrt(var + eps) + b


def _np_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _np_mha(
    Q: np.ndarray,           # (T_q, d_model)
    K: np.ndarray,           # (T_k, d_model)
    V: np.ndarray,           # (T_k, d_model)
    Wq: np.ndarray,          # (d_model, d_model)
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    n_heads: int,
    mask: Optional[np.ndarray] = None,   # (T_q, T_k) bool — True = mask out
) -> np.ndarray:
    """
    Multi-head attention (numpy, single-batch).
    mask: True positions are filled with -1e9 before softmax.
    """
    T_q, d = Q.shape
    T_k    = K.shape[0]
    d_h    = d // n_heads

    q = (Q @ Wq).reshape(T_q, n_heads, d_h).transpose(1, 0, 2)  # (H, T_q, d_h)
    k = (K @ Wk).reshape(T_k, n_heads, d_h).transpose(1, 0, 2)  # (H, T_k, d_h)
    v = (V @ Wv).reshape(T_k, n_heads, d_h).transpose(1, 0, 2)  # (H, T_k, d_h)

    scale = 1.0 / math.sqrt(d_h)
    scores = np.matmul(q, k.transpose(0, 2, 1)) * scale          # (H, T_q, T_k)
    if mask is not None:
        scores[:, mask] = -1e9

    attn = _np_softmax(scores, axis=-1)                           # (H, T_q, T_k)
    out  = np.matmul(attn, v)                                     # (H, T_q, d_h)
    out  = out.transpose(1, 0, 2).reshape(T_q, d)                 # (T_q, d)
    return out @ Wo


# ============================================================================
# LayoutTransformerNumpy  —  full numpy inference (no PyTorch required)
# ============================================================================

class _ARTransformerBlock:
    """
    One transformer block with:
      1. Causal self-attention  (sequence attends to its own prefix)
      2. Cross-attention        (sequence attends to GNN node embeddings)
      3. Feed-forward network   (2-layer MLP with GELU)
    """

    def __init__(self, npz: np.lib.npyio.NpzFile, i: int) -> None:
        p = f"blk{i}"
        # ── Self-attention weights ─────────────────────────────────
        self.sa_Wq   = npz[f"{p}_sa_Wq"].astype(np.float32)
        self.sa_Wk   = npz[f"{p}_sa_Wk"].astype(np.float32)
        self.sa_Wv   = npz[f"{p}_sa_Wv"].astype(np.float32)
        self.sa_Wo   = npz[f"{p}_sa_Wo"].astype(np.float32)
        self.sa_ln_w = npz[f"{p}_sa_ln_w"].astype(np.float32)
        self.sa_ln_b = npz[f"{p}_sa_ln_b"].astype(np.float32)
        # ── Cross-attention weights ────────────────────────────────
        self.ca_Wq   = npz[f"{p}_ca_Wq"].astype(np.float32)
        self.ca_Wk   = npz[f"{p}_ca_Wk"].astype(np.float32)
        self.ca_Wv   = npz[f"{p}_ca_Wv"].astype(np.float32)
        self.ca_Wo   = npz[f"{p}_ca_Wo"].astype(np.float32)
        self.ca_ln_w = npz[f"{p}_ca_ln_w"].astype(np.float32)
        self.ca_ln_b = npz[f"{p}_ca_ln_b"].astype(np.float32)
        # ── FFN weights ───────────────────────────────────────────
        self.ff_w1   = npz[f"{p}_ff_w1"].astype(np.float32)
        self.ff_b1   = npz[f"{p}_ff_b1"].astype(np.float32)
        self.ff_w2   = npz[f"{p}_ff_w2"].astype(np.float32)
        self.ff_b2   = npz[f"{p}_ff_b2"].astype(np.float32)
        self.ff_ln_w = npz[f"{p}_ff_ln_w"].astype(np.float32)
        self.ff_ln_b = npz[f"{p}_ff_ln_b"].astype(np.float32)
        # ── Infer n_heads from weight shape ───────────────────────
        self.n_heads = N_HEADS

    def forward(
        self,
        h: np.ndarray,           # (T, d_model)  sequence hidden states
        ctx: np.ndarray,         # (N, d_model)  GNN node embeddings (cross-attn keys/values)
        causal_mask: np.ndarray, # (T, T) bool   True = masked out
    ) -> np.ndarray:
        T = h.shape[0]

        # 1. Causal self-attention (pre-norm)
        h_norm = _np_layernorm(h, self.sa_ln_w, self.sa_ln_b)
        sa_out = _np_mha(h_norm, h_norm, h_norm,
                         self.sa_Wq, self.sa_Wk, self.sa_Wv, self.sa_Wo,
                         self.n_heads, mask=causal_mask)
        h = h + sa_out

        # 2. Cross-attention to GNN context (pre-norm)
        h_norm = _np_layernorm(h, self.ca_ln_w, self.ca_ln_b)
        ca_out = _np_mha(h_norm, ctx, ctx,
                         self.ca_Wq, self.ca_Wk, self.ca_Wv, self.ca_Wo,
                         self.n_heads, mask=None)  # no mask: attend to all GNN nodes
        h = h + ca_out

        # 3. FFN (pre-norm)
        h_norm = _np_layernorm(h, self.ff_ln_w, self.ff_ln_b)
        ff_out = _np_gelu(h_norm @ self.ff_w1.T + self.ff_b1) @ self.ff_w2.T + self.ff_b2
        h = h + ff_out

        return h


class LayoutTransformerNumpy:
    """
    Full numpy implementation of the LayoutTransformer for deployment inference
    WITHOUT requiring PyTorch.

    The weights are loaded from a .npz file (produced by LayoutTransformer's
    save_numpy_weights() method).

    Usage
    -----
        tr = LayoutTransformerNumpy("path/to/ar_transformer.npz")
        boxes, type_ids = tr.generate(
            gnn_node_emb,    # (N, 256) from GNNEncoderNumpy
            gnn_global_emb,  # (1, 256)
            tokenizer,       # LayoutTokenizer instance
            n_rooms,         # how many rooms to generate
        )
    """

    def __init__(self, npz_path: str) -> None:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"AR transformer weights not found: {npz_path}")
        npz = np.load(npz_path, allow_pickle=False)

        # ── Token embedding tables ─────────────────────────────────────────
        self.type_emb   = npz["type_emb"].astype(np.float32)    # (NUM_ROOM_TYPES, D_MODEL)
        self.role_emb   = npz["role_emb"].astype(np.float32)    # (NUM_TOKEN_ROLES, D_MODEL)
        self.global_W   = npz["global_W"].astype(np.float32)    # (6, D_MODEL)
        self.global_b   = npz["global_b"].astype(np.float32)    # (D_MODEL,)
        self.cont_W     = npz["cont_W"].astype(np.float32)      # (D_MODEL//2, D_MODEL)
        self.cont_b     = npz["cont_b"].astype(np.float32)      # (D_MODEL,)
        self.ctx_W      = npz["ctx_W"].astype(np.float32)       # (GNN_DIM, D_MODEL)
        self.ctx_b      = npz["ctx_b"].astype(np.float32)       # (D_MODEL,)
        self.pos_emb    = npz["pos_emb"].astype(np.float32)     # (MAX_SEQ_LEN, D_MODEL)

        # ── Final layer norm ───────────────────────────────────────────────
        self.final_ln_w = npz["final_ln_w"].astype(np.float32)  # (D_MODEL,)
        self.final_ln_b = npz["final_ln_b"].astype(np.float32)

        # ── Output heads ───────────────────────────────────────────────────
        self.type_head_W = npz["type_head_W"].astype(np.float32)  # (D_MODEL, 16)
        self.type_head_b = npz["type_head_b"].astype(np.float32)  # (16,)

        # MoG heads for each continuous dimension: cx, cy, w, h
        self.mog_cx = MixtureOfGaussiansHead(npz, "mog_cx")
        self.mog_cy = MixtureOfGaussiansHead(npz, "mog_cy")
        self.mog_w  = MixtureOfGaussiansHead(npz, "mog_w")
        self.mog_h  = MixtureOfGaussiansHead(npz, "mog_h")

        # ── Transformer blocks ─────────────────────────────────────────────
        self.blocks: List[_ARTransformerBlock] = [
            _ARTransformerBlock(npz, i) for i in range(N_LAYERS)
        ]

    # ── Token embedding helpers ────────────────────────────────────────────

    def _embed_global(self, ctx_vec: np.ndarray) -> np.ndarray:
        """Project the 6-dim global context vector to d_model."""
        return ctx_vec @ self.global_W.T + self.global_b      # (D_MODEL,)

    def _embed_type(self, type_id: int) -> np.ndarray:
        """Lookup room-type embedding."""
        return self.type_emb[int(type_id)]                    # (D_MODEL,)

    def _embed_continuous(self, value: float) -> np.ndarray:
        """
        Sinusoidal encoding (D_MODEL//2 dims) → linear projection → (D_MODEL,).
        Encodes a single continuous value in [0, 1].
        """
        sin_enc = _sinusoidal_embed(
            np.array([value], dtype=np.float32),
            D_MODEL // 2,
        )[0]                                                   # (D_MODEL//2,)
        return sin_enc @ self.cont_W.T + self.cont_b          # (D_MODEL,)

    # ── Forward pass ──────────────────────────────────────────────────────

    def _build_sequence(
        self,
        tokenizer: LayoutTokenizer,
        n_rooms: int,
        known_rooms: Optional[List[Tuple[int, float, float, float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build the token embedding sequence up to the current generation step.

        Args:
            tokenizer  : LayoutTokenizer with plot dimensions
            n_rooms    : total number of rooms to generate (for GLOBAL context)
            known_rooms: list of (type_id, cx, cy, w, h) for already-generated rooms

        Returns:
            seq  : (T, D_MODEL) sequence embeddings
            roles: (T,) token role IDs
            T    : current sequence length
        """
        known_rooms = known_rooms or []
        T = 1 + len(known_rooms) * TOKENS_PER_ROOM
        seq   = np.zeros((T, D_MODEL), dtype=np.float32)
        roles = np.zeros(T, dtype=np.int32)

        # ── Position 0: GLOBAL token ─────────────────────────────────────
        ctx_vec = tokenizer.global_context(n_rooms)
        seq[0]  = self._embed_global(ctx_vec)
        roles[0] = TOKEN_GLOBAL

        # ── Positions 1..T-1: already-placed rooms ───────────────────────
        for k, (tid, cx, cy, w, h) in enumerate(known_rooms):
            base = 1 + k * TOKENS_PER_ROOM
            seq[base + 0] = self._embed_type(tid)
            seq[base + 1] = self._embed_continuous(cx)
            seq[base + 2] = self._embed_continuous(cy)
            seq[base + 3] = self._embed_continuous(w)
            seq[base + 4] = self._embed_continuous(h)
            roles[base + 0] = TOKEN_TYPE
            roles[base + 1] = TOKEN_CX
            roles[base + 2] = TOKEN_CY
            roles[base + 3] = TOKEN_W
            roles[base + 4] = TOKEN_H

        # ── Add role embeddings + positional embeddings ───────────────────
        for t in range(T):
            seq[t] += self.role_emb[roles[t]]
            seq[t] += self.pos_emb[t]

        return seq, roles, T

    def _causal_mask(self, T: int) -> np.ndarray:
        """Upper-triangular boolean mask: True = masked out (future positions)."""
        return np.triu(np.ones((T, T), dtype=bool), k=1)

    def _project_context(self, gnn_node_emb: np.ndarray, gnn_global_emb: np.ndarray) -> np.ndarray:
        """
        Project GNN embeddings from GNN_DIM to D_MODEL and concatenate.
        ctx: (N+1, D_MODEL)
        """
        combined = np.concatenate([gnn_node_emb, gnn_global_emb], axis=0)  # (N+1, GNN_DIM)
        return combined @ self.ctx_W.T + self.ctx_b                         # (N+1, D_MODEL)

    def _forward(
        self,
        seq: np.ndarray,    # (T, D_MODEL)
        ctx: np.ndarray,    # (N+1, D_MODEL) projected GNN context
    ) -> np.ndarray:
        """Run the sequence through all transformer blocks. Returns (T, D_MODEL)."""
        T = seq.shape[0]
        causal_mask = self._causal_mask(T)
        h = seq.copy()
        for block in self.blocks:
            h = block.forward(h, ctx, causal_mask)
        return _np_layernorm(h, self.final_ln_w, self.final_ln_b)

    # ── Autoregressive generation ──────────────────────────────────────────

    def generate(
        self,
        gnn_node_emb:   np.ndarray,       # (N, GNN_DIM=256)
        gnn_global_emb: np.ndarray,       # (1, GNN_DIM)
        tokenizer:      LayoutTokenizer,
        n_rooms:        int,
        type_ids_forced: Optional[List[int]] = None,  # if given, skip type sampling
        temperature:    float = 1.0,
        top_p:          float = 0.9,
        seed:           int   = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoregressively generate n_rooms rooms.

        At each step the model generates 5 tokens for the next room:
          TYPE → CX → CY → W → H

        For TYPE, if type_ids_forced is provided, the forced types are used
        (this is the normal inference path where room types are known from the
        enricher).  Otherwise, the model samples from the type head.

        For the continuous tokens (CX, CY, W, H), the MoG head samples a value.

        Args:
            gnn_node_emb    : (N, 256) GNN per-room embeddings
            gnn_global_emb  : (1, 256) GNN global plan embedding
            tokenizer       : LayoutTokenizer with net_w, net_l
            n_rooms         : number of rooms to generate
            type_ids_forced : list of int room type IDs (length n_rooms).
                              Provided by the enricher from the user's requirements.
                              If None, the model samples room types freely.
            temperature     : sampling temperature (lower = more deterministic)
            top_p           : nucleus sampling cutoff for type tokens
            seed            : random seed for reproducibility

        Returns:
            boxes_norm : (n_rooms, 4) float32 — normalised [cx, cy, w, h]
            type_ids   : (n_rooms,)  int32   — room type indices
        """
        rng = np.random.default_rng(seed)
        ctx = self._project_context(gnn_node_emb, gnn_global_emb)  # (N+1, D_MODEL)

        known_rooms: List[Tuple[int, float, float, float, float]] = []
        out_boxes   = np.zeros((n_rooms, 4), dtype=np.float32)
        out_types   = np.zeros(n_rooms, dtype=np.int32)

        for room_idx in range(n_rooms):
            # Build sequence up to current position
            seq, roles, T = self._build_sequence(tokenizer, n_rooms, known_rooms)
            h_seq = self._forward(seq, ctx)   # (T, D_MODEL)
            h_last = h_seq[[-1]]              # (1, D_MODEL) — last hidden state

            # ── Step A: predict room TYPE ─────────────────────────────────
            if type_ids_forced is not None and room_idx < len(type_ids_forced):
                tid = int(type_ids_forced[room_idx])
            else:
                logits = h_last @ self.type_head_W + self.type_head_b  # (1, 16)
                logits = logits[0] / max(temperature, 1e-6)

                # Nucleus (top-p) sampling
                probs = _np_softmax(logits)
                sorted_idx = np.argsort(-probs)
                cum = 0.0
                keep_mask = np.zeros(NUM_ROOM_TYPES, dtype=bool)
                for idx in sorted_idx:
                    cum += probs[idx]
                    keep_mask[idx] = True
                    if cum >= top_p:
                        break
                probs_filtered = probs * keep_mask
                probs_filtered /= probs_filtered.sum()
                tid = int(rng.choice(NUM_ROOM_TYPES, p=probs_filtered))

            # ── Add TYPE token, rebuild sequence, predict CX ──────────────
            # We need 4 more forward passes (CX, CY, W, H), each conditioned
            # on the previous token.  We extend the known_rooms incrementally
            # using placeholder values and replace them after each prediction.

            # 1. Predict CX using the TYPE hidden state (index -5)
            temp_known = known_rooms + [(tid, 0.5, 0.5, 0.1, 0.1)]
            seq2, _, _ = self._build_sequence(tokenizer, n_rooms, temp_known)
            h2 = self._forward(seq2, ctx)
            h_type = h2[[-5]]  # TYPE hidden state predicts CX
            cx = float(np.clip(self.mog_cx.sample(h_type, temperature, rng)[0], 0.05, 0.95))

            # 2. Predict CY using the CX hidden state (index -4)
            temp_known = known_rooms + [(tid, cx, 0.5, 0.1, 0.1)]
            seq2, _, _ = self._build_sequence(tokenizer, n_rooms, temp_known)
            h2 = self._forward(seq2, ctx)
            h_cx = h2[[-4]]  # CX hidden state predicts CY
            cy = float(np.clip(self.mog_cy.sample(h_cx, temperature, rng)[0], 0.05, 0.95))

            # 3. Predict W using the CY hidden state (index -3)
            temp_known = known_rooms + [(tid, cx, cy, 0.1, 0.1)]
            seq2, _, _ = self._build_sequence(tokenizer, n_rooms, temp_known)
            h2 = self._forward(seq2, ctx)
            h_cy_state = h2[[-3]]  # CY hidden state predicts W
            w = float(np.clip(self.mog_w.sample(h_cy_state, temperature, rng)[0], 0.02, 0.95))

            # 4. Predict H using the W hidden state (index -2)
            temp_known = known_rooms + [(tid, cx, cy, w, 0.1)]
            seq2, _, _ = self._build_sequence(tokenizer, n_rooms, temp_known)
            h2 = self._forward(seq2, ctx)
            h_w = h2[[-2]]  # W hidden state predicts H
            h_val = float(np.clip(self.mog_h.sample(h_w, temperature, rng)[0], 0.02, 0.95))

            # Store results
            out_types[room_idx] = tid
            out_boxes[room_idx] = [cx, cy, w, h_val]
            known_rooms.append((tid, cx, cy, w, h_val))

        return out_boxes, out_types


# ============================================================================
# PyTorch LayoutTransformer  (training)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class _MoGHeadTorch(nn.Module):
        """
        Mixture of Gaussians output head for a single continuous dimension.

        Projects the transformer hidden state to K mixture components,
        each parameterised by (log_weight, mean, log_std).

        Training loss: negative log-likelihood under the predicted distribution.
        Inference: sample from the distribution (with temperature scaling).
        """

        def __init__(self, d_model: int = D_MODEL, n_components: int = N_MOG_COMPS) -> None:
            super().__init__()
            self.K = n_components
            self.head = nn.Linear(d_model, 3 * n_components)

        def forward(
            self, h: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Returns (log_weights, means, log_stds) each of shape (B, K).
            """
            out = self.head(h)                  # (B, 3K)
            log_w = out[:, :self.K]
            means = out[:, self.K:2*self.K]
            log_s = out[:, 2*self.K:].clamp(-6.0, 2.0)
            return log_w, means, log_s

        def log_prob(self, h: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            """
            Compute log-probability of each target value under the predicted MoG.

            Args:
                h      : (B, d_model) hidden states
                target : (B,) ground-truth continuous values

            Returns:
                log_probs : (B,) negative log-likelihood (to be minimised)
            """
            log_w, means, log_s = self.forward(h)
            stds   = torch.exp(log_s)
            log_wn = F.log_softmax(log_w, dim=-1)   # (B, K) normalised log weights

            # Per-component Gaussian log prob: -(x-mu)^2/(2sigma^2) - log_sigma - 0.5*log(2pi)
            t_exp  = target.unsqueeze(-1).expand_as(means)                     # (B, K)
            comp_log_p = (
                -0.5 * ((t_exp - means) / stds)**2
                - log_s
                - 0.5 * math.log(2 * math.pi)
            )                                                                   # (B, K)
            # log-sum-exp over components
            log_mix = torch.logsumexp(log_wn + comp_log_p, dim=-1)             # (B,)
            return log_mix   # positive means more likely — negate for loss

        def sample(
            self,
            h: "torch.Tensor",
            temperature: float = 1.0,
        ) -> "torch.Tensor":
            """Sample one value per batch element from the predicted MoG."""
            log_w, means, log_s = self.forward(h)
            weights = F.softmax(log_w / max(temperature, 1e-6), dim=-1)  # (B, K)
            comp    = torch.multinomial(weights, 1).squeeze(-1)            # (B,)
            stds    = torch.exp(log_s)
            means_s = means.gather(1, comp.unsqueeze(-1)).squeeze(-1)      # (B,)
            stds_s  = stds.gather(1, comp.unsqueeze(-1)).squeeze(-1)       # (B,)
            return (means_s + stds_s * torch.randn_like(means_s) * temperature).clamp(-0.5, 1.5)


    class _ARBlock(nn.Module):
        """
        One LayoutTransformer block.

        Sub-layers (all pre-norm):
          1. Causal self-attention  (sequence positions attend to earlier positions)
          2. Cross-attention        (each position attends to all N GNN node embeddings)
          3. Position-wise FFN      (2-layer MLP, GELU activation)

        Supports optional KV-cache for incremental inference (not used during
        teacher-forcing training where the full sequence is processed at once).
        """

        def __init__(
            self,
            d_model: int  = D_MODEL,
            n_heads: int  = N_HEADS,
            d_ff:    int  = D_FF,
            dropout: float = DROPOUT,
        ) -> None:
            super().__init__()
            self.ln_sa = nn.LayerNorm(d_model)
            self.ln_ca = nn.LayerNorm(d_model)
            self.ln_ff = nn.LayerNorm(d_model)

            self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.drop = nn.Dropout(dropout)

        def forward(
            self,
            h:    "torch.Tensor",           # (B, T, d_model)
            ctx:  "torch.Tensor",           # (B, N, d_model) projected GNN context
            mask: "torch.Tensor",           # (T, T) causal attention mask (additive)
        ) -> "torch.Tensor":
            # 1. Causal self-attention
            # NOTE: is_causal=True is NOT used here — it is unsupported on MPS and
            # causes a silent hang.  The upper-triangular -inf attn_mask already
            # enforces causality, so is_causal is redundant AND harmful on MPS.
            h_n = self.ln_sa(h)
            sa, _ = self.self_attn(h_n, h_n, h_n, attn_mask=mask)
            h = h + self.drop(sa)

            # 2. Cross-attention to GNN context
            h_n = self.ln_ca(h)
            ca, _ = self.cross_attn(h_n, ctx, ctx)
            h = h + self.drop(ca)

            # 3. FFN
            h = h + self.ff(self.ln_ff(h))
            return h


    class LayoutTransformer(nn.Module):
        """
        GNN-conditioned Autoregressive Layout Transformer.

        This is the PyTorch training module.  The numpy inference counterpart
        is LayoutTransformerNumpy (above).

        Input
        -----
        The model takes two kinds of input simultaneously:

          seq_input  : (B, T, d_model)   — pre-embedded token sequence
          gnn_context: (B, N, d_model)   — projected GNN node embeddings

        During training, the ENTIRE sequence is fed at once (teacher forcing).
        During inference, tokens are generated one by one.

        Output heads
        ------------
        The model returns hidden states (B, T, d_model).  Separate output
        heads are applied at TYPE and CONTINUOUS positions:

          type_head : Linear → (B, T_type, 16) — room type logits
          mog_cx    : MoG head → distribution over CX
          mog_cy    : MoG head → distribution over CY
          mog_w     : MoG head → distribution over W
          mog_h     : MoG head → distribution over H

        Loss
        ----
        total_loss = CrossEntropy(type_logits, type_targets)
                   + weight_cx * (-MoG_log_prob(cx_hidden, cx_targets))
                   + weight_cy * (-MoG_log_prob(cy_hidden, cy_targets))
                   + weight_w  * (-MoG_log_prob(w_hidden,  w_targets))
                   + weight_h  * (-MoG_log_prob(h_hidden,  h_targets))
        """

        def __init__(
            self,
            d_model:     int   = D_MODEL,
            n_heads:     int   = N_HEADS,
            n_layers:    int   = N_LAYERS,
            d_ff:        int   = D_FF,
            n_mog_comps: int   = N_MOG_COMPS,
            dropout:     float = DROPOUT,
            gnn_dim:     int   = GNN_DIM,
            max_seq_len: int   = MAX_SEQ_LEN,
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.n_layers = n_layers

            # ── Token embedding tables ─────────────────────────────────────
            # Room type embedding (learnable, one per class)
            self.type_emb = nn.Embedding(NUM_ROOM_TYPES, d_model)
            # Token role embedding (GLOBAL, TYPE, CX, CY, W, H)
            self.role_emb = nn.Embedding(NUM_TOKEN_ROLES, d_model)
            # Learnable positional embedding
            self.pos_emb  = nn.Embedding(max_seq_len, d_model)

            # ── Input projections ──────────────────────────────────────────
            # Global context token (6-dim plot info → d_model)
            self.global_proj = nn.Sequential(
                nn.Linear(6, d_model),
                nn.LayerNorm(d_model),
            )
            # Continuous value token (sinusoidal d_model//2 → d_model)
            self.cont_proj = nn.Sequential(
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
            # GNN context projection (gnn_dim → d_model)
            self.ctx_proj = nn.Sequential(
                nn.Linear(gnn_dim, d_model),
                nn.LayerNorm(d_model),
            )

            # ── Transformer blocks ─────────────────────────────────────────
            self.blocks = nn.ModuleList([
                _ARBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.final_ln = nn.LayerNorm(d_model)

            # ── Output heads ───────────────────────────────────────────────
            self.type_head = nn.Linear(d_model, NUM_ROOM_TYPES)
            self.mog_cx    = _MoGHeadTorch(d_model, n_mog_comps)
            self.mog_cy    = _MoGHeadTorch(d_model, n_mog_comps)
            self.mog_w     = _MoGHeadTorch(d_model, n_mog_comps)
            self.mog_h     = _MoGHeadTorch(d_model, n_mog_comps)

            self.drop = nn.Dropout(dropout)
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        # ── Input embedding ────────────────────────────────────────────────

        def embed_sequence(
            self,
            global_ctx:  "torch.Tensor",   # (B, 6)
            type_ids:    "torch.Tensor",   # (B, N)  — int room type IDs
            boxes_norm:  "torch.Tensor",   # (B, N, 4) — normalised [cx, cy, w, h]
            n_rooms:     int,
        ) -> "torch.Tensor":
            """
            Build the full (B, T, d_model) input embedding for teacher-forcing
            training.

            Sequence: [GLOBAL | R0_type, R0_cx, R0_cy, R0_w, R0_h | R1_type, ...]

            Fully vectorised — zero Python loops over rooms or dimensions.
            MPS-safe: all operations are single large tensor ops.
            """
            B   = global_ctx.shape[0]
            T   = 1 + n_rooms * TOKENS_PER_ROOM
            dev = global_ctx.device

            # ── Position embeddings ──────────────────────────────────────
            pos   = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)  # (B, T)
            pos_e = self.pos_emb(pos)                                        # (B, T, d)

            # ── Role IDs — built without any Python loop ────────────────
            # Pattern per room: [TYPE, CX, CY, W, H]
            room_roles = torch.tensor(
                [TOKEN_TYPE, TOKEN_CX, TOKEN_CY, TOKEN_W, TOKEN_H],
                dtype=torch.long, device=dev,
            ).unsqueeze(0).expand(n_rooms, -1).reshape(-1)   # (N*5,)
            role_ids = torch.cat([
                torch.tensor([TOKEN_GLOBAL], dtype=torch.long, device=dev),
                room_roles,
            ])                                                # (T,)
            role_e = self.role_emb(
                role_ids.unsqueeze(0).expand(B, -1)
            )                                                 # (B, T, d)

            # ── Sequence tensor ─────────────────────────────────────────
            seq = torch.zeros(B, T, self.d_model, device=dev, dtype=torch.float32)

            # Global token (position 0)
            seq[:, 0] = self.global_proj(global_ctx)         # (B, d)

            # ── Type tokens — all N rooms in one embedding lookup ───────
            # type_ids: (B, N) → type_all: (B, N, d)
            type_all = self.type_emb(type_ids)               # (B, N, d)
            # Positions 1, 6, 11, ... (every TOKENS_PER_ROOM)
            type_pos = 1 + torch.arange(n_rooms, device=dev) * TOKENS_PER_ROOM  # (N,)
            seq[:, type_pos] = type_all                      # advanced index assign

            # ── Continuous tokens — all B*N*4 values in one shot ────────
            # Precompute sinusoidal frequencies once
            d_sin = self.d_model // 2
            freq  = torch.exp(
                torch.arange(0, d_sin, 2, device=dev, dtype=torch.float32)
                * -(math.log(10000.0) / d_sin)
            )                                                 # (d_sin//2,)

            # Flatten (B, N, 4) → (B*N*4, 1) and encode in one batch
            vals_flat = boxes_norm.reshape(B * n_rooms * 4, 1)   # (M, 1)
            sin_enc   = torch.zeros(B * n_rooms * 4, d_sin, device=dev)
            sin_enc[:, 0::2] = torch.sin(vals_flat * freq)       # (M, d_sin//2)
            sin_enc[:, 1::2] = torch.cos(vals_flat * freq)       # (M, d_sin//2)
            cont_emb  = self.cont_proj(sin_enc)                   # (M, d_model)
            cont_all  = cont_emb.reshape(B, n_rooms, 4, self.d_model)  # (B,N,4,d)

            # Assign cx/cy/w/h into sequence — only 4 iterations (not N*4)
            for off in range(4):
                seq[:, type_pos + off + 1] = cont_all[:, :, off, :]   # (B,N,d)

            return self.drop(seq + role_e + pos_e)

        # ── Context projection ─────────────────────────────────────────────

        def project_context(
            self,
            node_emb:   "torch.Tensor",   # (B, N, GNN_DIM)
            global_emb: "torch.Tensor",   # (B, 1, GNN_DIM)
        ) -> "torch.Tensor":
            """Project GNN embeddings to d_model and concatenate."""
            combined = torch.cat([node_emb, global_emb], dim=1)   # (B, N+1, GNN_DIM)
            return self.ctx_proj(combined)                          # (B, N+1, d_model)

        # ── Causal attention mask ──────────────────────────────────────────

        @staticmethod
        def causal_mask(T: int, device: "torch.device") -> "torch.Tensor":
            """
            Additive causal mask: 0 for positions to attend, -inf for future.
            Shape (T, T).
            """
            m = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            return m

        # ── Forward pass (teacher-forcing training) ─────────────────────

        def forward(
            self,
            seq: "torch.Tensor",    # (B, T, d_model)  — from embed_sequence()
            ctx: "torch.Tensor",    # (B, N+1, d_model) — from project_context()
        ) -> "torch.Tensor":
            """
            Run the sequence through all transformer blocks.

            Args:
                seq : (B, T, d_model) input token embeddings
                ctx : (B, N+1, d_model) projected GNN context

            Returns:
                h   : (B, T, d_model) output hidden states
            """
            T    = seq.shape[1]
            mask = self.causal_mask(T, seq.device)   # (T, T)
            h    = seq
            for block in self.blocks:
                h = block(h, ctx, mask)
            return self.final_ln(h)

        # ── Loss computation ───────────────────────────────────────────────

        def compute_loss(
            self,
            h:           "torch.Tensor",   # (B, T, d_model)  from forward()
            type_ids:    "torch.Tensor",   # (B, N)  target room type IDs
            boxes_norm:  "torch.Tensor",   # (B, N, 4)  target [cx, cy, w, h]
            n_rooms:     int,
        ) -> "torch.Tensor":
            """
            Compute the combined training loss over TYPE and CONTINUOUS tokens.

            TYPE loss:
              CrossEntropy at positions [0, 5, 10, ...] (i.e. each room's TYPE slot)
              Prediction at position t-1 targets the TYPE at position t.

            CONTINUOUS loss (MoG NLL):
              At each TYPE position p, the following 4 positions (p+1, p+2, p+3, p+4)
              target cx, cy, w, h respectively.
            """
            B = h.shape[0]
            type_loss = 0.0
            cx_loss = cy_loss = w_loss = h_loss = 0.0
            count   = 0

            for k in range(n_rooms):
                base = 1 + k * TOKENS_PER_ROOM  # index of TYPE token for room k

                # TYPE: hidden state at base-1 predicts type at base
                # (position 0 is GLOBAL → predicts TYPE of room 0)
                h_type = h[:, base - 1]          # (B, d_model)
                type_logits = self.type_head(h_type)   # (B, 16)
                type_loss += F.cross_entropy(type_logits, type_ids[:, k])

                # CONTINUOUS: hidden states at base..base+3 predict cx,cy,w,h
                h_cx = h[:, base + 0]
                h_cy = h[:, base + 1]
                h_w  = h[:, base + 2]
                h_h  = h[:, base + 3]

                cx_loss -= self.mog_cx.log_prob(h_cx, boxes_norm[:, k, 0]).mean()
                cy_loss -= self.mog_cy.log_prob(h_cy, boxes_norm[:, k, 1]).mean()
                w_loss  -= self.mog_w.log_prob( h_w,  boxes_norm[:, k, 2]).mean()
                h_loss  -= self.mog_h.log_prob( h_h,  boxes_norm[:, k, 3]).mean()
                count   += 1

            n = max(count, 1)
            total = (type_loss + cx_loss + cy_loss + w_loss + h_loss) / n
            return total

        def masked_compute_loss(
            self,
            h:          "torch.Tensor",            # (B, T, d_model)  from forward()
            type_ids:   "torch.Tensor",            # (B, N)  target room type IDs
            boxes_norm: "torch.Tensor",            # (B, N, 4)  target [cx, cy, w, h]
            n_rooms:    int,
            mask:       Optional["torch.Tensor"] = None,  # (B, N) bool — True = real room
        ) -> "torch.Tensor":
            """
            Compute the masked training loss for variable-length batches.

            VECTORISED — no Python loop over rooms.  All N rooms are processed
            in a single batched index-gather + head forward pass.  This gives
            ~3-5x speedup per training step compared to the per-room loop.

            mask[b, k] = True  → sample b has a real room at position k.
            mask[b, k] = False → position k is padding — excluded from the loss.

            Falls back to compute_loss() when mask is None (all positions valid).

            Token indexing (causal, teacher-forcing):
              Sequence position t predicts position t+1.

              For room k:
                h[:, base-1]   → predicts TYPE at base     (base = 1 + k*TOKENS_PER_ROOM)
                h[:, base+0]   → predicts CX   at base+1
                h[:, base+1]   → predicts CY   at base+2
                h[:, base+2]   → predicts W    at base+3
                h[:, base+3]   → predicts H    at base+4

            This is the only place in the codebase where token offsets are
            computed — use this method everywhere to avoid off-by-one errors.
            """
            if mask is None:
                return self.compute_loss(h, type_ids, boxes_norm, n_rooms)

            B, T, d = h.shape
            dev     = h.device

            # ── Compute all base positions vectorised ─────────────────────
            # base[k] = 1 + k * TOKENS_PER_ROOM  (position of TYPE token for room k)
            room_indices = torch.arange(n_rooms, device=dev)          # (N,)
            bases        = 1 + room_indices * TOKENS_PER_ROOM         # (N,)

            # ── Gather hidden states for TYPE predictions ─────────────────
            # TYPE is predicted from h[:, base-1] for each room k
            type_positions = (bases - 1).unsqueeze(0).expand(B, -1)   # (B, N)
            h_type_all = torch.gather(
                h, 1, type_positions.unsqueeze(-1).expand(-1, -1, d)
            )                                                         # (B, N, d)

            # ── Gather hidden states for continuous predictions ────────────
            # CX at base+0, CY at base+1, W at base+2, H at base+3
            cont_offsets = torch.arange(4, device=dev)                # (4,)
            # cont_positions[k, off] = bases[k] + off → (N, 4)
            cont_positions = bases.unsqueeze(1) + cont_offsets.unsqueeze(0)  # (N, 4)
            # Expand for batch: (B, N, 4)
            cont_pos_batch = cont_positions.unsqueeze(0).expand(B, -1, -1)
            # Flatten to (B, N*4) for gathering
            cont_pos_flat = cont_pos_batch.reshape(B, n_rooms * 4)    # (B, N*4)
            h_cont_flat = torch.gather(
                h, 1, cont_pos_flat.unsqueeze(-1).expand(-1, -1, d)
            )                                                         # (B, N*4, d)
            h_cont_all = h_cont_flat.reshape(B, n_rooms, 4, d)       # (B, N, 4, d)

            # ── Apply mask: flatten valid entries ──────────────────────────
            # mask: (B, N) bool — True = valid room
            valid_flat = mask.reshape(-1)                              # (B*N,)
            if not valid_flat.any():
                return torch.zeros(1, device=dev, requires_grad=True).squeeze()

            # ── TYPE loss (vectorised) ────────────────────────────────────
            h_type_flat   = h_type_all.reshape(B * n_rooms, d)[valid_flat]   # (M, d)
            type_tgt_flat = type_ids.reshape(B * n_rooms)[valid_flat].clamp(min=0)
            type_logits   = self.type_head(h_type_flat)               # (M, 16)
            type_loss     = F.cross_entropy(type_logits, type_tgt_flat)

            # ── Continuous losses (vectorised per dimension) ──────────────
            h_cont_bnd = h_cont_all.reshape(B * n_rooms, 4, d)       # (B*N, 4, d)
            h_valid    = h_cont_bnd[valid_flat]                       # (M, 4, d)

            boxes_flat = boxes_norm.reshape(B * n_rooms, 4)[valid_flat]  # (M, 4)
            cx_t = boxes_flat[:, 0].clamp(0.01, 0.99)
            cy_t = boxes_flat[:, 1].clamp(0.01, 0.99)
            w_t  = boxes_flat[:, 2].clamp(0.01, 0.98)
            h_t  = boxes_flat[:, 3].clamp(0.01, 0.98)

            cx_loss = -self.mog_cx.log_prob(h_valid[:, 0], cx_t).mean()
            cy_loss = -self.mog_cy.log_prob(h_valid[:, 1], cy_t).mean()
            w_loss  = -self.mog_w.log_prob( h_valid[:, 2], w_t).mean()
            h_loss  = -self.mog_h.log_prob( h_valid[:, 3], h_t).mean()

            total_loss = (
                TYPE_LOSS_WEIGHT * type_loss
                + CONT_LOSS_WEIGHT * (cx_loss + cy_loss + w_loss + h_loss)
            )
            return total_loss

        # ── Weight export for numpy inference ─────────────────────────────

        def save_numpy_weights(self, path: str) -> None:
            """
            Export all model weights to a flat .npz file for LayoutTransformerNumpy.

            Key naming convention:
              type_emb       : (16, d_model)
              role_emb       : (6, d_model)
              global_W, global_b
              cont_W, cont_b
              ctx_W, ctx_b
              pos_emb        : (max_seq_len, d_model)
              final_ln_w, final_ln_b
              type_head_W, type_head_b
              mog_{cx,cy,w,h}_{w,b}  (3K for each of 3 sub-keys)
              blk{i}_sa_Wq ... blk{i}_ff_ln_b   (per block)
            """
            sd = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
            out: Dict[str, np.ndarray] = {}

            # ── Embedding tables ───────────────────────────────────────────
            out["type_emb"] = sd["type_emb.weight"]
            out["role_emb"] = sd["role_emb.weight"]
            out["pos_emb"]  = sd["pos_emb.weight"]

            # ── Projections ────────────────────────────────────────────────
            out["global_W"] = sd["global_proj.0.weight"]
            out["global_b"] = sd["global_proj.0.bias"]
            out["cont_W"]   = sd["cont_proj.0.weight"]
            out["cont_b"]   = sd["cont_proj.0.bias"]
            out["ctx_W"]    = sd["ctx_proj.0.weight"]
            out["ctx_b"]    = sd["ctx_proj.0.bias"]

            # ── Final layer norm ───────────────────────────────────────────
            out["final_ln_w"] = sd["final_ln.weight"]
            out["final_ln_b"] = sd["final_ln.bias"]

            # ── Output heads ───────────────────────────────────────────────
            out["type_head_W"] = sd["type_head.weight"]
            out["type_head_b"] = sd["type_head.bias"]
            for dim in ["cx", "cy", "w", "h"]:
                out[f"mog_{dim}_w"] = sd[f"mog_{dim}.head.weight"]
                out[f"mog_{dim}_b"] = sd[f"mog_{dim}.head.bias"]

            # ── Transformer blocks ─────────────────────────────────────────
            for i in range(self.n_layers):
                p = f"blk{i}"
                b = f"blocks.{i}"
                # Self-attention
                out[f"{p}_sa_Wq"]   = sd[f"{b}.self_attn.in_proj_weight"][:self.d_model]
                out[f"{p}_sa_Wk"]   = sd[f"{b}.self_attn.in_proj_weight"][self.d_model:2*self.d_model]
                out[f"{p}_sa_Wv"]   = sd[f"{b}.self_attn.in_proj_weight"][2*self.d_model:]
                out[f"{p}_sa_Wo"]   = sd[f"{b}.self_attn.out_proj.weight"]
                out[f"{p}_sa_ln_w"] = sd[f"{b}.ln_sa.weight"]
                out[f"{p}_sa_ln_b"] = sd[f"{b}.ln_sa.bias"]
                # Cross-attention
                out[f"{p}_ca_Wq"]   = sd[f"{b}.cross_attn.in_proj_weight"][:self.d_model]
                out[f"{p}_ca_Wk"]   = sd[f"{b}.cross_attn.in_proj_weight"][self.d_model:2*self.d_model]
                out[f"{p}_ca_Wv"]   = sd[f"{b}.cross_attn.in_proj_weight"][2*self.d_model:]
                out[f"{p}_ca_Wo"]   = sd[f"{b}.cross_attn.out_proj.weight"]
                out[f"{p}_ca_ln_w"] = sd[f"{b}.ln_ca.weight"]
                out[f"{p}_ca_ln_b"] = sd[f"{b}.ln_ca.bias"]
                # FFN
                out[f"{p}_ff_w1"]   = sd[f"{b}.ff.0.weight"]
                out[f"{p}_ff_b1"]   = sd[f"{b}.ff.0.bias"]
                out[f"{p}_ff_w2"]   = sd[f"{b}.ff.3.weight"]
                out[f"{p}_ff_b2"]   = sd[f"{b}.ff.3.bias"]
                out[f"{p}_ff_ln_w"] = sd[f"{b}.ln_ff.weight"]
                out[f"{p}_ff_ln_b"] = sd[f"{b}.ln_ff.bias"]

            np.savez_compressed(path, **out)
            print(f"[LayoutTransformer] Saved {len(out)} arrays to {path}  "
                  f"({os.path.getsize(path)/1024:.1f} KB)")
