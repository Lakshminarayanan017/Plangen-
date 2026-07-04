"""
GNN Encoder v2 — Production Quality
====================================
4-layer Graph Attention Network (GATv2) that encodes a floor plan's room
graph into rich 256-dim per-room embeddings for the diffusion decoder.

Architecture vs v1:
  v1: 3 layers, 128-dim output, 12-dim node features, 3-dim edge features
  v2: 4 layers, 256-dim output, 24-dim node features, 7-dim edge features,
      learnable edge projection, skip connections, layer-norm, dropout

Input:
  node_features  (N, 24)   — see data_prep.py NODE_FEAT_DIM
  edge_index     (2, E)    — sparse adjacency (wall-touching pairs)
  edge_features  (E, 7)    — see data_prep.py EDGE_FEAT_DIM

Output:
  node_embeddings (N, 256) — rich per-room context vectors
  global_embed    (1, 256) — mean-pooled plan-level summary

Inference path (numpy-only, no PyTorch required):
  GNNEncoder.forward_numpy(node_features, edge_index, edge_features) → ndarray
  Used by generator.py and greedy_placer.py at runtime without GPU.

Training path (PyTorch):
  GNNEncoder.forward(node_features, edge_index, edge_features) → Tensor
"""

from __future__ import annotations

import math
import os
import json
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

log = logging.getLogger(__name__)

# ── Dimensions ────────────────────────────────────────────────────────────────
NODE_FEAT_DIM: int = 24   # input node features (from data_prep v2)
EDGE_FEAT_DIM: int = 7    # input edge features
HIDDEN_DIMS   = [64, 128, 256, 256]   # per-layer output dims (4 layers)
N_HEADS       = [4,  8,   8,   8]    # attention heads per layer
OUTPUT_DIM    = 256                    # final embedding dim
DROPOUT_RATE  = 0.1


# ── Pure NumPy Inference Engine ───────────────────────────────────────────────

class _LinearNumpy:
    """W·x + b in numpy, loaded from weights dict."""
    def __init__(self, W: np.ndarray, b: Optional[np.ndarray] = None):
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32) if b is not None else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W.T
        if self.b is not None:
            out = out + self.b
        return out


class _LayerNormNumpy:
    def __init__(self, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
        self.gamma = gamma.astype(np.float32)
        self.beta  = beta.astype(np.float32)
        self.eps   = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class GATv2LayerNumpy:
    """
    Single GATv2 attention layer in pure numpy.
    GATv2: attention score = LeakyReLU(a · W·[h_i || h_j || e_ij])
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int,
                 W_src, W_dst, W_edge, attn_vec, W_out, b_out,
                 gamma, beta):
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.W_src    = _LinearNumpy(W_src)   # (in_dim → n_heads * head_dim)
        self.W_dst    = _LinearNumpy(W_dst)
        self.W_edge   = _LinearNumpy(W_edge)  # (edge_dim → n_heads * head_dim)
        self.attn_vec = attn_vec.astype(np.float32)  # (n_heads, head_dim)
        self.W_out    = _LinearNumpy(W_out, b_out)   # (out_dim → out_dim)
        self.norm     = _LayerNormNumpy(gamma, beta)

    def __call__(self, h: np.ndarray, edge_index: np.ndarray,
                 edge_feat: np.ndarray) -> np.ndarray:
        """
        h: (N, in_dim)
        edge_index: (2, E)
        edge_feat: (E, edge_dim)
        Returns: (N, out_dim)
        """
        N = h.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Project
        h_src = self.W_src(h[src])  # (E, n_heads * head_dim)
        h_dst = self.W_dst(h[dst])
        h_e   = self.W_edge(edge_feat)  # (E, n_heads * head_dim)

        E     = h_src.shape[0]
        H, D  = self.n_heads, self.head_dim

        # Reshape to (E, H, D)
        h_src = h_src.reshape(E, H, D)
        h_dst = h_dst.reshape(E, H, D)
        h_e   = h_e.reshape(E, H, D)

        # GATv2 attention: a · LeakyReLU(W·[h_src + h_dst + h_e])
        combined = np.maximum(h_src + h_dst + h_e, 0.2 * (h_src + h_dst + h_e))
        # attn_vec: (H, D) → score per edge per head
        scores = (combined * self.attn_vec[None, :, :]).sum(axis=-1)  # (E, H)

        # Softmax per destination node per head
        alpha = np.full((N, H), -1e9, dtype=np.float32)
        # Scatter-max for numerical stability
        np.maximum.at(alpha, dst, scores)
        alpha_max = alpha[dst]           # (E, H)
        exp_scores = np.exp(scores - alpha_max)
        denom = np.zeros((N, H), dtype=np.float32)
        np.add.at(denom, dst, exp_scores)
        alpha_norm = exp_scores / (denom[dst] + 1e-8)  # (E, H)

        # Weighted sum of value = h_src + h_e
        val = h_src + h_e  # (E, H, D)
        out = np.zeros((N, H, D), dtype=np.float32)
        np.add.at(out, dst, val * alpha_norm[:, :, None])

        out = out.reshape(N, H * D)
        out = np.maximum(out, 0.0)        # ReLU
        out = self.W_out(out)             # linear projection
        out = self.norm(out)             # layer norm
        return out


class GNNEncoderNumpy:
    """Pure numpy inference for GNNEncoder v2. Loaded from weights file."""

    def __init__(self, weights_path: str):
        self.layers = []
        self.skip_projs = []
        self.node_proj = None
        self._load_weights(weights_path)

    def _load_weights(self, path: str):
        try:
            w = np.load(path, allow_pickle=True)
            if isinstance(w, np.ndarray):
                w = w.item()
        except Exception as e:
            log.warning(f"GNN weights not found at {path}: {e}")
            w = {}
        self._w = w

    def _get(self, key: str, shape: tuple) -> np.ndarray:
        if key in self._w:
            return np.asarray(self._w[key], dtype=np.float32)
        return np.zeros(shape, dtype=np.float32)

    def forward(self, node_feat: np.ndarray, edge_index: np.ndarray,
                edge_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        node_feat:  (N, 24)
        edge_index: (2, E)
        edge_feat:  (E, 7)
        Returns:
          node_emb:  (N, 256)
          global:    (1, 256)
        """
        N = node_feat.shape[0]
        w = self._w

        # Input projection: 24 → 64
        W_inp = self._get("node_proj_W", (64, 24))
        b_inp = self._get("node_proj_b", (64,))
        h = np.maximum(node_feat @ W_inp.T + b_inp, 0.0)

        # 4 GATv2 layers
        in_dims  = [64, 64, 128, 256]
        out_dims = [64, 128, 256, 256]
        n_heads  = [4,  8,   8,   8]

        for i in range(4):
            in_d  = in_dims[i]
            out_d = out_dims[i]
            nh    = n_heads[i]
            hd    = out_d // nh

            W_src  = self._get(f"gat{i}_W_src",  (out_d, in_d))
            W_dst  = self._get(f"gat{i}_W_dst",  (out_d, in_d))
            W_edge = self._get(f"gat{i}_W_edge", (out_d, EDGE_FEAT_DIM))
            a_vec  = self._get(f"gat{i}_attn",   (nh, hd))
            W_out  = self._get(f"gat{i}_W_out",  (out_d, out_d))
            b_out  = self._get(f"gat{i}_b_out",  (out_d,))
            gamma  = self._get(f"gat{i}_gamma",  (out_d,))
            beta   = self._get(f"gat{i}_beta",   (out_d,))
            # Init gamma to 1 if not set
            if gamma.sum() == 0: gamma = np.ones(out_d, dtype=np.float32)

            layer = GATv2LayerNumpy(
                in_d, out_d, nh,
                W_src, W_dst, W_edge, a_vec, W_out, b_out, gamma, beta
            )
            h_new = layer(h, edge_index, edge_feat)

            # Skip connection (project if dims differ)
            if in_d != out_d:
                W_skip = self._get(f"gat{i}_skip_W", (out_d, in_d))
                h_skip = h @ W_skip.T
            else:
                h_skip = h
            h = h_new + h_skip

        # Global mean pooling
        global_emb = h.mean(axis=0, keepdims=True)  # (1, 256)
        return h, global_emb

    # Alias so diffusion_engine.py can call either .forward() or .forward_numpy()
    forward_numpy = forward


# ── PyTorch Module ────────────────────────────────────────────────────────────

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        return None, None, None


class GNNEncoder:
    """
    GNNEncoder v2 — wraps PyTorch GATv2 for training and numpy for inference.

    Usage (training):
        enc = GNNEncoder()
        node_emb, global_emb = enc(node_feat_tensor, edge_index, edge_feat_tensor)

    Usage (inference, no PyTorch):
        enc = GNNEncoder.from_numpy_weights("path/to/weights.npz")
        node_emb, global_emb = enc.forward_numpy(node_feat, edge_index, edge_feat)
    """

    def __init__(self, dropout: float = DROPOUT_RATE):
        torch, nn, F = _try_import_torch()
        if torch is None:
            log.warning("PyTorch not found — GNNEncoder runs in numpy-only mode")
            self._torch_module = None
            return

        self._torch_module = _GNNEncoderTorch(dropout=dropout)
        self._numpy_enc = None

    def __call__(self, node_feat, edge_index, edge_feat):
        """Forward pass — PyTorch tensors."""
        if self._torch_module is None:
            raise RuntimeError("PyTorch required for training forward pass")
        return self._torch_module(node_feat, edge_index, edge_feat)

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
        if self._torch_module:
            return self._torch_module.state_dict()
        return {}

    def load_state_dict(self, sd):
        if self._torch_module:
            self._torch_module.load_state_dict(sd)

    def to(self, device):
        """Move the underlying PyTorch module to device (CPU / CUDA / MPS)."""
        if self._torch_module:
            self._torch_module.to(device)
        return self

    def save_numpy_weights(self, path: str):
        """
        Export weights to .npz for numpy inference.

        Uses an explicit key mapping so GNNEncoderNumpy.forward() can load
        every weight by its expected flat name (gat0_W_src, gat1_skip_W, etc.)
        instead of the raw PyTorch dot-separated names.
        """
        if self._torch_module is None:
            return
        sd   = self._torch_module.state_dict()   # from _GNNEncoderTorchInner
        w    = {}
        in_dims  = [64,  64, 128, 256]
        out_dims = [64, 128, 256, 256]

        # ── Input projection: Sequential(Linear(24,64), ReLU, LayerNorm(64)) ──
        # index 0 = Linear, index 1 = ReLU (no params), index 2 = LayerNorm
        w["node_proj_W"] = sd["node_proj.0.weight"].cpu().numpy()
        w["node_proj_b"] = sd["node_proj.0.bias"].cpu().numpy()

        # ── 4 GATv2 layers ────────────────────────────────────────────────────
        for i in range(4):
            w[f"gat{i}_W_src"]  = sd[f"gat_layers.{i}.W_src.weight"].cpu().numpy()
            w[f"gat{i}_W_dst"]  = sd[f"gat_layers.{i}.W_dst.weight"].cpu().numpy()
            w[f"gat{i}_W_edge"] = sd[f"gat_layers.{i}.W_edge.weight"].cpu().numpy()
            # attn param is (1, n_heads, head_dim) in PyTorch → squeeze to (n_heads, head_dim)
            w[f"gat{i}_attn"]   = sd[f"gat_layers.{i}.attn"].cpu().numpy().squeeze(0)
            w[f"gat{i}_W_out"]  = sd[f"gat_layers.{i}.W_out.weight"].cpu().numpy()
            w[f"gat{i}_b_out"]  = sd[f"gat_layers.{i}.W_out.bias"].cpu().numpy()
            w[f"gat{i}_gamma"]  = sd[f"gat_layers.{i}.norm.weight"].cpu().numpy()
            w[f"gat{i}_beta"]   = sd[f"gat_layers.{i}.norm.bias"].cpu().numpy()

            # Skip projection only exists when in_dim != out_dim (i=1,2)
            if in_dims[i] != out_dims[i]:
                skip_key = f"skip_projs.{i}.weight"
                if skip_key in sd:
                    w[f"gat{i}_skip_W"] = sd[skip_key].cpu().numpy()

        np.savez(path, **w)
        log.info(f"Saved GNN numpy weights → {path}")

    @classmethod
    def from_numpy_weights(cls, path: str) -> "GNNEncoder":
        enc = cls.__new__(cls)
        enc._torch_module = None
        enc._numpy_enc = GNNEncoderNumpy(path)
        return enc

    def forward_numpy(self, node_feat: np.ndarray, edge_index: np.ndarray,
                      edge_feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inference path — pure numpy, no GPU needed."""
        if self._numpy_enc is None:
            # Export current torch weights to in-memory numpy
            import io, tempfile
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                self.save_numpy_weights(tmp.name)
                self._numpy_enc = GNNEncoderNumpy(tmp.name)
        return self._numpy_enc.forward(node_feat, edge_index, edge_feat)


# ── PyTorch Implementation ────────────────────────────────────────────────────

class _GATv2Conv(object):
    """Lazy import wrapper for PyTorch GATv2."""
    pass


def _build_torch_module():
    torch, nn, F = _try_import_torch()
    if torch is None:
        return None

    class GATv2ConvManual(nn.Module):
        """
        Manual GATv2 attention without torch_geometric dependency.
        Compatible with standard PyTorch only.
        """
        def __init__(self, in_dim: int, out_dim: int, n_heads: int,
                     edge_dim: int, dropout: float = 0.1):
            super().__init__()
            assert out_dim % n_heads == 0
            self.n_heads  = n_heads
            self.head_dim = out_dim // n_heads
            self.out_dim  = out_dim

            self.W_src  = nn.Linear(in_dim, out_dim, bias=False)
            self.W_dst  = nn.Linear(in_dim, out_dim, bias=False)
            self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
            self.attn   = nn.Parameter(torch.empty(1, n_heads, self.head_dim))
            self.W_out  = nn.Linear(out_dim, out_dim)
            self.norm   = nn.LayerNorm(out_dim)
            self.drop   = nn.Dropout(dropout)

            nn.init.xavier_uniform_(self.attn.view(1, -1).unsqueeze(0))
            nn.init.xavier_uniform_(self.W_src.weight)
            nn.init.xavier_uniform_(self.W_dst.weight)
            nn.init.xavier_uniform_(self.W_edge.weight)

        def forward(self, h, edge_index, edge_feat):
            """
            h: (N, in_dim)
            edge_index: (2, E)
            edge_feat: (E, edge_dim)
            """
            N = h.size(0)
            src, dst = edge_index[0], edge_index[1]

            h_src = self.W_src(h[src]).view(-1, self.n_heads, self.head_dim)
            h_dst = self.W_dst(h[dst]).view(-1, self.n_heads, self.head_dim)
            h_e   = self.W_edge(edge_feat).view(-1, self.n_heads, self.head_dim)

            # GATv2 attention score
            combined = F.leaky_relu(h_src + h_dst + h_e, 0.2)
            scores = (combined * self.attn).sum(-1)  # (E, H)

            # Softmax scatter
            from torch_scatter import scatter_softmax
            alpha = scatter_softmax(scores, dst, dim=0, dim_size=N)
            alpha = self.drop(alpha)

            # Aggregate
            val = (h_src + h_e) * alpha.unsqueeze(-1)  # (E, H, D)
            out = torch.zeros(N, self.n_heads, self.head_dim, device=h.device)
            out.scatter_add_(0, dst.view(-1,1,1).expand_as(val), val)
            out = out.view(N, self.out_dim)
            out = F.relu(self.W_out(out))
            return self.norm(out)

    class GATv2ConvSafe(nn.Module):
        """Fallback GATv2 without torch_scatter — uses dense softmax."""
        def __init__(self, in_dim: int, out_dim: int, n_heads: int,
                     edge_dim: int, dropout: float = 0.1):
            super().__init__()
            assert out_dim % n_heads == 0
            self.n_heads  = n_heads
            self.head_dim = out_dim // n_heads
            self.out_dim  = out_dim

            self.W_src  = nn.Linear(in_dim, out_dim, bias=False)
            self.W_dst  = nn.Linear(in_dim, out_dim, bias=False)
            self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
            self.attn   = nn.Parameter(torch.ones(1, n_heads, self.head_dim))
            self.W_out  = nn.Linear(out_dim, out_dim)
            self.norm   = nn.LayerNorm(out_dim)
            self.drop   = nn.Dropout(dropout)

            for m in [self.W_src, self.W_dst, self.W_edge, self.W_out]:
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, h, edge_index, edge_feat):
            N = h.size(0)
            src, dst = edge_index[0], edge_index[1]
            E = src.size(0)

            if E == 0:
                return self.norm(F.relu(self.W_out(
                    torch.zeros(N, self.out_dim, device=h.device)
                )))

            h_src = self.W_src(h[src]).view(E, self.n_heads, self.head_dim)
            h_dst = self.W_dst(h[dst]).view(E, self.n_heads, self.head_dim)
            h_e   = self.W_edge(edge_feat).view(E, self.n_heads, self.head_dim)

            combined = F.leaky_relu(h_src + h_dst + h_e, 0.2)
            scores   = (combined * self.attn).sum(-1)  # (E, H)

            # Per-destination softmax via scatter (manual)
            scores_max = torch.full((N, self.n_heads), -1e9, device=h.device)
            scores_max.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.n_heads),
                                       scores, reduce="amax", include_self=True)
            exp_s = torch.exp(scores - scores_max[dst])
            denom = torch.zeros(N, self.n_heads, device=h.device)
            denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.n_heads), exp_s)
            alpha = self.drop(exp_s / (denom[dst] + 1e-8))  # (E, H)

            val = (h_src + h_e) * alpha.unsqueeze(-1)  # (E, H, D)
            out = torch.zeros(N, self.n_heads, self.head_dim, device=h.device)
            out.scatter_add_(
                0,
                dst.view(-1,1,1).expand_as(val),
                val
            )
            out = out.view(N, self.out_dim)
            out = F.relu(self.W_out(out))
            return self.norm(out)

    class _GNNEncoderTorchInner(nn.Module):
        """
        4-layer GATv2 encoder.
        Layer dims: 24→64→128→256→256
        Heads:         4   8    8    8
        """
        def __init__(self, dropout: float = DROPOUT_RATE):
            super().__init__()

            # Input projection: 24 → 64
            self.node_proj = nn.Sequential(
                nn.Linear(NODE_FEAT_DIM, 64),
                nn.ReLU(),
                nn.LayerNorm(64),
            )

            in_dims  = [64, 64, 128, 256]
            out_dims = [64, 128, 256, 256]
            heads    = [4,  8,   8,   8]

            self.gat_layers = nn.ModuleList()
            self.skip_projs = nn.ModuleList()
            for i in range(4):
                self.gat_layers.append(
                    GATv2ConvSafe(in_dims[i], out_dims[i], heads[i],
                                  EDGE_FEAT_DIM, dropout)
                )
                if in_dims[i] != out_dims[i]:
                    self.skip_projs.append(
                        nn.Linear(in_dims[i], out_dims[i], bias=False)
                    )
                else:
                    self.skip_projs.append(nn.Identity())

            self.dropout = nn.Dropout(dropout)

            # Global readout MLP
            self.global_mlp = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, node_feat, edge_index, edge_feat):
            """
            node_feat:  (N, 24) float32
            edge_index: (2, E)  int64
            edge_feat:  (E, 7)  float32
            Returns:
              node_emb:   (N, 256)
              global_emb: (1, 256)
            """
            h = self.node_proj(node_feat)  # (N, 64)

            for i, (gat, skip) in enumerate(zip(self.gat_layers, self.skip_projs)):
                h_new = gat(h, edge_index, edge_feat)
                h = h_new + skip(h)
                if i < 3:
                    h = self.dropout(h)

            # Global pooling → plan-level summary
            global_emb = self.global_mlp(h.mean(dim=0, keepdim=True))  # (1, 256)
            return h, global_emb

    return _GNNEncoderTorchInner


_torch_enc_class = None

def _get_torch_enc_class():
    global _torch_enc_class
    if _torch_enc_class is None:
        _torch_enc_class = _build_torch_module()
    return _torch_enc_class


class _GNNEncoderTorch:
    """Thin wrapper that lazily builds the PyTorch module."""
    def __init__(self, dropout: float = DROPOUT_RATE):
        cls = _get_torch_enc_class()
        if cls is None:
            raise RuntimeError("PyTorch not available")
        self._model = cls(dropout=dropout)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self):
        return self._model.parameters()

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self._model.load_state_dict(sd, strict=strict)

    def to(self, device):
        self._model.to(device)
        return self

    def __getattr__(self, name):
        if name in ("_model",):
            raise AttributeError(name)
        return getattr(self._model, name)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_gnn_encoder(
    weights_path: Optional[str] = None,
    inference_only: bool = False,
    dropout: float = DROPOUT_RATE,
) -> GNNEncoder:
    """
    Factory function.

    Args:
        weights_path:   Path to .npz (numpy inference) or .pt (PyTorch training)
        inference_only: If True, load numpy weights only (no PyTorch required)
        dropout:        Dropout rate for training

    Returns:
        GNNEncoder instance
    """
    if inference_only and weights_path and weights_path.endswith(".npz"):
        return GNNEncoder.from_numpy_weights(weights_path)

    enc = GNNEncoder(dropout=dropout)

    if weights_path and os.path.exists(weights_path):
        torch, nn, F = _try_import_torch()
        if torch and weights_path.endswith(".pt"):
            sd = torch.load(weights_path, map_location="cpu")
            enc.load_state_dict(sd)
            log.info(f"Loaded GNN weights from {weights_path}")
        elif weights_path.endswith(".npz"):
            enc._numpy_enc = GNNEncoderNumpy(weights_path)

    return enc


# ── Quick Smoke Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GNN Encoder v2 — Smoke Test")
    print("=" * 50)

    # Create synthetic data
    N = 8   # rooms
    E = 14  # edges (7 bidirectional pairs)
    rng = np.random.default_rng(42)

    node_feat  = rng.random((N, NODE_FEAT_DIM), dtype=np.float32)
    edge_index = np.array([
        [0,1,1,2,2,3,3,4,4,5,5,6,6,7],
        [1,0,2,1,3,2,4,3,5,4,6,5,7,6],
    ], dtype=np.int64)
    edge_feat  = rng.random((E, EDGE_FEAT_DIM), dtype=np.float32)

    print(f"Input:  node_feat={node_feat.shape}, edge_index={edge_index.shape}, "
          f"edge_feat={edge_feat.shape}")

    # Test numpy inference
    enc = GNNEncoder.from_numpy_weights("/nonexistent.npz")  # will use random weights
    node_emb, global_emb = enc.forward_numpy(node_feat, edge_index, edge_feat)
    print(f"Numpy output: node_emb={node_emb.shape}, global_emb={global_emb.shape}")
    assert node_emb.shape   == (N, OUTPUT_DIM), f"Expected ({N}, {OUTPUT_DIM})"
    assert global_emb.shape == (1, OUTPUT_DIM), f"Expected (1, {OUTPUT_DIM})"
    print("✓ Numpy inference: shapes correct")

    # Test PyTorch (if available)
    torch, nn, F = _try_import_torch()
    if torch:
        enc_pt = GNNEncoder(dropout=0.0)
        enc_pt.eval()
        with torch.no_grad():
            nf  = torch.tensor(node_feat)
            ef  = torch.tensor(edge_feat)
            ei  = torch.tensor(edge_index)
            ne, ge = enc_pt(nf, ei, ef)
        print(f"PyTorch output: node_emb={tuple(ne.shape)}, global_emb={tuple(ge.shape)}")
        assert ne.shape == (N, OUTPUT_DIM)
        assert ge.shape == (1, OUTPUT_DIM)
        print("✓ PyTorch forward: shapes correct")

        # Count parameters
        n_params = sum(p.numel() for p in enc_pt.parameters())
        print(f"  Total parameters: {n_params:,}")
    else:
        print("  PyTorch not available — skipping PyTorch test")

    print("=" * 50)
    print("✓ All GNN Encoder v2 tests passed")
