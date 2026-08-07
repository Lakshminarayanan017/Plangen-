"""
numpy_infer.py — pure-NumPy inference mirroring PlacerNet (no torch).

Production loads the .npz weights and runs this. It reproduces the torch
forward op-for-op; test_placer_numpy_equality asserts logit agreement < 1e-4
with the torch model on random weights, so "matches torch" is a gate, not a
hope. Exposes the same encode/decode_step/cell_count/size_count interface the
constrained decoder (masked_decode) uses.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import erf

from ml.tier2_placer.config import SEED_GRID, PlacerConfig
from ml.tier2_placer.export import load_npz


# ── primitive ops (match torch semantics) ────────────────────────────────────

def linear(x, w, b=None):
    y = x @ w.T
    return y + b if b is not None else y


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                 # biased, matches torch LN
    return (x - mu) / np.sqrt(var + eps) * w + b


def relu(x):
    return np.maximum(x, 0.0)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1.0))


def leaky_relu(x, slope=0.2):
    return np.where(x > 0, x, slope * x)


def gelu(x):                                        # exact erf GELU (torch dflt)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def conv2d(x, w, b, stride=1, pad=1):
    # x: (Cin,H,W); w: (Cout,Cin,kh,kw); b: (Cout,)
    cin, h, wd = x.shape
    cout, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    ho = (h + 2 * pad - kh) // stride + 1
    wo = (wd + 2 * pad - kw) // stride + 1
    # im2col
    cols = np.empty((cin * kh * kw, ho * wo), dtype=x.dtype)
    idx = 0
    for c in range(cin):
        for i in range(kh):
            for j in range(kw):
                patch = xp[c, i:i + stride * ho:stride,
                           j:j + stride * wo:stride]
                cols[idx] = patch.reshape(-1)
                idx += 1
    wcol = w.reshape(cout, -1)
    out = wcol @ cols + b[:, None]
    return out.reshape(cout, ho, wo)


def multihead_attention(q, kv, in_w, in_b, out_w, out_b, n_heads,
                        attn_mask=None):
    """Reproduces nn.MultiheadAttention(batch_first=True) for a single
    sequence. q:(Lq,d), kv:(Lk,d). attn_mask:(Lq,Lk) additive or None."""
    d = q.shape[-1]
    hd = d // n_heads
    wq, wk, wv = in_w[:d], in_w[d:2 * d], in_w[2 * d:]
    bq, bk, bv = in_b[:d], in_b[d:2 * d], in_b[2 * d:]
    Q = linear(q, wq, bq).reshape(-1, n_heads, hd).transpose(1, 0, 2)
    K = linear(kv, wk, bk).reshape(-1, n_heads, hd).transpose(1, 0, 2)
    V = linear(kv, wv, bv).reshape(-1, n_heads, hd).transpose(1, 0, 2)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(hd)      # (H,Lq,Lk)
    if attn_mask is not None:
        scores = scores + attn_mask[None]
    attn = softmax(scores, axis=-1)
    ctx = attn @ V                                       # (H,Lq,hd)
    ctx = ctx.transpose(1, 0, 2).reshape(q.shape[0], d)
    return linear(ctx, out_w, out_b)


# ── segment ops for the GATv2 layer ──────────────────────────────────────────

def _seg_sum(vals, idx, n):
    out = np.zeros((n,) + vals.shape[1:], dtype=vals.dtype)
    np.add.at(out, idx, vals)
    return out


def _seg_max(vals, idx, n):
    out = np.full((n,) + vals.shape[1:], -np.inf, dtype=vals.dtype)
    np.maximum.at(out, idx, vals)
    return np.nan_to_num(out, neginf=0.0)


# ── the model ────────────────────────────────────────────────────────────────

class NumpyPlacer:
    def __init__(self, weights: Dict[str, np.ndarray], config: dict):
        self.w = weights
        self.cfg = PlacerConfig.from_dict(config)
        self.cell_count = self.cfg.cell_count
        self.size_count = self.cfg.size_count

    @classmethod
    def from_npz(cls, path: str) -> "NumpyPlacer":
        weights, config = load_npz(path)
        return cls(weights, config)

    def g(self, name):
        return self.w[name]

    # ── program encoder (GATv2) ──────────────────────────────────────────
    def _gat_layer(self, x, edge_index, i):
        p = f"program.layers.{i}."
        heads = self.cfg.gnn_heads
        d = self.cfg.node_dim
        hd = d // heads
        n = x.shape[0]
        xl = linear(x, self.g(p + "lin_l.weight"),
                    self.g(p + "lin_l.bias")).reshape(n, heads, hd)
        xr = linear(x, self.g(p + "lin_r.weight"),
                    self.g(p + "lin_r.bias")).reshape(n, heads, hd)
        src, dst = edge_index[0], edge_index[1]
        e = leaky_relu(xl[src] + xr[dst])
        scores = (e * self.g(p + "att")).sum(-1)             # (E,H)
        scores = scores - _seg_max(scores, dst, n)[dst]
        alpha = np.exp(scores)
        denom = np.clip(_seg_sum(alpha, dst, n), 1e-16, None)
        alpha = alpha / denom[dst]
        msg = (xl[src] * alpha[..., None]).reshape(src.shape[0], -1)
        return _seg_sum(msg, dst, n).reshape(n, d)

    def _program(self, arrays):
        d = self.cfg.node_dim
        h = (self.g("program.type_emb.weight")[arrays["type_ids"]]
             + self.g("program.zone_emb.weight")[arrays["zone_ids"]]
             + self.g("program.floor_emb.weight")[
                 np.clip(arrays["floor_ids"], 0, 7)])
        n = h.shape[0]
        ei = arrays["edge_index"]
        loops = np.stack([np.arange(n), np.arange(n)])
        ei = np.concatenate([ei, loops], axis=1) if ei.size else loops
        for i in range(self.cfg.gnn_layers):
            out = elu(self._gat_layer(h, ei, i))
            h = layernorm(h + out, self.g(f"program.norms.{i}.weight"),
                          self.g(f"program.norms.{i}.bias"))
        return linear(h, self.g("program.out.weight"),
                      self.g("program.out.bias"))

    # ── boundary encoder (CNN) ───────────────────────────────────────────
    def _boundary(self, boundary, glob):
        x = boundary
        for idx in (0, 2, 4, 6):
            x = relu(conv2d(x, self.g(f"boundary.cnn.{idx}.weight"),
                            self.g(f"boundary.cnn.{idx}.bias"),
                            stride=(2 if idx < 6 else 1), pad=1))
        c = x.shape[0]
        tokens = x.reshape(c, -1).T                          # (64, c)
        tokens = linear(tokens, self.g("boundary.proj.weight"),
                        self.g("boundary.proj.bias"))
        g0 = relu(linear(glob, self.g("boundary.global_mlp.0.weight"),
                         self.g("boundary.global_mlp.0.bias")))
        gtok = linear(g0, self.g("boundary.global_mlp.2.weight"),
                      self.g("boundary.global_mlp.2.bias"))[None]
        return np.concatenate([tokens, gtok], axis=0)

    def encode(self, arrays):
        prog = self._program(arrays)
        bnd = self._boundary(arrays["boundary"], arrays["global"])
        return np.concatenate([prog, bnd], axis=0)

    # ── decoder ──────────────────────────────────────────────────────────
    def _dec_input(self, type_ids, zone_ids, prev_cell, prev_size):
        d = self.cfg.d_model
        n = len(type_ids)
        pos = np.clip(np.arange(n), 0, self.cfg.max_rooms - 1)
        return (self.g("decoder.type_emb.weight")[type_ids]
                + self.g("decoder.zone_emb.weight")[zone_ids]
                + self.g("decoder.prev_cell_emb.weight")[prev_cell]
                + self.g("decoder.prev_size_emb.weight")[prev_size]
                + self.g("decoder.pos_emb.weight")[pos])

    def _dec_layer(self, x, memory, causal, i):
        p = f"decoder.layers.{i}."
        h = layernorm(x, self.g(p + "n1.weight"), self.g(p + "n1.bias"))
        x = x + multihead_attention(
            h, h, self.g(p + "self_attn.in_proj_weight"),
            self.g(p + "self_attn.in_proj_bias"),
            self.g(p + "self_attn.out_proj.weight"),
            self.g(p + "self_attn.out_proj.bias"),
            self.cfg.dec_heads, attn_mask=causal)
        h = layernorm(x, self.g(p + "n2.weight"), self.g(p + "n2.bias"))
        x = x + multihead_attention(
            h, memory, self.g(p + "cross_attn.in_proj_weight"),
            self.g(p + "cross_attn.in_proj_bias"),
            self.g(p + "cross_attn.out_proj.weight"),
            self.g(p + "cross_attn.out_proj.bias"),
            self.cfg.dec_heads, attn_mask=None)
        h = layernorm(x, self.g(p + "n3.weight"), self.g(p + "n3.bias"))
        ff = linear(gelu(linear(h, self.g(p + "ff.0.weight"),
                                self.g(p + "ff.0.bias"))),
                    self.g(p + "ff.3.weight"), self.g(p + "ff.3.bias"))
        return x + ff

    def decode_step(self, memory, arrays, prev_cell, prev_size):
        m = prev_cell.shape[0]
        x = self._dec_input(arrays["type_ids"][:m], arrays["zone_ids"][:m],
                            prev_cell, prev_size)
        causal = np.triu(np.full((m, m), -np.inf), k=1)
        for i in range(self.cfg.dec_layers):
            x = self._dec_layer(x, memory, causal, i)
        h = layernorm(x, self.g("decoder.norm.weight"),
                      self.g("decoder.norm.bias"))
        cell_logits = linear(h, self.g("decoder.cell_head.weight"),
                             self.g("decoder.cell_head.bias"))
        size_logits = linear(h, self.g("decoder.size_head.weight"),
                             self.g("decoder.size_head.bias"))
        return cell_logits, size_logits

    def teacher_forced_logits(self, arrays):
        """Full-sequence logits with ground-truth prev — the equality target."""
        from ml.tier2_placer.model.decoder import _START_CELL  # noqa: F401
        n = int(arrays["n_rooms"])
        tc, ts = arrays["target_cell"], arrays["target_size"]
        prev_cell = np.full(n, self.cell_count, dtype=np.int64)
        prev_size = np.full(n, self.size_count, dtype=np.int64)
        if n > 1:
            prev_cell[1:] = tc[:-1]
            prev_size[1:] = ts[:-1]
        memory = self.encode(arrays)
        return self.decode_step(memory, arrays, prev_cell, prev_size)
