"""
gbt.py — gradient-boosted decision trees, in NumPy, with no ML dependency.

Why not scikit-learn / XGBoost: production here is deliberately torch-free
and sklearn-free (the same reason tier2_placer ships a NumPy inference
path). A critic that forces a 100 MB dependency into the server to score a
40-feature vector is a bad trade. This is ~200 lines of second-order
boosting — the XGBoost objective, exactly:

    gain  = 1/2 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ] - gamma
    leaf  = -G / (H + lambda)

with logistic loss (g = p - y, h = p(1-p)). Split search is histogram-based
over per-feature quantile bins, so training a few hundred trees on a few
thousand plans takes seconds.

Trees serialize to plain JSON. That is the point of choosing trees first:
`explain()` reports which features carried a decision, so a critic that
learns something silly is caught by reading it, not by trusting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


@dataclass
class Node:
    """Internal node (feature/threshold/children) or leaf (value)."""
    feature: int = -1
    threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: float = 0.0
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def to_dict(self) -> Dict:
        if self.is_leaf:
            return {"v": round(self.value, 6)}
        return {"f": self.feature, "t": round(self.threshold, 6),
                "g": round(self.gain, 6),
                "l": self.left.to_dict(), "r": self.right.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict) -> "Node":
        if "v" in d:
            return cls(value=float(d["v"]))
        return cls(feature=int(d["f"]), threshold=float(d["t"]),
                   gain=float(d.get("g", 0.0)),
                   left=cls.from_dict(d["l"]), right=cls.from_dict(d["r"]))


@dataclass
class GBTConfig:
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.08
    reg_lambda: float = 1.0        # L2 on leaf weights
    gamma: float = 0.0             # minimum gain to split
    min_child_weight: float = 2.0  # minimum sum of hessians in a child
    max_bins: int = 32
    early_stopping_rounds: int = 20

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class GBTModel:
    """A trained booster. Pure data + a NumPy forward pass."""
    trees: List[Node] = field(default_factory=list)
    base_score: float = 0.0                    # logit of the prior
    learning_rate: float = 0.08
    feature_names: List[str] = field(default_factory=list)
    train_report: Dict = field(default_factory=dict)

    # ── inference ────────────────────────────────────────────────────────
    def decision(self, X: np.ndarray) -> np.ndarray:
        """Raw logits for a (n, d) feature matrix."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.full(X.shape[0], self.base_score, dtype=np.float64)
        for tree in self.trees:
            out += self.learning_rate * _tree_predict(tree, X)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision(X))

    def score_one(self, features: Sequence[float]) -> float:
        return float(self.predict_proba(np.asarray(features)[None, :])[0])

    # ── interpretation ───────────────────────────────────────────────────
    def gain_importance(self) -> Dict[str, float]:
        """Total split gain per feature, normalized to sum 1. The whole
        reason for choosing trees over a black box: you can read this."""
        totals: Dict[int, float] = {}

        def walk(node: Node) -> None:
            if node.is_leaf:
                return
            totals[node.feature] = totals.get(node.feature, 0.0) + node.gain
            walk(node.left)
            walk(node.right)

        for tree in self.trees:
            walk(tree)
        s = sum(totals.values()) or 1.0
        named = {(self.feature_names[i] if i < len(self.feature_names)
                  else f"f{i}"): v / s for i, v in totals.items()}
        return dict(sorted(named.items(), key=lambda kv: -kv[1]))

    def explain(self, top: int = 10) -> str:
        lines = [f"GBT: {len(self.trees)} trees, base "
                 f"{self.base_score:+.3f}, lr {self.learning_rate}"]
        for name, imp in list(self.gain_importance().items())[:top]:
            lines.append(f"  {imp:6.1%}  {name}")
        return "\n".join(lines)

    # ── persistence (plain JSON — inspectable, diffable, no pickle) ──────
    def to_dict(self) -> Dict:
        return {"base_score": self.base_score,
                "learning_rate": self.learning_rate,
                "feature_names": list(self.feature_names),
                "train_report": self.train_report,
                "trees": [t.to_dict() for t in self.trees]}

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path

    @classmethod
    def from_dict(cls, d: Dict) -> "GBTModel":
        return cls(trees=[Node.from_dict(t) for t in d["trees"]],
                   base_score=float(d["base_score"]),
                   learning_rate=float(d["learning_rate"]),
                   feature_names=list(d.get("feature_names", [])),
                   train_report=dict(d.get("train_report", {})))

    @classmethod
    def load(cls, path: str) -> "GBTModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _tree_predict(node: Node, X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=np.float64)
    _fill(node, X, np.arange(X.shape[0]), out)
    return out


def _fill(node: Node, X: np.ndarray, idx: np.ndarray,
          out: np.ndarray) -> None:
    if idx.size == 0:
        return
    if node.is_leaf:
        out[idx] = node.value
        return
    go_left = X[idx, node.feature] <= node.threshold
    _fill(node.left, X, idx[go_left], out)
    _fill(node.right, X, idx[~go_left], out)


# ── training ─────────────────────────────────────────────────────────────────

def _bin_edges(X: np.ndarray, max_bins: int) -> List[np.ndarray]:
    """Per-feature candidate thresholds from quantiles of the training data.
    Constant features get no candidates and are never split on."""
    edges = []
    qs = np.linspace(0, 100, min(max_bins, 64) + 1)[1:-1]
    for j in range(X.shape[1]):
        col = X[:, j]
        uniq = np.unique(col)
        if uniq.size <= 1:
            edges.append(np.empty(0))
        elif uniq.size <= max_bins:
            edges.append((uniq[:-1] + uniq[1:]) / 2.0)
        else:
            edges.append(np.unique(np.percentile(col, qs)))
    return edges


def _build_tree(X: np.ndarray, g: np.ndarray, h: np.ndarray,
                idx: np.ndarray, edges: List[np.ndarray], depth: int,
                cfg: GBTConfig) -> Node:
    G, H = float(g[idx].sum()), float(h[idx].sum())
    leaf = Node(value=-G / (H + cfg.reg_lambda))
    if depth >= cfg.max_depth or idx.size < 2:
        return leaf

    parent = G * G / (H + cfg.reg_lambda)
    best = (cfg.gamma, -1, 0.0)          # (gain, feature, threshold)
    for j, cand in enumerate(edges):
        if cand.size == 0:
            continue
        col = X[idx, j]
        # histogram over candidate thresholds: cumulative sums of g and h
        # in threshold order give every left/right split in one pass
        order = np.searchsorted(cand, col, side="left")
        gsum = np.bincount(order, weights=g[idx], minlength=cand.size + 1)
        hsum = np.bincount(order, weights=h[idx], minlength=cand.size + 1)
        GL, HL = np.cumsum(gsum)[:-1], np.cumsum(hsum)[:-1]
        GR, HR = G - GL, H - HL
        ok = (HL >= cfg.min_child_weight) & (HR >= cfg.min_child_weight)
        if not ok.any():
            continue
        gains = 0.5 * (GL ** 2 / (HL + cfg.reg_lambda)
                       + GR ** 2 / (HR + cfg.reg_lambda)
                       - parent)
        gains = np.where(ok, gains, -np.inf)
        k = int(np.argmax(gains))
        if gains[k] > best[0]:
            best = (float(gains[k]), j, float(cand[k]))

    gain, feature, threshold = best
    if feature < 0:
        return leaf
    mask = X[idx, feature] <= threshold
    left_idx, right_idx = idx[mask], idx[~mask]
    if left_idx.size == 0 or right_idx.size == 0:
        return leaf
    return Node(
        feature=feature, threshold=threshold, gain=gain,
        left=_build_tree(X, g, h, left_idx, edges, depth + 1, cfg),
        right=_build_tree(X, g, h, right_idx, edges, depth + 1, cfg))


def train_gbt(X: np.ndarray, y: np.ndarray, *,
              config: Optional[GBTConfig] = None,
              feature_names: Optional[Sequence[str]] = None,
              validation: Optional[Tuple[np.ndarray, np.ndarray]] = None
              ) -> GBTModel:
    """Fit a binary classifier. `validation` enables early stopping on
    held-out log-loss — without it the tree count is taken at face value."""
    cfg = config or GBTConfig()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X {X.shape} and y {y.shape} disagree")

    prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    model = GBTModel(base_score=float(np.log(prior / (1 - prior))),
                     learning_rate=cfg.learning_rate,
                     feature_names=list(feature_names or
                                        [f"f{i}" for i in range(X.shape[1])]))

    edges = _bin_edges(X, cfg.max_bins)
    raw = np.full(X.shape[0], model.base_score)
    all_idx = np.arange(X.shape[0])

    best_val, best_round, history = np.inf, 0, []
    val_raw = None
    if validation is not None:
        val_raw = np.full(validation[0].shape[0], model.base_score)

    for round_i in range(cfg.n_estimators):
        p = sigmoid(raw)
        tree = _build_tree(X, p - y, np.maximum(p * (1 - p), 1e-6),
                           all_idx, edges, 0, cfg)
        model.trees.append(tree)
        raw += cfg.learning_rate * _tree_predict(tree, X)

        if validation is not None:
            val_raw += cfg.learning_rate * _tree_predict(tree, validation[0])
            loss = log_loss(validation[1], sigmoid(val_raw))
            history.append(round(loss, 5))
            if loss < best_val - 1e-6:
                best_val, best_round = loss, round_i
            elif round_i - best_round >= cfg.early_stopping_rounds:
                # keep only the trees up to the best round — the rest are
                # measured overfitting, not a judgement call
                del model.trees[best_round + 1:]
                break

    model.train_report = {
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "positive_rate": round(float(y.mean()), 4),
        "n_trees": len(model.trees), "config": cfg.to_dict(),
        "val_logloss": round(float(best_val), 5) if history else None,
        "val_history": history[:200],
    }
    return model


# ── metrics ──────────────────────────────────────────────────────────────────

def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold) == (y >= 0.5)))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (ties averaged). 0.5 = coin flip."""
    y = np.asarray(y) >= 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(np.asarray(p), kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks within ties so identical scores cannot fake separation
    sorted_p = np.asarray(p)[order]
    start = 0
    for i in range(1, len(sorted_p) + 1):
        if i == len(sorted_p) or sorted_p[i] != sorted_p[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
