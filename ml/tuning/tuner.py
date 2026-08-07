"""
tuner.py — convex pairwise learning-to-rank over the 20 soft weights.

Model. A candidate's score is `s(c) = w · g(c)` (the constant 100 cancels
in every comparison). Given a human preference i ≻ j on a brief, the
Bradley-Terry / RankNet likelihood is `P(i ≻ j) = σ(w · (g_i - g_j))`. The
negative log-likelihood

    L_data(w) = Σ_{i≻j}  softplus( -w · (g_i - g_j) )

is convex in w. We add a convex, scale-aware pull toward the hand-tuned
defaults w0 (defaults are a strong prior; labels are scarce):

    L_reg(w)  = λ · Σ_k ( (w_k - w0_k) / w0_k )²

and constrain w_k ≥ 0 so a penalty weight can never flip into a reward.
The sum is convex with a simple bound constraint → L-BFGS-B finds the
global optimum. We supply the exact gradient.

λ is chosen by leave-one-brief-out cross-validation, scoring folds by
held-out top-1 agreement (does argmax_c w·g match the human's pick) with
pairwise agreement as the tie-break — the honest "did the selector get
better" metric, never raw score. The final weights are refit on all briefs
at the chosen λ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from modules.step4_generate.engine.contracts import EngineConfig
from ml.tuning.corpus import Corpus
from ml.tuning.features import WEIGHT_KEYS, with_weights
from ml.tuning.labeling import Preferences

# Default λ grid for cross-validation (relative-deviation regularizer).
DEFAULT_LAMBDAS: Tuple[float, ...] = (
    0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

# Upper bound on any weight: no weight may exceed this multiple of its
# default. Keeps the fit interpretable and bounded when a brief's pairs
# would otherwise push a weight arbitrarily high.
_MAX_WEIGHT_MULTIPLE = 12.0


# ── per-brief numeric view ───────────────────────────────────────────────────

@dataclass
class BriefMatrix:
    """A brief's candidates as a feature matrix plus the human's ordering.

    G has shape (n_candidates, 20) in WEIGHT_KEYS order. `pref_pairs` are
    (winner_row, loser_row) index pairs implied by the human label;
    `best_row` is the human's single top pick (or None if unlabeled)."""
    name: str
    cand_ids: List[str]
    G: np.ndarray
    best_row: Optional[int]
    pref_pairs: List[Tuple[int, int]]


def _feature_row(features: Dict[str, float]) -> List[float]:
    return [features.get(k, 0.0) for k in WEIGHT_KEYS]


def build_matrices(corpus: Corpus, prefs: Preferences) -> List[BriefMatrix]:
    """Turn the corpus + labels into per-brief numeric matrices, keeping
    only briefs that have both ≥2 candidates and a usable preference."""
    matrices: List[BriefMatrix] = []
    for brief in corpus.briefs:
        if len(brief.candidates) < 2:
            continue
        pref = prefs.for_brief(brief.name)
        if pref is None:
            continue
        cand_ids = [c.cand_id for c in brief.candidates]
        index = {cid: r for r, cid in enumerate(cand_ids)}
        G = np.array([_feature_row(c.features) for c in brief.candidates],
                     dtype=float)
        pairs = _pairs_from_pref(pref, index)
        if not pairs:
            continue
        best_row = index.get(pref.best) if pref.best else None
        matrices.append(BriefMatrix(
            name=brief.name, cand_ids=cand_ids, G=G,
            best_row=best_row, pref_pairs=pairs))
    return matrices


def _pairs_from_pref(pref, index: Dict[str, int]) -> List[Tuple[int, int]]:
    """Ordered (winner_row, loser_row) pairs implied by one brief's label.

    A full `ranking` yields every ordered pair that respects it; a lone
    `best` pick yields (best ≻ other); `bad` marks add (acceptable ≻ bad)
    for every non-bad candidate. Ids not in this corpus (stale labels) are
    skipped, and duplicate pairs are collapsed."""
    bad_rows = {index[c] for c in pref.bad if c in index}
    pairs: set[Tuple[int, int]] = set()

    if pref.ranking:
        rows = [index[c] for c in pref.ranking if c in index]
        pairs.update((rows[a], rows[b])
                     for a in range(len(rows))
                     for b in range(a + 1, len(rows)))
    elif pref.best and pref.best in index:
        w = index[pref.best]
        pairs.update((w, r) for r in index.values() if r != w)

    # every acceptable candidate outranks every explicitly-bad one
    acceptable = [r for r in index.values() if r not in bad_rows]
    pairs.update((a, b) for b in bad_rows for a in acceptable)
    return sorted(pairs)


# ── the convex objective ─────────────────────────────────────────────────────

def _diff_stack(matrices: Sequence[BriefMatrix]) -> np.ndarray:
    """All preference pairs across briefs as winner-minus-loser rows D, so
    the data loss is Σ softplus(-w · D_row)."""
    diffs = [m.G[w] - m.G[l]
             for m in matrices for (w, l) in m.pref_pairs]
    if not diffs:
        return np.zeros((0, len(WEIGHT_KEYS)))
    return np.asarray(diffs, dtype=float)


def _objective(w: np.ndarray, D: np.ndarray, w0: np.ndarray,
               lam: float) -> Tuple[float, np.ndarray]:
    """Convex NLL + relative-L2 regularizer, with exact gradient."""
    # data term: Σ softplus(-z),  z = D w
    if D.shape[0]:
        z = D @ w
        # softplus(-z) = log(1+e^{-z}); stable form
        data = np.logaddexp(0.0, -z).sum()
        # d/dw softplus(-z) = -σ(-z) · D_row ;  σ(-z) = 1/(1+e^{z})
        sig_neg = 1.0 / (1.0 + np.exp(z))          # σ(-z)
        grad_data = -(D * sig_neg[:, None]).sum(axis=0)
    else:
        data = 0.0
        grad_data = np.zeros_like(w)
    # regularizer: λ Σ ((w-w0)/w0)^2
    rel = (w - w0) / w0
    reg = lam * float(rel @ rel)
    grad_reg = lam * 2.0 * rel / w0
    return data + reg, grad_data + grad_reg


def fit_weights(matrices: Sequence[BriefMatrix], w0: np.ndarray,
                lam: float) -> np.ndarray:
    """Global optimum of the convex objective for a fixed λ (L-BFGS-B with
    bounds 0 ≤ w_k ≤ _MAX_WEIGHT_MULTIPLE · w0_k)."""
    D = _diff_stack(matrices)
    bounds = [(0.0, _MAX_WEIGHT_MULTIPLE * float(b)) for b in w0]
    res = minimize(_objective, x0=w0.copy(), args=(D, w0, lam),
                   jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-10})
    return np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])


# ── agreement metrics (the honest objective) ─────────────────────────────────

def top1_agreement(matrices: Sequence[BriefMatrix], w: np.ndarray) -> float:
    """Fraction of labeled briefs whose argmax_c w·g matches the human's
    top pick. Briefs without a single `best` pick are skipped."""
    hits = total = 0
    for m in matrices:
        if m.best_row is None:
            continue
        total += 1
        if int(np.argmax(m.G @ w)) == m.best_row:
            hits += 1
    return hits / total if total else 0.0


def pairwise_agreement(matrices: Sequence[BriefMatrix],
                       w: np.ndarray) -> float:
    """Fraction of preference pairs the weights order correctly
    (w · (g_winner - g_loser) > 0)."""
    hits = total = 0
    for m in matrices:
        s = m.G @ w
        for (win, lose) in m.pref_pairs:
            total += 1
            if s[win] > s[lose]:
                hits += 1
    return hits / total if total else 0.0


# ── cross-validated tuning ───────────────────────────────────────────────────

#: a weight this close to its lower bound (0) counts as "collapsed" —
#: the data offered no gradient to keep it, not evidence the rule is unwanted
_FLOOR_EPS = 1e-6

#: labeled briefs per free weight below which a collapsed-weight warning
#: fires; with 20 weights this asks for a small multiple of that as briefs
_MIN_BRIEFS_PER_WEIGHT = 0.5


@dataclass
class TuneResult:
    weights: Dict[str, float]              # tuned weight dict (WEIGHT_KEYS)
    chosen_lambda: float
    base_top1: float                       # default weights, in-sample
    tuned_top1_cv: float                   # tuned, leave-one-brief-out (honest)
    base_pairwise: float
    tuned_pairwise_cv: float
    n_briefs_labeled: int
    n_pairs: int
    zeroed_weights: List[str] = field(default_factory=list)
    per_lambda: List[Dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def apply_to(self, base: EngineConfig) -> EngineConfig:
        return with_weights(base, self.weights)


def _cv_scores(matrices: Sequence[BriefMatrix], w0: np.ndarray,
               lam: float) -> Tuple[float, float]:
    """Leave-one-brief-out held-out (top1, pairwise) for a given λ."""
    # top-1 CV counts only briefs that carry a single best pick
    top1_hits = top1_total = 0
    pair_hits = pair_total = 0
    for held in matrices:
        train = [m for m in matrices if m is not held]
        if not train:
            continue
        w = fit_weights(train, w0, lam)
        if held.best_row is not None:
            top1_total += 1
            if int(np.argmax(held.G @ w)) == held.best_row:
                top1_hits += 1
        s = held.G @ w
        for (win, lose) in held.pref_pairs:
            pair_total += 1
            if s[win] > s[lose]:
                pair_hits += 1
    top1 = top1_hits / top1_total if top1_total else 0.0
    pair = pair_hits / pair_total if pair_total else 0.0
    return top1, pair


def tune(corpus: Corpus, prefs: Preferences,
         lambdas: Sequence[float] = DEFAULT_LAMBDAS,
         base_config: Optional[EngineConfig] = None) -> TuneResult:
    """Fit the selector weights to human preferences with CV-chosen
    regularization. Returns the tuned weights plus honest before/after
    agreement metrics."""
    base_config = base_config or EngineConfig.from_dict(corpus.base_config)
    w0 = np.array([getattr(base_config, k) for k in WEIGHT_KEYS], dtype=float)

    matrices = build_matrices(corpus, prefs)
    n_pairs = sum(len(m.pref_pairs) for m in matrices)
    if not matrices:
        raise ValueError(
            "no labeled briefs with usable preferences — label the review "
            "UI first (need ≥2 candidates and a pick per brief)")

    base_top1 = top1_agreement(matrices, w0)
    base_pair = pairwise_agreement(matrices, w0)

    # Evaluate every lambda; report them smallest-first (readable sweep),
    # but SELECT the largest lambda among those tied for best held-out
    # score. Sparse labels (few briefs, 20 free weights) make lambda=0
    # under-determined: L-BFGS-B can drive weights that don't discriminate
    # the observed pairs to their 0 boundary without hurting held-out score
    # at all — a spurious "this rule doesn't matter" that regularization
    # exists precisely to prevent. Preferring the most-regularized tied
    # optimum keeps every weight closer to the trusted hand-tuned defaults
    # unless the data actually earns the deviation.
    per_lambda: List[Dict] = []
    scored: List[Tuple[float, Tuple[float, float]]] = []
    for lam in lambdas:
        cv_top1, cv_pair = _cv_scores(matrices, w0, lam)
        rounded = (round(cv_top1, 4), round(cv_pair, 4))
        per_lambda.append({"lambda": lam, "cv_top1": rounded[0],
                           "cv_pairwise": rounded[1]})
        scored.append((lam, rounded))

    best_key = max(key for _lam, key in scored)
    chosen_lambda = max(lam for lam, key in scored if key == best_key)
    cv_top1, cv_pair = best_key
    final_w = fit_weights(matrices, w0, chosen_lambda)
    weights = {k: float(v) for k, v in zip(WEIGHT_KEYS, final_w)}
    zeroed = [k for k, v in weights.items() if v < _FLOOR_EPS]

    notes: List[str] = []
    if base_top1 >= cv_top1:
        notes.append(
            "cross-validated top-1 did not beat the defaults — the current "
            "weights already rank these briefs as well as the labels can "
            "justify; treat tuned weights as low-confidence.")
    n_labeled = len(matrices)
    if n_labeled < len(WEIGHT_KEYS) * _MIN_BRIEFS_PER_WEIGHT:
        notes.append(
            f"only {n_labeled} labeled briefs for {len(WEIGHT_KEYS)} free "
            "weights — CV estimates are noisy; label more briefs (or raise "
            "k) before trusting the deltas.")
    if len(zeroed) >= len(WEIGHT_KEYS) // 3:
        notes.append(
            f"{len(zeroed)}/{len(WEIGHT_KEYS)} weights collapsed to zero: "
            f"{', '.join(zeroed)}. With this few labeled briefs relative to "
            "20 free weights the fit is underdetermined — a weight at zero "
            "means the labels gave it no gradient to survive on, NOT that "
            "the rule is unwanted (raising regularization erases the CV "
            "gain entirely rather than fixing this — see per_lambda). "
            "DO NOT wire tuned_config.json into the live app as-is; the "
            "safe path is to gather more labels (more briefs and/or a "
            "second labeling pass with fresh candidate pools) before this "
            "fit is trustworthy.")

    return TuneResult(
        weights=weights, chosen_lambda=chosen_lambda,
        base_top1=round(base_top1, 4), tuned_top1_cv=round(cv_top1, 4),
        base_pairwise=round(base_pair, 4), tuned_pairwise_cv=round(cv_pair, 4),
        n_briefs_labeled=n_labeled, n_pairs=n_pairs, zeroed_weights=zeroed,
        per_lambda=per_lambda, notes=notes)
