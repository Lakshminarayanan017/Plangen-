"""
train.py — build the corpus, fit the critic, and report honestly.

    python -m critic.train                     # build + train + evaluate
    python -m critic.train --corpus out.npz    # reuse a saved corpus
    python -m critic.train --no-save           # evaluate without shipping

The report is the deliverable, not the model. It prints:
  • held-out AUC/accuracy, split BY BRIEF (never by sample — a perturbation
    shares almost every feature with its parent)
  • per-damage-type recall, so a critic that only ever learns to spot one
    perturbation is obvious instead of hiding behind an average
  • gain importance, so what it keyed on can be read
  • RANKING lift: on held-out briefs, how often the critic puts the clean
    plan above its damaged siblings, against the soft score doing the same
    job. A critic that does not beat the rules it was meant to augment is
    not shipped, and this is where that shows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Dict, List

import numpy as np

from modules.step4_generate.core.console import force_utf8_console
from modules.step4_generate.critic import features as feat
from modules.step4_generate.critic.critic import DEFAULT_MODEL_PATH, WEIGHTS_DIR
from modules.step4_generate.critic.dataset import Corpus, build_corpus
from modules.step4_generate.critic.gbt import GBTConfig, accuracy, roc_auc, train_gbt
from modules.step4_generate.critic.preferences import agreement, read_log
from modules.step4_generate.engine.contracts import EngineConfig
from ml.harness.briefs import golden_briefs

REPORT_PATH = os.path.join(WEIGHTS_DIR, "critic_report.json")


def per_kind_ranking(corpus: Corpus, probs: np.ndarray) -> Dict[str, float]:
    """For each damage type: fraction of (clean, damaged) pairs FROM THE
    SAME BRIEF that the critic orders correctly.

    Deliberately threshold-free. Recall at a 0.5 cut was the first version
    and it was actively misleading: with ~4.7 negatives per positive the
    model's calibrated probabilities sit below 0.5 almost everywhere, so
    'clean' scored 0.0 while the ranking — the only thing the critic is
    ever used for — was 91%. A metric that damns a working model is worse
    than no metric.
    """
    by_brief: Dict[str, List[int]] = {}
    for i, brief in enumerate(corpus.briefs):
        by_brief.setdefault(brief, []).append(i)

    wins: Dict[str, List[float]] = {}
    for idx in by_brief.values():
        clean = [i for i in idx if corpus.y[i] >= 0.5]
        for d in idx:
            if corpus.y[d] >= 0.5:
                continue
            kind = corpus.kinds[d]
            for c in clean:
                wins.setdefault(kind, []).append(
                    1.0 if probs[c] > probs[d]
                    else 0.5 if probs[c] == probs[d] else 0.0)
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(wins.items())}


def balanced_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    """The cut that maximizes balanced accuracy — the honest operating
    point for a class-imbalanced ranking model."""
    best = (0.0, 0.5)
    for t in np.unique(np.round(probs, 4)):
        tp = float(((probs >= t) & (y >= 0.5)).sum())
        tn = float(((probs < t) & (y < 0.5)).sum())
        pos, neg = float((y >= 0.5).sum()), float((y < 0.5).sum())
        if pos and neg:
            score = 0.5 * (tp / pos + tn / neg)
            if score > best[0]:
                best = (score, float(t))
    return best[1]


def ranking_lift(corpus: Corpus, probs: np.ndarray) -> Dict[str, float]:
    """Within each brief, how often a clean plan outranks a damaged one.

    The soft score is not in the feature vector, so it is recomputed here
    from what IS: `area_drift` and the violation counts are its inputs. To
    keep the comparison fair and simple we use the reviewer's own ordering
    proxy — the count of rule violations — which is what a rules-only
    ranking would fall back on."""
    by_brief: Dict[str, List[int]] = {}
    for i, brief in enumerate(corpus.briefs):
        by_brief.setdefault(brief, []).append(i)

    viol_idx = [feat.FEATURE_NAMES.index(k) for k in
                ("freespace_violations", "privacy_doors", "wet_social_doors",
                 "nbc_violations", "daylightless")]
    critic_ok = rules_ok = pairs = 0
    for idx in by_brief.values():
        clean = [i for i in idx if corpus.y[i] >= 0.5]
        damaged = [i for i in idx if corpus.y[i] < 0.5]
        for c in clean:
            rules_c = float(corpus.X[c, viol_idx].sum())
            for d in damaged:
                pairs += 1
                critic_ok += int(probs[c] > probs[d])
                rules_d = float(corpus.X[d, viol_idx].sum())
                rules_ok += int(rules_c < rules_d) + \
                    0.5 * int(rules_c == rules_d)
    if not pairs:
        return {"pairs": 0, "critic": 0.0, "rules_only": 0.0}
    return {"pairs": pairs,
            "critic": round(critic_ok / pairs, 4),
            "rules_only": round(rules_ok / pairs, 4)}


def main(argv=None) -> int:
    force_utf8_console()
    p = argparse.ArgumentParser(prog="critic.train")
    p.add_argument("--corpus", default=None,
                   help="load a saved corpus .npz instead of generating")
    p.add_argument("--save-corpus", default=os.path.join(
        WEIGHTS_DIR, "critic_corpus.npz"))
    p.add_argument("--out", default=DEFAULT_MODEL_PATH)
    p.add_argument("--k", type=int, default=6, help="candidates per brief")
    p.add_argument("--positives-per-brief", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    if args.corpus and os.path.exists(args.corpus):
        corpus = Corpus.load(args.corpus)
        print(f"corpus loaded from {args.corpus}")
    else:
        print("generating plans and perturbations "
              "(this runs the whole engine) ...")
        corpus = build_corpus(
            golden_briefs(k=args.k), config=EngineConfig(),
            max_positives_per_brief=args.positives_per_brief)
        if args.save_corpus:
            corpus.save(args.save_corpus)
            print(f"corpus saved -> {os.path.abspath(args.save_corpus)}")

    if corpus.feature_names != feat.FEATURE_NAMES:
        raise SystemExit("saved corpus predates the current feature set; "
                         "rebuild it without --corpus")

    info = corpus.summary()
    print(f"  {info['n_samples']} samples · {info['n_features']} features · "
          f"{info['n_briefs']} briefs · {info['positives']} positive")
    print(f"  by kind: {info['by_kind']}")

    train, val = corpus.split_by_brief(args.val_fraction, args.seed)
    if len(train) == 0 or len(val) == 0:
        raise SystemExit("split produced an empty side — more briefs needed")
    print(f"  split by brief: {len(train)} train / {len(val)} val")

    model = train_gbt(
        train.X, train.y,
        config=GBTConfig(n_estimators=args.trees, max_depth=args.depth,
                         learning_rate=args.lr),
        feature_names=feat.FEATURE_NAMES, validation=(val.X, val.y))

    val_p = model.predict_proba(val.X)
    train_p = model.predict_proba(train.X)
    cut = balanced_threshold(val.y, val_p)
    report = {
        "n_trees": len(model.trees),
        "balanced_threshold": round(cut, 4),
        "train": {"auc": round(roc_auc(train.y, train_p), 4),
                  "acc": round(accuracy(train.y, train_p, cut), 4)},
        "val": {"auc": round(roc_auc(val.y, val_p), 4),
                "acc": round(accuracy(val.y, val_p, cut), 4)},
        "val_per_kind": per_kind_ranking(val, val_p),
        "val_ranking": ranking_lift(val, val_p),
        "importance": {k: round(v, 4)
                       for k, v in list(model.gain_importance().items())[:12]},
        "corpus": info,
    }
    model.train_report.update(report)

    print("-" * 70)
    print(f"  trees kept        : {report['n_trees']}")
    print(f"  train AUC / acc   : {report['train']['auc']} / "
          f"{report['train']['acc']}")
    print(f"  HELD-OUT AUC / acc: {report['val']['auc']} / "
          f"{report['val']['acc']}  (at threshold {cut:.3f})")
    print(f"  per damage type (clean ranked above this damage):")
    for kind, value in report["val_per_kind"].items():
        print(f"      {kind:<22}{value:.1%}")
    rank = report["val_ranking"]
    print(f"  ranking (clean above damaged, {rank['pairs']} pairs):")
    print(f"      critic     {rank['critic']:.1%}")
    print(f"      rules only {rank['rules_only']:.1%}")
    print("-" * 70)
    print(model.explain(8))

    prefs = read_log()
    if prefs:
        agree = agreement(prefs, model.score_one)
        report["preference_agreement"] = agree
        print("-" * 70)
        print(f"  logged user picks : {agree['n']}")
        print(f"  critic top-1      : {agree['critic_top1']:.1%}")
        print(f"  soft-score top-1  : {agree['soft_top1']:.1%}")
        print(f"  random-of-K       : {agree['random_top1']:.1%}")
    else:
        print("\n  no user picks logged yet (critic/preferences.jsonl) — the "
              "\n  ranking lift above is against perturbations, not taste.")

    ships = rank["critic"] > rank["rules_only"] and report["val"]["auc"] > 0.7
    print("-" * 70)
    print(f"  VERDICT: {'ship' if ships else 'DO NOT SHIP'} — "
          f"held-out AUC {report['val']['auc']}, ranking "
          f"{rank['critic']:.1%} vs rules {rank['rules_only']:.1%}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"  report -> {os.path.abspath(REPORT_PATH)}")

    if args.no_save:
        print("  --no-save: weights not written")
    elif ships:
        model.save(args.out)
        print(f"  weights -> {os.path.abspath(args.out)}")
    else:
        print("  weights NOT written (the critic did not beat the rules)")
    return 0 if ships else 1


if __name__ == "__main__":
    raise SystemExit(main())
