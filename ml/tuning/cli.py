"""
cli.py — the Phase-A workflow driver.

Run from the project root:

  python -m tuning.cli export              # build corpus + labeling.html
  #  … open tuning/results/labeling.html, pick a best plan per brief,
  #     Export labels.json into tuning/results/ …
  python -m tuning.cli tune                # fit weights → tuned_config.json
  python -m tuning.cli verify              # base-vs-tuned re-ranking report

Every step is explicit about what it needs and where it wrote things. The
tune step prints the honest cross-validated selection-agreement numbers,
never raw score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from modules.step4_generate.engine.contracts import EngineConfig
from ml.tuning.corpus import CORPUS_PATH, Corpus, build_corpus
from ml.tuning.labeling import (
    LABELING_HTML_PATH, LABELS_PATH, Preferences, render_labeling_html,
)
from ml.tuning.tuner import tune
from ml.tuning.verify import (
    VERIFY_HTML_PATH, compare, render_verify_html, weight_diff_table,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TUNED_CONFIG_PATH = os.path.join(RESULTS_DIR, "tuned_config.json")
TUNE_REPORT_PATH = os.path.join(RESULTS_DIR, "tune_report.json")


def _load_config(path: str | None) -> EngineConfig:
    if not path:
        return EngineConfig()
    with open(path, encoding="utf-8") as f:
        return EngineConfig.from_dict(json.load(f))


# ── export ───────────────────────────────────────────────────────────────────

def cmd_export(args) -> int:
    config = _load_config(args.config)
    print(f"Building corpus over the golden briefs (k={args.k}, "
          f"seed={args.seed}) — this runs the full engine per brief…")
    corpus = build_corpus(config, k=args.k, seed=args.seed, render=True)

    corpus.save(CORPUS_PATH, include_svg=True)
    render_labeling_html(corpus)

    n_labelable = len(corpus.labelable())
    n_cands = sum(len(b.candidates) for b in corpus.briefs)
    print(f"\n  {n_cands} candidates across {len(corpus.briefs)} briefs "
          f"({n_labelable} have ≥2 candidates to choose between).")
    print(f"  corpus   -> {os.path.abspath(CORPUS_PATH)}")
    print(f"  labeling -> {os.path.abspath(LABELING_HTML_PATH)}")
    print("\n  Next: open the labeling page, pick the best plan per brief, "
          "click\n  'Export labels.json', and save it to "
          f"{os.path.abspath(LABELS_PATH)}")
    return 0


# ── tune ─────────────────────────────────────────────────────────────────────

def cmd_tune(args) -> int:
    if not os.path.exists(args.corpus):
        print(f"no corpus at {args.corpus} — run `export` first.",
              file=sys.stderr)
        return 2
    if not os.path.exists(args.labels):
        print(f"no labels at {args.labels} — label the review UI and export "
              "labels.json there first.", file=sys.stderr)
        return 2

    corpus = Corpus.load(args.corpus)
    prefs = Preferences.load(args.labels)
    base = EngineConfig.from_dict(corpus.base_config)
    result = tune(corpus, prefs, base_config=base)

    tuned = result.apply_to(base)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TUNED_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(tuned.to_dict(), f, indent=1)
    with open(TUNE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "chosen_lambda": result.chosen_lambda,
            "n_briefs_labeled": result.n_briefs_labeled,
            "n_pairs": result.n_pairs,
            "base_top1": result.base_top1,
            "tuned_top1_cv": result.tuned_top1_cv,
            "base_pairwise": result.base_pairwise,
            "tuned_pairwise_cv": result.tuned_pairwise_cv,
            "zeroed_weights": result.zeroed_weights,
            "per_lambda": result.per_lambda,
            "weights": result.weights,
            "notes": result.notes,
        }, f, indent=1)

    _print_tune_report(result)
    print(f"\n  tuned config -> {os.path.abspath(TUNED_CONFIG_PATH)}")
    print(f"  report       -> {os.path.abspath(TUNE_REPORT_PATH)}")
    print("\n  Next: `python -m tuning.cli verify` to eyeball the plans "
          "whose best pick changed.")
    return 0


def _print_tune_report(result) -> None:
    print("=" * 68)
    print("SELECTOR TUNING — honest, cross-validated (never raw score)")
    print("=" * 68)
    print(f"  labeled briefs / pref pairs : {result.n_briefs_labeled} "
          f"/ {result.n_pairs}")
    print(f"  chosen lambda (regulariser) : {result.chosen_lambda}")
    print(f"  top-1 selection agreement   : {result.base_top1:.0%} (default) "
          f"-> {result.tuned_top1_cv:.0%} (tuned, leave-one-brief-out)")
    print(f"  pairwise agreement          : {result.base_pairwise:.0%} -> "
          f"{result.tuned_pairwise_cv:.0%} (held out)")
    print("-" * 68)
    print("  lambda sweep (held-out top-1 / pairwise):")
    for row in result.per_lambda:
        marker = "  <-- chosen" if row["lambda"] == result.chosen_lambda else ""
        print(f"    lam={row['lambda']:<6} top1={row['cv_top1']:.3f}  "
              f"pair={row['cv_pairwise']:.3f}{marker}")
    if result.notes:
        print("-" * 68)
        for n in result.notes:
            print(f"  ! {n}")


# ── verify ───────────────────────────────────────────────────────────────────

def cmd_verify(args) -> int:
    if not os.path.exists(args.corpus):
        print(f"no corpus at {args.corpus} — run `export` first.",
              file=sys.stderr)
        return 2
    if not os.path.exists(args.tuned):
        print(f"no tuned config at {args.tuned} — run `tune` first.",
              file=sys.stderr)
        return 2

    corpus = Corpus.load(args.corpus)
    base = EngineConfig.from_dict(corpus.base_config)
    tuned = EngineConfig.from_dict(json.load(open(args.tuned, encoding="utf-8")))
    prefs = (Preferences.load(args.labels)
             if os.path.exists(args.labels) else None)

    report = compare(corpus, base, tuned, prefs)
    diff_rows = weight_diff_table(base, tuned)
    render_verify_html(corpus, report, diff_rows)

    print(report.summary_line())
    print("\n  weight changes (largest relative move first):")
    for r in diff_rows[:10]:
        if r["x"] != 1.0:
            print(f"    {r['weight']:<20} {r['base']:>8} -> {r['tuned']:>8} "
                  f"({r['x']}×)")
    print(f"\n  verify report -> {os.path.abspath(VERIFY_HTML_PATH)}")
    return 0


# ── argparse ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tuning.cli",
                                description="PlanGen Phase-A selector tuning.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="build the candidate corpus + "
                                      "labeling UI")
    e.add_argument("--k", type=int, default=8,
                   help="candidates per brief to generate (default 8)")
    e.add_argument("--seed", type=int, default=20260716)
    e.add_argument("--config", default=None,
                   help="EngineConfig JSON to generate under (default: "
                        "engine defaults)")
    e.set_defaults(func=cmd_export)

    t = sub.add_parser("tune", help="fit selector weights to labels.json")
    t.add_argument("--corpus", default=CORPUS_PATH)
    t.add_argument("--labels", default=LABELS_PATH)
    t.set_defaults(func=cmd_tune)

    v = sub.add_parser("verify", help="base-vs-tuned re-ranking report")
    v.add_argument("--corpus", default=CORPUS_PATH)
    v.add_argument("--tuned", default=TUNED_CONFIG_PATH)
    v.add_argument("--labels", default=LABELS_PATH)
    v.set_defaults(func=cmd_verify)
    return p


def _force_utf8_console() -> None:
    """See core.console — one implementation, every CLI."""
    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()


def main(argv=None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
