"""
run_harness.py — run the engine over the golden brief set and record the
baseline. Every engine change is judged here: the previous run's numbers
are loaded and deltas printed, so regressions are visible immediately.

Outputs:
  harness/results/latest.json    — machine-readable baseline (git-friendly)
  harness/results/review.html    — visual grid of every brief's best plan

Run from the project root:  python -m ml.harness.run_harness
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs
from modules.step4_generate.render.svg_render import render_svg

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LATEST_PATH = os.path.join(RESULTS_DIR, "latest.json")
REVIEW_PATH = os.path.join(RESULTS_DIR, "review.html")


def run(critic=None) -> Dict:
    """`critic` (a critic.LearnedCritic) reorders the candidates the rules
    accepted. Note what that CANNOT show up as: a higher mean soft score.
    The critic's whole purpose is to disagree with the soft score, so
    judging it by the soft score is circular — `critic_top1_changed` counts
    how often it picked a different plan, and whether those picks are better
    is a question only logged user choices answer (critic/preferences.py)."""
    orch = Orchestrator(critic=critic)
    rows: List[Dict] = []
    svgs: List[str] = []

    t_all = time.perf_counter()
    for request in golden_briefs():
        t0 = time.perf_counter()
        result = orch.generate(request)
        elapsed = time.perf_counter() - t0

        row = {
            "brief": request.name,
            "kept": len(result.ranked),
            "k": request.k,
            "best_score": result.best.verdict.soft_score if result.best else 0,
            "fidelity": round(result.best.fidelity, 3) if result.best else 0,
            "drift": (result.best.verdict.breakdown.get("area_drift", 1.0)
                      if result.best else 1.0),
            "worst_aspect": (result.best.verdict.breakdown.get(
                "worst_aspect", 0) if result.best else 0),
            "privacy_doors": (result.best.verdict.breakdown.get(
                "privacy_doors", 0) if result.best else 0),
            "seconds": round(elapsed, 2),
            "discard_reasons": sorted({
                c.notes[-1][:80] for c in result.discarded if c.notes}),
        }
        if critic is not None and result.ranked:
            by_rules = max(result.ranked,
                           key=lambda c: c.verdict.soft_score)
            row["critic_changed_pick"] = int(by_rules is not result.best)
            row["rules_top1_score"] = by_rules.verdict.soft_score
        rows.append(row)
        if result.best:
            svgs.append(f"<div class='cell'><h3>{request.name} — "
                        f"score {row['best_score']}</h3>"
                        f"{render_svg(result.best.plan)}</div>")
        else:
            svgs.append(f"<div class='cell fail'><h3>{request.name} — "
                        f"NO PLAN</h3><pre>"
                        f"{chr(10).join(row['discard_reasons'])}</pre></div>")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_briefs": len(rows),
        "briefs_with_plan": sum(1 for r in rows if r["kept"] > 0),
        "mean_best_score": round(
            sum(r["best_score"] for r in rows) / len(rows), 2),
        "mean_fidelity": round(
            sum(r["fidelity"] for r in rows) / len(rows), 3),
        "mean_drift": round(sum(r["drift"] for r in rows) / len(rows), 3),
        "total_privacy_doors": sum(r["privacy_doors"] for r in rows),
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "rows": rows,
    }
    if critic is not None:
        summary["critic"] = critic.model.train_report.get("val", {})
        summary["critic_top1_changed"] = sum(
            r.get("critic_changed_pick", 0) for r in rows)
    return summary, svgs


def print_report(summary: Dict, previous: Dict = None) -> None:
    print("=" * 78)
    print(f"GOLDEN HARNESS — {summary['n_briefs']} briefs   "
          f"({summary['timestamp']})")
    print("=" * 78)
    print(f"  {'brief':<18}{'kept':<7}{'score':<9}{'fid':<7}{'drift':<8}"
          f"{'aspect':<8}{'privdrs':<8}{'sec':<6}")
    for r in summary["rows"]:
        flag = "" if r["kept"] > 0 else "  << NO PLAN"
        print(f"  {r['brief']:<18}{r['kept']}/{r['k']:<5}"
              f"{r['best_score']:<9}{r['fidelity']:<7}{r['drift']:<8}"
              f"{r['worst_aspect']:<8}{r['privacy_doors']:<8}"
              f"{r['seconds']:<6}{flag}")
    print("-" * 78)
    print(f"  briefs with a valid plan : "
          f"{summary['briefs_with_plan']}/{summary['n_briefs']}")
    print(f"  mean best score          : {summary['mean_best_score']}")
    print(f"  mean fidelity            : {summary['mean_fidelity']}")
    print(f"  mean drift               : {summary['mean_drift']}")
    print(f"  total privacy doors      : {summary['total_privacy_doors']}")
    print(f"  total wall time          : {summary['total_seconds']}s")

    if previous:
        print("-" * 78)
        print("  vs previous run:")
        for key in ("briefs_with_plan", "mean_best_score", "mean_fidelity",
                    "mean_drift", "total_privacy_doors"):
            old, new = previous.get(key), summary.get(key)
            if old is not None and old != new:
                print(f"    {key:<24} {old} -> {new}")


def main(argv=None) -> None:
    import argparse

    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()
    parser = argparse.ArgumentParser(prog="harness.run_harness")
    parser.add_argument("--critic", action="store_true",
                        help="rank with the trained critic as well as the "
                             "rules (reports how often it changes the pick)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="do not overwrite results/latest.json")
    args = parser.parse_args(argv)

    critic = None
    if args.critic:
        from modules.step4_generate.critic.critic import LearnedCritic
        critic = LearnedCritic.load_if_available()
        if critic is None:
            raise SystemExit("no trained critic weights "
                             "(run: python -m critic.train)")

    previous = None
    if os.path.exists(LATEST_PATH):
        with open(LATEST_PATH, encoding="utf-8") as f:
            previous = json.load(f)

    summary, svgs = run(critic)
    print_report(summary, previous)
    if critic is not None:
        print(f"  critic changed the pick  : "
              f"{summary['critic_top1_changed']}/{summary['n_briefs']} briefs")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not (args.no_baseline or critic is not None):
        # a critic run is a different ruler; it must not become the baseline
        with open(LATEST_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    with open(REVIEW_PATH, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'>"
            "<title>PlanGen harness review</title><style>"
            "body{font-family:Segoe UI,sans-serif;background:#fafafa}"
            ".grid{display:flex;flex-wrap:wrap;gap:24px}"
            ".cell{background:#fff;padding:12px;border:1px solid #ddd}"
            ".cell.fail{border-color:#c00}"
            "h3{margin:4px 0;font-size:14px}</style>"
            f"<h1>Harness review — {summary['timestamp']}</h1>"
            f"<p>score {summary['mean_best_score']} · fidelity "
            f"{summary['mean_fidelity']} · drift {summary['mean_drift']}</p>"
            f"<div class='grid'>{''.join(svgs)}</div>"
        )
    print(f"\n  baseline -> {os.path.abspath(LATEST_PATH)}")
    print(f"  review   -> {os.path.abspath(REVIEW_PATH)}")


if __name__ == "__main__":
    main()
