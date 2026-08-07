"""
merge_gate.py — decide whether the trained placer replaces the prior proposer.

implementation_plan_v2.md §2.5 states the gate, and this module is the ONLY
thing allowed to answer it. Both proposers run the identical golden harness,
same briefs, same seeds, same engine config, so the only variable is who
proposes the seed cells:

    PASS  iff  mean best score improves
          and  no single brief regresses by more than --max-regression (5)
          and  mean fidelity >= --min-fidelity (0.8)

Two arms are measured for the model, because they answer different questions:
  • deployed  (tau as configured) — what users would actually get, fallback
    included. This is the arm the gate judges.
  • model-only (tau = -1, never falls back) — what the network itself can do.
    Diagnostic: a deployed arm that merely ties the baseline because it fell
    back on every brief is NOT a model that learned anything, and the
    fallback rate printed here is what distinguishes the two cases.

Usage:
    python -m tier2_placer.merge_gate --weights tier2_placer/weights/placer_last.npz
    python -m tier2_placer.merge_gate --weights ... --json out.json --html out.html

Exit code 0 = gate passed (safe to merge), 1 = did not pass, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

from modules.step4_generate.engine.contracts import EngineConfig
from modules.step4_generate.engine.fallbacks import PriorProposer
from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "gate_results")

# Gate thresholds (implementation_plan_v2.md §2.5).
MAX_REGRESSION = 5.0
MIN_FIDELITY = 0.8


# ── one arm of the comparison ────────────────────────────────────────────────

def run_arm(proposer, briefs, config: EngineConfig) -> Dict:
    """Run the whole brief set through one proposer. Returns per-brief rows
    plus aggregates — the same fields run_harness records, so gate numbers
    and harness numbers are directly comparable."""
    orch = Orchestrator(proposer=proposer, config=config)
    rows: List[Dict] = []
    t0 = time.perf_counter()
    for request in briefs:
        t_brief = time.perf_counter()
        result = orch.generate(request)
        best = result.best
        rows.append({
            "brief": request.name,
            "kept": len(result.ranked),
            "best_score": round(best.verdict.soft_score, 2) if best else 0.0,
            "fidelity": round(best.fidelity, 3) if best else 0.0,
            "hard_fails": len(result.discarded),
            "seconds": round(time.perf_counter() - t_brief, 2),
        })
    scored = [r for r in rows if r["kept"] > 0]
    return {
        "rows": rows,
        "n_briefs": len(rows),
        "briefs_with_plan": len(scored),
        # briefs that produced NO plan score 0 and are counted in the mean:
        # a proposer that wins on the briefs it solves while failing others
        # has not earned a merge.
        "mean_best_score": round(
            sum(r["best_score"] for r in rows) / max(1, len(rows)), 2),
        "mean_fidelity": round(
            sum(r["fidelity"] for r in scored) / max(1, len(scored)), 3),
        "total_seconds": round(time.perf_counter() - t0, 1),
    }


def load_model_placer(weights: str, tau: float, temperature: float = 1.0):
    """Build a Tier2Placer on the production NumPy backend (no torch)."""
    from ml.tier2_placer.numpy_infer import NumpyPlacer
    from ml.tier2_placer.tier2_placer import Tier2Placer
    return Tier2Placer(NumpyPlacer.from_npz(weights), tau=tau,
                       temperature=temperature)


# ── the gate ─────────────────────────────────────────────────────────────────

def evaluate(baseline: Dict, deployed: Dict, *,
             max_regression: float = MAX_REGRESSION,
             min_fidelity: float = MIN_FIDELITY) -> Dict:
    """Apply §2.5 to two completed arms. Returns the verdict + every check
    that was made, passing or not — a gate that only reports its failures
    cannot be audited."""
    by_brief = {r["brief"]: r for r in baseline["rows"]}
    regressions = []
    improvements = []
    for row in deployed["rows"]:
        base = by_brief.get(row["brief"])
        if base is None:
            continue
        delta = round(row["best_score"] - base["best_score"], 2)
        entry = {"brief": row["brief"], "delta": delta,
                 "baseline": base["best_score"], "model": row["best_score"]}
        if delta < -max_regression:
            regressions.append(entry)
        elif delta > 0:
            improvements.append(entry)

    score_delta = round(deployed["mean_best_score"]
                        - baseline["mean_best_score"], 2)
    lost_plans = baseline["briefs_with_plan"] - deployed["briefs_with_plan"]

    checks = [
        {"check": "mean best score improves",
         "pass": score_delta > 0,
         "detail": f"{baseline['mean_best_score']} -> "
                   f"{deployed['mean_best_score']} ({score_delta:+.2f})"},
        {"check": f"no brief regresses > {max_regression} pts",
         "pass": not regressions,
         "detail": "clean" if not regressions else ", ".join(
             f"{r['brief']} {r['delta']:+.2f}" for r in regressions)},
        {"check": f"mean fidelity >= {min_fidelity}",
         "pass": deployed["mean_fidelity"] >= min_fidelity,
         "detail": f"{deployed['mean_fidelity']}"},
        {"check": "no brief loses its plan",
         "pass": lost_plans <= 0,
         "detail": f"{baseline['briefs_with_plan']} -> "
                   f"{deployed['briefs_with_plan']} briefs with a plan"},
    ]
    return {
        "passed": all(c["pass"] for c in checks),
        "checks": checks,
        "score_delta": score_delta,
        "regressions": regressions,
        "improvements": improvements,
    }


# ── reporting ────────────────────────────────────────────────────────────────

def print_report(report: Dict) -> None:
    base, dep = report["baseline"], report["deployed"]
    only = report.get("model_only")
    by_brief = {r["brief"]: r for r in base["rows"]}

    print("=" * 78)
    print("TIER-2 MERGE GATE  —  implementation_plan_v2.md 2.5")
    print(f"weights: {report['weights']}")
    print("=" * 78)
    head = f"  {'brief':<18}{'prior':>8}{'model':>8}{'delta':>8}{'fid':>7}"
    if only:
        head += f"{'model-only':>12}"
    print(head)
    only_by_brief = {r["brief"]: r for r in only["rows"]} if only else {}
    for row in dep["rows"]:
        b = by_brief.get(row["brief"], {})
        delta = row["best_score"] - b.get("best_score", 0.0)
        line = (f"  {row['brief']:<18}{b.get('best_score', 0):>8.1f}"
                f"{row['best_score']:>8.1f}{delta:>+8.1f}"
                f"{row['fidelity']:>7.2f}")
        if only:
            line += f"{only_by_brief.get(row['brief'], {}).get('best_score', 0):>12.1f}"
        print(line + ("   << NO PLAN" if row["kept"] == 0 else ""))

    print("-" * 78)
    print(f"  mean score   prior {base['mean_best_score']:>7.2f}   "
          f"model {dep['mean_best_score']:>7.2f}   "
          f"({report['verdict']['score_delta']:+.2f})")
    if only:
        print(f"  model-only (never falls back): "
              f"{only['mean_best_score']:.2f}   "
              f"fidelity {only['mean_fidelity']:.3f}")
    print(f"  fallback rate: {report['fallback_rate']:.0%}   "
          f"mean confidence {report['mean_confidence']:.3f} "
          f"(tau {report['tau']})")
    print("-" * 78)
    for c in report["verdict"]["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']:<34}"
              f"{c['detail']}")
    print("-" * 78)
    verdict = report["verdict"]["passed"]
    print(f"  GATE: {'PASSED — safe to merge' if verdict else 'NOT PASSED — model stays on its branch'}")
    for note in report.get("notes", []):
        print(f"  note: {note}")


def write_html(report: Dict, path: str) -> None:
    v = report["verdict"]
    by_brief = {r["brief"]: r for r in report["baseline"]["rows"]}
    rows = "".join(
        f"<tr><td>{r['brief']}</td>"
        f"<td>{by_brief.get(r['brief'], {}).get('best_score', 0):.1f}</td>"
        f"<td>{r['best_score']:.1f}</td>"
        f"<td class='{'up' if r['best_score'] >= by_brief.get(r['brief'], {}).get('best_score', 0) else 'down'}'>"
        f"{r['best_score'] - by_brief.get(r['brief'], {}).get('best_score', 0):+.1f}</td>"
        f"<td>{r['fidelity']:.2f}</td></tr>"
        for r in report["deployed"]["rows"])
    checks = "".join(
        f"<li class='{'up' if c['pass'] else 'down'}'>"
        f"<b>{'PASS' if c['pass'] else 'FAIL'}</b> {c['check']} — {c['detail']}</li>"
        for c in v["checks"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'><title>Tier-2 merge gate</title>"
            "<style>body{font:14px Segoe UI,sans-serif;margin:32px;color:#222}"
            "table{border-collapse:collapse}td,th{border:1px solid #ddd;"
            "padding:4px 10px;text-align:right}td:first-child{text-align:left}"
            ".up{color:#0a0}.down{color:#c00}"
            f".verdict{{font-size:20px;font-weight:600;color:{'#0a0' if v['passed'] else '#c00'}}}"
            "</style>"
            f"<h1>Tier-2 merge gate</h1><p class='verdict'>"
            f"{'PASSED' if v['passed'] else 'NOT PASSED'}</p>"
            f"<p>weights <code>{report['weights']}</code> · "
            f"fallback rate {report['fallback_rate']:.0%} · "
            f"mean confidence {report['mean_confidence']:.3f} "
            f"(tau {report['tau']})</p>"
            f"<ul>{checks}</ul>"
            "<table><tr><th>brief</th><th>prior</th><th>model</th>"
            f"<th>delta</th><th>fidelity</th></tr>{rows}</table>")


# ── CLI ──────────────────────────────────────────────────────────────────────

def run_gate(weights: str, *, tau: Optional[float] = None,
             temperature: float = 1.0, k: int = 6,
             max_regression: float = MAX_REGRESSION,
             min_fidelity: float = MIN_FIDELITY,
             model_only: bool = True) -> Dict:
    from ml.tier2_placer.tier2_placer import DEFAULT_TAU
    tau = DEFAULT_TAU if tau is None else tau
    config = EngineConfig()
    briefs = golden_briefs(k=k)

    baseline = run_arm(PriorProposer(), briefs, config)
    placer = load_model_placer(weights, tau, temperature)
    deployed = run_arm(placer, briefs, config)

    only = None
    if model_only:
        raw = load_model_placer(weights, -1.0, temperature)
        only = run_arm(raw, briefs, config)

    verdict = evaluate(baseline, deployed, max_regression=max_regression,
                       min_fidelity=min_fidelity)
    notes = []
    if placer.fallback_rate >= 0.999:
        notes.append(
            "every proposal fell back — the deployed arm IS the prior "
            "proposer, so this gate says nothing about the model. Compare "
            "the model-only column and lower --tau to test the network.")
    elif placer.fallback_rate > 0.5:
        notes.append(f"{placer.fallback_rate:.0%} of proposals fell back; the "
                     f"deployed arm is mostly the prior proposer.")
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": os.path.abspath(weights),
        "tau": tau,
        "temperature": temperature,
        "k": k,
        "fallback_rate": round(placer.fallback_rate, 3),
        "mean_confidence": round(placer.mean_confidence, 3),
        "baseline": baseline,
        "deployed": deployed,
        "model_only": only,
        "verdict": verdict,
        "notes": notes,
    }


def main(argv=None) -> int:
    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()
    p = argparse.ArgumentParser(prog="tier2_placer.merge_gate")
    p.add_argument("--weights", required=True, help="placer .npz weights")
    p.add_argument("--tau", type=float, default=None,
                   help="confidence threshold (default: DEFAULT_TAU)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--k", type=int, default=6, help="candidates per brief")
    p.add_argument("--max-regression", type=float, default=MAX_REGRESSION)
    p.add_argument("--min-fidelity", type=float, default=MIN_FIDELITY)
    p.add_argument("--no-model-only", action="store_true",
                   help="skip the never-fall-back diagnostic arm (faster)")
    p.add_argument("--json", default=os.path.join(RESULTS_DIR, "latest.json"))
    p.add_argument("--html", default=os.path.join(RESULTS_DIR, "gate.html"))
    p.add_argument("--quiet-engine", action="store_true", default=True,
                   help="mute engine warnings (tight-program noise)")
    args = p.parse_args(argv)

    if not os.path.exists(args.weights):
        print(f"weights not found: {args.weights}\n"
              f"export them first:  python -m tier2_placer.export "
              f"--checkpoint <checkpoint.pt>")
        return 2
    if args.quiet_engine:
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)

    report = run_gate(args.weights, tau=args.tau,
                      temperature=args.temperature, k=args.k,
                      max_regression=args.max_regression,
                      min_fidelity=args.min_fidelity,
                      model_only=not args.no_model_only)
    print_report(report)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"\n  report -> {os.path.abspath(args.json)}")
    if args.html:
        write_html(report, args.html)
        print(f"  html   -> {os.path.abspath(args.html)}")
    return 0 if report["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
