"""
merge_gate.py — does Placer v3 replace the prior proposer?

Same gate as v2, deliberately unchanged, so the two are comparable: identical
briefs, identical seeds, identical engine config. The ONLY variable is who
proposes the seed cells.

    PASS iff  mean best score improves
         and  no single brief regresses by more than --max-regression (5.0)
         and  mean fidelity >= --min-fidelity (0.80)
         and  no brief loses its plan

Three arms are measured, because they answer different questions:

  baseline    PriorProposer — the incumbent. v2 measured it at 75.49.
  deployed    v3 at the real tau, fallback included. THIS is what the gate
              judges, because it is what users would actually get.
  model_only  v3 with tau = -1, never falling back. Diagnostic: a deployed
              arm that ties the baseline only because it fell back on every
              brief has learned nothing, and the fallback rate is what
              distinguishes the two cases.

Exit code 0 = passed (safe to merge), 1 = did not pass, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np

from modules.step4_generate.engine.contracts import EngineConfig
from modules.step4_generate.engine.fallbacks import PriorProposer
from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs

MAX_REGRESSION = 5.0
MIN_FIDELITY = 0.80


def run_arm(proposer, briefs, config: EngineConfig) -> Dict:
    orch = Orchestrator(config=config, proposer=proposer)
    rows: List[Dict] = []
    t0 = time.perf_counter()
    for request in briefs:
        try:
            result = orch.generate(request)
        except Exception as exc:
            rows.append({"brief": request.name, "score": 0.0, "fidelity": 0.0,
                         "kept": 0, "error": str(exc)[:80]})
            continue
        best = result.best
        rows.append({
            "brief": request.name,
            "score": round(best.verdict.soft_score, 2) if best else 0.0,
            "fidelity": round(best.fidelity or 0.0, 3) if best else 0.0,
            "kept": len(result.ranked),
        })
    scored = [r["score"] for r in rows]
    fids = [r["fidelity"] for r in rows if r["kept"]]
    return {
        "rows": rows,
        "mean_score": round(float(np.mean(scored)) if scored else 0.0, 2),
        "mean_fidelity": round(float(np.mean(fids)) if fids else 0.0, 3),
        "briefs_with_plan": sum(1 for r in rows if r["kept"] > 0),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def judge(baseline: Dict, deployed: Dict, max_regression: float,
          min_fidelity: float) -> Dict:
    by_brief = {r["brief"]: r["score"] for r in baseline["rows"]}
    deltas = []
    for row in deployed["rows"]:
        before = by_brief.get(row["brief"])
        if before is None:
            continue
        deltas.append({"brief": row["brief"],
                       "delta": round(row["score"] - before, 2),
                       "baseline": before, "model": row["score"]})
    regressions = [d for d in deltas if d["delta"] < -max_regression]
    improvements = sorted([d for d in deltas if d["delta"] > 0],
                          key=lambda d: -d["delta"])[:5]
    score_delta = round(deployed["mean_score"] - baseline["mean_score"], 2)

    checks = [
        {"check": "mean best score improves", "pass": score_delta > 0,
         "detail": f"{baseline['mean_score']} -> {deployed['mean_score']} "
                   f"({score_delta:+.2f})"},
        {"check": f"no brief regresses > {max_regression}",
         "pass": not regressions,
         "detail": ", ".join(f"{d['brief']} {d['delta']}"
                             for d in regressions) or "none"},
        {"check": f"mean fidelity >= {min_fidelity}",
         "pass": deployed["mean_fidelity"] >= min_fidelity,
         "detail": str(deployed["mean_fidelity"])},
        {"check": "no brief loses its plan",
         "pass": deployed["briefs_with_plan"] >= baseline["briefs_with_plan"],
         "detail": f"{baseline['briefs_with_plan']} -> "
                   f"{deployed['briefs_with_plan']} briefs with a plan"},
    ]
    return {"passed": all(c["pass"] for c in checks), "checks": checks,
            "score_delta": score_delta, "regressions": regressions,
            "improvements": improvements}


def main(argv=None) -> int:
    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()

    p = argparse.ArgumentParser(prog="ml.placer_v3.merge_gate")
    p.add_argument("--weights", required=True,
                   help=".npz from ml.placer_v3.export, or a .pt checkpoint")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--max-regression", type=float, default=MAX_REGRESSION)
    p.add_argument("--min-fidelity", type=float, default=MIN_FIDELITY)
    p.add_argument("--json", default=None)
    p.add_argument("--html", default=None)
    args = p.parse_args(argv)

    logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)

    try:
        net = _load_net(args.weights)
    except Exception as exc:
        print(f"could not load {args.weights}: {exc}")
        return 2

    from ml.placer_v3.decode import DEFAULT_TAU
    from ml.placer_v3.proposer import PlacerV3Proposer
    tau = DEFAULT_TAU if args.tau is None else args.tau

    config = EngineConfig()
    briefs = golden_briefs(k=args.k)
    print(f"gate: {len(briefs)} golden briefs · k={args.k} · tau={tau}\n")

    print("  baseline   (PriorProposer) ...")
    baseline = run_arm(PriorProposer(), briefs, config)

    print("  deployed   (v3 @ tau, fallback allowed) ...")
    deployed_prop = PlacerV3Proposer(net, tau=tau,
                                     temperature=args.temperature)
    deployed = run_arm(deployed_prop, briefs, config)
    deployed["fallback_rate"] = round(deployed_prop.fallback_rate, 3)
    deployed["mean_confidence"] = round(deployed_prop.mean_confidence, 3)

    print("  model_only (v3, never falls back) ...")
    model_prop = PlacerV3Proposer(net, tau=-1.0,
                                  temperature=args.temperature)
    model_only = run_arm(model_prop, briefs, config)
    model_only["mean_confidence"] = round(model_prop.mean_confidence, 3)

    verdict = judge(baseline, deployed, args.max_regression,
                    args.min_fidelity)

    print("\n" + "=" * 74)
    print(f"  {'arm':<12}{'mean score':>12}{'fidelity':>11}{'plans':>8}"
          f"{'fallback':>11}")
    for name, arm in (("baseline", baseline), ("deployed", deployed),
                      ("model_only", model_only)):
        print(f"  {name:<12}{arm['mean_score']:>12.2f}"
              f"{arm['mean_fidelity']:>11.3f}"
              f"{arm['briefs_with_plan']:>8}"
              f"{arm.get('fallback_rate', '-'):>11}")
    print("=" * 74)
    for c in verdict["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']:<34} "
              f"{c['detail']}")
    print("=" * 74)
    print(f"  VERDICT: {'PASS — safe to merge' if verdict['passed'] else 'DID NOT PASS'}")
    if deployed.get("fallback_rate", 0) > 0.25:
        print(f"  NOTE: fell back on {deployed['fallback_rate']:.0%} of "
              f"proposals — the deployed arm is largely the OLD proposer.\n"
              f"        Compare model_only ({model_only['mean_score']:.2f}) "
              f"to see what the network itself did.")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": os.path.abspath(args.weights),
        "tau": tau, "temperature": args.temperature, "k": args.k,
        "baseline": baseline, "deployed": deployed, "model_only": model_only,
        "verdict": verdict,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".",
                    exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"  report -> {os.path.abspath(args.json)}")
    if args.html:
        _write_html(report, args.html)
        print(f"  html   -> {os.path.abspath(args.html)}")
    return 0 if verdict["passed"] else 1


def _load_net(path: str):
    import torch
    from ml.placer_v3.config import PlacerV3Config
    from ml.placer_v3.model.placer_net import PlacerNetV3

    if path.endswith(".npz"):
        blob = np.load(path, allow_pickle=False)
        meta = json.loads(str(blob["__meta__"]))
        cfg = PlacerV3Config.from_dict(meta["config"])
        net = PlacerNetV3(cfg)
        net.load_state_dict({k: torch.from_numpy(blob[k])
                             for k in blob.files if k != "__meta__"})
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PlacerV3Config.from_dict(
            payload.get("config") or PlacerV3Config().to_dict())
        net = PlacerNetV3(cfg)
        net.load_state_dict(payload["model"])
    net.eval()
    print(f"loaded {net.num_params() / 1e6:.2f}M params from "
          f"{os.path.basename(path)}")
    return net


def _write_html(report: Dict, path: str) -> None:
    rows = {r["brief"]: r for r in report["baseline"]["rows"]}
    body = []
    for dep in report["deployed"]["rows"]:
        base = rows.get(dep["brief"], {})
        delta = round(dep["score"] - base.get("score", 0), 2)
        colour = "#0a7" if delta > 0 else ("#c33" if delta < -5 else "#666")
        body.append(
            f"<tr><td>{dep['brief']}</td><td>{base.get('score', 0)}</td>"
            f"<td>{dep['score']}</td>"
            f"<td style='color:{colour};font-weight:600'>{delta:+.2f}</td>"
            f"<td>{dep['fidelity']}</td></tr>")
    v = report["verdict"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><meta charset='utf-8'><title>Placer v3 gate</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:40px;max-width:900px}"
            "table{border-collapse:collapse;width:100%}"
            "td,th{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}"
            "th{font-size:12px;text-transform:uppercase;color:#666}"
            ".v{padding:12px;border-radius:4px;font-weight:600;margin:16px 0}"
            "</style>"
            f"<h1>Placer v3 merge gate</h1><p>{report['timestamp']} · "
            f"tau {report['tau']}</p>"
            f"<div class='v' style='background:"
            f"{'#e3f1ea' if v['passed'] else '#f8e5e5'}'>"
            f"{'PASS — safe to merge' if v['passed'] else 'DID NOT PASS'} "
            f"({v['score_delta']:+.2f})</div>"
            "<table><tr><th>brief</th><th>baseline</th><th>v3</th>"
            "<th>delta</th><th>fidelity</th></tr>"
            + "".join(body) + "</table>")


if __name__ == "__main__":
    raise SystemExit(main())
