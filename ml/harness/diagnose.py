"""
diagnose.py — why does a brief score the way it does?

The A/B harness reports one number per brief. When that number is bad, the
number alone tells you nothing about what to fix. This runs the golden briefs
and prints, per brief: the winner's hard violations (there should be none),
what the DISCARDED candidates failed on, and the winner's soft-score
breakdown ranked by contribution.

A note on the field named `hard_fails` in the merge-gate JSON: it is
`len(result.discarded)` — the number of candidate plans the validator threw
out, NOT violations in the plan that ships. A brief can report `hard_fails: 5`
and still ship a plan with a clean bill of health. Discarding is the reviewer
working, not failing. This tool prints both so the two can never be confused
again.

Run:
    python -m ml.harness.diagnose
    python -m ml.harness.diagnose --brief 25x50_E_3bhk --brief 25x40_S_2bhk
    python -m ml.harness.diagnose --worst 5
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from typing import Dict, List

from ml.harness.briefs import golden_briefs
from modules.step4_generate.engine.orchestrator import Orchestrator


def _aspect(req) -> float:
    lo = min(req.plot_w_ft, req.plot_h_ft)
    return max(req.plot_w_ft, req.plot_h_ft) / lo if lo else 1.0


def _numeric(breakdown) -> Dict[str, float]:
    """Keep only the numeric evidence — some rules record strings."""
    out: Dict[str, float] = {}
    for k, v in (breakdown or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if abs(v) > 1e-9:
            out[k] = float(v)
    return out


def diagnose(names: List[str] | None = None, worst: int = 0) -> List[Dict]:
    orch = Orchestrator()
    rows: List[Dict] = []

    for req in golden_briefs():
        if names and req.name not in names:
            continue
        result = orch.generate(req)
        best = result.best

        discarded_reasons: Counter = Counter()
        for cand in result.discarded:
            verdict = getattr(cand, "verdict", None)
            for violation in (verdict.hard if verdict else []):
                # Violations read like "STR-003: parking has no vehicle gate"
                discarded_reasons[str(violation).split(":")[0].strip()] += 1

        rows.append({
            "brief": req.name,
            "score": round(best.verdict.soft_score, 2) if best else 0.0,
            "aspect": round(_aspect(req), 2),
            "rooms": len(req.rooms),
            "area_sqft": round(req.plot_w_ft * req.plot_h_ft),
            "kept": len(result.ranked),
            "discarded": len(result.discarded),
            "winner_hard": list(best.verdict.hard) if best else ["NO PLAN"],
            "discarded_reasons": dict(discarded_reasons.most_common()),
            "breakdown": _numeric(best.verdict.breakdown) if best else {},
        })

    rows.sort(key=lambda r: r["score"])
    if worst:
        rows = rows[:worst]
    return rows


def report(rows: List[Dict]) -> str:
    out: List[str] = []
    clean = sum(1 for r in rows if not r["winner_hard"])
    out.append("%d brief(s); %d shipped a plan with ZERO hard violations."
               % (len(rows), clean))
    out.append("")
    for r in rows:
        density = r["area_sqft"] / max(r["rooms"], 1)
        out.append("== %-15s score=%7.2f  aspect=%.2f  %d rooms on %d sqft "
                   "(%.0f sqft/room)" % (r["brief"], r["score"], r["aspect"],
                                         r["rooms"], r["area_sqft"], density))
        out.append("   winner hard violations : %s"
                   % (r["winner_hard"] or "none"))
        out.append("   candidates kept/discarded: %d/%d   discarded for: %s"
                   % (r["kept"], r["discarded"], r["discarded_reasons"] or "-"))
        top = sorted(r["breakdown"].items(), key=lambda kv: -abs(kv[1]))[:8]
        if top:
            out.append("   score drivers:")
            for k, v in top:
                out.append("       %-28s %+8.2f" % (k, v))
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain golden-brief scores")
    ap.add_argument("--brief", action="append", default=None,
                    help="brief name (repeatable); default = all")
    ap.add_argument("--worst", type=int, default=0,
                    help="show only the N lowest-scoring briefs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    print(report(diagnose(names=args.brief, worst=args.worst)))


if __name__ == "__main__":
    main()
