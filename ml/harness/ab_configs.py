"""
ab_configs.py — run the golden briefs under several EngineConfig variants
and print the difference.

Every config knob is supposed to earn its default. This is the tool that
makes it prove it: same briefs, same seeds, same everything except the named
config fields, with a per-brief delta table so a change that lifts the mean
while quietly wrecking one plot cannot hide behind an average.

    python -m harness.ab_configs --vary cpsat_mode=off,repair,always
    python -m harness.ab_configs --vary settle_sweeps=30,60 --k 4

Also usable as a library: `compare({"name": config, ...})`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List

from modules.step4_generate.core.console import force_utf8_console
from modules.step4_generate.engine.contracts import EngineConfig
from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_config(config: EngineConfig, briefs) -> Dict:
    orch = Orchestrator(config=config)
    rows: List[Dict] = []
    t0 = time.perf_counter()
    for request in briefs:
        t = time.perf_counter()
        result = orch.generate(request)
        best = result.best
        rows.append({
            "brief": request.name,
            "kept": len(result.ranked),
            "k": request.k,
            "score": round(best.verdict.soft_score, 2) if best else 0.0,
            "fidelity": round(best.fidelity, 3) if best else 0.0,
            "nbc": (best.verdict.breakdown.get("nbc_violations", 0)
                    if best else 0),
            "seconds": round(time.perf_counter() - t, 2),
        })
    return {
        "rows": rows,
        "mean_score": round(sum(r["score"] for r in rows) / len(rows), 2),
        "planned": sum(1 for r in rows if r["kept"] > 0),
        "total_kept": sum(r["kept"] for r in rows),
        "nbc_violations": sum(r["nbc"] for r in rows),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def compare(configs: Dict[str, EngineConfig], k: int = 6) -> Dict:
    briefs = golden_briefs(k=k)
    return {name: run_config(cfg, briefs) for name, cfg in configs.items()}


def print_comparison(results: Dict[str, Dict]) -> None:
    names = list(results)
    base = names[0]
    width = max(12, max(len(n) for n in names) + 2)
    print("=" * (24 + width * len(names)))
    print("ENGINE CONFIG A/B  —  golden briefs, identical seeds")
    print("=" * (24 + width * len(names)))
    print(f"  {'brief':<20}" + "".join(f"{n:>{width}}" for n in names))
    for i, row in enumerate(results[base]["rows"]):
        line = f"  {row['brief']:<20}"
        for n in names:
            r = results[n]["rows"][i]
            cell = f"{r['score']:.1f}" if r["kept"] else "NO PLAN"
            if n != base:
                delta = r["score"] - results[base]["rows"][i]["score"]
                cell += f" ({delta:+.1f})" if abs(delta) >= 0.01 else "  (=)"
            line += f"{cell:>{width}}"
        print(line)
    print("-" * (24 + width * len(names)))
    for label, key, fmt in (("mean score", "mean_score", "{:.2f}"),
                            ("briefs planned", "planned", "{}"),
                            ("candidates kept", "total_kept", "{}"),
                            ("NBC violations", "nbc_violations", "{}"),
                            ("wall time (s)", "seconds", "{:.1f}")):
        line = f"  {label:<20}"
        for n in names:
            line += f"{fmt.format(results[n][key]):>{width}}"
        print(line)


def _apply(field: str, raw: str) -> EngineConfig:
    """EngineConfig with one field overridden, parsed to the field's type."""
    cfg = EngineConfig()
    if not hasattr(cfg, field):
        raise SystemExit(f"EngineConfig has no field {field!r}")
    current = getattr(cfg, field)
    if isinstance(current, bool):
        value = raw.lower() in ("1", "true", "yes", "on")
    elif isinstance(current, int):
        value = int(raw)
    elif isinstance(current, float):
        value = float(raw)
    else:
        value = raw
    setattr(cfg, field, value)
    return cfg


def main(argv=None) -> int:
    force_utf8_console()
    p = argparse.ArgumentParser(prog="harness.ab_configs")
    p.add_argument("--vary", required=True,
                   help="FIELD=value1,value2[,...] — one EngineConfig field")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--json", default=os.path.join(RESULTS_DIR, "ab.json"))
    p.add_argument("--quiet-engine", action="store_true", default=True)
    args = p.parse_args(argv)

    if "=" not in args.vary:
        raise SystemExit("--vary must look like FIELD=v1,v2")
    field, values = args.vary.split("=", 1)
    configs = {f"{field}={v}": _apply(field, v) for v in values.split(",")}

    if args.quiet_engine:
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)

    results = compare(configs, k=args.k)
    print_comparison(results)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"\n  -> {os.path.abspath(args.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
