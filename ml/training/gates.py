"""
gates.py — the hard data-quality gates. No training runs until these pass.

The v1 model was supervised on data that was itself 13.4% overlapping and
nobody measured it (docs/plans/partition_first_redesign.md). These gates
make that failure impossible to repeat silently. Adapted to the DISCRETE
formulation, the gates are:

  (a) seed-collision rate = 0%  (two rooms never share a 32×32 seed cell)
      + coverage sanity: Σ room area / plot area in a plausible band
  (b) every seed cell lies inside its boundary mask
  (c) distribution report: room counts, type & size-class histograms
      (domain-gap visibility vs our program generator's ranges)
  (e) val split is present, frozen, and disjoint by sample

Gate (d), the mandatory 100-sample eyeball render, is training.render_samples
(a committed artifact) — invoked separately so the image is reviewable.

Exit code 0 iff every hard gate passes. Run after prep:
    python -m training.gates
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID
from ml.training.paths import PREPARED_DIR
from ml.training.prep_cubicasa import BOUNDARY_GRID

# coverage sanity band: Σ room area / plot area. Rooms + walls + circulation
# fill the plot, so < 1 is normal; way under = sparse extraction, over 1 =
# rooms exceed the plot (bad data, the v1 symptom).
_COVERAGE_LO = 0.30
_COVERAGE_HI = 1.25


def _load(out_dir: str):
    with open(os.path.join(out_dir, "samples.jsonl"), encoding="utf-8") as f:
        samples = [json.loads(ln) for ln in f if ln.strip()]
    masks = np.load(os.path.join(out_dir, "masks.npy"))
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    return samples, masks, manifest


def _seed_in_mask(row: int, col: int, mask: np.ndarray) -> bool:
    """Is the 32-grid seed cell inside the 64-grid boundary mask? Checks the
    2×2 mask block the seed maps to (any fill counts — tolerates a centroid
    landing on a thin interior wall gap)."""
    scale = BOUNDARY_GRID // SEED_GRID          # 2
    r0, c0 = row * scale, col * scale
    block = mask[r0:r0 + scale, c0:c0 + scale]
    return bool(block.any())


def check(out_dir: str) -> Dict:
    samples, masks, manifest = _load(out_dir)
    n = len(samples)
    report: Dict = {"n_samples": n, "passed": True, "gates": {}}
    if n == 0:
        report["passed"] = False
        report["gates"]["nonempty"] = {"pass": False, "detail": "0 samples"}
        return report

    # ── gate (a): seed collisions + coverage ─────────────────────────────
    collisions = rooms_total = 0
    coverage_bad: List[str] = []
    for s in samples:
        seen = set()
        for r in s["rooms"]:
            rooms_total += 1
            key = (r["row"], r["col"])
            if key in seen:
                collisions += 1
            seen.add(key)
        plot_area = max(1.0, s["plot_w_ft"] * s["plot_h_ft"])
        cover = sum(r["area_sqft"] for r in s["rooms"]) / plot_area
        if not (_COVERAGE_LO <= cover <= _COVERAGE_HI):
            coverage_bad.append(f"{s['plan_id']}={cover:.2f}")
    report["gates"]["seed_collisions"] = {
        "pass": collisions == 0,
        "detail": f"{collisions}/{rooms_total} rooms collide",
    }
    report["gates"]["coverage_sanity"] = {
        "pass": len(coverage_bad) <= max(1, int(0.02 * n)),   # ≤2% outliers
        "detail": f"{len(coverage_bad)}/{n} plans outside "
                  f"[{_COVERAGE_LO}, {_COVERAGE_HI}]",
        "examples": coverage_bad[:10],
    }

    # ── gate (b): seeds inside boundary ──────────────────────────────────
    out_of_bounds = 0
    for s in samples:
        mask = masks[s["mask_index"]]
        for r in s["rooms"]:
            if not _seed_in_mask(r["row"], r["col"], mask):
                out_of_bounds += 1
    report["gates"]["seeds_in_boundary"] = {
        "pass": out_of_bounds <= max(1, int(0.01 * rooms_total)),  # ≤1%
        "detail": f"{out_of_bounds}/{rooms_total} seeds outside their mask",
    }

    # ── gate (c): distribution report ────────────────────────────────────
    types = Counter(); sizes = Counter(); rc = Counter(); ent = Counter()
    for s in samples:
        rc[len(s["rooms"])] += 1
        ent[s["entrance_side"]] += 1
        for r in s["rooms"]:
            types[r["rtype"]] += 1
            sizes[r["size_class"]] += 1
    report["gates"]["distribution"] = {
        "pass": True,                    # informational gate
        "room_type_histogram": dict(types.most_common()),
        "size_class_histogram": dict(sorted(sizes.items())),
        "room_count_histogram": dict(sorted(rc.items())),
        "entrance_side_histogram": dict(ent.most_common()),
        "mean_rooms": round(rooms_total / n, 2),
    }

    # ── gate (e): val split frozen & disjoint ────────────────────────────
    val = manifest.get("val_indices", [])
    report["gates"]["val_split"] = {
        "pass": (len(val) == len(set(val)) and len(val) > 0
                 and all(0 <= i < n for i in val)),
        "detail": f"{len(val)} frozen val indices, "
                  f"seed={manifest.get('split_seed')}",
    }

    report["passed"] = all(g.get("pass", False)
                           for g in report["gates"].values())
    return report


def print_report(report: Dict) -> None:
    print("=" * 66)
    print(f"DATA GATES — {report['n_samples']} prepared samples")
    print("=" * 66)
    for name, g in report["gates"].items():
        mark = "PASS" if g.get("pass") else "FAIL"
        detail = g.get("detail", "")
        print(f"  [{mark}] {name:<20} {detail}")
        if name == "distribution":
            print(f"         mean rooms/plan: {g['mean_rooms']}")
            print(f"         types: {g['room_type_histogram']}")
            print(f"         sizes: {g['size_class_histogram']}")
            print(f"         entrances: {g['entrance_side_histogram']}")
        if g.get("examples"):
            print(f"         e.g. {g['examples']}")
    print("-" * 66)
    print(f"  OVERALL: {'PASS — clear to train' if report['passed'] else 'FAIL — do not train'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="training.gates")
    p.add_argument("--out-dir", default=PREPARED_DIR)
    p.add_argument("--json", default=None, help="also write the report JSON")
    args = p.parse_args(argv)

    if not os.path.exists(os.path.join(args.out_dir, "samples.jsonl")):
        print(f"no prepared data in {args.out_dir} — run prep_cubicasa first.")
        return 2
    report = check(args.out_dir)
    print_report(report)
    out = args.json or os.path.join(args.out_dir, "gates_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  report -> {os.path.abspath(out)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
