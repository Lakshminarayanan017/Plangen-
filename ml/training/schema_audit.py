"""
schema_audit.py — inspect the REAL corpus schema before writing any parser.

The v1 disaster was invisible because nobody looked at the data
(docs/plans/partition_first_redesign.md §1). This script looks: it streams
the whole corpus once (memory-safe via ijson) and reports exactly what is
in it — field inventory of one sample, then distributions across every plan:
room-type histogram, undefined fraction, room-count spread, BHK mix,
per-room real-world scale availability, and coordinate ranges. This is both
the "never trust the docs' schema" gate and gate (c) of §3 (the
distribution report / domain-gap visibility).

Run:  python -m training.schema_audit
      python -m training.schema_audit --limit 500   (quick sample)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, Iterator, Tuple

import ijson

from ml.training.paths import AUDIT_DIR, CUBICASA_NORMALIZED, ensure_dir


def stream_plans(path: str, limit: int = 0) -> Iterator[Tuple[str, dict]]:
    """Yield (plan_id, plan_record) from the normalized extraction without
    loading the file. `plans` is a JSON object keyed by plan id."""
    with open(path, "rb") as f:
        n = 0
        # use_float=True → plain floats, not Decimal (JSON-serializable and
        # correct for the downstream geometry math)
        for plan_id, record in ijson.kvitems(f, "plans", use_float=True):
            yield plan_id, record
            n += 1
            if limit and n >= limit:
                return


def _sample_fields(record: dict) -> Dict:
    """Full field inventory of one plan record (shapes, not values)."""
    out = {}
    for k, v in record.items():
        if isinstance(v, list):
            child = (sorted(v[0].keys()) if v and isinstance(v[0], dict)
                     else [type(v[0]).__name__] if v else [])
            out[k] = {"list_len": len(v), "item_fields": child}
        elif isinstance(v, dict):
            out[k] = {"dict_keys": sorted(v.keys())}
        else:
            out[k] = {"type": type(v).__name__, "value": v}
    return out


def audit(path: str, limit: int = 0) -> Dict:
    types = Counter()
    original_types = Counter()
    bhk = Counter()
    room_counts = Counter()
    undefined_plans = 0          # plans that are >=50% undefined rooms
    scale_available = 0
    n_plans = 0
    n_rooms = 0
    coord_max_x = coord_max_y = 0.0
    sample_fields = None
    sample_space = None

    for plan_id, rec in stream_plans(path, limit):
        n_plans += 1
        if sample_fields is None:
            sample_fields = _sample_fields(rec)
            if rec.get("spaces"):
                sample_space = rec["spaces"][0]

        spaces = rec.get("spaces", [])
        room_counts[len(spaces)] += 1
        n_undef = 0
        for s in spaces:
            n_rooms += 1
            t = s.get("type", "MISSING")
            types[t] += 1
            original_types[s.get("original_type", "MISSING")] += 1
            if t == "undefined":
                n_undef += 1
            dims = s.get("dimensions") or {}
            if dims.get("width_ft"):
                scale_available += 1
            bb = s.get("bbox") or {}
            coord_max_x = max(coord_max_x, bb.get("x", 0) + bb.get("w", 0))
            coord_max_y = max(coord_max_y, bb.get("y", 0) + bb.get("h", 0))
        if spaces and n_undef / len(spaces) >= 0.5:
            undefined_plans += 1
        meta = rec.get("metadata") or {}
        bhk[meta.get("bhk_category", "MISSING")] += 1

    return {
        "corpus": os.path.basename(path),
        "n_plans": n_plans,
        "n_rooms": n_rooms,
        "sample_plan_fields": sample_fields,
        "sample_space_record": sample_space,
        "room_type_histogram": dict(types.most_common()),
        "original_type_histogram": dict(original_types.most_common(25)),
        "undefined_room_fraction": round(
            types.get("undefined", 0) / max(1, n_rooms), 4),
        "plans_majority_undefined": undefined_plans,
        "plans_majority_undefined_pct": round(
            100 * undefined_plans / max(1, n_plans), 1),
        "per_room_scale_available_pct": round(
            100 * scale_available / max(1, n_rooms), 1),
        "bhk_histogram": dict(bhk.most_common()),
        "room_count_histogram": dict(sorted(room_counts.items())),
        "coord_max_xy": [round(coord_max_x, 1), round(coord_max_y, 1)],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="training.schema_audit")
    p.add_argument("--path", default=CUBICASA_NORMALIZED)
    p.add_argument("--limit", type=int, default=0,
                   help="audit only the first N plans (0 = all)")
    p.add_argument("--out", default=None,
                   help="write the report JSON here (default: audit/)")
    args = p.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"corpus not found: {args.path}")
        return 2

    print(f"auditing {args.path} "
          f"({'all' if not args.limit else args.limit} plans)…")
    report = audit(args.path, args.limit)

    out = args.out or os.path.join(
        ensure_dir(AUDIT_DIR), "cubicasa_schema_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # human-readable summary
    print("=" * 70)
    print(f"  plans: {report['n_plans']}   rooms: {report['n_rooms']}")
    print(f"  undefined rooms: {report['undefined_room_fraction']:.1%} of all "
          f"rooms; {report['plans_majority_undefined_pct']}% of plans are "
          f">=50% undefined")
    print(f"  per-room ft scale available: "
          f"{report['per_room_scale_available_pct']}%")
    print(f"  coord max (x,y): {report['coord_max_xy']}")
    print("  room-type histogram:")
    for t, c in report["room_type_histogram"].items():
        print(f"    {t:<16} {c}")
    print("  BHK mix:", report["bhk_histogram"])
    print("-" * 70)
    print(f"  full report -> {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
