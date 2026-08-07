"""
plan_indexer.py
===============
Builds `data/plan_index/` — the retrieval index step 2 searches.

Why this file exists
--------------------
`semantic_matcher.py` and `feature_encoder.py` were both written against a
`plan_indexer.py` that is not in the repository. Their error messages tell you
to run `python -m modules.data_prep.plan_indexer`; that module did not exist.
The consequence was that step 2 retrieved 0 reference plans on every request
and the enricher silently fell back to NBC constants, so the 5,000-plan corpus
contributed nothing at all. This rebuilds the missing half.

What it produces
----------------
    data/plan_index/
      plan_vectors.npy         (N, 28) float32 — the searchable matrix
      plan_metadata.json       plan_keys + per-plan metadata + norm_params
      index_metadata.json      version, counts, provenance
      room_stats_by_bhk.json   room_type -> width/area percentiles (feet)
      adjacency_weights.json   room_a -> {room_b: weight}
      zone_probs.json          room_type -> {front, middle, back}
      circulation_meta.json    circulation benchmarks

Where the numbers come from
---------------------------
Nothing here is invented. Two sources, both already in the repo:

  data/zone_patterns_features.json   per-plan features for 4,983 plans
                                     (aspect ratio, spreads, doors/room,
                                     compartmentalization, bhk, zone balance)
  ml/data/normalized_extraction.json per-plan room composition, streamed with
                                     ijson (396 MB — never loaded whole)

plus the distilled aggregates in data/{learned,zone,circulation}_patterns.json,
which were computed from the same corpus.

The 28-dim layout MUST stay identical to feature_encoder.py::FeatureEncoder.
That file is the query side of this index; if the two disagree, every
similarity score is meaningless. The dimension table below is the contract.

Run:
    python -m modules.data_prep.plan_indexer
    python -m modules.data_prep.plan_indexer --limit 500   (quick check)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:                          # pragma: no cover
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.step2_match.feature_encoder import (  # noqa: E402
    BHK_TO_NUM,
    FEATURE_DIM,
    ROOM_FLAGS,
)

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "plan_index"
FEATURES_FILE = DATA_DIR / "zone_patterns_features.json"
LEARNED_FILE = DATA_DIR / "learned_patterns.json"
ZONE_FILE = DATA_DIR / "zone_patterns.json"
CIRC_FILE = DATA_DIR / "circulation_patterns.json"

INDEX_VERSION = "1.0.0"
SQFT_PER_M2 = 10.763910417

log = logging.getLogger("PlanGen.Indexer")

# ── corpus room type → encoder room flag ─────────────────────────────────────
# Corpus vocabulary confirmed by sampling the extraction directly. Types not
# listed here (outdoor, undefined, sauna, special) map to no flag on purpose —
# they are not rooms a user ever asks for.
CORPUS_TYPE_TO_FLAG: Dict[str, str] = {
    "kitchen": "kitchen",
    "living_room": "living_room",
    "dining": "dining",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "balcony": "balcony",
    "utility": "utility",
    "storage": "storage",
    "office": "study",        # feature_encoder ROOM_ALIAS maps office -> study
    "garage": "garage",
    "stairs": "staircase",
    "corridor": "entry",      # circulation space; encoder folds passage -> entry
    "entry": "entry",
}


# ══════════════════════════════════════════════════════════════════════════
# Room composition (streamed from the corpus)
# ══════════════════════════════════════════════════════════════════════════

def collect_room_composition(limit: int = 0) -> Dict[str, Dict[str, Any]]:
    """
    Stream the normalized extraction and return, per plan:
        {plan_key: {"flags": {flag: bool}, "n_bedrooms": int,
                    "n_bathrooms": int, "n_rooms": int}}

    Streamed with ijson so the 396 MB file is never held in memory.
    """
    try:
        import ijson  # noqa: F401
        from ml.training.paths import CUBICASA_NORMALIZED
        from ml.training.schema_audit import stream_plans
    except ImportError as exc:
        log.error("Cannot stream the corpus (%s). Is ijson installed and "
                  "ml/data/normalized_extraction.json present?", exc)
        return {}

    if not Path(CUBICASA_NORMALIZED).exists():
        log.error("Corpus not found: %s", CUBICASA_NORMALIZED)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    t0 = time.perf_counter()
    for i, (plan_id, rec) in enumerate(stream_plans(CUBICASA_NORMALIZED, limit=limit)):
        flags = {f: False for f in ROOM_FLAGS}
        n_bed = n_bath = 0
        spaces = rec.get("spaces") or []
        for sp in spaces:
            rtype = str(sp.get("type") or "").strip().lower()
            flag = CORPUS_TYPE_TO_FLAG.get(rtype)
            if flag:
                flags[flag] = True
            if rtype == "bedroom":
                n_bed += 1
            elif rtype == "bathroom":
                n_bath += 1
        out[plan_id] = {
            "flags": flags,
            "n_bedrooms": n_bed,
            "n_bathrooms": n_bath,
            "n_rooms": len(spaces),
        }
        if (i + 1) % 1000 == 0:
            log.info("  streamed %d plans (%.1fs)", i + 1, time.perf_counter() - t0)

    log.info("Room composition for %d plans in %.1fs",
             len(out), time.perf_counter() - t0)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Feature vectors
# ══════════════════════════════════════════════════════════════════════════

def build_raw_vector(feat: Dict[str, Any], comp: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    Build ONE un-normalised 28-dim vector.

    Dimension contract — mirrors feature_encoder.FeatureEncoder.encode():
        0   plan_aspect_ratio scaled (ar - 0.2) / 4.8, clipped
        1   room_count / 20
        2   bhk numeric
        3   depth_spread
        4   lateral_spread
        5   doors_per_room / 4
        6   compartmentalization
        7-9 zone balance front / middle / back
        10-21 room presence flags (ROOM_FLAGS order)
        22  multi_floor — always 0.0, the corpus is single-floor and the
            encoder suppresses this dimension for the same reason
        23  bedroom_count_ratio
        24  bathroom_count_ratio
        25-27 reserved (0.0)
    """
    v = np.zeros(FEATURE_DIM, dtype=np.float32)

    ar = float(feat.get("plan_aspect_ratio") or 1.3)
    # The encoder folds aspect ratio to >= 1 implicitly (max/min); the corpus
    # stores raw width/height ratios which can be < 1. Fold here so both sides
    # of the index describe "how elongated", not "which way round".
    if ar < 1.0 and ar > 1e-6:
        ar = 1.0 / ar
    v[0] = float(np.clip((ar - 0.2) / (5.0 - 0.2), 0.0, 1.0))

    room_count = float(feat.get("room_count") or 0)
    v[1] = float(np.clip(room_count / 20.0, 0.0, 1.0))

    v[2] = float(BHK_TO_NUM.get(str(feat.get("bhk", "")).lower(), 0.5))
    v[3] = float(np.clip(feat.get("depth_spread") or 0.0, 0.0, 1.0))
    v[4] = float(np.clip(feat.get("lateral_spread") or 0.0, 0.0, 1.0))
    v[5] = float(np.clip((feat.get("doors_per_room") or 0.0) / 4.0, 0.0, 1.0))
    v[6] = float(np.clip(feat.get("compartmentalization") or 0.0, 0.0, 1.0))

    zb = feat.get("zone_balance") or {}
    v[7] = float(np.clip(zb.get("front", 0.33), 0.0, 1.0))
    v[8] = float(np.clip(zb.get("middle", 0.34), 0.0, 1.0))
    v[9] = float(np.clip(zb.get("back", 0.33), 0.0, 1.0))

    if comp:
        flags = comp["flags"]
        for i, flag in enumerate(ROOM_FLAGS):
            v[10 + i] = 1.0 if flags.get(flag) else 0.0
        denom = max(comp["n_rooms"], 1)
        v[23] = float(np.clip(comp["n_bedrooms"] / denom, 0.0, 1.0))
        v[24] = float(np.clip(comp["n_bathrooms"] / denom, 0.0, 1.0))

    v[22] = 0.0     # multi_floor — suppressed on both sides. See docstring.
    return v


def build_vectors(
    features: Dict[str, Dict[str, Any]],
    composition: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]], Dict[str, List[float]]]:
    """
    Assemble the full matrix, its metadata, and the normalisation parameters.

    Returns (vectors_normalised, plan_keys, metadata, norm_params).
    """
    plan_keys = sorted(features.keys())
    raw = np.zeros((len(plan_keys), FEATURE_DIM), dtype=np.float32)
    metadata: List[Dict[str, Any]] = []
    matched = 0

    for i, key in enumerate(plan_keys):
        feat = features[key]
        comp = composition.get(key)
        if comp:
            matched += 1
        raw[i] = build_raw_vector(feat, comp)

        ar = float(feat.get("plan_aspect_ratio") or 1.0)
        metadata.append({
            "bhk": str(feat.get("bhk", "unknown")),
            "room_count": int(feat.get("room_count") or 0),
            "aspect_ratio": round(ar, 4),
            "zone_balance": feat.get("zone_balance") or {},
        })

    log.info("Vectors: %d plans, %d joined with room composition (%.1f%%)",
             len(plan_keys), matched,
             100.0 * matched / max(len(plan_keys), 1))
    if matched == 0:
        log.warning("NO plans joined with the corpus — room presence flags "
                    "(dims 10-21) and bedroom/bathroom ratios are all zero. "
                    "The index will still work but retrieval quality is "
                    "materially reduced.")

    # Normalisation must be computed on the index and then applied to BOTH
    # sides. FeatureEncoder re-applies these exact params to every query, so
    # they are part of the index contract, not an implementation detail.
    col_min = raw.min(axis=0)
    col_max = raw.max(axis=0)
    col_range = np.where(col_max - col_min < 1e-8, 1.0, col_max - col_min)
    normed = np.clip((raw - col_min) / col_range, 0.0, 1.0).astype(np.float32)

    norm_params = {
        "col_min": [round(float(x), 6) for x in col_min],
        "col_max": [round(float(x), 6) for x in col_max],
    }
    return normed, plan_keys, metadata, norm_params


# ══════════════════════════════════════════════════════════════════════════
# Aggregate statistics
# ══════════════════════════════════════════════════════════════════════════

def build_room_stats(learned: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    room_type -> width/area percentiles in FEET, the shape StatsAggregator
    reads (min/p25/median/p75/max for both width and area, plus sample_count).

    The corpus stores areas in m2 and aspect ratios separately, so widths are
    derived rather than measured:

        width = sqrt(area / aspect_ratio)

    i.e. the SHORT side of a rectangle with that area and that proportion —
    which is what "room width" means in the NBC minimums this is compared
    against. Using the median aspect ratio per room type keeps the derivation
    stable against the long tail (some 'undefined' rooms have aspect > 9).
    """
    areas = learned.get("room_areas_m2", {})
    aspects = learned.get("room_aspect_ratios", {})
    out: Dict[str, Dict[str, float]] = {}

    for rtype, a in areas.items():
        if rtype in ("outdoor", "undefined"):
            continue    # not rooms anyone requests

        asp = aspects.get(rtype, {})
        ar = float(asp.get("median") or 1.4)
        ar = max(ar, 1.0)      # fold; aspect < 1 is the same shape rotated

        def ft2(key: str, default_m2: float) -> float:
            return float(a.get(key, default_m2)) * SQFT_PER_M2

        min_a = ft2("min_m2", 4.0)
        p25_a = ft2("p25_m2", 7.0)
        med_a = ft2("median_m2", 9.0)
        p75_a = ft2("p75_m2", 12.0)
        max_a = ft2("max_m2", 25.0)

        def width(area_sqft: float) -> float:
            return math.sqrt(max(area_sqft, 1.0) / ar)

        out[rtype] = {
            "min_width_ft":     round(width(min_a), 2),
            "p25_width_ft":     round(width(p25_a), 2),
            "median_width_ft":  round(width(med_a), 2),
            "p75_width_ft":     round(width(p75_a), 2),
            "max_width_ft":     round(width(max_a), 2),
            "min_area_sqft":    round(min_a, 2),
            "p25_area_sqft":    round(p25_a, 2),
            "median_area_sqft": round(med_a, 2),
            "p75_area_sqft":    round(p75_a, 2),
            "max_area_sqft":    round(max_a, 2),
            "aspect_ratio_median": round(ar, 3),
            "sample_count":     int(a.get("samples", 0)),
        }
    return out


def build_adjacency(learned: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    "room_a|room_b" -> {weight: w}  becomes  {room_a: {room_b: w}}, symmetric.

    StatsAggregator looks up `self._adj_weights[a][b]`, so both directions
    must be present.
    """
    raw = learned.get("adjacency_weights", {})
    out: Dict[str, Dict[str, float]] = {}
    for pair, info in raw.items():
        if "|" not in pair:
            continue
        a, b = (p.strip() for p in pair.split("|", 1))
        if not a or not b or a == b:
            continue
        w = float(info.get("weight", 0.0)) if isinstance(info, dict) else float(info)
        out.setdefault(a, {})[b] = round(w, 4)
        out.setdefault(b, {})[a] = round(w, 4)
    return out


def build_zone_probs(zone: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """room_type -> {front, middle, back}, renormalised to sum to 1."""
    dists = zone.get("zone_distributions", {})
    out: Dict[str, Dict[str, float]] = {}
    for rtype, info in dists.items():
        zd = (info or {}).get("zone_distribution") or {}
        front = float(zd.get("front", 0.0))
        middle = float(zd.get("middle", 0.0))
        back = float(zd.get("back", 0.0))
        total = front + middle + back
        if total <= 1e-6:
            continue
        out[rtype] = {
            "front": round(front / total, 4),
            "middle": round(middle / total, 4),
            "back": round(back / total, 4),
        }
    return out


def build_circulation_meta(circ: Dict[str, Any]) -> Dict[str, Any]:
    """Pass through the benchmark sections the enricher actually consumes."""
    keys = ("depth_distributions", "journey_efficiency", "privacy_violation_rates",
            "hub_type_distribution", "score_statistics", "guest_bathroom_access",
            "total_plans_analyzed")
    return {k: circ[k] for k in keys if k in circ}


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════

def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        log.warning("Missing input: %s", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_index(limit: int = 0, skip_corpus: bool = False) -> Dict[str, Any]:
    """Build every index artefact and write it to data/plan_index/."""
    t0 = time.perf_counter()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    features = _load(FEATURES_FILE)
    if not features:
        raise SystemExit(
            f"Cannot build the index: {FEATURES_FILE} is required and absent. "
            "It holds the per-plan features for all indexed plans."
        )
    if limit:
        features = dict(list(features.items())[:limit])

    composition = {} if skip_corpus else collect_room_composition(limit=limit)

    vectors, plan_keys, metadata, norm_params = build_vectors(features, composition)

    np.save(INDEX_DIR / "plan_vectors.npy", vectors)
    (INDEX_DIR / "plan_metadata.json").write_text(json.dumps({
        "plan_keys": plan_keys,
        "metadata": metadata,
        "norm_params": norm_params,
    }), encoding="utf-8")

    learned = _load(LEARNED_FILE)
    zone = _load(ZONE_FILE)
    circ = _load(CIRC_FILE)

    room_stats = build_room_stats(learned)
    adjacency = build_adjacency(learned)
    zone_probs = build_zone_probs(zone)
    circ_meta = build_circulation_meta(circ)

    for name, payload in (
        ("room_stats_by_bhk.json", room_stats),
        ("adjacency_weights.json", adjacency),
        ("zone_probs.json", zone_probs),
        ("circulation_meta.json", circ_meta),
    ):
        (INDEX_DIR / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    joined = sum(1 for k in plan_keys if k in composition)
    index_meta = {
        "version": INDEX_VERSION,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_plans": len(plan_keys),
        "feature_dim": FEATURE_DIM,
        "plans_with_room_composition": joined,
        "sources": {
            "features": str(FEATURES_FILE.relative_to(PROJECT_ROOT)),
            "corpus": "ml/data/normalized_extraction.json",
            "aggregates": ["learned_patterns.json", "zone_patterns.json",
                           "circulation_patterns.json"],
        },
        "counts": {
            "room_stats": len(room_stats),
            "adjacency_rooms": len(adjacency),
            "zone_probs": len(zone_probs),
        },
        "notes": (
            "Dim 22 (multi_floor) is fixed at 0.0 on both index and query "
            "sides: CubiCasa5K is single-floor, so a non-zero query value "
            "would penalise the entire index."
        ),
    }
    (INDEX_DIR / "index_metadata.json").write_text(
        json.dumps(index_meta, indent=1), encoding="utf-8")

    log.info("Index built in %.1fs -> %s", time.perf_counter() - t0, INDEX_DIR)
    return index_meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data/plan_index/")
    ap.add_argument("--limit", type=int, default=0,
                    help="index only the first N plans (smoke test)")
    ap.add_argument("--skip-corpus", action="store_true",
                    help="skip streaming the 396MB corpus; room presence "
                         "flags will be zero (degrades retrieval quality)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    meta = build_index(limit=args.limit, skip_corpus=args.skip_corpus)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
