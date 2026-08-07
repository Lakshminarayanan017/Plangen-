"""
preferences.py — log which candidate the user actually chose.

"User picks among K logged from day one" (implementation_plan_v2.md §2.5,
§4.3). Perturbation labels teach the critic to recognize damage; only real
preferences teach it taste, and preference data cannot be collected
retroactively. So the log starts now, before there is anything to train on.

The format is append-only JSONL: one line per pick, holding the feature
vectors of every candidate that was on screen plus the index of the chosen
one. Feature vectors, not plan blobs, because the vectors are what a future
pairwise model consumes and they stay valid without re-running the engine.
Vectors are stamped with the feature-set version they were captured under,
so a later FEATURE_NAMES change cannot silently mix incompatible rows.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from modules.step4_generate.critic import features as feat
from modules.step4_generate.engine.contracts import Candidate

DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "preferences.jsonl")

# Bumped whenever FEATURE_NAMES changes. Rows tagged with an older version
# are readable but excluded from training pairs.
FEATURE_SET_VERSION = 1


@dataclass
class PreferenceRecord:
    brief: str
    chosen: int
    vectors: List[List[float]]
    soft_scores: List[float]
    timestamp: str
    version: int = FEATURE_SET_VERSION
    note: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "brief": self.brief, "chosen": self.chosen,
            "vectors": [[round(v, 6) for v in row] for row in self.vectors],
            "soft_scores": [round(s, 3) for s in self.soft_scores],
            "timestamp": self.timestamp, "version": self.version,
            "note": self.note})

    @classmethod
    def from_json(cls, line: str) -> "PreferenceRecord":
        d = json.loads(line)
        return cls(brief=d["brief"], chosen=int(d["chosen"]),
                   vectors=d["vectors"], soft_scores=d.get("soft_scores", []),
                   timestamp=d.get("timestamp", ""),
                   version=int(d.get("version", 0)), note=d.get("note", ""))


def log_choice(candidates: Sequence[Candidate], chosen_index: int, *,
               brief: str = "", note: str = "",
               path: str = DEFAULT_LOG,
               config=None) -> Optional[str]:
    """Append one pick. Returns the path written, or None if the pick could
    not be featurized — logging must never break the request that produced
    it, so every failure here is swallowed after being recorded as None."""
    if not (0 <= chosen_index < len(candidates)):
        raise IndexError(f"chosen_index {chosen_index} outside "
                         f"0..{len(candidates) - 1}")
    try:
        vectors = [feat.extract(c.plan, c.request, c.room_ids, c.verdict,
                                config).tolist() for c in candidates]
    except Exception:
        return None
    record = PreferenceRecord(
        brief=brief, chosen=chosen_index, vectors=vectors,
        soft_scores=[c.verdict.soft_score if c.verdict else 0.0
                     for c in candidates],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), note=note)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")
    return path


def read_log(path: str = DEFAULT_LOG) -> List[PreferenceRecord]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(PreferenceRecord.from_json(line))
    return out


def pairwise_samples(records: Sequence[PreferenceRecord]
                     ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """(chosen, rejected) feature-vector pairs from compatible records.

    This is the training signal for preference fine-tuning: every pick over
    K candidates yields K-1 ordered pairs."""
    for rec in records:
        if rec.version != FEATURE_SET_VERSION:
            continue
        if not (0 <= rec.chosen < len(rec.vectors)):
            continue
        chosen = np.asarray(rec.vectors[rec.chosen], dtype=np.float64)
        for i, vec in enumerate(rec.vectors):
            if i != rec.chosen:
                yield chosen, np.asarray(vec, dtype=np.float64)


def agreement(records: Sequence[PreferenceRecord],
              score_fn) -> Dict[str, float]:
    """How often a scorer's top pick matches the user's, versus chance.

    `score_fn(vector) -> float`. This is M6's gate: the critic's top-1 must
    beat random-of-K on the user's own picks. Returns both numbers plus the
    soft score's agreement, so a critic that merely ties the rules is
    visible as such."""
    usable = [r for r in records if r.version == FEATURE_SET_VERSION
              and len(r.vectors) > 1]
    if not usable:
        return {"n": 0, "critic_top1": 0.0, "random_top1": 0.0,
                "soft_top1": 0.0}
    hits = soft_hits = 0
    chance = 0.0
    for rec in usable:
        scores = [score_fn(np.asarray(v, dtype=np.float64))
                  for v in rec.vectors]
        hits += int(int(np.argmax(scores)) == rec.chosen)
        if rec.soft_scores and len(rec.soft_scores) == len(rec.vectors):
            soft_hits += int(int(np.argmax(rec.soft_scores)) == rec.chosen)
        chance += 1.0 / len(rec.vectors)
    n = len(usable)
    return {"n": n,
            "critic_top1": round(hits / n, 4),
            "random_top1": round(chance / n, 4),
            "soft_top1": round(soft_hits / n, 4)}
