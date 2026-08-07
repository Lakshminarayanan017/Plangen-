"""
corpus.py — the candidate corpus: what the human labels and the tuner fits.

For each golden brief we run the engine ONCE (with a base config) and keep
every valid candidate it produced, recording for each:
  • a stable id  (brief name + rank)
  • the engine's soft_score under the base config
  • the raw breakdown evidence (for the labeling UI's "why" panel)
  • the signed feature vector g  (tuning.features — the tuner's input)
  • a rendered SVG                (the labeling UI's picture)

Crucial property that makes this corpus reusable: the 20 soft weights do
NOT affect candidate GENERATION (the proposer variants, carve retries,
dedup and top-up are all weight-independent) — they only re-rank a fixed
set. So one corpus, built once, is valid for evaluating ANY weight vector,
and the same corpus + labels is the training set for the Phase C critic.

The corpus round-trips through JSON. SVGs are heavy and only the labeling
UI needs them, so they are kept out of corpus.json and written alongside
by the labeling module; corpus.json stays lean (ids + features + scores).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest
from modules.step4_generate.engine.orchestrator import Orchestrator
from ml.harness.briefs import golden_briefs
from modules.step4_generate.render.svg_render import render_svg
from ml.tuning.features import extract_features, room_ids_for

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CORPUS_PATH = os.path.join(RESULTS_DIR, "corpus.json")


@dataclass
class CandidateRecord:
    """One valid candidate, everything the labeler and tuner need."""
    cand_id: str                       # e.g. "30x40_S_2bhk#0"
    rank: int                          # engine's base-config rank (0 = best)
    base_score: float                  # soft_score under the base config
    fidelity: float
    features: Dict[str, float]         # signed g vector (weight-keyed)
    breakdown: Dict[str, float]        # raw rule evidence (for the UI)
    svg: Optional[str] = None          # populated only for the labeling UI


@dataclass
class BriefCorpus:
    """All valid candidates for one brief."""
    name: str
    request: Dict                      # EngineRequest.to_dict()
    candidates: List[CandidateRecord] = field(default_factory=list)


@dataclass
class Corpus:
    """The full labelling/tuning corpus across every brief."""
    base_config: Dict                  # EngineConfig.to_dict()
    k: int
    seed: int
    briefs: List[BriefCorpus] = field(default_factory=list)

    # ── serialization ────────────────────────────────────────────────────
    def to_json(self, *, include_svg: bool) -> Dict:
        def cand(c: CandidateRecord) -> Dict:
            d = asdict(c)
            if not include_svg:
                d.pop("svg", None)
            return d
        return {
            "base_config": self.base_config,
            "k": self.k,
            "seed": self.seed,
            "briefs": [
                {"name": b.name, "request": b.request,
                 "candidates": [cand(c) for c in b.candidates]}
                for b in self.briefs
            ],
        }

    @classmethod
    def from_json(cls, data: Dict) -> "Corpus":
        briefs = [
            BriefCorpus(
                name=b["name"], request=b["request"],
                candidates=[CandidateRecord(**c) for c in b["candidates"]],
            )
            for b in data["briefs"]
        ]
        return cls(base_config=data["base_config"], k=data["k"],
                   seed=data["seed"], briefs=briefs)

    def save(self, path: str = CORPUS_PATH, *, include_svg: bool = False
             ) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(include_svg=include_svg), f, indent=1)
        return path

    @classmethod
    def load(cls, path: str = CORPUS_PATH) -> "Corpus":
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def labelable(self) -> List[BriefCorpus]:
        """Briefs worth labeling: at least two candidates to choose between.
        A single-candidate brief carries no preference signal."""
        return [b for b in self.briefs if len(b.candidates) >= 2]


def build_corpus(config: Optional[EngineConfig] = None, *, k: int = 8,
                 seed: int = 20260716, render: bool = True,
                 progress: bool = True) -> Corpus:
    """Run the engine over the golden briefs and collect the labeling/tuning
    corpus. `render=True` embeds an SVG per candidate (needed to build the
    labeling UI); pass False for a lean, fast, tuner-only corpus."""
    config = config or EngineConfig()
    orch = Orchestrator(config=config)
    corpus = Corpus(base_config=config.to_dict(), k=k, seed=seed)

    for req in golden_briefs(k=k, seed=seed):
        result = orch.generate(req)
        brief = BriefCorpus(name=req.name, request=req.to_dict())
        for rank, cand in enumerate(result.ranked):
            room_ids = room_ids_for(cand.plan, req)
            features = extract_features(cand.plan, req, room_ids, config)
            brief.candidates.append(CandidateRecord(
                cand_id=f"{req.name}#{rank}",
                rank=rank,
                base_score=cand.verdict.soft_score,
                fidelity=round(cand.fidelity, 3) if cand.fidelity else 0.0,
                features=features,
                breakdown=dict(cand.verdict.breakdown),
                svg=render_svg(cand.plan) if render else None,
            ))
        corpus.briefs.append(brief)
        if progress:
            print(f"  {req.name:16} {len(brief.candidates)} candidates")
    return corpus
