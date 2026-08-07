"""
dataset.py — build the critic's training corpus from engine output.

For every brief the engine generates K candidates. The ones that survive the
hard gate are POSITIVES; each is then damaged in every applicable way
(critic/perturb.py) and every damaged copy is a NEGATIVE. Both classes come
from the same generator on the same plot, so the only thing separating them
is the injected flaw.

The split is by BRIEF, never by sample: a perturbation of a plan shares
almost every feature with its parent, so splitting by sample would put near
duplicates on both sides and report a validation score that means nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.step4_generate.critic import features as feat
from modules.step4_generate.critic.perturb import perturb_all
from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.engine.validator import BasicValidator

log = logging.getLogger("PlanGen.Critic")


@dataclass
class Corpus:
    X: np.ndarray                       # (n, N_FEATURES)
    y: np.ndarray                       # 1 = kept plan, 0 = perturbed
    briefs: List[str] = field(default_factory=list)      # per sample
    kinds: List[str] = field(default_factory=list)       # "clean" | damage
    feature_names: List[str] = field(default_factory=lambda:
                                     list(feat.FEATURE_NAMES))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def summary(self) -> Dict:
        kinds: Dict[str, int] = {}
        for k in self.kinds:
            kinds[k] = kinds.get(k, 0) + 1
        return {"n_samples": len(self), "n_features": int(self.X.shape[1]),
                "n_briefs": len(set(self.briefs)),
                "positives": int(self.y.sum()), "by_kind": kinds}

    def split_by_brief(self, val_fraction: float = 0.25, seed: int = 0
                       ) -> Tuple["Corpus", "Corpus"]:
        names = sorted(set(self.briefs))
        rng = np.random.default_rng(seed)
        rng.shuffle(names)
        n_val = max(1, int(round(len(names) * val_fraction)))
        val_names = set(names[:n_val])
        mask = np.array([b in val_names for b in self.briefs])
        return self._subset(~mask), self._subset(mask)

    def _subset(self, mask: np.ndarray) -> "Corpus":
        idx = np.nonzero(mask)[0]
        return Corpus(X=self.X[idx], y=self.y[idx],
                      briefs=[self.briefs[i] for i in idx],
                      kinds=[self.kinds[i] for i in idx],
                      feature_names=list(self.feature_names))

    def save(self, path: str) -> str:
        np.savez_compressed(path, X=self.X, y=self.y,
                            briefs=np.array(self.briefs),
                            kinds=np.array(self.kinds),
                            feature_names=np.array(self.feature_names))
        return path

    @classmethod
    def load(cls, path: str) -> "Corpus":
        d = np.load(path, allow_pickle=False)
        return cls(X=d["X"], y=d["y"], briefs=list(d["briefs"]),
                   kinds=list(d["kinds"]),
                   feature_names=list(d["feature_names"]))


def build_corpus(briefs: Sequence[EngineRequest], *,
                 config: Optional[EngineConfig] = None,
                 orchestrator: Optional[Orchestrator] = None,
                 max_positives_per_brief: int = 3) -> Corpus:
    """Generate plans, keep the survivors, damage them, and featurize."""
    config = config or EngineConfig()
    orch = orchestrator or Orchestrator(config=config)
    validator = BasicValidator(config)

    rows: List[np.ndarray] = []
    labels: List[float] = []
    brief_of: List[str] = []
    kind_of: List[str] = []

    for request in briefs:
        result = orch.generate(request)
        for cand in result.ranked[:max_positives_per_brief]:
            # the candidate carries the request it was BUILT from (implicit
            # rooms included) and the reviewer's own name -> face map
            req, room_ids = cand.request, cand.room_ids
            rows.append(feat.extract(cand.plan, req, room_ids, cand.verdict,
                                     config))
            labels.append(1.0)
            brief_of.append(request.name)
            kind_of.append("clean")

            for damaged in perturb_all(
                    cand.plan, req, room_ids,
                    seed=abs(hash(request.name)) % 10_000):
                # face ids survive the deep copy; a rename swap moves names
                # between them, which is exactly the damage being modelled
                ids = {damaged.plan.rooms[rid].name: rid
                       for rid in room_ids.values()}
                dreq = _relabel(req, damaged.plan, ids)
                try:
                    verdict = validator.check(damaged.plan, dreq, ids)
                    rows.append(feat.extract(damaged.plan, dreq, ids,
                                             verdict, config))
                except Exception as exc:      # damage the reviewer refuses
                    log.debug("skipped %s on %s: %s", damaged.kind,
                              request.name, exc)
                    continue
                labels.append(0.0)
                brief_of.append(request.name)
                kind_of.append(damaged.kind)

    if not rows:
        raise ValueError("no candidates survived — cannot build a corpus")
    return Corpus(X=np.vstack(rows), y=np.asarray(labels, dtype=np.float64),
                  briefs=brief_of, kinds=kind_of)


def _relabel(request: EngineRequest, plan, room_ids: Dict[str, int]
             ) -> EngineRequest:
    """The request re-stated against a perturbed plan's labelling.

    A type swap moves a name AND its type onto another face, so the room
    list must follow the plan, not the brief. Target areas and zones stay
    attached to the name (the program did not change — only where it
    landed), which is precisely why the swap shows up as area drift."""
    import dataclasses

    from modules.step4_generate.engine.contracts import RoomSpec
    by_name = {s.name: s for s in request.rooms}
    rooms = []
    for name, rid in room_ids.items():
        spec = by_name.get(name)
        rtype = plan.rooms[rid].rtype
        if spec is None:
            rooms.append(RoomSpec(name=name, rtype=rtype, target_sqft=50.0,
                                  zone="service"))
        elif spec.rtype != rtype:
            rooms.append(dataclasses.replace(spec, rtype=rtype))
        else:
            rooms.append(spec)
    return dataclasses.replace(request, rooms=rooms, wishes=[])
