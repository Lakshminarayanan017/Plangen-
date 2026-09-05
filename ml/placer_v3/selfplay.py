"""
selfplay.py — R2b: manufacture Indian-scored training data from the engine.

The problem this solves. There is no Indian floor-plan corpus. CubiCasa5K is
Finnish (its own census lists 123 saunas), and imitating it harder is what
made v2 regress worst on 20x45 and 30x60 — the narrow deep plots that are the
Indian urban archetype.

There IS, however, an Indian-tuned reward: 33 reviewer rules encoding NBC
minimums, circulation depth, the openness gradient and the stated quality
bar, plus a learned critic at AUC 0.912. So:

    when you cannot get the dataset, the reward IS the dataset.

This module sweeps briefs across the real Indian plot distribution, runs the
existing algorithmic engine at k candidates per brief, scores every survivor
with rules + critic, and writes the winners out as (brief -> placement)
training records. Room seeds come from the CARVED geometry, so the model
learns where rooms actually ended up in a plan the reviewer liked — not where
a Finnish architect once put them.

Run:
    python -m ml.placer_v3.selfplay --briefs 4000 --workers 8
    python -m ml.placer_v3.selfplay --briefs 200 --out /tmp/sp   # smoke test

Resumable: re-running appends only briefs whose id is not already in the
corpus, so an interrupted 17-hour sweep costs you nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.engine.contracts import (
    EngineConfig, EngineRequest, RoomSpec, size_class_for,
)
from ml.placer_v3.config import (
    MAX_BANDS, MAX_ROOMS, STATE_GRID, aspect_class, orientation_class,
)

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "corpus")

# ── the Indian plot distribution ────────────────────────────────────────────
# Frontage x depth in feet. These are the sizes that actually get built on:
# 30x40 and 20x30 dominate plotted urban development, the narrow-deep shapes
# are where v2 failed hardest, and the large ones keep the spacious regime
# represented so the model does not learn "always compress".
_PLOTS: Sequence[Tuple[int, int]] = (
    (15, 30), (18, 30), (20, 30), (20, 35), (20, 40), (20, 45), (20, 50),
    (22, 35), (24, 36), (25, 35), (25, 40), (25, 45), (25, 50), (25, 60),
    (28, 40), (30, 35), (30, 40), (30, 45), (30, 50), (30, 55), (30, 60),
    (32, 48), (33, 45), (35, 40), (35, 50), (35, 55), (36, 54), (38, 45),
    (40, 30), (40, 40), (40, 50), (40, 60), (42, 55), (45, 45), (45, 60),
    (48, 36), (50, 40), (50, 50), (50, 70), (55, 45), (60, 40), (60, 60),
    (60, 90), (70, 50),
)
_SIDES = ("N", "E", "S", "W")

# Program fractions of net room area, Indian residential convention.
_FRACTIONS = {
    "living": 0.20, "drawing": 0.13, "dining": 0.11, "kitchen": 0.09,
    "passage": 0.05, "master": 0.16, "bedroom": 0.13, "bath": 0.045,
    "parking": 0.12, "pooja": 0.03, "store": 0.04, "utility": 0.04,
    "study": 0.08,
}


def _program(rng: random.Random, bhk: int, plot_sqft: float
             ) -> List[RoomSpec]:
    """One plausible program for this plot. Optional rooms appear with
    probabilities that rise with plot size, so the corpus contains the
    program VARIETY the model has to generalise over — a corpus of identical
    7-room 3BHKs would teach it one layout."""
    net = plot_sqft * 0.80
    f = _FRACTIONS
    rooms: List[RoomSpec] = [
        RoomSpec("Living Room", "living_room", net * f["living"],
                 zone="public"),
        RoomSpec("Kitchen", "kitchen", net * f["kitchen"], zone="service"),
    ]
    if plot_sqft >= 800:
        rooms.append(RoomSpec("Dining Room", "dining_room",
                              net * f["dining"], zone="service"))
    if plot_sqft >= 900 and rng.random() < 0.75:
        rooms.append(RoomSpec("Parking", "parking", net * f["parking"],
                              zone="public"))
    if bhk >= 2:
        rooms.append(RoomSpec("Passage", "hallway", net * f["passage"],
                              zone="private"))
    rooms.append(RoomSpec("Master Bedroom", "master_bedroom",
                          net * f["master"], zone="private"))
    for i in range(2, bhk + 1):
        rooms.append(RoomSpec(f"Bedroom {i}", "bedroom", net * f["bedroom"],
                              zone="private"))
    n_baths = max(1, (bhk + 1) // 2 + (1 if bhk >= 3 else 0))
    for i in range(1, n_baths + 1):
        rooms.append(RoomSpec(f"Bath {i}", "bathroom", net * f["bath"],
                              zone="private"))

    optional = [
        (0.30 if plot_sqft >= 1000 else 0.12, "Pooja Room", "pooja_room",
         f["pooja"], "public"),
        (0.25 if plot_sqft >= 1100 else 0.08, "Store", "store",
         f["store"], "service"),
        (0.22 if plot_sqft >= 1200 else 0.06, "Utility", "utility",
         f["utility"], "service"),
        (0.20 if plot_sqft >= 1500 else 0.04, "Study", "study",
         f["study"], "private"),
    ]
    for p, name, rtype, frac, zone in optional:
        if rng.random() < p and len(rooms) < MAX_ROOMS - 1:
            rooms.append(RoomSpec(name, rtype, net * frac, zone=zone))
    return rooms


def brief_at(index: int, seed: int = 20260905) -> EngineRequest:
    """Deterministic brief #index. Determinism matters twice over: a resumed
    sweep must regenerate the same briefs, and the held-out split has to mean
    something."""
    rng = random.Random(seed * 1_000_003 + index)
    w, h = rng.choice(_PLOTS)
    side = rng.choice(_SIDES)
    plot_sqft = w * h
    if plot_sqft < 700:
        bhk = rng.choice([1, 1, 2])
    elif plot_sqft < 1300:
        bhk = rng.choice([2, 2, 3])
    elif plot_sqft < 2200:
        bhk = rng.choice([2, 3, 3, 4])
    else:
        bhk = rng.choice([3, 4, 4])
    return EngineRequest(
        plot_w_ft=w, plot_h_ft=h, entrance_side=side,
        rooms=_program(rng, bhk, plot_sqft),
        k=1, seed=rng.randrange(1 << 30), name=f"sp{index:06d}")


# ── reading a carved plan back into targets ─────────────────────────────────

def _depth_key(plan, rid: int, side: str) -> Tuple[int, int]:
    """The room's extent along the entrance-depth axis. Rooms carved into the
    same band share this interval exactly, which is what makes band recovery
    a grouping rather than a guess."""
    x0, y0, x1, y1 = plan.face_bbox(rid)
    if side in ("N", "S"):
        return (y0, y1)
    return (x0, x1)


def _depth_from_entrance(plan, rid: int, side: str) -> float:
    x0, y0, x1, y1 = plan.face_bbox(rid)
    if side == "S":
        return plan.h - (y0 + y1) / 2
    if side == "N":
        return (y0 + y1) / 2
    if side == "E":
        return plan.w - (x0 + x1) / 2
    return (x0 + x1) / 2


def plan_to_record(plan, request: EngineRequest, room_ids: Dict[str, int],
                   reward: float, breakdown: Dict) -> Optional[Dict]:
    """A carved GridPlan -> one training record, in generation order.

    Seeds are the realised room CENTROIDS on the 32x32 grid: what the model
    learns to propose is what the engine, judged by the Indian reward,
    actually produced.
    """
    from ml.training.vocab import generation_sort_key

    side = request.entrance_side
    specs = [s for s in request.rooms if s.name in room_ids]
    if not specs or len(specs) > MAX_ROOMS:
        return None
    specs.sort(key=lambda s: generation_sort_key(s.rtype, s.target_sqft))

    # bands: group by shared depth interval, ordered from the entrance in
    bands = sorted({_depth_key(plan, room_ids[s.name], side) for s in specs},
                   key=lambda iv: -iv[0] if side in ("S", "E") else iv[0])
    band_of = {iv: min(i, MAX_BANDS - 1) for i, iv in enumerate(bands)}

    rooms = []
    for spec in specs:
        rid = room_ids[spec.name]
        x0, y0, x1, y1 = plan.face_bbox(rid)
        w_cells, h_cells = x1 - x0, y1 - y0
        row = int(np.clip(round((y0 + y1) / 2 / plan.h * (STATE_GRID - 1)),
                          0, STATE_GRID - 1))
        col = int(np.clip(round((x0 + x1) / 2 / plan.w * (STATE_GRID - 1)),
                          0, STATE_GRID - 1))
        rooms.append({
            "rtype": spec.rtype,
            "zone": spec.zone,
            "row": row, "col": col,
            "size_class": size_class_for(plan.area_sqft(rid)),
            "area_sqft": round(plan.area_sqft(rid), 1),
            "aspect_class": aspect_class(w_cells, h_cells),
            "orientation_class": orientation_class(w_cells, h_cells),
            "band_index": band_of[_depth_key(plan, rid, side)],
            "depth_ft": round(_depth_from_entrance(plan, rid, side)
                              / units.CELLS_PER_FOOT, 1),
        })

    # adjacency edges from the plan's own shared walls, restricted to the
    # requested rooms — the real graph, not the wish list
    idx = {room_ids[s.name]: i for i, s in enumerate(specs)}
    edges = []
    for (a, b), length in plan.adjacency().items():
        if a in idx and b in idx and length >= units.cells(2):
            edges.append([idx[a], idx[b]])

    return {
        "brief": request.name,
        "plot_w_ft": request.plot_w_ft,
        "plot_h_ft": request.plot_h_ft,
        "entrance_side": side,
        "program_sqft": round(sum(s.target_sqft for s in specs), 1),
        "rooms": rooms,
        "edges": edges,
        "reward": round(float(reward), 3),
        "breakdown": {k: v for k, v in (breakdown or {}).items()
                      if isinstance(v, (int, float))},
    }


# ── generation ──────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    k: int = 16                  # candidates per brief
    keep: int = 2                # top-N kept as positives
    min_reward: float = 55.0     # below this the plan is not worth imitating
    critic_weight: float = 0.4


def run_brief(index: int, cfg: SweepConfig, seed: int) -> List[Dict]:
    """Generate, score and harvest one brief. Import-inside so this is safe
    as a multiprocessing worker target on Windows spawn."""
    from modules.step4_generate.critic.critic import LearnedCritic
    from modules.step4_generate.engine.orchestrator import (
        Orchestrator, blended_score,
    )

    request = brief_at(index, seed)
    request.k = cfg.k
    engine_cfg = EngineConfig()
    critic = _worker_critic(engine_cfg)
    orch = Orchestrator(config=engine_cfg, critic=critic)
    try:
        result = orch.generate(request)
    except Exception:
        return []

    scored = []
    for cand in result.ranked:
        reward = cand.verdict.soft_score
        if critic is not None:
            try:
                reward = blended_score(reward, critic.score_candidate(cand),
                                       cfg.critic_weight)
            except Exception:
                pass
        scored.append((reward, cand))
    scored.sort(key=lambda t: -t[0])

    out = []
    for reward, cand in scored[:cfg.keep]:
        if reward < cfg.min_reward:
            continue
        rec = plan_to_record(cand.plan, cand.request or request,
                             cand.room_ids, reward, cand.verdict.breakdown)
        if rec:
            out.append(rec)
    return out


_CRITIC_CACHE: Dict[int, object] = {}


def _worker_critic(engine_cfg):
    """One critic per process, loaded once. Reloading the GBT per brief
    dominated the sweep before this cache existed."""
    key = id(type(engine_cfg))
    if key not in _CRITIC_CACHE:
        from modules.step4_generate.critic.critic import LearnedCritic
        _CRITIC_CACHE[key] = LearnedCritic.load_if_available(
            config=engine_cfg)
    return _CRITIC_CACHE[key]


def _existing_briefs(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["brief"])
                except Exception:
                    continue
    return seen


def main(argv=None) -> int:
    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()

    p = argparse.ArgumentParser(prog="ml.placer_v3.selfplay")
    p.add_argument("--briefs", type=int, default=4000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--min-reward", type=float, default=55.0)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--strict-reward", action="store_true",
                   help="refuse to build a corpus if the reward has drifted "
                        "from the frozen snapshot")
    args = p.parse_args(argv)

    # The corpus IS the reward, distilled. Record which reward produced it,
    # so a later run cannot silently mix rows graded by two different rulers.
    from modules.step4_generate.engine import reward as reward_mod
    drift = reward_mod.require(strict=args.strict_reward)
    reward_fp = reward_mod.fingerprint()

    os.makedirs(args.out, exist_ok=True)
    corpus_path = os.path.join(args.out, "selfplay.jsonl")
    done = _existing_briefs(corpus_path)
    if done:
        print(f"resuming — {len(done)} brief(s) already in {corpus_path}")

    todo = [i for i in range(args.start, args.start + args.briefs)
            if f"sp{i:06d}" not in done]
    cfg = SweepConfig(k=args.k, keep=args.keep, min_reward=args.min_reward)
    print(f"sweeping {len(todo)} briefs · k={cfg.k} keep={cfg.keep} "
          f"workers={args.workers}")

    written = kept = 0
    logging_every = max(1, len(todo) // 50)

    def emit(records, fh):
        nonlocal written, kept
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
            kept += 1
        written += 1
        if written % logging_every == 0:
            fh.flush()
            print(f"  {written}/{len(todo)} briefs · {kept} records "
                  f"({kept / max(written, 1):.2f} per brief)")

    with open(corpus_path, "a", encoding="utf-8") as fh:
        if args.workers > 1:
            import multiprocessing as mp
            with mp.Pool(args.workers) as pool:
                jobs = [(i, cfg, args.seed) for i in todo]
                for records in pool.imap_unordered(_star_run, jobs,
                                                   chunksize=4):
                    emit(records, fh)
        else:
            for i in todo:
                emit(run_brief(i, cfg, args.seed), fh)

    total = sum(1 for _ in open(corpus_path, encoding="utf-8"))
    val_briefs = sorted({f"sp{i:06d}" for i in range(args.start,
                                                     args.start + args.briefs)
                         if (i * 2654435761 % 1000) / 1000.0 < args.val_frac})
    manifest = {
        "corpus": "selfplay.jsonl",
        "records": total,
        "briefs_swept": len(todo) + len(done),
        "k": cfg.k, "keep": cfg.keep, "min_reward": cfg.min_reward,
        "seed": args.seed,
        "val_briefs": val_briefs,
        "reward": "soft_score blended with learned critic @ 0.4",
        "reward_fingerprint": reward_fp,
        "reward_matches_frozen": drift.matches,
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\ncorpus  -> {os.path.abspath(corpus_path)}  ({total} records)")
    print(f"held-out briefs: {len(val_briefs)}")
    return 0


def _star_run(job):
    return run_brief(*job)


if __name__ == "__main__":
    raise SystemExit(main())
