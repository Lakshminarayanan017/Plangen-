"""
dataset.py — the two corpora, behind one interface.

  PreparedDataset  CubiCasa5K, stage (a)  — imitation pretraining
  SelfPlayDataset  engine output, stage (b) — Indian-scored distillation

Both yield the same array dict, so the training loop does not know or care
which stage it is in.

Augmentation note. The 8 dihedral transforms are applied to the SAMPLE (mask,
seed cells, entrance side) and the state stack is then rebuilt from the
transformed cells — never transformed itself. Rotating the state separately
would be a second implementation of the same geometry, and the two would
eventually disagree about which corner is which. Deriving it once is the only
way they cannot.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID
from ml.placer_v3.features import sample_to_arrays, selfplay_to_arrays
from ml.training.paths import PREPARED_DIR

_ROT90 = {"N": "E", "E": "S", "S": "W", "W": "N"}
_FLIP_H = {"N": "N", "S": "S", "E": "W", "W": "E"}


def _augment_rooms(rooms: List[Dict], flip: bool, k: int) -> List[Dict]:
    g = SEED_GRID
    out = [dict(r) for r in rooms]
    if flip:
        for r in out:
            r["col"] = g - 1 - r["col"]
            # a mirror swaps left/right but not the long axis, so a
            # horizontal room stays horizontal; orientation is untouched
    for _ in range(k % 4):
        for r in out:
            r["row"], r["col"] = r["col"], g - 1 - r["row"]
            # a quarter turn DOES swap the long axis
            if "orientation_class" in r and r["orientation_class"] in (1, 2):
                r["orientation_class"] = 3 - r["orientation_class"]
    return out


def _augment_side(side: str, flip: bool, k: int) -> str:
    if flip:
        side = _FLIP_H[side]
    for _ in range(k % 4):
        side = _ROT90[side]
    return side


class PreparedDataset:
    """CubiCasa5K prepared corpus, 8x dihedral augmentation."""

    def __init__(self, out_dir: str = PREPARED_DIR, split: str = "train",
                 augment: bool = True, max_rooms: Optional[int] = None):
        with open(os.path.join(out_dir, "samples.jsonl"),
                  encoding="utf-8") as f:
            self.samples = [json.loads(ln) for ln in f if ln.strip()]
        self.masks = np.load(os.path.join(out_dir, "masks.npy"))
        with open(os.path.join(out_dir, "manifest.json"),
                  encoding="utf-8") as f:
            manifest = json.load(f)
        val = set(manifest.get("val_indices", []))
        idx = [i for i in range(len(self.samples))
               if (i in val) == (split == "val")]
        if max_rooms:
            idx = [i for i in idx
                   if len(self.samples[i]["rooms"]) <= max_rooms]
        self.index = idx
        self.mult = 8 if augment else 1
        self.split = split

    def __len__(self) -> int:
        return len(self.index) * self.mult

    def __getitem__(self, i: int) -> Dict:
        sample = self.samples[self.index[i // self.mult]]
        mask = self.masks[sample["mask_index"]]
        aug = i % self.mult
        if aug:
            flip, k = aug >= 4, aug % 4
            mask = mask[:, ::-1] if flip else mask
            for _ in range(k):
                mask = np.rot90(mask, k=-1)
            sample = dict(sample)
            sample["rooms"] = _augment_rooms(sample["rooms"], flip, k)
            sample["entrance_side"] = _augment_side(sample["entrance_side"],
                                                    flip, k)
            if k % 2 == 1:
                sample["plot_w_ft"], sample["plot_h_ft"] = \
                    sample["plot_h_ft"], sample["plot_w_ft"]
            mask = np.ascontiguousarray(mask)
        return sample_to_arrays(sample, mask)


class SelfPlayDataset:
    """Engine-generated, reward-filtered corpus (ml.placer_v3.selfplay).

    Splitting is BY BRIEF, never by record: the two candidates kept from one
    brief share a plot, a program and most of a layout, so a record-level
    split would put near-duplicates on both sides and report a validation
    number that means nothing. Same discipline the critic's corpus uses.
    """

    def __init__(self, corpus_dir: str, split: str = "train",
                 augment: bool = True, min_reward: float = 0.0,
                 max_rooms: Optional[int] = None):
        path = os.path.join(corpus_dir, "selfplay.jsonl")
        with open(path, encoding="utf-8") as f:
            records = [json.loads(ln) for ln in f if ln.strip()]
        man_path = os.path.join(corpus_dir, "manifest.json")
        val_briefs = set()
        if os.path.exists(man_path):
            with open(man_path, encoding="utf-8") as f:
                val_briefs = set(json.load(f).get("val_briefs", []))
        keep = []
        for r in records:
            if r.get("reward", 0.0) < min_reward:
                continue
            if max_rooms and len(r["rooms"]) > max_rooms:
                continue
            if (r["brief"] in val_briefs) == (split == "val"):
                keep.append(r)
        self.records = keep
        self.mult = 8 if augment else 1
        self.split = split

    def __len__(self) -> int:
        return len(self.records) * self.mult

    def __getitem__(self, i: int) -> Dict:
        record = self.records[i // self.mult]
        aug = i % self.mult
        if aug:
            flip, k = aug >= 4, aug % 4
            record = dict(record)
            record["rooms"] = _augment_rooms(record["rooms"], flip, k)
            record["entrance_side"] = _augment_side(record["entrance_side"],
                                                    flip, k)
            if k % 2 == 1:
                record["plot_w_ft"], record["plot_h_ft"] = \
                    record["plot_h_ft"], record["plot_w_ft"]
        return selfplay_to_arrays(record)

    def reward_stats(self) -> Dict[str, float]:
        if not self.records:
            return {"n": 0}
        r = np.array([x["reward"] for x in self.records])
        return {"n": len(r), "mean": round(float(r.mean()), 2),
                "p10": round(float(np.percentile(r, 10)), 2),
                "p90": round(float(np.percentile(r, 90)), 2),
                "max": round(float(r.max()), 2)}


def describe(ds) -> str:
    kind = type(ds).__name__
    extra = ""
    if isinstance(ds, SelfPlayDataset):
        extra = f" · reward {ds.reward_stats()}"
    return f"{kind}[{ds.split}] {len(ds)} items (x{ds.mult} aug){extra}"
