"""
dataset.py — the prepared corpus as model examples, with ×8 augmentation.

Reads training/prepared/{samples.jsonl, masks.npy, manifest.json}. Each item
is one plan's numpy arrays (tier2_placer.features.sample_to_arrays). The 8
dihedral augmentations (flips/rotations) each remap BOTH the boundary mask
and the seed cells consistently and relabel the entrance side, multiplying
effective data 8× — cheap and high-value for placement learning.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID
from ml.tier2_placer.features import sample_to_arrays
from ml.training.paths import PREPARED_DIR

# entrance side relabeling under each transform (see _augment)
_ROT90 = {"N": "E", "E": "S", "S": "W", "W": "N"}
_FLIP_H = {"N": "N", "S": "S", "E": "W", "W": "E"}   # mirror left-right


def _rot_side(side: str, k: int) -> str:
    for _ in range(k % 4):
        side = _ROT90[side]
    return side


def load_prepared(out_dir: str = PREPARED_DIR):
    with open(os.path.join(out_dir, "samples.jsonl"), encoding="utf-8") as f:
        samples = [json.loads(ln) for ln in f if ln.strip()]
    masks = np.load(os.path.join(out_dir, "masks.npy"))
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    return samples, masks, manifest


def _augment(sample: Dict, mask: np.ndarray, aug: int):
    """aug in 0..7: {0..3}=rot90*k, {4..7}=flip then rot90*k. Returns a new
    (sample, mask) with mask, seed cells (row/col), and entrance consistently
    transformed."""
    flip = aug >= 4
    k = aug % 4
    g = SEED_GRID
    m = mask
    rooms = [dict(r) for r in sample["rooms"]]

    def tf_rc(row, col):
        return row, col

    if flip:                      # horizontal flip (mirror columns)
        m = m[:, ::-1]
        for r in rooms:
            r["col"] = g - 1 - r["col"]
        entrance = _FLIP_H[sample["entrance_side"]]
    else:
        entrance = sample["entrance_side"]

    for _ in range(k):            # rotate 90° clockwise
        m = np.rot90(m, k=-1)
        for r in rooms:
            r["row"], r["col"] = r["col"], g - 1 - r["row"]
        entrance = _ROT90[entrance]

    new = dict(sample)
    new["rooms"] = rooms
    new["entrance_side"] = entrance
    # rotation swaps plot w/h
    if k % 2 == 1:
        new["plot_w_ft"], new["plot_h_ft"] = \
            sample["plot_h_ft"], sample["plot_w_ft"]
    return new, np.ascontiguousarray(m)


class PreparedDataset:
    """Indexable dataset of model arrays. `augment=True` yields 8× items."""

    def __init__(self, out_dir: str = PREPARED_DIR, split: str = "train",
                 augment: bool = True, max_rooms: Optional[int] = None):
        samples, masks, manifest = load_prepared(out_dir)
        val = set(manifest.get("val_indices", []))
        idx = [i for i in range(len(samples))
               if (i in val) == (split == "val")]
        if max_rooms:
            idx = [i for i in idx if len(samples[i]["rooms"]) <= max_rooms]
        self.samples = samples
        self.masks = masks
        self.index = idx
        self.mult = 8 if augment else 1

    def __len__(self) -> int:
        return len(self.index) * self.mult

    def __getitem__(self, i: int) -> Dict:
        base = self.index[i // self.mult]
        aug = i % self.mult
        s = self.samples[base]
        mask = self.masks[s["mask_index"]]
        if aug:
            s, mask = _augment(s, mask, aug)
        return sample_to_arrays(s, mask)

    def room_count(self, i: int) -> int:
        return len(self.samples[self.index[i // self.mult]]["rooms"])
