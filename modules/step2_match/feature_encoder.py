"""
feature_encoder.py
==================
Encodes a BuildingRequirements object into the same 28-dimensional
feature space used by the plan index (built by plan_indexer.py).

This is the QUERY ENCODER — at runtime it converts what the user
asked for into a vector, which is then compared against all 4983
indexed plan vectors using cosine similarity.

The 28-dim feature space is identical to the one defined in
plan_indexer.py — any change there must be mirrored here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from models import BuildingRequirements

# ── Paths ──────────────────────────────────────────────────────────────
BASE      = Path(__file__).parents[2]
INDEX_DIR = BASE / "extracted data" / "plan_index"

# ── BHK numeric mapping (mirrors plan_indexer.py) ──────────────────────
BHK_TO_NUM: Dict[str, float] = {
    "1bhk": 0.25, "2bhk": 0.5, "3bhk": 0.75, "4bhk": 1.0, "4bhk+": 1.0,
    "studio": 0.0, "1": 0.25, "2": 0.5, "3": 0.75, "4": 1.0,
}

# Room type flags — same order as plan_indexer.py (indices 10-21)
ROOM_FLAGS = [
    "kitchen", "living_room", "dining", "bedroom", "bathroom",
    "balcony", "utility", "storage", "study", "garage", "staircase", "entry",
]

# Aliases: user-supplied room names → canonical flag name
ROOM_ALIAS: Dict[str, str] = {
    "kitchen":           "kitchen",
    "dining":            "dining",
    "dining room":       "dining",
    "dining_room":       "dining",
    "living":            "living_room",
    "living room":       "living_room",
    "living_room":       "living_room",
    "drawing room":      "living_room",
    "drawing_room":      "living_room",
    "hall":              "living_room",
    "bedroom":           "bedroom",
    "master bedroom":    "bedroom",
    "master_bedroom":    "bedroom",
    "kids bedroom":      "bedroom",
    "guest bedroom":     "bedroom",
    "bathroom":          "bathroom",
    "bath":              "bathroom",
    "toilet":            "bathroom",
    "wc":                "bathroom",
    "balcony":           "balcony",
    "utility":           "utility",
    "utility room":      "utility",
    "laundry":           "utility",
    "storage":           "storage",
    "store":             "storage",
    "store room":        "storage",
    "store_room":        "storage",
    "study":             "study",
    "study room":        "study",
    "study_room":        "study",
    "office":            "study",
    "garage":            "garage",
    "car parking":       "garage",
    "car_parking":       "garage",
    "parking":           "garage",
    "staircase":         "staircase",
    "stairs":            "staircase",
    "entry":             "entry",
    "foyer":             "entry",
    "lobby":             "entry",
    "passage":           "entry",
    "pooja":             "utility",
    "pooja room":        "utility",
    "puja":              "utility",
}

FEATURE_DIM = 28


class FeatureEncoder:
    """
    Encodes a BuildingRequirements → normalised 28-dim float32 vector.

    Loads the normalisation parameters saved by plan_indexer so that
    query vectors land in the same space as the index vectors.
    """

    def __init__(self) -> None:
        self._norm_params: Optional[Dict[str, Any]] = None
        self._load_norm_params()

    def _load_norm_params(self) -> None:
        """Load normalisation params from plan_metadata.json."""
        meta_file = INDEX_DIR / "plan_metadata.json"
        if meta_file.exists():
            with meta_file.open() as f:
                data = json.load(f)
            self._norm_params = data.get("norm_params")
        # If index not built yet, use None — encoder still works without it
        # (cosine similarity is scale-invariant on the feature values)

    def encode(self, reqs: BuildingRequirements) -> np.ndarray:
        """
        Convert BuildingRequirements → float32 vector of shape (28,).
        All values are in [0, 1] before normalisation.
        """
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        dims = reqs.plot_dimensions
        ctx  = reqs.plot_context

        # ── 0: plan_aspect_ratio ────────────────────────────────────────
        if dims and dims.length and dims.width:
            length_ft = dims.length if dims.unit == "ft" else dims.length * 3.28084
            width_ft  = dims.width  if dims.unit == "ft" else dims.width  * 3.28084
            ar = max(length_ft, width_ft) / max(min(length_ft, width_ft), 0.1)
        else:
            ar = 1.3   # typical Indian plot default
        vec[0] = float(np.clip((ar - 0.2) / (5.0 - 0.2), 0.0, 1.0))

        # ── 1: room_count / 20 ──────────────────────────────────────────
        total_rooms = sum(r.quantity for r in reqs.rooms)
        vec[1] = float(np.clip(total_rooms / 20.0, 0.0, 1.0))

        # ── 2: bhk_numeric ──────────────────────────────────────────────
        bhk_num = self._infer_bhk_numeric(reqs)
        vec[2] = bhk_num

        # ── 3: depth_spread (proxy from plot shape) ─────────────────────
        # Deeper plots → higher depth_spread. Estimate from aspect ratio.
        # depth_spread ≈ 1 - 1/ar (normalised)
        vec[3] = float(np.clip(1.0 - 1.0 / max(ar, 1.0), 0.0, 1.0))

        # ── 4: lateral_spread (proxy — moderate for most Indian plots) ──
        vec[4] = 0.4

        # ── 5: doors_per_room / 4 ───────────────────────────────────────
        vec[5] = 0.25   # typical 1 door per room / 4

        # ── 6: compartmentalization ──────────────────────────────────────
        # More rooms → more compartmentalized
        vec[6] = float(np.clip(total_rooms / 15.0, 0.2, 0.9))

        # ── 7-9: zone_balance ────────────────────────────────────────────
        # We use the typical Indian residential zone pattern
        # (higher back zone for bedrooms, front for living)
        n_bedrooms = self._count_bedrooms(reqs)
        n_living   = self._count_living(reqs)

        # Heuristic: bedrooms → back zone, living/public → front zone
        total_rooms_safe = max(total_rooms, 1)
        front_ratio  = min(n_living / total_rooms_safe, 0.6)
        back_ratio   = min(n_bedrooms / total_rooms_safe, 0.6)
        middle_ratio = max(0.0, 1.0 - front_ratio - back_ratio)

        vec[7] = float(front_ratio)
        vec[8] = float(middle_ratio)
        vec[9] = float(back_ratio)

        # ── 10-21: room presence flags ───────────────────────────────────
        present = self._detect_room_presence(reqs)
        for i, flag in enumerate(ROOM_FLAGS):
            vec[10 + i] = 1.0 if present.get(flag, False) else 0.0

        # ── 22: multi_floor ──────────────────────────────────────────────
        # ALWAYS set to 0.0 — the CubiCasa5K plan index has every plan at
        # multi_floor=0 (single-floor apartment floorplates). Setting this
        # to 1.0 for multi-floor queries creates a permanent cosine similarity
        # penalty against the ENTIRE database, artificially capping similarity
        # at ~0.66. Until a multi-floor Indian residential plan index is built,
        # this dimension is suppressed so it does not hurt matching quality.
        # See: plan_indexer.py line 45 — "multi_floor set to 0 as CubiCasa5K
        # is single-floor".
        vec[22] = 0.0

        # ── 23: bedroom_count_ratio ──────────────────────────────────────
        vec[23] = float(np.clip(n_bedrooms / max(total_rooms_safe, 1), 0.0, 1.0))

        # ── 24: bathroom_count_ratio ─────────────────────────────────────
        n_bath = self._count_bathrooms(reqs)
        vec[24] = float(np.clip(n_bath / max(total_rooms_safe, 1), 0.0, 1.0))

        # ── 25-27: reserved ──────────────────────────────────────────────
        # (left as 0.0)

        # ── Apply normalisation params (if available) ─────────────────────
        if self._norm_params:
            col_min = np.array(self._norm_params["col_min"], dtype=np.float32)
            col_max = np.array(self._norm_params["col_max"], dtype=np.float32)
            col_range = np.where(col_max - col_min < 1e-8, 1.0, col_max - col_min)
            vec = np.clip((vec - col_min) / col_range, 0.0, 1.0)

        return vec.astype(np.float32)

    # ── Private helpers ────────────────────────────────────────────────

    def _infer_bhk_numeric(self, reqs: BuildingRequirements) -> float:
        """Infer BHK numeric score from room list."""
        bedrooms = self._count_bedrooms(reqs)
        if bedrooms == 0:
            return 0.0
        if bedrooms == 1:
            return 0.25
        if bedrooms == 2:
            return 0.5
        if bedrooms == 3:
            return 0.75
        return 1.0   # 4+ bedrooms

    def _count_bedrooms(self, reqs: BuildingRequirements) -> int:
        count = 0
        for room in reqs.rooms:
            norm = room.room_type.lower().replace(" ", "_")
            if "bedroom" in norm:
                count += room.quantity
        return count

    def _count_living(self, reqs: BuildingRequirements) -> int:
        count = 0
        for room in reqs.rooms:
            norm = room.room_type.lower()
            if any(k in norm for k in ["living", "drawing", "hall", "dining", "foyer", "lobby"]):
                count += room.quantity
        return count

    def _count_bathrooms(self, reqs: BuildingRequirements) -> int:
        count = 0
        for room in reqs.rooms:
            norm = room.room_type.lower()
            if any(k in norm for k in ["bathroom", "bath", "toilet", "wc"]):
                count += room.quantity
        return count

    def _detect_room_presence(self, reqs: BuildingRequirements) -> Dict[str, bool]:
        """Map user room list to room flags using ROOM_ALIAS."""
        present: Dict[str, bool] = {flag: False for flag in ROOM_FLAGS}

        for room in reqs.rooms:
            norm_name = room.room_type.lower().strip()
            # Try alias map
            flag = ROOM_ALIAS.get(norm_name)
            if flag is None:
                # Try partial match
                for alias, f in ROOM_ALIAS.items():
                    if alias in norm_name or norm_name in alias:
                        flag = f
                        break
            if flag and flag in present:
                present[flag] = True

        # Implicit room inferences
        # If there's a bedroom, there's likely a bathroom
        if present.get("bedroom"):
            present["bathroom"] = True
        # If there's a kitchen, there's likely an entry
        if present.get("kitchen"):
            present["entry"] = True

        return present

    def describe(self, vec: np.ndarray) -> Dict[str, float]:
        """Return a human-readable breakdown of a feature vector."""
        names = [
            "plan_aspect_ratio", "room_count", "bhk_numeric",
            "depth_spread", "lateral_spread", "doors_per_room",
            "compartmentalization",
            "zone_front", "zone_middle", "zone_back",
            "has_kitchen", "has_living_room", "has_dining", "has_bedroom",
            "has_bathroom", "has_balcony", "has_utility", "has_storage",
            "has_study", "has_garage", "has_staircase", "has_entry",
            "multi_floor", "bedroom_count_ratio", "bathroom_count_ratio",
            "reserved_1", "reserved_2", "reserved_3",
        ]
        return {name: round(float(val), 4) for name, val in zip(names, vec)}
