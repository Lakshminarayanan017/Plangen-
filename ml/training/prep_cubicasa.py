"""
prep_cubicasa.py — CubiCasa5K normalized_extraction.json → discrete targets.

Per plan, emits a PreparedSample: a 64×64 interior boundary mask + a program
graph (rooms with type/zone/adjacency) + the SUPERVISED targets the placer
learns — for each room, a seed cell on the 32×32 grid (from the polygon
centroid) and a size class (from the polygon area). Rooms are ordered by the
canonical generation order (training.vocab).

This is the polygon-TRUE prep the v1 pipeline never did: we take real room
polygons and reduce them to discrete cell+class targets a downstream
deterministic carver realizes exactly. Overlap is not merely rare — two
rooms sharing a seed cell is the only failure mode, measured and gated
(training.gates), and structurally cannot become geometric overlap because
the carver owns geometry.

Coordinates are SVG px (x right, y down). Scale (px→ft) is recovered from
per-room `dimensions` (present for 99.7% of rooms, per the audit).

Output (training/prepared/):
    samples.jsonl   one sample per line (rooms/edges/meta + mask index)
    masks.npy       (N, 64, 64) uint8 boundary masks
    manifest.json   counts, val split (by plan id), prep config

Run:  python -m training.prep_cubicasa
      python -m training.prep_cubicasa --limit 200 --out-dir /tmp/prep_test
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID, size_class_for
from ml.training import geometry as geo
from ml.training.paths import CUBICASA_NORMALIZED, PREPARED_DIR, ensure_dir
from ml.training.schema_audit import stream_plans
from ml.training.vocab import MIN_REAL_ROOMS, generation_sort_key, map_room

BOUNDARY_GRID = 64

# adjacency thresholds in SVG px (walls are ~15px thick in this corpus)
_ADJ_GAP_PX = 40.0
_ADJ_OVERLAP_PX = 24.0


@dataclass
class PreparedRoom:
    rtype: str
    zone: str
    row: int                 # seed cell row on SEED_GRID (0 = north/top)
    col: int                 # seed cell col on SEED_GRID (0 = west/left)
    size_class: int          # 1..SIZE_CLASS_MAX
    area_sqft: float


@dataclass
class PreparedSample:
    plan_id: str
    plot_w_ft: float
    plot_h_ft: float
    entrance_side: str       # N/E/S/W
    rooms: List[PreparedRoom] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)  # room-idx pairs

    def to_json(self, mask_index: int) -> Dict:
        d = {
            "plan_id": self.plan_id,
            "plot_w_ft": round(self.plot_w_ft, 2),
            "plot_h_ft": round(self.plot_h_ft, 2),
            "entrance_side": self.entrance_side,
            "mask_index": mask_index,
            "rooms": [asdict(r) for r in self.rooms],
            "edges": [list(e) for e in self.edges],
        }
        return d


class PrepError(ValueError):
    """A plan cannot be prepared (skipped with a counted reason)."""


# ── scale ────────────────────────────────────────────────────────────────────

def _ft_per_px(spaces: List[dict]) -> float:
    """Median ft-per-px from rooms that carry real dimensions."""
    ratios = []
    for s in spaces:
        dims = s.get("dimensions") or {}
        bb = s.get("bbox") or {}
        for ft, px in (("width_ft", "w"), ("height_ft", "h")):
            fv, pv = dims.get(ft), bb.get(px)
            if fv and pv and pv > 1e-6:
                ratios.append(fv / pv)
    if not ratios:
        raise PrepError("no per-room dimensions to recover scale")
    return float(np.median(ratios))


# ── entrance side ────────────────────────────────────────────────────────────

def _entrance_side(record: dict, plot_bbox: dict,
                   interior: List[dict]) -> str:
    """Side (N/E/S/W) of the main exterior door; falls back to the foyer/
    entry room's nearest edge, then to 'S'."""
    ext_wall_ids = {w.get("wall_id") for w in record.get("walls", [])
                    if w.get("is_external")}
    ext_doors = [d for d in record.get("doors", [])
                 if d.get("parent_wall_id") in ext_wall_ids
                 and d.get("centroid")]
    if ext_doors:
        main = max(ext_doors, key=lambda d: d.get("area", 0.0))
        return _nearest_side(main["centroid"], plot_bbox)
    # fallback: the entry/foyer room's centroid
    foyers = [s for s in interior if s.get("_rtype") == "foyer"]
    if foyers and foyers[0].get("centroid"):
        return _nearest_side(foyers[0]["centroid"], plot_bbox)
    return "S"


def _nearest_side(centroid: dict, bbox: dict) -> str:
    cx, cy = centroid["x"], centroid["y"]
    left = cx - bbox["x"]
    right = (bbox["x"] + bbox["w"]) - cx
    top = cy - bbox["y"]                 # SVG y-down: small y = north
    bottom = (bbox["y"] + bbox["h"]) - cy
    return min((("W", left), ("E", right), ("N", top), ("S", bottom)),
              key=lambda kv: kv[1])[0]


# ── adjacency ────────────────────────────────────────────────────────────────

def _cell_in_mask(row: int, col: int, mask: np.ndarray) -> bool:
    """Is the 32-grid cell inside the 64-grid boundary mask (any fill in its
    2×2 block)?"""
    scale = BOUNDARY_GRID // SEED_GRID
    r0, c0 = row * scale, col * scale
    return bool(mask[r0:r0 + scale, c0:c0 + scale].any())


def _place_seed(raw: Tuple[int, int], occupied: set,
                mask: np.ndarray) -> Tuple[int, int]:
    """Nearest cell to `raw` that is free (not in `occupied`) AND in-boundary,
    by expanding-ring (Chebyshev) search. Falls back to nearest free cell
    ignoring the boundary if the footprint is pathologically small."""
    r0, c0 = raw
    if (r0, c0) not in occupied and _cell_in_mask(r0, c0, mask):
        return r0, c0
    best_free = None
    for radius in range(1, SEED_GRID):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue                    # ring only
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < SEED_GRID and 0 <= c < SEED_GRID):
                    continue
                if (r, c) in occupied:
                    continue
                if _cell_in_mask(r, c, mask):
                    return r, c
                if best_free is None:
                    best_free = (r, c)          # free but out-of-boundary
    return best_free if best_free is not None else (r0, c0)


def _adjacent(a: dict, b: dict) -> bool:
    """True if two room bboxes are wall-adjacent (small gap on one axis with
    real overlap on the other) — the program-graph edge signal."""
    ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    y_overlap = min(ay + ah, by + bh) - max(ay, by)
    # vertical neighbors: gap along y, overlap along x
    if x_overlap >= _ADJ_OVERLAP_PX:
        gap = max(ay - (by + bh), by - (ay + ah))
        if -_ADJ_OVERLAP_PX <= gap <= _ADJ_GAP_PX:
            return True
    # horizontal neighbors: gap along x, overlap along y
    if y_overlap >= _ADJ_OVERLAP_PX:
        gap = max(ax - (bx + bw), bx - (ax + aw))
        if -_ADJ_OVERLAP_PX <= gap <= _ADJ_GAP_PX:
            return True
    return False


# ── per-plan preparation ─────────────────────────────────────────────────────

def prepare_plan(plan_id: str, record: dict
                 ) -> Tuple[PreparedSample, np.ndarray]:
    """Prepare one plan → (sample, 64×64 mask). Raises PrepError to skip."""
    plot_bbox = (record.get("metadata") or {}).get("plan_bbox")
    if not plot_bbox or plot_bbox.get("w", 0) < 1 or plot_bbox.get("h", 0) < 1:
        raise PrepError("missing/degenerate plan_bbox")

    spaces = record.get("spaces", [])
    # keep only mapped interior rooms; annotate rtype/zone
    interior = []
    for s in spaces:
        mapped = map_room(s.get("type", ""))
        if mapped is None or not s.get("polygon") or not s.get("bbox"):
            continue
        s = dict(s)
        s["_rtype"], s["_zone"] = mapped
        interior.append(s)
    if len(interior) < MIN_REAL_ROOMS:
        raise PrepError(f"only {len(interior)} real rooms (< {MIN_REAL_ROOMS})")

    fpp = _ft_per_px(spaces)             # ft per px (use all spaces for scale)

    # Frame everything on the INTERIOR footprint (union of kept rooms), not
    # plan_bbox — plan_bbox includes the outdoor/yard, which would shrink the
    # boundary mask and waste the seed grid. The interior bbox = the building
    # footprint the carver actually fills (and its mask captures L-shapes).
    polys = [s["polygon"] for s in interior]
    minx, miny, maxx, maxy = geo.bbox_of(polys)
    ext_w, ext_h = maxx - minx, maxy - miny
    if ext_w < 1 or ext_h < 1:
        raise PrepError("degenerate interior footprint")
    plot_w_ft = ext_w * fpp
    plot_h_ft = ext_h * fpp

    # boundary mask = rasterized union of interior room polygons
    mask = geo.rasterize_union(
        polys, BOUNDARY_GRID, origin=(minx, miny), extent=(ext_w, ext_h))
    if int(mask.sum()) < 4:
        raise PrepError("empty boundary mask after rasterization")

    entrance = _entrance_side(record, plot_bbox, interior)

    # per-room raw seed cell (from centroid) + size, then canonical order
    interim = []
    for s in interior:
        cx, cy = geo.centroid(s["polygon"])
        raw_col = int(np.clip(int((cx - minx) / ext_w * SEED_GRID),
                              0, SEED_GRID - 1))
        raw_row = int(np.clip(int((cy - miny) / ext_h * SEED_GRID),
                              0, SEED_GRID - 1))
        area_sqft = geo.area(s["polygon"]) * (fpp ** 2)
        interim.append({"space": s, "rtype": s["_rtype"], "zone": s["_zone"],
                        "raw": (raw_row, raw_col), "area": area_sqft})
    interim.sort(key=lambda d: generation_sort_key(d["rtype"], d["area"]))

    # Resolve seed cells IN generation order: each room keeps its cell if it
    # is in-boundary and free, else nudges to the nearest such cell. This
    # guarantees 0% seed collisions and 100% in-boundary by construction —
    # anchors (generated first) hold prime cells; later rooms yield. It also
    # mirrors the model's decode-time claimed-cell mask (training/inference
    # agree on what "already placed" means).
    occupied: set = set()
    rooms: List[PreparedRoom] = []
    keep_spaces: List[dict] = []
    for d in interim:
        row, col = _place_seed(d["raw"], occupied, mask)
        occupied.add((row, col))
        rooms.append(PreparedRoom(
            rtype=d["rtype"], zone=d["zone"], row=row, col=col,
            size_class=size_class_for(d["area"]),
            area_sqft=round(d["area"], 1)))
        keep_spaces.append(d["space"])

    # adjacency edges (indices into the reordered room list)
    edges: List[Tuple[int, int]] = []
    for i in range(len(keep_spaces)):
        for j in range(i + 1, len(keep_spaces)):
            if _adjacent(keep_spaces[i]["bbox"], keep_spaces[j]["bbox"]):
                edges.append((i, j))

    sample = PreparedSample(
        plan_id=plan_id, plot_w_ft=plot_w_ft, plot_h_ft=plot_h_ft,
        entrance_side=entrance, rooms=rooms, edges=edges)
    return sample, mask


# ── driver ───────────────────────────────────────────────────────────────────

def run(path: str, out_dir: str, limit: int = 0,
        val_frac: float = 0.10, seed: int = 20260722) -> Dict:
    ensure_dir(out_dir)
    samples_path = os.path.join(out_dir, "samples.jsonl")
    masks: List[np.ndarray] = []
    skip_reasons: Dict[str, int] = {}
    n_ok = 0

    with open(samples_path, "w", encoding="utf-8") as sf:
        for plan_id, record in stream_plans(path, limit):
            try:
                sample, mask = prepare_plan(plan_id, record)
            except PrepError as e:
                key = str(e).split("(")[0].strip()[:40]
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
                continue
            except Exception as e:                       # never crash the run
                skip_reasons[f"ERROR:{type(e).__name__}"] = \
                    skip_reasons.get(f"ERROR:{type(e).__name__}", 0) + 1
                continue
            sf.write(json.dumps(sample.to_json(n_ok)) + "\n")
            masks.append(mask)
            n_ok += 1

    masks_arr = (np.stack(masks) if masks
                 else np.zeros((0, BOUNDARY_GRID, BOUNDARY_GRID), np.uint8))
    np.save(os.path.join(out_dir, "masks.npy"), masks_arr)

    # deterministic val split by sample index (plan ids are 1:1 with samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ok)
    n_val = int(round(n_ok * val_frac))
    val_idx = sorted(int(i) for i in perm[:n_val])
    manifest = {
        "corpus": os.path.basename(path),
        "n_prepared": n_ok,
        "n_val": n_val,
        "val_indices": val_idx,
        "skip_reasons": dict(sorted(skip_reasons.items(),
                                    key=lambda kv: -kv[1])),
        "seed_grid": SEED_GRID,
        "boundary_grid": BOUNDARY_GRID,
        "val_frac": val_frac,
        "split_seed": seed,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="training.prep_cubicasa")
    p.add_argument("--path", default=CUBICASA_NORMALIZED)
    p.add_argument("--out-dir", default=PREPARED_DIR)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"corpus not found: {args.path}")
        return 2

    print(f"preparing {args.path} -> {args.out_dir} …")
    m = run(args.path, args.out_dir, args.limit)
    print("=" * 66)
    print(f"  prepared samples : {m['n_prepared']}   (val {m['n_val']})")
    print("  skipped:")
    for reason, c in m["skip_reasons"].items():
        print(f"    {c:>6}  {reason}")
    print(f"  -> {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
