"""
test_training_prep.py — M4 data-pipeline unit tests on hand-built fixtures.

No dependency on the multi-GB corpus: tiny synthetic plans exercise the
polygon→seed math, the vocab map, the size-class clamp, adjacency, and the
end-to-end prepare_plan on a known 2-room plan.
"""

from __future__ import annotations

import unittest

import numpy as np

from modules.step4_generate.engine.contracts import SEED_GRID, SIZE_CLASS_MAX, size_class_for
from ml.training import geometry as geo
from ml.training.prep_cubicasa import prepare_plan, PrepError
from ml.training.vocab import map_room, generation_sort_key, MIN_REAL_ROOMS


class TestGeometry(unittest.TestCase):
    def test_area_of_unit_square(self):
        sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertAlmostEqual(geo.area(sq), 100.0)

    def test_area_ignores_winding_and_closure(self):
        cw = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]  # closed, clockwise
        self.assertAlmostEqual(geo.area(cw), 100.0)

    def test_centroid_of_square(self):
        sq = [[0, 0], [4, 0], [4, 4], [0, 4]]
        cx, cy = geo.centroid(sq)
        self.assertAlmostEqual(cx, 2.0)
        self.assertAlmostEqual(cy, 2.0)

    def test_centroid_of_L_shape_is_inside_bbox(self):
        L = [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]]
        cx, cy = geo.centroid(L)
        self.assertTrue(0 <= cx <= 6 and 0 <= cy <= 6)

    def test_rasterize_square_fills_grid(self):
        sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
        m = geo.rasterize_union([sq], 8, origin=(0, 0), extent=(10, 10))
        self.assertEqual(m.shape, (8, 8))
        self.assertEqual(int(m.sum()), 64)          # full square fills all

    def test_rasterize_half_square(self):
        # right half only → ~half filled
        half = [[5, 0], [10, 0], [10, 10], [5, 10]]
        m = geo.rasterize_union([half], 8, origin=(0, 0), extent=(10, 10))
        self.assertTrue(28 <= int(m.sum()) <= 36)   # ~32


class TestVocab(unittest.TestCase):
    def test_known_types_map(self):
        self.assertEqual(map_room("bedroom"), ("bedroom", "private"))
        self.assertEqual(map_room("kitchen"), ("kitchen", "service"))
        self.assertEqual(map_room("entry"), ("foyer", "public"))

    def test_dropped_types(self):
        for t in ("undefined", "outdoor", "balcony", "sauna", "special"):
            self.assertIsNone(map_room(t), t)

    def test_unknown_type_dropped(self):
        self.assertIsNone(map_room("teleporter_room"))

    def test_generation_order_anchors_first(self):
        rooms = [("bathroom", 40), ("living_room", 200), ("bedroom", 120)]
        order = sorted(rooms, key=lambda r: generation_sort_key(*r))
        self.assertEqual([r[0] for r in order],
                         ["living_room", "bedroom", "bathroom"])

    def test_generation_order_larger_first_within_tier(self):
        rooms = [("bedroom", 100), ("bedroom", 180)]
        order = sorted(rooms, key=lambda r: generation_sort_key(*r))
        self.assertEqual([r[1] for r in order], [180, 100])


class TestSizeClass(unittest.TestCase):
    def test_normal_rooms_unclamped(self):
        self.assertEqual(size_class_for(120), 5)
        self.assertEqual(size_class_for(348), 14)

    def test_floor_at_one(self):
        self.assertEqual(size_class_for(1), 1)
        self.assertEqual(size_class_for(0), 1)

    def test_clamped_at_max(self):
        self.assertEqual(size_class_for(1000), SIZE_CLASS_MAX)
        self.assertEqual(size_class_for(5000), SIZE_CLASS_MAX)


def _rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _space(rtype, x, y, w, h, width_ft, height_ft):
    poly = _rect(x, y, w, h)
    return {
        "type": rtype, "polygon": poly,
        "bbox": {"x": x, "y": y, "w": w, "h": h},
        "centroid": {"x": x + w / 2, "y": y + h / 2},
        "area": w * h,
        "dimensions": {"width_ft": width_ft, "height_ft": height_ft},
    }


class TestPreparePlan(unittest.TestCase):
    def _plan(self):
        # 2 real rooms side by side + 1 more so it passes MIN_REAL_ROOMS;
        # 100px = 10ft → 0.1 ft/px
        spaces = [
            _space("living_room", 0, 0, 100, 100, 10, 10),
            _space("kitchen", 100, 0, 100, 100, 10, 10),
            _space("bedroom", 0, 100, 200, 100, 20, 10),
        ]
        return {
            "metadata": {"plan_bbox": {"x": 0, "y": 0, "w": 200, "h": 200}},
            "spaces": spaces,
            "walls": [{"wall_id": 0, "is_external": True}],
            "doors": [{"door_id": 0, "parent_wall_id": 0, "area": 50,
                       "centroid": {"x": 100, "y": 200}}],  # south edge
        }

    def test_prepares_three_rooms_in_generation_order(self):
        sample, mask = prepare_plan("fixture/1", self._plan())
        self.assertEqual([r.rtype for r in sample.rooms],
                         ["living_room", "kitchen", "bedroom"])

    def test_seed_cells_in_range_and_distinct(self):
        sample, _ = prepare_plan("fixture/1", self._plan())
        cells = set()
        for r in sample.rooms:
            self.assertTrue(0 <= r.row < SEED_GRID and 0 <= r.col < SEED_GRID)
            cells.add((r.row, r.col))
        self.assertEqual(len(cells), 3, "distinct seed cells")

    def test_scale_and_dims_recovered(self):
        sample, _ = prepare_plan("fixture/1", self._plan())
        # interior footprint is the full 200×200px = 20×20ft
        self.assertAlmostEqual(sample.plot_w_ft, 20.0, delta=0.5)
        self.assertAlmostEqual(sample.plot_h_ft, 20.0, delta=0.5)

    def test_entrance_side_south(self):
        sample, _ = prepare_plan("fixture/1", self._plan())
        self.assertEqual(sample.entrance_side, "S")

    def test_adjacency_detected(self):
        sample, _ = prepare_plan("fixture/1", self._plan())
        # living & kitchen share a vertical wall; bedroom spans below both
        self.assertGreaterEqual(len(sample.edges), 2)

    def test_too_few_rooms_raises(self):
        plan = self._plan()
        plan["spaces"] = plan["spaces"][:1] + [
            _space("undefined", 100, 0, 100, 100, 10, 10)]
        with self.assertRaises(PrepError):
            prepare_plan("fixture/thin", plan)

    def test_mask_is_64_and_nonempty(self):
        _, mask = prepare_plan("fixture/1", self._plan())
        self.assertEqual(mask.shape, (64, 64))
        self.assertGreater(int(mask.sum()), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
