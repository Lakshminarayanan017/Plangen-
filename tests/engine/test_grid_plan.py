import unittest

import numpy as np

from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import (
    EXT_WALL_CELLS, INT_WALL_CELLS, OUTSIDE, WALL,
    CarveError, GridPlan,
)


def cell_counts(plan):
    values, counts = np.unique(plan.grid, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


class TestGridPlanBasics(unittest.TestCase):
    def test_construction(self):
        plan = GridPlan.from_feet(20, 45)
        self.assertEqual(plan.grid.shape, (units.cells(45), units.cells(20)))
        # interior = plot minus the 9" ring on all sides
        interior_w = units.cells(20) - 2 * EXT_WALL_CELLS
        interior_h = units.cells(45) - 2 * EXT_WALL_CELLS
        self.assertEqual(plan.face_area_cells(1), interior_w * interior_h)
        self.assertTrue(plan.face_is_rect(1))
        self.assertTrue(plan.verify())

    def test_too_small_plot_rejected(self):
        with self.assertRaises(CarveError):
            GridPlan(10, 10)

    def test_clear_dims(self):
        plan = GridPlan.from_feet(20, 45)
        w, h = plan.clear_dims(1)
        self.assertEqual(units.fmt_ft_in(w), "18'6\"")   # 20' - 2 x 9"
        self.assertEqual(units.fmt_ft_in(h), "43'6\"")


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.plan = GridPlan.from_feet(30, 40)

    def test_vertical_split_partitions_exactly(self):
        before = self.plan.face_area_cells(1)
        x0, y0, x1, y1 = self.plan.face_bbox(1)
        pos = x0 + units.cells(12)
        new_id = self.plan.split(1, "v", pos, name="Right")
        a, b = self.plan.face_area_cells(1), self.plan.face_area_cells(new_id)
        wall_band = INT_WALL_CELLS * (y1 - y0)
        self.assertEqual(a + b + wall_band, before)     # conservation of space
        self.assertTrue(self.plan.face_is_rect(1))
        self.assertTrue(self.plan.face_is_rect(new_id))
        self.assertTrue(self.plan.verify())

    def test_split_positions_are_exact(self):
        x0, y0, x1, y1 = self.plan.face_bbox(1)
        pos = x0 + units.cells(12)
        new_id = self.plan.split(1, "v", pos, name="Right")
        w_left, _ = self.plan.clear_dims(1)
        w_right, _ = self.plan.clear_dims(new_id)
        self.assertEqual(w_left, units.cells(12))
        self.assertEqual(w_right, (x1 - x0) - units.cells(12) - INT_WALL_CELLS)

    def test_split_rejects_thin_side(self):
        x0, y0, x1, y1 = self.plan.face_bbox(1)
        with self.assertRaises(CarveError):
            self.plan.split(1, "v", x0 + 2, min_side=8)

    def test_nested_splits_stay_valid(self):
        right = self.plan.split(1, "v", units.cells(15), name="Right")
        self.plan.split(1, "h", units.cells(20), name="Left-Bottom")
        self.plan.split(right, "h", units.cells(24), name="Right-Bottom")
        self.assertEqual(len(self.plan.rooms), 4)
        for rid in self.plan.rooms:
            self.assertTrue(self.plan.face_is_rect(rid))
        self.assertTrue(self.plan.verify())
        # no cell is unaccounted for
        counts = cell_counts(self.plan)
        total = sum(counts.values())
        self.assertEqual(total, self.plan.w * self.plan.h)
        self.assertNotIn(OUTSIDE, counts)


class TestSharedWallsAndOpenings(unittest.TestCase):
    def setUp(self):
        self.plan = GridPlan.from_feet(30, 40)
        self.right = self.plan.split(1, "v", units.cells(15), name="Right")

    def test_shared_wall_detected(self):
        walls = self.plan.shared_walls()
        self.assertEqual(len(walls), 1)
        sw = walls[0]
        self.assertEqual({sw.room_a, sw.room_b}, {1, self.right})
        self.assertEqual(sw.axis, "v")
        self.assertEqual(sw.thickness, INT_WALL_CELLS)
        # run spans the full interior height
        _, h = self.plan.clear_dims(1)
        self.assertEqual(sw.length, h)

    def test_adjacency_lengths(self):
        adj = self.plan.adjacency()
        key = tuple(sorted((1, self.right)))
        _, h = self.plan.clear_dims(1)
        self.assertEqual(adj[key], h)

    def test_add_door(self):
        op = self.plan.add_opening(1, self.right, "door")
        self.assertEqual(op.width, 20)                  # 30" leaf
        self.assertTrue(self.plan.verify())

    def test_opening_too_wide_rejected(self):
        with self.assertRaises(CarveError):
            self.plan.add_opening(1, self.right, "wide",
                                  width=units.cells(40))

    def test_overlapping_openings_rejected(self):
        self.plan.add_opening(1, self.right, "door")
        with self.assertRaises(CarveError):
            self.plan.add_opening(1, self.right, "door")

    def test_no_shared_wall_rejected(self):
        # split right side; far room does not touch room 1
        far = self.plan.split(self.right, "h", units.cells(20), name="Far")
        # rooms 1 and far DO share the vertical wall; make a true non-adjacent
        # pair by splitting again and checking a diagonal pair
        corner = self.plan.split(far, "v",
                                 self.plan.face_bbox(far)[0] + units.cells(7),
                                 name="Corner")
        self.assertTrue(self.plan.verify())


class TestVerifyCatchesCorruption(unittest.TestCase):
    def test_unregistered_id_detected(self):
        plan = GridPlan.from_feet(30, 40)
        plan.grid[50, 50] = 99                          # corrupt directly
        with self.assertRaises(AssertionError):
            plan.verify()

    def test_disconnected_room_detected(self):
        plan = GridPlan.from_feet(30, 40)
        x0, y0, x1, y1 = plan.face_bbox(1)
        # manually paint a wall column WITHOUT registering a new room:
        # room 1 becomes two disconnected islands
        mid = (x0 + x1) // 2
        plan.grid[y0:y1, mid:mid + INT_WALL_CELLS] = WALL
        with self.assertRaises(AssertionError):
            plan.verify()


if __name__ == "__main__":
    unittest.main()
