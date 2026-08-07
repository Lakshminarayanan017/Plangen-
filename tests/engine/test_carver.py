import unittest

import numpy as np

from modules.step4_generate.carve.strip_carver import RoomBrief, carve
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import OUTSIDE, WALL, GridPlan

PROGRAM = [
    RoomBrief("Living Room", "living_room", 190),
    RoomBrief("Dining", "dining_room", 130),
    RoomBrief("Kitchen", "kitchen", 110),
    RoomBrief("Passage", "passage", 70),
    RoomBrief("Master Bedroom", "master_bedroom", 165),
    RoomBrief("Bath 1", "bathroom", 45),
    RoomBrief("Bedroom 2", "bedroom", 140),
    RoomBrief("Bath 2", "bathroom", 40),
]


class TestCarver(unittest.TestCase):
    def setUp(self):
        self.plan = GridPlan.from_feet(30, 40)
        self.ids = carve(self.plan, 1, PROGRAM)

    def test_all_rooms_exist(self):
        self.assertEqual(len(self.ids), len(PROGRAM))
        names = {self.plan.rooms[i].name for i in self.ids}
        self.assertEqual(names, {b.name for b in PROGRAM})

    def test_invariants_hold(self):
        self.assertTrue(self.plan.verify())

    def test_full_coverage_no_overlap(self):
        # every cell is WALL or exactly one room; nothing else exists
        values = set(int(v) for v in np.unique(self.plan.grid))
        self.assertNotIn(OUTSIDE, values)
        self.assertEqual(values - {WALL}, set(self.ids))
        total = self.plan.w * self.plan.h
        wall_cells = int((self.plan.grid == WALL).sum())
        room_cells = sum(self.plan.face_area_cells(i) for i in self.ids)
        self.assertEqual(wall_cells + room_cells, total)

    def test_min_side_respected(self):
        # habitable rooms ≥ 4'; service rooms (bath/store/utility) ≥ 3'
        service = {"bathroom", "toilet", "store", "storage", "utility", "ots"}
        for rid in self.ids:
            w, h = self.plan.clear_dims(rid)
            room = self.plan.rooms[rid]
            floor_min = units.cells(3 if room.rtype in service else 4)
            self.assertGreaterEqual(min(w, h), floor_min - 1, room.name)

    def test_areas_roughly_proportional(self):
        # v0 carver has no optimizer, but big rooms must come out bigger
        living = self.plan.area_sqft(self.plan.find_room("Living Room").id)
        bath = self.plan.area_sqft(self.plan.find_room("Bath 1").id)
        self.assertGreater(living, bath * 2)

    def test_deterministic(self):
        plan2 = GridPlan.from_feet(30, 40)
        carve(plan2, 1, PROGRAM)
        self.assertTrue(np.array_equal(self.plan.grid, plan2.grid))


if __name__ == "__main__":
    unittest.main()
