"""
test_engine_bridge.py — api/engine_bridge.py room-type handling.

_build_floor_request only reads a small duck-typed slice of EnrichedPlan /
EnrichedRoom (get_rooms_on_floor, room_type, display_name,
target_area_sqft) — lightweight stubs exercise its actual branching logic
without constructing the full pydantic model graph (BuildingRequirements,
KnowledgeBundle, etc.), which is unrelated to what this module does.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.engine_bridge import _build_floor_request  # noqa: E402


def _room(room_type: str, display_name: str, sqft: float = 100.0,
         floor: int = 0) -> SimpleNamespace:
    return SimpleNamespace(room_type=room_type, display_name=display_name,
                           target_area_sqft=sqft, preferred_floor=floor)


def _enriched(rooms):
    return SimpleNamespace(
        get_rooms_on_floor=lambda i: [r for r in rooms
                                      if r.preferred_floor == i],
        entrance_direction="south",
        total_floors=1)


class TestUnrecognizedRoomTypes(unittest.TestCase):
    def test_unrecognized_type_is_still_carved(self):
        """A word with no engine rule must never be silently dropped —
        the user asked for a room and should get one, even generic."""
        enriched = _enriched([
            _room("living_room", "Living Room"),
            _room("spaceship_dock", "Spaceship Dock"),
        ])
        req, warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        names = [s.name for s in req.rooms]
        self.assertIn("Spaceship Dock", names)

    def test_unrecognized_type_produces_a_warning(self):
        enriched = _enriched([
            _room("living_room", "Living Room"),
            _room("spaceship_dock", "Spaceship Dock"),
        ])
        _req, warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        self.assertTrue(
            any("Spaceship Dock" in w and "no specific engine rule" in w
                for w in warnings),
            warnings)

    def test_recognized_type_produces_no_warning(self):
        enriched = _enriched([_room("living_room", "Living Room")])
        _req, warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        self.assertEqual(warnings, [])


class TestUnsupportedRoomTypes(unittest.TestCase):
    def test_garden_is_omitted_with_a_warning_not_mis_carved(self):
        enriched = _enriched([
            _room("living_room", "Living Room"),
            _room("garden", "Garden"),
        ])
        req, warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        names = [s.name for s in req.rooms]
        self.assertNotIn("Garden", names)
        self.assertTrue(any("Garden" in w and "omitted" in w
                            for w in warnings), warnings)

    def test_swimming_pool_is_omitted(self):
        enriched = _enriched([
            _room("living_room", "Living Room"),
            _room("swimming_pool", "Swimming Pool"),
        ])
        req, warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        names = [s.name for s in req.rooms]
        self.assertNotIn("Swimming Pool", names)
        self.assertTrue(any("Swimming Pool" in w for w in warnings))


class TestParkingTypeMapping(unittest.TestCase):
    def test_parking_maps_to_public_zone(self):
        enriched = _enriched([
            _room("living_room", "Living Room"),
            _room("car_parking", "Parking"),
        ])
        req, _warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        parking = next(s for s in req.rooms if s.name == "Parking")
        self.assertEqual(parking.rtype, "parking")
        self.assertEqual(parking.zone, "public")

    def test_garage_maps_to_public_zone(self):
        # Normally room_resolver's alias table already routes "garage" to
        # "car_parking" upstream; this exercises the bridge's own
        # defense-in-depth mapping directly, in case a raw "garage" type
        # ever reaches the bridge unnormalized.
        enriched = _enriched([_room("garage", "Garage")])
        req, _warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        garage = next(s for s in req.rooms if s.name == "Garage")
        self.assertEqual(garage.zone, "public")


if __name__ == "__main__":
    unittest.main(verbosity=2)
