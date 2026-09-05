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
          floor: int = 0, vastu=None, min_sqft=None,
          max_sqft=None) -> SimpleNamespace:
    """A stand-in for EnrichedRoom.

    `min_area_sqft` / `max_area_sqft` are on the real model and the bridge
    reads both: the ceiling is what stops the settler inflating a 45 sqft
    bathroom to 266 on a large plot. Defaulted here from the room's own
    target so a stub behaves like a real room rather than an unbounded one.
    """
    return SimpleNamespace(
        room_type=room_type, display_name=display_name, vastu=vastu,
        target_area_sqft=sqft, preferred_floor=floor,
        min_area_sqft=min_sqft if min_sqft is not None else sqft * 0.7,
        max_area_sqft=max_sqft if max_sqft is not None else sqft * 1.6)


def _enriched(rooms, north_direction="N", vastu_enabled=False):
    """A stand-in for EnrichedPlan.

    `north_direction` and `vastu_enabled` are as required here as they are on
    the real model: the bridge reads them to give the engine its compass
    frame, and a stub without them is not an EnrichedPlan. Kept explicit
    rather than defaulted away in the bridge, so a genuinely malformed plan
    still fails loudly."""
    return SimpleNamespace(
        get_rooms_on_floor=lambda i: [r for r in rooms
                                      if r.preferred_floor == i],
        entrance_direction="south",
        north_direction=north_direction,
        vastu_enabled=vastu_enabled,
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
        """No UNRECOGNIZED-TYPE warning — which is what this test is about.

        Asserting `warnings == []` was too broad: since phase 02 a floor may
        also, legitimately, report that surplus area became a courtyard. That
        is information the user wants, not a defect, so the assertion now
        names the warning it actually cares about.
        """
        enriched = _enriched([_room("living_room", "Living Room")])
        _req, warnings = _build_floor_request(
            enriched, 0, "run1", plot_w=30, plot_h=40, multi=False)
        self.assertFalse(
            [w for w in warnings if "no specific engine rule" in w],
            warnings)


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
