"""
test_room_resolver.py — curated room-name alias coverage.

Confirms the curated additions to sources/enricher_rules.json's
room_name_aliases (added alongside the parking/unknown-room-type fix)
resolve to the intended canonical types, and that the existing partial-
match fallback still catches compound phrases built from a known alias.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.step3_enrich.room_resolver import RoomResolver  # noqa: E402


class TestCuratedAliases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = RoomResolver()

    def test_meditation_room_maps_to_study(self):
        self.assertEqual(
            self.resolver._normalize_type("Meditation Room"), "study_room")

    def test_yoga_room_maps_to_gym(self):
        self.assertEqual(
            self.resolver._normalize_type("Yoga Room"), "gym_room")

    def test_prayer_hall_maps_to_pooja_room(self):
        self.assertEqual(
            self.resolver._normalize_type("Prayer Hall"), "pooja_room")

    def test_garage_maps_to_car_parking(self):
        self.assertEqual(
            self.resolver._normalize_type("Garage"), "car_parking")

    def test_compound_phrase_matches_via_substring_fallback(self):
        # "garage" is a known alias; a novel compound phrase containing it
        # should still resolve via the existing partial-match fallback.
        self.assertEqual(
            self.resolver._normalize_type("EV Charging Garage"),
            "car_parking")

    def test_genuinely_novel_word_falls_through_unchanged(self):
        # No alias, no partial match — sanitized passthrough, NOT a crash
        # and NOT a silent misassignment to an unrelated known type.
        self.assertEqual(
            self.resolver._normalize_type("Spaceship Dock"),
            "spaceship_dock")


if __name__ == "__main__":
    unittest.main(verbosity=2)
