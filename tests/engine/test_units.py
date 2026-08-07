import unittest

from modules.step4_generate.core import units


class TestUnits(unittest.TestCase):
    def test_feet_to_cells(self):
        self.assertEqual(units.cells(10), 80)
        self.assertEqual(units.cells(1), 8)
        self.assertEqual(units.cells(4, 6), 36)      # 4'6" = 54" = 36 cells
        self.assertEqual(units.cells(0, 4.5), 3)     # internal wall
        self.assertEqual(units.cells(0, 9), 6)       # external wall

    def test_off_lattice_rejected(self):
        with self.assertRaises(ValueError):
            units.cells(0, 4.0)                       # 4" not on 1.5" lattice
        with self.assertRaises(ValueError):
            units.cells(1, 1)                         # 13"

    def test_format_ft_in(self):
        self.assertEqual(units.fmt_ft_in(80), "10'0\"")
        self.assertEqual(units.fmt_ft_in(122), "15'3\"")
        self.assertEqual(units.fmt_ft_in(3), "0'4.5\"")
        self.assertEqual(units.fmt_ft_in(60), "7'6\"")

    def test_area(self):
        # 10' x 10' room = 80 x 80 cells = 6400 cells^2 = 100 sqft
        self.assertAlmostEqual(units.area_sqft(80 * 80), 100.0)

    def test_snap_to_module(self):
        self.assertEqual(units.snap_to_module(7), 8)
        self.assertEqual(units.snap_to_module(6), 6)
        self.assertEqual(units.snap_to_module(1), 2)


if __name__ == "__main__":
    unittest.main()
