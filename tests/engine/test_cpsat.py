"""
CP-SAT side-tool tests (M3).

Two things must hold no matter what: the solver NEVER changes a result the
greedy path already produced (mode "off" is the reference), and its absence
is never an error. The optimality claims are checked on tiny instances where
the optimum can be written down by hand.
"""

import unittest

from modules.step4_generate.carve.hub_carver import _distribute
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import CarveError
from modules.step4_generate.engine import cpsat
from modules.step4_generate.engine.cpsat.bands import BandRoom, exact_bands, solution_signature
from modules.step4_generate.engine.cpsat.distribute import exact_distribute

HAVE_SOLVER = cpsat.available()
skip_no_solver = unittest.skipUnless(HAVE_SOLVER, "ortools not installed")


class TestAvailability(unittest.TestCase):
    def test_import_never_fails(self):
        self.assertIn(cpsat.available(), (True, False))

    def test_entry_points_return_none_without_a_solver(self):
        if HAVE_SOLVER:
            self.skipTest("solver present; the absent path is exercised by "
                          "the None-returns below")
        self.assertIsNone(exact_distribute(100, [1, 1], [10, 10]))
        self.assertIsNone(exact_bands([BandRoom(100, 10)], across_span=50,
                                      depth_span=50, wall=3, max_per_band=2))


@skip_no_solver
class TestExactDistribute(unittest.TestCase):
    def test_sums_to_the_span_and_respects_minimums(self):
        parts = exact_distribute(200, [1.0, 2.0, 3.0], [20, 20, 20])
        self.assertEqual(sum(parts), 200)
        for part, minimum in zip(parts, (20, 20, 20)):
            self.assertGreaterEqual(part, minimum)

    def test_returns_none_when_minimums_do_not_fit(self):
        self.assertIsNone(exact_distribute(30, [1, 1, 1], [20, 20, 20]))

    def test_honors_a_minimum_the_greedy_split_would_violate(self):
        # weights say the last part gets ~4 cells; its minimum is 40
        parts = exact_distribute(200, [10.0, 10.0, 0.2], [10, 10, 40])
        self.assertGreaterEqual(parts[2], 40)
        self.assertEqual(sum(parts), 200)

    def test_module_snapping(self):
        parts = exact_distribute(240, [1.0, 1.0, 1.0], [16, 16, 16])
        on_module = [p % units.MODULE_CELLS == 0 for p in parts]
        self.assertGreaterEqual(sum(on_module), len(parts) - 1)

    def test_single_part_takes_everything(self):
        self.assertEqual(exact_distribute(97, [1.0], [10]), [97])

    def test_is_deterministic(self):
        args = (240, [1.0, 2.5, 0.7, 3.0], [16, 16, 16, 16])
        self.assertEqual(exact_distribute(*args), exact_distribute(*args))

    def test_never_worse_than_greedy_on_relative_error(self):
        """The claim the exact split actually makes: over the SAME feasible
        set, its total relative deviation from the proportional ideal is
        <= the greedy split's. Both run with snapping off, because the
        exact version enforces the 3" module as a hard constraint while
        greedy only snaps opportunistically — comparing them with snapping
        on measures the constraint, not the optimizer."""
        import random

        rng = random.Random(20260730)
        checked = 0
        for _ in range(25):
            n = rng.randint(2, 5)
            weights = [rng.uniform(0.5, 8.0) for _ in range(n)]
            minimums = [rng.choice([16, 24, 32, 40]) for _ in range(n)]
            total = int(sum(minimums) * rng.uniform(1.2, 3.0))
            exact = exact_distribute(total, weights, minimums, snap=False)
            try:
                greedy = _distribute(total, weights, minimums, snap=False)
            except CarveError:
                self.assertIsNotNone(exact, "exact should still solve this")
                continue
            self.assertIsNotNone(exact)
            wsum = sum(weights)
            ideals = [max(1.0, total * w / wsum) for w in weights]

            def err(parts):
                return sum(abs(p - i) / i for p, i in zip(parts, ideals))

            self.assertLessEqual(err(exact), err(greedy) + 1e-6,
                                 f"total={total} w={weights} min={minimums}")
            checked += 1
        self.assertGreater(checked, 10, "too few comparable instances")


@skip_no_solver
class TestExactBands(unittest.TestCase):
    ROOMS = [BandRoom(4000, 60), BandRoom(3000, 60), BandRoom(1500, 24),
             BandRoom(2500, 60), BandRoom(1200, 24)]

    def solve(self, **kw):
        params = dict(across_span=160, depth_span=320, wall=3,
                      max_per_band=3, min_band_depth=56)
        params.update(kw)
        return exact_bands(self.ROOMS, **params)

    def test_every_room_lands_in_exactly_one_band(self):
        sol = self.solve()
        self.assertIsNotNone(sol)
        placed = [i for band in sol.bands for i in band]
        self.assertEqual(sorted(placed), list(range(len(self.ROOMS))))

    def test_depth_order_is_preserved(self):
        sol = self.solve()
        flat = [i for band in sol.bands for i in band]
        self.assertEqual(flat, sorted(flat))

    def test_depths_fill_the_plot_exactly(self):
        sol = self.solve()
        wall_total = (len(sol.bands) - 1) * 3
        self.assertEqual(sum(sol.depths) + wall_total, 320)

    def test_no_band_is_empty_or_overfull(self):
        sol = self.solve()
        for band in sol.bands:
            self.assertGreaterEqual(len(band), 1)
            self.assertLessEqual(len(band), 3)

    def test_every_band_can_hold_its_rooms(self):
        sol = self.solve()
        for band, depth in zip(sol.bands, sol.depths):
            area = sum(self.ROOMS[i].area_cells for i in band)
            self.assertGreaterEqual(depth * 160, area)
            for i in band:
                self.assertGreaterEqual(depth, self.ROOMS[i].min_side)

    def test_a_run_requirement_forces_a_deep_band(self):
        rooms = list(self.ROOMS) + [BandRoom(4160, 52, 80)]
        sol = exact_bands(rooms, across_span=160, depth_span=400, wall=3,
                          max_per_band=3, min_band_depth=56)
        self.assertIsNotNone(sol)
        for band, depth in zip(sol.bands, sol.depths):
            if len(rooms) - 1 in band:
                self.assertGreaterEqual(depth, 80)

    def test_is_deterministic(self):
        self.assertEqual(solution_signature(self.solve()),
                         solution_signature(self.solve()))

    def test_impossible_program_returns_none(self):
        huge = [BandRoom(90_000, 300) for _ in range(4)]
        self.assertIsNone(exact_bands(huge, across_span=80, depth_span=80,
                                      wall=3, max_per_band=2))

    def test_objective_actually_discriminates(self):
        """The first version minimized TOTAL slack, which is a constant for
        any feasible packing — every solution tied and the solver returned
        an arbitrary one (measured: -16 points on the harness). The minimax
        objective must give different packings different scores."""
        balanced = exact_bands(
            [BandRoom(2000, 24), BandRoom(2000, 24), BandRoom(2000, 24),
             BandRoom(2000, 24)],
            across_span=160, depth_span=200, wall=3, max_per_band=2)
        lopsided = exact_bands(
            [BandRoom(200, 24), BandRoom(200, 24), BandRoom(6000, 24),
             BandRoom(6000, 24)],
            across_span=160, depth_span=200, wall=3, max_per_band=2)
        self.assertIsNotNone(balanced)
        self.assertIsNotNone(lopsided)
        self.assertNotEqual(balanced.waste_cells, lopsided.waste_cells)


class TestEngineIntegration(unittest.TestCase):
    def test_default_is_off_so_the_solver_changes_nothing(self):
        from modules.step4_generate.engine.contracts import EngineConfig
        self.assertEqual(EngineConfig().cpsat_mode, "off")

    def test_off_and_repair_agree_where_greedy_succeeds(self):
        """`repair` may only fire where greedy RAISED. On a brief the greedy
        path handles, the two must produce the identical plan."""
        import logging

        from modules.step4_generate.engine.contracts import EngineConfig
        from modules.step4_generate.engine.orchestrator import Orchestrator
        from ml.harness.briefs import golden_briefs
        logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)
        brief = golden_briefs(k=3)[5]
        off = Orchestrator(config=EngineConfig(cpsat_mode="off"))
        repair = Orchestrator(config=EngineConfig(cpsat_mode="repair"))
        a, b = off.generate(brief), repair.generate(brief)
        self.assertEqual(a.best.plan.grid.tobytes(),
                         b.best.plan.grid.tobytes())


if __name__ == "__main__":
    unittest.main()
