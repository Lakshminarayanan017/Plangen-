"""
Irregular plots and CAD export (M8).

M8's gate: irregular-plot briefs pass the hard gate. The geometry claims —
rasterized area converging on the true area, the inscribed rectangle really
being the largest — are checked against cases whose answers are known
independently, not against the implementation's own output.
"""

import logging
import os
import tempfile
import unittest

import numpy as np

from modules.step4_generate.core import polygon as poly
from modules.step4_generate.core import units
from modules.step4_generate.core.grid_plan import CarveError, GridPlan
from modules.step4_generate.engine.contracts import EngineRequest, EngineRequestError, RoomSpec
from modules.step4_generate.engine.orchestrator import Orchestrator
from modules.step4_generate.engine.site import build_site, core_request, site_for
from ml.harness.briefs import polygon_briefs
from modules.step4_generate.render.dxf_export import DxfExporter, dxf_string, export_dxf

logging.getLogger("PlanGen.Engine").setLevel(logging.ERROR)

SQUARE = [(0, 0), (40, 0), (40, 40), (0, 40)]
L_SHAPE = [(0, 0), (40, 0), (40, 30), (22, 30), (22, 50), (0, 50)]
CHAMFER = [(0, 0), (35, 0), (35, 38), (27, 45), (0, 45)]


class TestPolygonGeometry(unittest.TestCase):
    def test_shoelace_area(self):
        self.assertAlmostEqual(poly.signed_area_sqft(SQUARE), 1600.0)
        # the L is a 40x30 leg plus a 22x20 leg
        self.assertAlmostEqual(abs(poly.signed_area_sqft(L_SHAPE)),
                               40 * 30 + 22 * 20)

    def test_winding_is_normalized(self):
        ccw = poly.normalize(SQUARE)
        cw = poly.normalize(list(reversed(SQUARE)))
        self.assertGreater(poly.signed_area_sqft(ccw), 0)
        self.assertGreater(poly.signed_area_sqft(cw), 0)

    def test_a_closing_duplicate_point_is_dropped(self):
        self.assertEqual(len(poly.normalize(SQUARE + [SQUARE[0]])), 4)

    def test_degenerate_rings_are_refused(self):
        with self.assertRaises(poly.PolygonError):
            poly.normalize([(0, 0), (1, 1)])
        with self.assertRaises(poly.PolygonError):
            poly.normalize([(0, 0), (1, 0), (2, 0)])

    def test_rasterized_area_matches_the_true_area(self):
        for ring in (SQUARE, L_SHAPE, CHAMFER):
            norm = poly.normalize(ring)
            w, h, ox, oy = poly.bbox_cells(norm)
            mask = poly.rasterize(norm, w, h, origin=(ox, oy))
            got = units.area_sqft(int(mask.sum()))
            want = abs(poly.signed_area_sqft(norm))
            self.assertLess(abs(got - want) / want, 0.01, ring)

    def test_erode_shrinks_by_the_requested_ring(self):
        mask = np.ones((40, 40), dtype=bool)
        self.assertEqual(int(poly.erode(mask, 1).sum()), 38 * 38)
        self.assertEqual(int(poly.erode(mask, 3).sum()), 34 * 34)

    def test_largest_inscribed_rectangle_is_exact(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 1:5] = True            # 6 rows x 4 cols = 24
        mask[3:5, 5:9] = True            # 2 rows x 4 cols, overlapping band
        rect = poly.largest_inscribed_rectangle(mask)
        x0, y0, x1, y1 = rect
        self.assertTrue(mask[y0:y1, x0:x1].all())
        self.assertEqual((x1 - x0) * (y1 - y0), 24)

    def test_inscribed_rectangle_of_an_empty_mask_is_none(self):
        self.assertIsNone(poly.largest_inscribed_rectangle(
            np.zeros((5, 5), dtype=bool)))

    def test_connected_components_separate_disjoint_pieces(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:3, 0:3] = True
        mask[7:10, 7:10] = True
        pieces = poly.connected_components(mask)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(sum(int(p.sum()) for p in pieces), 18)


class TestMaskedLattice(unittest.TestCase):
    """GridPlan.from_polygon — the site model."""

    def test_the_plot_becomes_a_core_plus_open_space(self):
        plan = GridPlan.from_polygon(L_SHAPE)
        self.assertIsNotNone(plan.buildable_core)
        self.assertTrue(plan.verify())
        types = {room.rtype for room in plan.rooms.values()}
        self.assertIn("ots", types)

    def test_the_core_is_rectangular(self):
        plan = GridPlan.from_polygon(L_SHAPE)
        self.assertTrue(plan.face_is_rect(1))

    def test_cells_outside_the_boundary_are_outside(self):
        from modules.step4_generate.core.grid_plan import OUTSIDE
        plan = GridPlan.from_polygon(L_SHAPE)
        # the notch of the L must be empty
        self.assertEqual(int(plan.grid[units.cells(45), units.cells(35)]),
                         OUTSIDE)

    def test_a_plot_with_no_buildable_rectangle_is_refused(self):
        sliver = [(0, 0), (60, 0), (60, 4), (0, 4)]
        with self.assertRaises(CarveError):
            GridPlan.from_polygon(sliver)


class TestSite(unittest.TestCase):
    def test_core_is_inside_the_plot_and_smaller(self):
        site = build_site(L_SHAPE, setback_ft=3.0)
        self.assertLess(site.core_area_sqft, site.plot_area_sqft)
        self.assertGreater(site.coverage, 0.2)
        self.assertLess(site.coverage, 1.0)

    def test_a_bigger_setback_shrinks_the_core(self):
        small = build_site(L_SHAPE, setback_ft=0.0).core_area_sqft
        large = build_site(L_SHAPE, setback_ft=6.0).core_area_sqft
        self.assertLess(large, small)

    def test_open_space_is_the_remainder(self):
        site = build_site(CHAMFER, setback_ft=3.0)
        self.assertAlmostEqual(
            site.open_space_sqft + site.core_area_sqft,
            site.plot_area_sqft, places=3)

    def test_an_unbuildable_plot_raises_a_named_error(self):
        with self.assertRaises(EngineRequestError):
            build_site([(0, 0), (60, 0), (60, 4), (0, 4)])

    def test_core_request_is_rectangular(self):
        request = EngineRequest(
            plot_w_ft=40, plot_h_ft=50, entrance_side="S",
            rooms=[RoomSpec("Living Room", "living_room", 200,
                            zone="public")],
            plot_polygon=L_SHAPE, setback_ft=3.0)
        site = site_for(request)
        core = core_request(request, site)
        self.assertIsNone(core.plot_polygon)
        self.assertLessEqual(core.plot_w_ft, request.plot_w_ft)
        self.assertLessEqual(core.plot_h_ft, request.plot_h_ft)

    def test_rectangular_requests_have_no_site(self):
        self.assertIsNone(site_for(EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S",
            rooms=[RoomSpec("Living Room", "living_room", 200,
                            zone="public")])))


class TestIrregularBriefsGate(unittest.TestCase):
    """M8's gate: every irregular brief passes the hard gate."""

    @classmethod
    def setUpClass(cls):
        orch = Orchestrator()
        cls.results = [(b, orch.generate(b)) for b in polygon_briefs(k=4)]

    def test_every_irregular_brief_produces_a_plan(self):
        failed = [b.name for b, r in self.results if not r.best]
        self.assertEqual(failed, [])

    def test_no_hard_violation_in_a_kept_candidate(self):
        for brief, result in self.results:
            for cand in result.ranked:
                self.assertEqual(cand.verdict.hard, [], brief.name)

    def test_the_building_fits_inside_its_envelope(self):
        for brief, result in self.results:
            site = site_for(brief)
            plan = result.best.plan
            self.assertLessEqual(plan.w / units.CELLS_PER_FOOT,
                                 site.core_w_ft + 1e-6, brief.name)
            self.assertLessEqual(plan.h / units.CELLS_PER_FOOT,
                                 site.core_h_ft + 1e-6, brief.name)

    def test_the_site_is_reported_to_the_caller(self):
        for brief, result in self.results:
            self.assertTrue(any("buildable core" in w
                                for w in result.warnings), brief.name)


class TestDxfExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        brief, result = polygon_briefs(k=3)[0], None
        result = Orchestrator().generate(brief)
        cls.plan = result.best.plan
        cls.site = site_for(brief)

    def _pairs(self, text):
        lines = text.splitlines()
        return [(lines[i].strip(), lines[i + 1].strip())
                for i in range(0, len(lines) - 1, 2)]

    def test_structure_is_a_valid_r12_document(self):
        text = dxf_string(self.plan)
        pairs = self._pairs(text)
        codes = [c for c, _ in pairs]
        self.assertEqual(len(codes), len(pairs))       # even number of lines
        values = [v for c, v in pairs if c == "0"]
        self.assertEqual(values.count("SECTION"), 3)
        self.assertEqual(values.count("ENDSEC"), 3)
        self.assertEqual(values[-1], "EOF")
        self.assertIn(("1", "AC1009"), pairs)

    def test_every_layer_is_declared(self):
        from modules.step4_generate.render.dxf_export import LAYERS
        pairs = self._pairs(dxf_string(self.plan))
        declared = {v for c, v in pairs if c == "2"}
        for layer in LAYERS:
            self.assertIn(layer, declared)

    def test_entities_are_emitted(self):
        pairs = self._pairs(dxf_string(self.plan))
        kinds = [v for c, v in pairs if c == "0"]
        self.assertGreater(kinds.count("LINE"), 20)
        self.assertGreater(kinds.count("TEXT"), 3)

    def test_units_change_the_coordinates(self):
        ft = dxf_string(self.plan, unit="ft")
        mm = dxf_string(self.plan, unit="mm")
        self.assertNotEqual(ft, mm)
        self.assertIn(("70", "2"), self._pairs(ft))     # $INSUNITS feet
        self.assertIn(("70", "4"), self._pairs(mm))     # $INSUNITS mm

    def test_an_unknown_unit_is_refused(self):
        with self.assertRaises(ValueError):
            DxfExporter(self.plan, unit="furlongs")

    def test_the_plot_boundary_is_drawn_when_a_site_is_given(self):
        without = dxf_string(self.plan)
        with_site = dxf_string(self.plan, site=self.site,
                               origin_ft=self.site.core_offset_ft())
        self.assertNotIn("PLOT", [v for c, v in self._pairs(without)
                                  if c == "8"])
        self.assertIn("PLOT", [v for c, v in self._pairs(with_site)
                               if c == "8"])

    def test_text_values_never_break_the_pairing(self):
        pairs = self._pairs(dxf_string(self.plan))
        for code, value in pairs:
            self.assertNotIn("\n", value)
            self.assertNotIn('"', value)

    def test_writes_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = export_dxf(self.plan, os.path.join(tmp, "a", "plan.dxf"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main()
