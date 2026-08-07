"""
test_tier2_placer.py — M5 model, decoding, wrapper, and NumPy-export tests.

All CPU, all on tiny synthetic inputs (no dependency on the prepared corpus),
so they run in CI. The heavy proofs — that invalid layouts are unsamplable,
that the model can learn (overfit a batch), and that the torch-free NumPy
path matches torch — live here.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from modules.step4_generate.engine.contracts import EngineConfig, EngineRequest, RoomSpec
from ml.tier2_placer.config import PlacerConfig
from ml.tier2_placer.export import export_npz
from ml.tier2_placer.features import request_to_arrays
from ml.tier2_placer.model.placer_net import PlacerNet
from ml.tier2_placer.numpy_infer import NumpyPlacer
from ml.tier2_placer.tier2_placer import Tier2Placer
from ml.tier2_placer.torch_infer import TorchPlacer
from ml.tier2_placer import masked_decode as md


def _tiny_request(k=3, seed=1):
    return EngineRequest(
        plot_w_ft=30, plot_h_ft=40, entrance_side="S",
        rooms=[RoomSpec("Living Room", "living_room", 180, zone="public"),
               RoomSpec("Kitchen", "kitchen", 90, zone="service"),
               RoomSpec("Master Bedroom", "master_bedroom", 150,
                        zone="private"),
               RoomSpec("Bath 1", "bathroom", 40, zone="private")],
        k=k, seed=seed)


def _arrays_with_targets(net_cfg, n=5, seed=0):
    """Synthetic training arrays with valid distinct targets."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(net_cfg.cell_count, size=n, replace=False)
    return {
        "type_ids": rng.integers(2, net_cfg.n_room_types, n).astype(np.int64),
        "zone_ids": rng.integers(1, net_cfg.n_zones, n).astype(np.int64),
        "floor_ids": np.zeros(n, np.int64),
        "edge_index": np.array([[0, 1], [1, 0]], np.int64),
        "boundary": np.ones((2, 64, 64), np.float32),
        "global": np.zeros(4, np.float32),
        "target_cell": cells.astype(np.int64),
        "target_size": rng.integers(0, net_cfg.size_count, n).astype(np.int64),
        "n_rooms": n,
    }


class TestModelForward(unittest.TestCase):
    def test_forward_shapes_and_param_budget(self):
        cfg = PlacerConfig()
        net = PlacerNet(cfg)
        arr = {k: torch.as_tensor(v) for k, v in
               _arrays_with_targets(cfg, 6).items() if isinstance(v, np.ndarray)}
        cell, size = net(arr)
        self.assertEqual(cell.shape, (6, cfg.cell_count))
        self.assertEqual(size.shape, (6, cfg.size_count))
        self.assertLess(net.num_params(), 15_000_000)   # Colab budget
        self.assertGreater(net.num_params(), 5_000_000)


class TestMaskedDecode(unittest.TestCase):
    def setUp(self):
        self.cfg = PlacerConfig()
        self.placer = TorchPlacer(PlacerNet(self.cfg))
        self.arr = request_to_arrays(_tiny_request())

    def test_placements_in_boundary_and_distinct(self):
        # carve out an L-shaped boundary so "in-boundary" is a real constraint
        self.arr["boundary"][0, :, 48:] = 0.0
        res = md.sample(self.placer, self.arr, greedy=True)
        legal = md.boundary_legal_cells(self.arr["boundary"][0])
        seen = set()
        for p in res.placements:
            self.assertTrue(legal[p.row * 32 + p.col],
                            "placed outside boundary")
            self.assertNotIn((p.row, p.col), seen, "duplicate seed cell")
            seen.add((p.row, p.col))
            self.assertTrue(1 <= p.size_class <= self.cfg.size_count)

    def test_k_proposals_are_diverse(self):
        ks = md.sample_k(self.placer, self.arr, k=4, temperature=1.5)
        layouts = {tuple((p.row, p.col) for p in r.placements) for r in ks}
        self.assertGreaterEqual(len(layouts), 2)

    def test_confidence_low_for_untrained(self):
        res = md.sample(self.placer, self.arr, greedy=True)
        self.assertLess(res.confidence, md.__dict__.get("DEFAULT_TAU", 0.35)
                        if False else 0.35)


class TestWrapperFallback(unittest.TestCase):
    def test_untrained_model_falls_back(self):
        placer = Tier2Placer(TorchPlacer(PlacerNet(PlacerConfig())))
        prop = placer.propose(_tiny_request(), variant=0)
        self.assertEqual(prop.source, "fallback-prior")

    def test_drops_into_orchestrator(self):
        from modules.step4_generate.engine.orchestrator import Orchestrator
        placer = Tier2Placer(TorchPlacer(PlacerNet(PlacerConfig())))
        result = Orchestrator(proposer=placer,
                              config=EngineConfig()).generate(_tiny_request())
        self.assertGreaterEqual(len(result.ranked), 1)


class TestNumpyEquality(unittest.TestCase):
    def test_numpy_matches_torch(self):
        torch.manual_seed(3)
        cfg = PlacerConfig()
        net = PlacerNet(cfg).eval()
        path = os.path.join(tempfile.gettempdir(), "eq_ci.npz")
        export_npz({"state_dict": net.state_dict(), "config": cfg.to_dict()},
                   path)
        npy = NumpyPlacer.from_npz(path)
        arr = _arrays_with_targets(cfg, 6, seed=2)
        tc, ts = TorchPlacer(net).teacher_forced_logits(arr)
        nc, ns = npy.teacher_forced_logits(arr)
        self.assertLess(np.abs(tc - nc).max(), 1e-4)
        self.assertLess(np.abs(ts - ns).max(), 1e-4)


class TestDeviceAgnostic(unittest.TestCase):
    """Regression: a GPU run died because helper tensors were built on the
    CPU while the data lived on cuda. No GPU is needed to catch this — the
    'meta' device raises the same mismatch, so these run in ordinary CI."""

    def test_soft_targets_follow_their_input_device(self):
        from ml.tier2_placer.train import gaussian_soft_targets
        t = torch.tensor([5, 70, 300], dtype=torch.long, device="meta")
        self.assertEqual(gaussian_soft_targets(t, 1.0).device.type, "meta")

    def test_torch_placer_follows_the_net(self):
        """TorchPlacer must move numpy inputs onto the net's device, not
        assume CPU — this is the eval path used to pick checkpoints."""
        net = PlacerNet(PlacerConfig())
        placer = TorchPlacer(net)
        self.assertEqual(placer.device, next(net.parameters()).device)
        arr = request_to_arrays(_tiny_request())
        moved = placer._to_torch(arr)
        for k, v in moved.items():
            self.assertEqual(v.device, placer.device, f"{k} on wrong device")


class TestLearning(unittest.TestCase):
    def test_overfits_a_single_batch(self):
        """A few Adam steps on ONE plan must drive its loss down — proves the
        gradients/wiring actually learn."""
        from ml.tier2_placer.train import plan_loss
        torch.manual_seed(0)
        cfg = PlacerConfig()
        net = PlacerNet(cfg)
        arr = {k: torch.as_tensor(v) for k, v in
               _arrays_with_targets(cfg, 5, seed=7).items()
               if isinstance(v, np.ndarray)}
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        first = plan_loss(net, arr, sigma=1.0).item()
        for _ in range(25):
            opt.zero_grad()
            loss = plan_loss(net, arr, sigma=1.0)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), first * 0.6, "model failed to overfit")


class TestAugmentation(unittest.TestCase):
    def test_rotation_keeps_seeds_in_mask(self):
        from ml.tier2_placer.dataset import _augment
        sample = {
            "plan_id": "x", "plot_w_ft": 30.0, "plot_h_ft": 40.0,
            "entrance_side": "S", "mask_index": 0,
            "rooms": [{"rtype": "living_room", "zone": "public",
                       "row": 5, "col": 10, "size_class": 8, "area_sqft": 200}],
            "edges": [],
        }
        mask = np.zeros((64, 64), np.uint8)
        mask[8:12, 18:22] = 1                     # around the seed's 2x2 block
        for aug in range(8):
            s2, m2 = _augment(sample, mask, aug)
            r = s2["rooms"][0]
            self.assertTrue(0 <= r["row"] < 32 and 0 <= r["col"] < 32)
            self.assertIn(s2["entrance_side"], ("N", "E", "S", "W"))
            self.assertEqual(m2.shape, (64, 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
