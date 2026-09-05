"""
test_placer_v3.py — the redesigned placer's contracts.

These test the things that, when they broke in v2, broke SILENTLY: the state
stack's causality, parity between the state and the legality mask, the
augmentation agreeing with the geometry it augments, and the checkpoint
manager noticing that the eval set changed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

from ml.placer_v3 import config as cfgmod
from ml.placer_v3.state import (
    PlacementState, build_state_stack, claim_radius, footprint_from_boundary,
)


class TestClaimParity(unittest.TestCase):
    def test_claim_radius_matches_v2(self):
        """v3 redefines claim_radius rather than importing it, so the two
        must be asserted equal — a silent divergence would mean the state the
        model sees and the mask applied to it disagree."""
        from ml.tier2_placer.masked_decode import _claim_radius as v2_radius
        for size_class in range(1, cfgmod.SIZE_COUNT + 1):
            self.assertEqual(claim_radius(size_class), v2_radius(size_class),
                             f"size_class {size_class}")


class TestState(unittest.TestCase):
    def setUp(self):
        self.boundary = np.ones((64, 64), dtype=np.float32)

    def test_footprint_downsamples_to_seed_grid(self):
        fp = footprint_from_boundary(self.boundary)
        self.assertEqual(fp.shape, (cfgmod.STATE_GRID, cfgmod.STATE_GRID))
        self.assertTrue(fp.all())

    def test_degenerate_footprint_never_blocks_everything(self):
        fp = footprint_from_boundary(np.zeros((64, 64), dtype=np.float32))
        self.assertTrue(fp.all(), "an empty footprint must not dead-end")

    def test_state_is_causal(self):
        """state[i] must reflect rooms 0..i-1 and NEVER room i. If it leaked
        the current room the model would be told the answer at training time
        and be blind at inference."""
        cells = [100, 200, 300, 400]
        sizes = [10, 20, 30, 40]
        zones = [1, 3, 1, 3]
        stack = build_state_stack(self.boundary, "S", cells, sizes, zones)
        self.assertEqual(stack.shape[0], len(cells))

        # step 0 has claimed nothing
        self.assertEqual(stack[0, cfgmod.STATE_CH_CLAIMED].sum(), 0.0)
        # claims only ever grow
        claimed = [float(stack[i, cfgmod.STATE_CH_CLAIMED].sum())
                   for i in range(len(cells))]
        self.assertEqual(claimed, sorted(claimed))
        # the cell of room i is NOT claimed in its own state
        for i, cell in enumerate(cells):
            r, c = divmod(cell, cfgmod.STATE_GRID)
            self.assertEqual(stack[i, cfgmod.STATE_CH_CLAIMED, r, c], 0.0,
                             f"state[{i}] leaked room {i}")

    def test_zone_channels_separate(self):
        stack = build_state_stack(self.boundary, "S", [100, 500],
                                  [20, 20],
                                  [cfgmod.ZONE_TO_ID["public"],
                                   cfgmod.ZONE_TO_ID["private"]])
        # after the public room, the public plane is hot and private is not
        self.assertGreater(stack[1, cfgmod.STATE_CH_PUBLIC].sum(), 0.0)
        self.assertEqual(stack[1, cfgmod.STATE_CH_PRIVATE].sum(), 0.0)

    def test_legal_cells_agree_with_claims(self):
        state = PlacementState(self.boundary, "S")
        before = int(state.legal_cells().sum())
        state.place(16, 16, 40, cfgmod.ZONE_TO_ID["public"])
        after = int(state.legal_cells().sum())
        self.assertLess(after, before, "placing a room must consume cells")

    def test_empty_program(self):
        stack = build_state_stack(self.boundary, "S", [], [], [])
        self.assertEqual(stack.shape[0], 0)


class TestConfigVocab(unittest.TestCase):
    def test_aspect_buckets_ordered(self):
        self.assertEqual(cfgmod.aspect_class(10, 10), 0)
        self.assertEqual(cfgmod.aspect_class(10, 1), cfgmod.ASPECT_COUNT - 1)
        seen = [cfgmod.aspect_class(r, 1.0)
                for r in (1.0, 1.3, 1.6, 2.0, 2.5, 4.0)]
        self.assertEqual(seen, sorted(seen))

    def test_aspect_is_symmetric(self):
        self.assertEqual(cfgmod.aspect_class(20, 8),
                         cfgmod.aspect_class(8, 20))

    def test_orientation(self):
        self.assertEqual(cfgmod.orientation_class(10, 10), 0)   # square
        self.assertEqual(cfgmod.orientation_class(20, 8), 1)    # wide
        self.assertEqual(cfgmod.orientation_class(8, 20), 2)    # tall

    def test_regimes_partition_the_range(self):
        self.assertEqual(cfgmod.regime_for(400), "compact")
        self.assertEqual(cfgmod.regime_for(1200), "normal")
        self.assertEqual(cfgmod.regime_for(2400), "spacious")

    def test_task_contract_matches_engine(self):
        """The seed grid and size-class cap are shared with the engine. If
        they drift, every trained weight silently means something else."""
        from modules.step4_generate.engine.contracts import (
            SEED_GRID, SIZE_CLASS_MAX,
        )
        self.assertEqual(cfgmod.CELL_COUNT, SEED_GRID * SEED_GRID)
        self.assertEqual(cfgmod.SIZE_COUNT, SIZE_CLASS_MAX)
        self.assertEqual(cfgmod.STATE_GRID, SEED_GRID)


class TestModel(unittest.TestCase):
    def test_forward_shapes_and_size(self):
        import torch
        from ml.placer_v3.model.placer_net import PlacerNetV3

        cfg = cfgmod.PlacerV3Config()
        net = PlacerNetV3(cfg)
        self.assertLess(net.num_params(), 15e6, "small preset drifted")

        n = 6
        cells = np.arange(n) * 37
        sizes = np.arange(1, n + 1) * 3
        zones = np.array([1, 2, 3, 1, 2, 3])
        boundary = np.ones((2, 64, 64), np.float32)
        arrays = {
            "type_ids": torch.arange(2, 2 + n),
            "zone_ids": torch.as_tensor(zones),
            "floor_ids": torch.zeros(n, dtype=torch.long),
            "vastu_dir_ids": torch.arange(n) % cfg.n_vastu_dirs,
            "vastu_strength_ids": torch.zeros(n, dtype=torch.long),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "boundary": torch.as_tensor(boundary),
            "global": torch.zeros(cfg.global_dim),
            "state": torch.as_tensor(
                build_state_stack(boundary[0], "S", cells, sizes, zones)),
            "target_cell": torch.as_tensor(cells),
            "target_size": torch.as_tensor(sizes - 1),
        }
        with torch.no_grad():
            out = net(arrays)
        self.assertEqual(tuple(out["cell"].shape), (n, cfg.cell_count))
        self.assertEqual(tuple(out["size"].shape), (n, cfg.size_count))
        self.assertEqual(tuple(out["aspect"].shape), (n, cfg.aspect_count))
        self.assertEqual(tuple(out["orientation"].shape),
                         (n, cfg.orientation_count))
        self.assertEqual(tuple(out["band"].shape), (n, cfg.band_count))

    def test_gaussian_targets_are_distributions(self):
        import torch
        from ml.placer_v3.train import gaussian_cell_targets
        target = torch.tensor([0, 512, 1023])
        soft = gaussian_cell_targets(target)
        self.assertEqual(tuple(soft.shape), (3, cfgmod.CELL_COUNT))
        self.assertTrue(torch.allclose(soft.sum(dim=1), torch.ones(3),
                                       atol=1e-5))
        # the true cell carries the most mass
        for i, t in enumerate(target):
            self.assertEqual(int(soft[i].argmax()), int(t))


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="v3ckpt")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _tiny(self):
        import torch
        return torch.nn.Linear(4, 4)

    def test_save_resume_roundtrip(self):
        import torch
        from ml.placer_v3.checkpoint import (
            CheckpointManager, TrainState, eval_key_for,
        )
        model = self._tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr = CheckpointManager(self.dir, keep_last=2, milestone_every=0)
        key = eval_key_for("t", 10, 1)

        state = TrainState(epoch=5, best_score=42.0, best_epoch=3,
                           eval_key=key)
        mgr.save(model=model, optimizer=opt, scheduler=None, scaler=None,
                 state=state, config={}, is_best=True)
        self.assertTrue(os.path.exists(mgr.last_path))
        self.assertTrue(os.path.exists(mgr.best_path))

        fresh = self._tiny()
        restored = mgr.resume(model=fresh, optimizer=None, eval_key=key)
        self.assertEqual(restored.epoch, 5)
        self.assertEqual(restored.best_score, 42.0)
        for a, b in zip(model.parameters(), fresh.parameters()):
            self.assertTrue(torch.allclose(a, b))

    def test_eval_key_change_resets_best(self):
        """The v2 bug: --eval-n was raised mid-run, the mean fell for that
        reason alone, and best_model froze at epoch 2 forever."""
        import torch
        from ml.placer_v3.checkpoint import (
            CheckpointManager, TrainState, eval_key_for,
        )
        model = self._tiny()
        mgr = CheckpointManager(self.dir, milestone_every=0)
        mgr.save(model=model, optimizer=None, scheduler=None, scaler=None,
                 state=TrainState(epoch=3, best_score=88.0, best_epoch=2,
                                  eval_key=eval_key_for("e", 50, 1)),
                 config={}, is_best=True)

        restored = mgr.resume(model=self._tiny(),
                              eval_key=eval_key_for("e", 443, 1))
        self.assertEqual(restored.epoch, 3, "epoch must survive")
        self.assertEqual(restored.best_score, float("-inf"),
                         "best_score must reset when the ruler changes")

    def test_no_checkpoint_starts_clean(self):
        from ml.placer_v3.checkpoint import CheckpointManager
        mgr = CheckpointManager(os.path.join(self.dir, "empty"))
        state = mgr.resume(model=self._tiny(), eval_key="k")
        self.assertEqual(state.epoch, 0)
        self.assertEqual(state.best_score, float("-inf"))

    def test_corrupt_checkpoint_is_skipped(self):
        import torch
        from ml.placer_v3.checkpoint import CheckpointManager, TrainState
        model = self._tiny()
        mgr = CheckpointManager(self.dir, milestone_every=1)
        mgr.save(model=model, optimizer=None, scheduler=None, scaler=None,
                 state=TrainState(epoch=1), config={}, is_best=False)
        with open(mgr.last_path, "wb") as f:      # truncate the newest
            f.write(b"not a checkpoint")
        payload = mgr.load_latest()
        self.assertIsNotNone(payload, "must fall back to the milestone")
        self.assertEqual(payload["state"]["epoch"], 1)


class TestSelfPlay(unittest.TestCase):
    def test_brief_generation_is_deterministic(self):
        from ml.placer_v3.selfplay import brief_at
        a, b = brief_at(7), brief_at(7)
        self.assertEqual(a.plot_w_ft, b.plot_w_ft)
        self.assertEqual(a.entrance_side, b.entrance_side)
        self.assertEqual([r.name for r in a.rooms], [r.name for r in b.rooms])

    def test_briefs_span_the_regimes(self):
        from ml.placer_v3.selfplay import brief_at
        regimes = {cfgmod.regime_for(brief_at(i).plot_w_ft
                                     * brief_at(i).plot_h_ft)
                   for i in range(120)}
        self.assertEqual(regimes, {"compact", "normal", "spacious"},
                         "the corpus must cover all three scale regimes")

    def test_programs_stay_within_max_rooms(self):
        from ml.placer_v3.selfplay import brief_at
        for i in range(200):
            self.assertLessEqual(len(brief_at(i).rooms), cfgmod.MAX_ROOMS)


class TestFeatures(unittest.TestCase):
    def test_request_arrays_shapes(self):
        from modules.step4_generate.engine.contracts import (
            EngineRequest, RoomSpec,
        )
        from ml.placer_v3.features import request_to_arrays

        req = EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S",
            rooms=[RoomSpec("Living Room", "living_room", 200, "public"),
                   RoomSpec("Kitchen", "kitchen", 100, "service"),
                   RoomSpec("Bed", "master_bedroom", 150, "private")])
        arrays = request_to_arrays(req)
        self.assertEqual(arrays["n_rooms"], 3)
        self.assertEqual(arrays["boundary"].shape, (2, 64, 64))
        self.assertEqual(arrays["global"].shape, (cfgmod.GLOBAL_DIM,))
        self.assertEqual(len(arrays["vastu_dir_ids"]), 3)
        self.assertTrue((arrays["vastu_dir_ids"] == 0).all(),
                        "no vastu supplied -> 'none', not a missing value")

    def test_vastu_conditioning_reaches_the_arrays(self):
        from modules.step4_generate.engine.contracts import (
            EngineRequest, RoomSpec,
        )
        from ml.placer_v3.features import request_to_arrays

        req = EngineRequest(
            plot_w_ft=30, plot_h_ft=40, entrance_side="S",
            rooms=[RoomSpec("Pooja Room", "pooja_room", 40, "public"),
                   RoomSpec("Kitchen", "kitchen", 100, "service")])
        arrays = request_to_arrays(
            req, vastu={"Pooja Room": {"direction": "NE", "strength": "hard"},
                        "Kitchen": {"direction": "SE", "strength": "soft"}})
        by_name = {s.name: i for i, s in enumerate(arrays["specs"])}
        self.assertEqual(arrays["vastu_dir_ids"][by_name["Pooja Room"]],
                         cfgmod.VASTU_DIR_TO_ID["NE"])
        self.assertEqual(arrays["vastu_strength_ids"][by_name["Kitchen"]],
                         cfgmod.VASTU_STRENGTH_TO_ID["soft"])

    def test_global_encodes_regime(self):
        from ml.placer_v3.features import build_global
        compact = build_global(20, 25, 6)      # 500 sqft
        spacious = build_global(50, 50, 6)     # 2500 sqft
        self.assertEqual(list(compact[4:7]), [1.0, 0.0, 0.0])
        self.assertEqual(list(spacious[4:7]), [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
