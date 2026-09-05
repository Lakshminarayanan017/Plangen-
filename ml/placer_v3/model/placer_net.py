"""
placer_net.py — encoders + decoder, as one model.

Two entry points, deliberately sharing every weight and every builder:

  forward(arrays)              teacher-forced training over a whole plan
  encode/decode_step(...)      one step at a time, for constrained sampling

v2 kept those two paths further apart than it should have, and the drift is
what hid the confidence-ceiling bug for a whole training run. Here the
sampler calls the SAME modules with a one-room state slice, so anything that
works in training works at inference or fails loudly in both.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ml.placer_v3.config import PlacerV3Config
from ml.placer_v3.model.decoder import PlacementDecoder, shift_prev
from ml.placer_v3.model.encoder import (
    BoundaryEncoder, ProgramEncoder, StateEncoder,
)


class PlacerNetV3(nn.Module):
    def __init__(self, cfg: Optional[PlacerV3Config] = None):
        super().__init__()
        self.cfg = cfg or PlacerV3Config()
        self.program = ProgramEncoder(self.cfg)
        self.boundary = BoundaryEncoder(self.cfg)
        self.state = StateEncoder(self.cfg)
        self.decoder = PlacementDecoder(self.cfg)

    # ── static memory (computed once per plan) ──────────────────────────
    def encode(self, a: Dict[str, torch.Tensor]) -> torch.Tensor:
        prog = self.program(a["type_ids"], a["zone_ids"], a["floor_ids"],
                            a["vastu_dir_ids"], a["vastu_strength_ids"],
                            a["edge_index"])
        bnd = self.boundary(a["boundary"], a["global"])
        return torch.cat([prog, bnd], dim=0)

    # ── training ────────────────────────────────────────────────────────
    def forward(self, a: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        memory = self.encode(a)
        state_tokens = self.state(a["state"])
        prev_cell, prev_size = shift_prev(
            a["target_cell"], a["target_size"],
            self.cfg.cell_count, self.cfg.size_count)
        x = self.decoder.step_input(a["type_ids"], a["zone_ids"],
                                    a["vastu_dir_ids"], prev_cell, prev_size)
        return self.decoder(x, memory, state_tokens)

    # ── sampling ────────────────────────────────────────────────────────
    def decode_prefix(self, memory: torch.Tensor,
                      a: Dict[str, torch.Tensor],
                      prev_cell: torch.Tensor, prev_size: torch.Tensor,
                      state_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Logits for a prefix of length m.

        `state_stack` is (m, C, 32, 32) — the board before each of the first
        m rooms. The caller (sampler) grows it one row at a time, so the
        model never sees a state that depends on a room it has not placed.
        """
        m = prev_cell.size(0)
        state_tokens = self.state(state_stack)
        x = self.decoder.step_input(
            a["type_ids"][:m], a["zone_ids"][:m], a["vastu_dir_ids"][:m],
            prev_cell, prev_size)
        return self.decoder(x, memory, state_tokens)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_report(self) -> Dict[str, int]:
        """Parameters by block — the number to watch when deciding whether
        R5 (more capacity) is warranted."""
        def count(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters())
        return {
            "program_encoder": count(self.program),
            "boundary_encoder": count(self.boundary),
            "state_encoder": count(self.state),
            "decoder": count(self.decoder),
            "total": self.num_params(),
        }
