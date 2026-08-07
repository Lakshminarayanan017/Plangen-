"""
placer_net.py — the full Tier-2 model: encoders + AR decoder.

Training forward (teacher-forced) returns per-room cell & size logits for a
single plan. The training loop (tier2_placer.train) loops plans in a
minibatch and sums the cross-entropy losses. Inference sampling lives in
tier2_placer.masked_decode, which calls the pieces here step by step with
legality masks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ml.tier2_placer.config import PlacerConfig
from ml.tier2_placer.model.decoder import PlacementDecoder, shift_prev
from ml.tier2_placer.model.encoder import BoundaryEncoder, ProgramEncoder


class PlacerNet(nn.Module):
    def __init__(self, cfg: PlacerConfig = None):
        super().__init__()
        self.cfg = cfg or PlacerConfig()
        self.program = ProgramEncoder(self.cfg)
        self.boundary = BoundaryEncoder(self.cfg)
        self.decoder = PlacementDecoder(self.cfg)

    def encode(self, arrays: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the cross-attention memory (program ‖ boundary ‖ global)."""
        prog = self.program(arrays["type_ids"], arrays["zone_ids"],
                            arrays["floor_ids"], arrays["edge_index"])
        bnd = self.boundary(arrays["boundary"], arrays["global"])
        return torch.cat([prog, bnd], dim=0)              # (N+65, d_model)

    def forward(self, arrays: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced training forward → (cell_logits, size_logits)."""
        memory = self.encode(arrays)
        prev_cell, prev_size = shift_prev(
            arrays["target_cell"], arrays["target_size"],
            self.cfg.cell_count, self.cfg.size_count)
        x = self.decoder.step_input(
            arrays["type_ids"], arrays["zone_ids"], prev_cell, prev_size)
        return self.decoder(x, memory)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
