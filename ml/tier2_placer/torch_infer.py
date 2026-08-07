"""
torch_infer.py — adapter exposing the torch model through the decode
interface masked_decode expects (encode / decode_step, numpy logits).

Used two ways: to sample from the freshly-trained torch model during
engine-metric checkpoint selection, and as the reference the NumPy inference
path is unit-tested against (logit equality < 1e-4).
"""

from __future__ import annotations

import numpy as np
import torch

from ml.tier2_placer.config import PlacerConfig
from ml.tier2_placer.model.decoder import shift_prev
from ml.tier2_placer.model.placer_net import PlacerNet


class TorchPlacer:
    """Wraps a PlacerNet for constrained decoding on numpy arrays."""

    def __init__(self, net: PlacerNet):
        self.net = net.eval()
        self.cfg: PlacerConfig = net.cfg
        self.cell_count = self.cfg.cell_count
        self.size_count = self.cfg.size_count

    @property
    def device(self) -> torch.device:
        """Follow the net — eval runs on whatever device training is using."""
        return next(self.net.parameters()).device

    def _to_torch(self, arrays: dict) -> dict:
        dev = self.device
        out = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.as_tensor(v).to(dev)
        return out

    @torch.no_grad()
    def encode(self, arrays: dict):
        return self.net.encode(self._to_torch(arrays))      # opaque memory

    @torch.no_grad()
    def decode_step(self, memory, arrays: dict,
                    prev_cell: np.ndarray, prev_size: np.ndarray):
        """Run the decoder over the current prefix; return numpy logits."""
        t = self._to_torch(arrays)
        dev = self.device
        m = prev_cell.shape[0]
        x = self.net.decoder.step_input(
            t["type_ids"][:m], t["zone_ids"][:m],
            torch.as_tensor(prev_cell).to(dev),
            torch.as_tensor(prev_size).to(dev))
        cell_logits, size_logits = self.net.decoder(x, memory)
        return cell_logits.cpu().numpy(), size_logits.cpu().numpy()

    @torch.no_grad()
    def teacher_forced_logits(self, arrays: dict):
        """Full-sequence logits with ground-truth prev (for the export
        equality test). Returns (cell_logits, size_logits) numpy."""
        t = self._to_torch(arrays)
        cell_logits, size_logits = self.net(t)
        return cell_logits.cpu().numpy(), size_logits.cpu().numpy()
