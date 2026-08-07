"""
tier2_placer — the Tier-2 placement transformer (M5) behind the
LayoutProposal contract.

The model decides DISCRETE things — for each room, in canonical generation
order, a seed cell on the 32x32 grid + a size class — conditioned on a 64x64
boundary raster and the program graph. A downstream deterministic carver
(engine/realizer.py) turns those decisions into exact geometry, so the v1
failure mode (overlapping continuous boxes) is structurally impossible.

Production inference has NO torch dependency: the trained weights are
exported to .npz and run through a pure-NumPy path (numpy_infer.py) that is
unit-tested for logit equality with the torch model. torch is a
DEV/TRAIN-only dependency (training/requirements-train.txt).

`tier2_placer.py::Tier2Placer` is the drop-in `propose(request, variant) ->
LayoutProposal` replacement for engine.fallbacks.PriorProposer, with a
confidence-threshold fallback to it. It is merged into the engine only after
beating the fallback on the golden harness (the merge gate).
"""
