"""
training — the M4 data pipeline for the Tier-2 placement transformer.

Turns real floor-plan corpora (CubiCasa5K, RPLAN) into the discrete
supervised targets the placer learns: per room, a seed cell on the 32x32
grid + a size class, conditioned on a 64x64 boundary raster and the program
graph. The v1 model died on continuous bbox targets that were themselves
13.4% overlapping (docs/plans/partition_first_redesign.md); this pipeline
produces DISCRETE targets and gates the output hard (0% seed collision,
seeds in-bounds, a committed eyeball render) before any training step runs.

Memory discipline: the source JSONs are hundreds of MB to multiple GB, so
everything streams (ijson) — no whole-file loads.

Scripts (each argument-driven, runnable, tested on fixtures):
    schema_audit.py   — inspect the real corpus schema + distribution report
    vocab.py          — corpus room type -> engine rtype (single source)
    prep_cubicasa.py  — CubiCasa5K normalized_extraction.json -> PreparedSamples
    gates.py          — the 5 hard quality gates
    render_samples.py — the mandatory 100-sample eyeball artifact
"""
