"""
ml/
===
Training-time machinery. Nothing here is imported by the request path
(api/ → modules/step1..step4); the running app only ever loads the *weights*
these packages produce.

    tier2_placer/  the Tier-2 seed proposer: model, train, export, inference
    training/      CubiCasa5K -> discrete training samples (prep, gates, vocab)
    tuning/        CMA-ES tuning of the selector weights
    harness/       golden briefs + A/B harness used to judge any of the above
    data/          bulk corpora (gitignored)
    notebooks/     Colab notebooks

Run from the project root, e.g.:
    python -m ml.harness.run_harness
    python -m ml.tuning.cli
"""
