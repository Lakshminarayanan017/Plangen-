"""
critic — Tier 5: the learned second opinion (implementation_plan_v2.md §4.3).

  features.py     what the critic may look at (rule breakdown + geometry)
  perturb.py      controlled damage — the negatives it learns from
  dataset.py      corpus assembly, split BY BRIEF
  gbt.py          gradient-boosted trees in NumPy (no ML dependency)
  critic.py       LearnedCritic + the ranking blend
  preferences.py  append-only log of real user picks
  train.py        the CLI, and the report that decides whether it ships

Nothing here can reject a plan. The rules decide legality; the critic only
reorders what the rules already accepted, and only when trained weights
exist.
"""

from modules.step4_generate.critic.critic import LearnedCritic, blended_score      # noqa: F401

__all__ = ["LearnedCritic", "blended_score"]
