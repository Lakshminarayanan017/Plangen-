"""
diagnostics.py
==============
One honest answer to the question "what is actually working right now?"

Why this module exists
----------------------
Every knowledge source in PlanGen fails soft. A missing index returns `{}`,
a missing rule file returns `{}`, an unavailable model falls back to a
deterministic proposer. Individually that is good engineering — a user asking
for a floor plan should get a floor plan, not a stack trace.

Collectively it produced a system that reported success while running on
constants: step 2 was retrieving from an index that had never been built,
Vastu was silently disabled on every request, and nothing anywhere said so.
The failure was not that the fallbacks existed. It was that they were
invisible.

So: nothing here changes behaviour. It only makes the current state legible,
at startup, over the API, and in the UI.

Usage
-----
    from modules.diagnostics import self_check
    report = self_check()
    report["healthy"]        # bool — no MISSING checks
    report["checks"]         # list of per-subsystem results
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "plan_index"

# Status vocabulary, worst last — the overall verdict is the worst seen.
OK = "ok"             # working as designed
DEGRADED = "degraded"  # running, but on a fallback rather than the real thing
MISSING = "missing"    # a capability the product claims to have is absent

_SEVERITY = {OK: 0, DEGRADED: 1, MISSING: 2}


@dataclass
class Check:
    """One subsystem's health, in terms a human can act on."""
    name: str
    status: str
    detail: str
    impact: str = ""          # what the user actually loses, in plain words
    remedy: str = ""          # what would fix it
    meta: Dict[str, Any] = field(default_factory=dict)


# ── individual checks ────────────────────────────────────────────────────────

_INDEX_FILES = [
    "plan_vectors.npy",
    "plan_metadata.json",
    "index_metadata.json",
    "room_stats_by_bhk.json",
    "adjacency_weights.json",
    "zone_probs.json",
    "circulation_meta.json",
]


def _check_plan_index() -> Check:
    present = [f for f in _INDEX_FILES if (INDEX_DIR / f).exists()]
    missing = [f for f in _INDEX_FILES if f not in present]

    if not INDEX_DIR.exists() or not present:
        return Check(
            name="step2_plan_index",
            status=MISSING,
            detail=f"{INDEX_DIR} has none of the {len(_INDEX_FILES)} index files.",
            impact="Step 2 retrieves 0 reference plans. Room sizes, adjacency "
                   "weights and zone probabilities all fall back to NBC "
                   "constants, so the corpus contributes nothing.",
            remedy="python -m modules.data_prep.plan_indexer",
            meta={"present": present, "missing": missing},
        )
    if missing:
        return Check(
            name="step2_plan_index",
            status=DEGRADED,
            detail=f"{len(present)}/{len(_INDEX_FILES)} index files present.",
            impact="Parts of step 2 fall back to constants.",
            remedy="python -m modules.data_prep.plan_indexer",
            meta={"present": present, "missing": missing},
        )

    n = None
    try:
        import json
        meta = json.loads((INDEX_DIR / "index_metadata.json").read_text("utf-8"))
        n = meta.get("n_plans")
    except Exception:
        pass
    return Check(
        name="step2_plan_index",
        status=OK,
        detail=f"Index complete{f' — {n} plans' if n else ''}.",
        meta={"n_plans": n},
    )


def _check_zone_priors() -> Check:
    p = DATA_DIR / "zone_patterns.json"
    if not p.exists():
        return Check(
            name="engine_zone_priors",
            status=DEGRADED,
            detail=f"{p} not found.",
            impact="The carver places rooms using built-in constants instead of "
                   "median positions measured from real plans. Layouts still "
                   "generate, but lose their statistical grounding.",
            remedy="Restore data/zone_patterns.json.",
        )
    return Check(name="engine_zone_priors", status=OK,
                 detail=f"Loaded ({p.stat().st_size // 1024} KB).")


def _check_vastu() -> Check:
    try:
        from sources.rule_loader import rules
        v = rules.get_vastu_rules()
        n = len(v.get("room_zone_rules", []))
    except Exception as exc:                                  # pragma: no cover
        return Check(name="vastu_rules", status=MISSING,
                     detail=f"Rule loader raised {exc!r}.",
                     impact="Vastu-compliant requests silently return "
                            "non-Vastu plans.",
                     remedy="Check sources/enricher_rules.json.")
    if n == 0:
        return Check(
            name="vastu_rules", status=MISSING,
            detail="No room_zone_rules could be built.",
            impact="Requests asking for Vastu compliance return plans with no "
                   "Vastu adjustment at all.",
            remedy="Populate zone_rules in sources/enricher_rules.json.",
        )
    return Check(name="vastu_rules", status=OK,
                 detail=f"{n} room rules from {v.get('source')}.",
                 meta={"n_rules": n, "strength": "soft"})


def _check_vastu_mandala() -> Check:
    """The 81-pada mandala behind the entrance-pada, Brahmasthan and marma
    rules. Separate from `vastu_rules` (per-room compass, which lives in
    enricher_rules.json) because they fail independently: you can have room
    directions with no mandala, and the product should say which."""
    try:
        from modules.step4_generate.engine.vastu import mandala as mod
        m = mod.shared()
    except Exception as exc:                                  # pragma: no cover
        return Check(name="vastu_mandala", status=MISSING,
                     detail=f"Loader raised {exc!r}.",
                     impact="Entrance-pada, Brahmasthan and marma rules do "
                            "not run. Per-room compass placement still does.",
                     remedy="Check data/vastuRules1.json.")
    if not m.available:
        return Check(
            name="vastu_mandala", status=DEGRADED,
            detail="data/vastuRules1.json not loaded.",
            impact="The 32-gate entrance check, the Brahmasthan protection "
                   "and the marma diagonals are all skipped. Room-sector "
                   "placement still works from enricher_rules.json.",
            remedy="Restore data/vastuRules1.json.")
    report = m.load_report()
    status = DEGRADED if report["warnings"] else OK
    detail = (f"{report['gates']} gates ({report['ideal_gates']} ideal, "
              f"{report['hard_block_gates']} blocked), "
              f"{report['energy_fields']} energy fields, "
              f"{report['marma_lines']} marma lines.")
    return Check(
        name="vastu_mandala", status=status, detail=detail,
        impact=("" if status == OK else
                f"{len(report['warnings'])} internal inconsistency(ies) in "
                f"the mandala data; endpoints resolved by id."),
        remedy=("" if status == OK else
                "Correct the pada names in data/vastuRules1.json."),
        meta=report)


def _check_reward_freeze() -> Check:
    """Is the live reward still the one the training corpora were graded by?

    Stage (b) distillation imitates plans this reward chose, and stage (c)
    optimises it directly. A weight retuned after a corpus was built makes
    that corpus quietly incomparable — the same class of bug as v2's
    mid-run eval-set change, caught here instead of suffered."""
    try:
        from modules.step4_generate.engine import reward as mod
        drift = mod.verify()
        frozen = mod.load()
    except Exception as exc:                                  # pragma: no cover
        return Check(name="reward_freeze", status=DEGRADED,
                     detail=f"Check failed: {exc!r}",
                     impact="Reward drift would go undetected.")
    if frozen is None:
        return Check(
            name="reward_freeze", status=DEGRADED,
            detail="No frozen reward snapshot.",
            impact="Training corpora carry no record of which reward graded "
                   "them, so a later weight change is undetectable.",
            remedy="python -m modules.step4_generate.engine.reward --freeze")
    if drift.matches:
        base = frozen.get("harness_baseline") or {}
        return Check(
            name="reward_freeze", status=OK,
            detail=(f"{frozen.get('version')} {drift.frozen_fingerprint} — "
                    f"{len(frozen.get('rules', []))} rules, "
                    f"{len(frozen.get('weights', {}))} weights"
                    + (f", baseline mean {base['mean_best_score']}"
                       if base.get("mean_best_score") else "")),
            meta={"fingerprint": drift.frozen_fingerprint})
    return Check(
        name="reward_freeze", status=DEGRADED,
        detail=f"Live reward {drift.live_fingerprint} != frozen "
               f"{drift.frozen_fingerprint}.",
        impact="Plans scored now are not comparable to the training corpora; "
               "re-gate before trusting any merge-gate result.",
        remedy="Restore the frozen weights, or re-freeze and rebuild the "
               "self-play corpus.",
        meta={"rules_added": drift.rules_added,
              "weights_changed": {k: list(v) for k, v
                                  in drift.weights_changed.items()}})


def _check_program_synth() -> Check:
    """The layer that decides what belongs on a plot. Pure code with a
    measured table — it cannot be "missing", but a broken import would
    silently return the pre-phase-02 behaviour, so it is checked."""
    try:
        from modules.step3_enrich.program_synth import (
            capacity_for, max_sqft_for, regime_for,
        )
        tiers = [capacity_for(a)[0] for a in (250, 500, 1300, 3000)]
        capped = max_sqft_for("bathroom", 45)
    except Exception as exc:                                  # pragma: no cover
        return Check(name="program_synth", status=MISSING,
                     detail=f"Import failed: {exc!r}",
                     impact="Room programs are neither adapted to the plot "
                            "nor capped, so a large plot inflates every room "
                            "by one uniform factor.",
                     remedy="Check modules/step3_enrich/program_synth.py.")
    return Check(
        name="program_synth", status=OK,
        detail=f"Capacity curve live ({' / '.join(tiers)}); "
               f"a 45 sqft bath is capped at {capped:.0f}.",
        meta={"tiers": tiers})


def _check_rule_book() -> Check:
    p = PROJECT_ROOT / "sources" / "enricher_rules.json"
    if not p.exists():
        return Check(name="rule_book", status=MISSING,
                     detail=f"{p} not found.",
                     impact="Sizes, aliases, adjacency and floor rules all fall "
                            "back to hard-coded values in the enricher.",
                     remedy="Restore sources/enricher_rules.json.")
    try:
        from sources.rule_loader import rules
        n = len(rules.raw("zone_rules") or {})
        return Check(name="rule_book", status=OK,
                     detail=f"Loaded — {n} zone rules.")
    except Exception as exc:                                  # pragma: no cover
        return Check(name="rule_book", status=MISSING,
                     detail=f"Failed to parse: {exc!r}",
                     impact="Enricher runs entirely on hard-coded fallbacks.",
                     remedy="Validate the JSON.")


def _check_nbc_plot_regs() -> Check:
    p = DATA_DIR / "nbc_plot_regulations.json"
    if not p.exists():
        return Check(
            name="nbc_plot_regulations", status=DEGRADED,
            detail=f"{p} not found.",
            impact="Plot-level FAR/coverage regulations fall back to the "
                   "defaults in enricher_rules.json rather than per-city NBC "
                   "tables.",
            remedy="Add data/nbc_plot_regulations.json.",
        )
    return Check(name="nbc_plot_regulations", status=OK, detail="Loaded.")


def _check_critic() -> Check:
    try:
        from modules.step4_generate.critic.critic import DEFAULT_MODEL_PATH
    except Exception as exc:                                  # pragma: no cover
        return Check(name="learned_critic", status=DEGRADED,
                     detail=f"Import failed: {exc!r}",
                     impact="Candidates ranked by the hand-tuned score only.")
    if not os.path.exists(DEFAULT_MODEL_PATH):
        return Check(
            name="learned_critic", status=DEGRADED,
            detail="critic_gbt.json not found.",
            impact="Candidate plans are ranked by the rule-based score alone, "
                   "without the learned preference model.",
            remedy="python -m modules.step4_generate.critic.train",
        )
    return Check(name="learned_critic", status=OK, detail="Weights present.")


def _check_tier2_placer() -> Check:
    """Not wired in by design — the trained placer failed its merge gate."""
    w = PROJECT_ROOT / "ml" / "tier2_placer" / "weights"
    npz = sorted(w.glob("*.npz")) if w.exists() else []
    return Check(
        name="tier2_placer",
        status=OK,          # intentionally off is not a fault
        detail=("Not wired into the request path (by design). "
                f"{len(npz)} weight file(s) on disk."),
        impact="Seeds come from the deterministic PriorProposer. The trained "
               "placer regressed mean score 75.5 -> 68.6 on the golden "
               "harness, so it stays off until it passes the gate.",
        remedy="python -m ml.tier2_placer.merge_gate  (after retraining)",
        meta={"weights": [p.name for p in npz]},
    )


def _check_llm_keys() -> Check:
    # The parser loads sources/.env via dotenv at import time; do the same here
    # so the check reports what the running app sees, not what a bare shell has.
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / "sources" / ".env")
    except Exception:                                         # pragma: no cover
        pass
    keys = [k for k in os.environ if k.startswith("GEMINI_API_KEY")]
    if not keys:
        return Check(
            name="llm_keys", status=DEGRADED,
            detail="No GEMINI_API_KEY* variables in the environment.",
            impact="Brief parsing uses the offline extractor; the gatekeeper "
                   "and question generator serve static replies. The pipeline "
                   "still runs end to end.",
            remedy="Set GEMINI_API_KEY_1.. in sources/.env",
        )
    return Check(name="llm_keys", status=OK,
                 detail=f"{len(keys)} key variable(s) present.",
                 meta={"count": len(keys)})


_CHECKS: List[Callable[[], Check]] = [
    _check_plan_index,
    _check_zone_priors,
    _check_vastu,
    _check_vastu_mandala,
    _check_program_synth,
    _check_reward_freeze,
    _check_rule_book,
    _check_nbc_plot_regs,
    _check_critic,
    _check_tier2_placer,
    _check_llm_keys,
]


# ── public API ───────────────────────────────────────────────────────────────

def self_check() -> Dict[str, Any]:
    """Run every subsystem check and return a JSON-serialisable report."""
    checks: List[Check] = []
    for fn in _CHECKS:
        try:
            checks.append(fn())
        except Exception as exc:                              # pragma: no cover
            checks.append(Check(name=getattr(fn, "__name__", "unknown"),
                                status=MISSING,
                                detail=f"Check itself failed: {exc!r}"))

    worst = max((_SEVERITY[c.status] for c in checks), default=0)
    overall = {v: k for k, v in _SEVERITY.items()}[worst]

    return {
        "healthy": worst == 0,
        "overall": overall,
        "counts": {
            OK: sum(1 for c in checks if c.status == OK),
            DEGRADED: sum(1 for c in checks if c.status == DEGRADED),
            MISSING: sum(1 for c in checks if c.status == MISSING),
        },
        "checks": [asdict(c) for c in checks],
    }


def format_report(report: Dict[str, Any] | None = None) -> str:
    """Render the report as aligned console text for startup logging."""
    report = report or self_check()
    mark = {OK: "OK  ", DEGRADED: "WARN", MISSING: "FAIL"}
    width = max(len(c["name"]) for c in report["checks"])
    lines = ["PlanGen self-check — overall: %s" % report["overall"].upper()]
    for c in report["checks"]:
        lines.append("  [%s] %-*s  %s" % (mark[c["status"]], width,
                                          c["name"], c["detail"]))
        if c["status"] != OK and c.get("impact"):
            lines.append("       %s* %s" % (" " * width, c["impact"]))
    return "\n".join(lines)


if __name__ == "__main__":                                    # pragma: no cover
    print(format_report())
