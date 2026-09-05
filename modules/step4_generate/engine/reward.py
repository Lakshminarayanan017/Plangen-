"""
reward.py — pin down what "good" means, so training can trust it.

Stages (b) and (c) of the placer both optimise the reviewer's score blended
with the learned critic. That only works if the reward holds still. Add a
rule, retune a weight, retrain the critic, and every number produced before
that moment silently stops being comparable to every number after it — the
self-play corpus was distilled from one ruler and the policy is now being
graded by another.

This is the same failure v2 hit with `--eval-n`, where a mid-run change to
the eval set moved the mean by ~12 points for no modelling reason and froze
`best_model` at epoch 2. The fix there was an eval KEY recorded in every
checkpoint. This is that idea applied to the reward itself.

    fingerprint()   a stable hash over the rule registry, every scoring
                    weight, and the critic's identity
    freeze()        write the snapshot that becomes reward v1
    verify()        compare live against frozen and say exactly what moved

Deliberately NOT in the hash: generation knobs. `settle_sweeps`,
`realize_attempts`, `cpsat_mode` and `vastu_bias` change which plans get
MADE, not how a plan is JUDGED. They are recorded as context so a run can be
reproduced, but changing them does not invalidate a corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from modules.step4_generate.engine.contracts import EngineConfig

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "reward_v1.json")

# Config fields that define the SCORING function. Everything matching `w_*`
# is picked up automatically, so a new weight cannot be forgotten; these are
# the non-`w_` fields that also change what a plan scores.
_SCORING_FLAGS = ("fsp001_hard", "vastu_hard", "critic_weight",
                  "parking_strict", "floor_height_ft")

# Recorded for reproducibility, but NOT hashed — see the module docstring.
_GENERATION_CONTEXT = ("settle_sweeps", "settle_aspect_limit",
                       "settle_aspect_weight", "max_repair_rounds",
                       "realize_attempts", "top_up_candidates",
                       "dedup_candidates", "cpsat_mode", "cpsat_timeout_ms",
                       "vastu_bias", "vertical_bias",
                       "window_habitable_min_ft", "window_habitable_max_ft",
                       "window_wet_ft")


def rule_signature() -> List[str]:
    """Every registered rule as "id|severity", sorted.

    Adding a rule, removing one, or flipping its severity all change this —
    and all three change what a plan scores.
    """
    from modules.step4_generate.engine.rules import RULES
    return sorted(f"{rule_id}|{severity}" for rule_id, severity, _ in RULES)


def critic_signature(path: Optional[str] = None) -> Dict[str, Any]:
    """The critic's identity: its file digest plus the shape of what it
    learned. A retrained critic is a different reward even when the weights
    file has the same name."""
    from modules.step4_generate.critic.critic import DEFAULT_MODEL_PATH
    path = path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        return {"present": False}
    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:16]
        with open(path, encoding="utf-8") as fh:
            model = json.load(fh)
        return {
            "present": True,
            "sha256_16": digest,
            "n_trees": len(model.get("trees", [])),
            "n_features": len(model.get("feature_names", [])),
        }
    except Exception as exc:                                # pragma: no cover
        return {"present": True, "unreadable": str(exc)[:80]}


def scoring_weights(config: Optional[EngineConfig] = None
                    ) -> Dict[str, Any]:
    """Every field that changes what a plan scores."""
    cfg = (config or EngineConfig()).to_dict()
    out = {k: v for k, v in cfg.items() if k.startswith("w_")}
    for key in _SCORING_FLAGS:
        if key in cfg:
            out[key] = cfg[key]
    return dict(sorted(out.items()))


def generation_context(config: Optional[EngineConfig] = None
                       ) -> Dict[str, Any]:
    cfg = (config or EngineConfig()).to_dict()
    return {k: cfg[k] for k in _GENERATION_CONTEXT if k in cfg}


def _payload(config: Optional[EngineConfig] = None,
             critic_path: Optional[str] = None) -> Dict[str, Any]:
    return {
        "rules": rule_signature(),
        "weights": scoring_weights(config),
        "critic": critic_signature(critic_path),
    }


def fingerprint(config: Optional[EngineConfig] = None,
                critic_path: Optional[str] = None) -> str:
    """A stable 16-hex digest of the reward. Same reward, same string."""
    blob = json.dumps(_payload(config, critic_path), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class RewardDrift:
    """What moved between a frozen reward and the live one."""
    matches: bool
    frozen_fingerprint: str = ""
    live_fingerprint: str = ""
    rules_added: List[str] = field(default_factory=list)
    rules_removed: List[str] = field(default_factory=list)
    weights_changed: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    critic_changed: bool = False
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.matches:
            return f"reward matches the frozen snapshot ({self.frozen_fingerprint})"
        lines = [f"REWARD DRIFT: frozen {self.frozen_fingerprint} -> live "
                 f"{self.live_fingerprint}"]
        for r in self.rules_added:
            lines.append(f"  + rule {r}")
        for r in self.rules_removed:
            lines.append(f"  - rule {r}")
        for key, (was, now) in sorted(self.weights_changed.items()):
            lines.append(f"  ~ {key}: {was} -> {now}")
        if self.critic_changed:
            lines.append("  ~ critic weights differ")
        lines.append("  Corpora and checkpoints produced under the frozen "
                     "reward are NOT comparable to ones produced now.")
        return "\n".join(lines)


def freeze(path: str = DEFAULT_PATH, *, version: str = "v1",
           config: Optional[EngineConfig] = None,
           critic_path: Optional[str] = None,
           baseline: Optional[Dict[str, Any]] = None,
           note: str = "") -> Dict[str, Any]:
    """Write the snapshot that downstream training treats as authoritative."""
    import time

    payload = _payload(config, critic_path)
    snapshot = {
        "version": version,
        "fingerprint": fingerprint(config, critic_path),
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
        **payload,
        "generation_context": generation_context(config),
        "harness_baseline": baseline or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=1)
    return snapshot


def load(path: str = DEFAULT_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def verify(path: str = DEFAULT_PATH,
           config: Optional[EngineConfig] = None,
           critic_path: Optional[str] = None) -> RewardDrift:
    """Compare the live reward against the frozen one, itemised.

    Reports WHAT moved rather than only that something did — "the reward
    changed" is not actionable; "w_vastu_sector 6.0 -> 8.0 and VAS-012 was
    added" is.
    """
    frozen = load(path)
    if frozen is None:
        return RewardDrift(matches=False, live_fingerprint=fingerprint(
            config, critic_path),
            notes=[f"no frozen reward at {path}"])

    live = _payload(config, critic_path)
    live_fp = fingerprint(config, critic_path)
    frozen_fp = frozen.get("fingerprint", "")
    if live_fp == frozen_fp:
        return RewardDrift(matches=True, frozen_fingerprint=frozen_fp,
                           live_fingerprint=live_fp)

    frozen_rules = set(frozen.get("rules", []))
    live_rules = set(live["rules"])
    frozen_w = frozen.get("weights", {})
    changed = {k: (frozen_w.get(k), v) for k, v in live["weights"].items()
               if k in frozen_w and frozen_w[k] != v}
    changed.update({k: (v, None) for k, v in frozen_w.items()
                    if k not in live["weights"]})
    changed.update({k: (None, v) for k, v in live["weights"].items()
                    if k not in frozen_w})

    return RewardDrift(
        matches=False,
        frozen_fingerprint=frozen_fp, live_fingerprint=live_fp,
        rules_added=sorted(live_rules - frozen_rules),
        rules_removed=sorted(frozen_rules - live_rules),
        weights_changed=changed,
        critic_changed=(frozen.get("critic") != live["critic"]))


def require(path: str = DEFAULT_PATH,
            config: Optional[EngineConfig] = None,
            critic_path: Optional[str] = None,
            *, strict: bool = False) -> RewardDrift:
    """Check the reward and say so on stdout. `strict` raises on drift.

    Training entry points call this: a corpus built under one reward and a
    policy graded by another is the quiet failure this whole module exists
    to prevent.
    """
    drift = verify(path, config, critic_path)
    if drift.matches:
        print(f"  reward: {drift.describe()}")
        return drift
    print(drift.describe())
    if strict:
        raise SystemExit(
            "reward drift with --strict-reward: re-freeze "
            "(python -m modules.step4_generate.engine.reward --freeze) "
            "or restore the frozen configuration")
    return drift


def main(argv=None) -> int:
    import argparse

    from modules.step4_generate.core.console import force_utf8_console
    force_utf8_console()

    p = argparse.ArgumentParser(prog="engine.reward")
    p.add_argument("--freeze", action="store_true",
                   help="write the snapshot from the CURRENT engine")
    p.add_argument("--version", default="v1")
    p.add_argument("--note", default="")
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--with-baseline", action="store_true",
                   help="attach the recorded golden-harness numbers")
    args = p.parse_args(argv)

    if not args.freeze:
        drift = verify(args.path)
        print(drift.describe())
        frozen = load(args.path)
        if frozen:
            print(f"\n  frozen {frozen.get('version')} at "
                  f"{frozen.get('frozen_at')}")
            print(f"  {len(frozen.get('rules', []))} rules · "
                  f"{len(frozen.get('weights', {}))} scoring weights")
            base = frozen.get("harness_baseline") or {}
            if base:
                print(f"  harness baseline: {base.get('n_briefs')} briefs, "
                      f"mean {base.get('mean_best_score')}")
        return 0 if drift.matches else 1

    baseline = {}
    if args.with_baseline:
        harness = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "ml", "harness", "results", "latest.json")
        try:
            with open(os.path.normpath(harness), encoding="utf-8") as fh:
                data = json.load(fh)
            baseline = {k: data[k] for k in
                        ("timestamp", "n_briefs", "briefs_with_plan",
                         "mean_best_score", "mean_fidelity", "mean_drift")
                        if k in data}
        except Exception as exc:
            print(f"  (harness baseline unavailable: {exc})")

    snap = freeze(args.path, version=args.version, baseline=baseline,
                  note=args.note)
    print(f"froze reward {snap['version']} — fingerprint "
          f"{snap['fingerprint']}")
    print(f"  {len(snap['rules'])} rules · {len(snap['weights'])} weights · "
          f"critic {snap['critic'].get('sha256_16', 'absent')}")
    if baseline:
        print(f"  baseline: {baseline.get('n_briefs')} briefs, mean "
              f"{baseline.get('mean_best_score')}")
    print(f"  -> {os.path.abspath(args.path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
