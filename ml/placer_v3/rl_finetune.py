"""
rl_finetune.py — stage (c): optimise the reward instead of imitating a corpus.

Stages (a) and (b) can only ever approach the thing they imitate. Stage (c)
removes that ceiling: the model samples k layouts per brief, the ENGINE
carves and the REVIEWER scores each one, and the model is pushed toward the
proposals that actually scored well on the Indian reward.

Algorithm — GRPO-shaped, deliberately:
    for each brief, sample a GROUP of k trajectories
    reward r_i  = blended soft score + critic for trajectory i
    advantage   = (r_i - mean(r)) / (std(r) + eps)      <- group baseline
    loss        = -A_i * log p(trajectory_i)  +  beta * KL(pi || pi_ref)

A group baseline means no value network to train, no value network to be
wrong, and the variance reduction comes free from the k samples the engine
was going to carve anyway.

The KL term is not optional. The reward is 33 hand-written rules, and a
policy optimised against hand-written rules will find their seams — parking
in a corner that technically gates, a passage that games the circulation
metric. Anchoring to the stage-(b) policy keeps it inside the distribution
that produced plans a human called good.

COST. Every reward requires a full carve + review, so throughput is bounded
by the engine (~0.5-1.5 s per candidate), not the GPU. Budget accordingly:
--briefs-per-epoch 64 --group 6 is ~400 carves per epoch, a few minutes.

Run:
    python -m ml.placer_v3.rl_finetune \
        --init runs/stage_b/ckpt_best.pt --out runs/stage_c --epochs 30
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ml.placer_v3.checkpoint import CheckpointManager, eval_key_for
from ml.placer_v3.config import PRESET_LARGE, PlacerV3Config, STATE_GRID
from ml.placer_v3.decode import sample as sample_trajectory
from ml.placer_v3.features import request_to_arrays
from ml.placer_v3.model.placer_net import PlacerNetV3
from ml.placer_v3.state import build_state_stack
from ml.placer_v3.train import engine_eval, to_device


def _trajectory_arrays(base: Dict, cells: List[int], sizes: List[int],
                       device: str) -> Dict[str, torch.Tensor]:
    """The model's own sampled trajectory, packaged as if it were a training
    target. Re-running the forward teacher-forced on the sample is how we get
    differentiable log-probs for actions that were drawn without grad."""
    zone_ids = np.asarray(base["zone_ids"])
    arrays = dict(base)
    arrays["target_cell"] = np.asarray(cells, dtype=np.int64)
    arrays["target_size"] = np.asarray([s - 1 for s in sizes], dtype=np.int64)
    arrays["state"] = build_state_stack(base["boundary"][0],
                                        base["entrance_side"],
                                        cells, sizes, zone_ids)
    arrays.pop("specs", None)
    tensors = to_device(
        {k: v for k, v in arrays.items()
         if k not in ("entrance_side", "target_aspect", "target_orientation",
                      "target_band")}, device)
    return tensors


def _logprob(net, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Summed log-probability of the trajectory's cell and size choices."""
    out = net(tensors)
    cell_lp = F.log_softmax(out["cell"], dim=-1)
    size_lp = F.log_softmax(out["size"], dim=-1)
    idx = torch.arange(tensors["target_cell"].size(0),
                       device=tensors["target_cell"].device)
    return (cell_lp[idx, tensors["target_cell"]].sum()
            + size_lp[idx, tensors["target_size"]].sum())


def _reward_for(plan_candidate, critic, critic_weight: float) -> float:
    from modules.step4_generate.engine.orchestrator import blended_score
    reward = plan_candidate.verdict.soft_score
    if critic is not None:
        try:
            return blended_score(reward, critic.score_candidate(plan_candidate),
                                 critic_weight)
        except Exception:
            pass
    return reward


def rollout_group(net, request, group: int, temperature: float,
                  device: str, critic, critic_weight: float, seed: int
                  ) -> Tuple[List[Dict], List[float]]:
    """Sample `group` trajectories and score each by carving it for real.

    Trajectories the carver cannot realise get the floor reward rather than
    being dropped: "this proposal produces no plan" is the single most
    important thing the policy can learn, and dropping it teaches nothing.
    """
    from modules.step4_generate.carve.hub_carver import carve_from_proposal
    from modules.step4_generate.core.grid_plan import GridPlan
    from modules.step4_generate.engine.contracts import (
        EngineConfig, LayoutProposal, Placement,
    )
    from modules.step4_generate.engine.connector import RuleConnector
    from modules.step4_generate.engine.contracts import Candidate
    from modules.step4_generate.engine.settle import SqueezeSettler
    from modules.step4_generate.engine.stairs import StairFitter
    from modules.step4_generate.engine.validator import BasicValidator

    cfg = EngineConfig()
    base = request_to_arrays(request)
    specs = base["specs"]

    trajectories: List[Dict] = []
    rewards: List[float] = []

    for g in range(group):
        rng = np.random.default_rng(seed * 977 + g)
        result = sample_trajectory(net, base, temperature=temperature,
                                   rng=rng, greedy=False, device=device)
        cells = [p.row * STATE_GRID + p.col for p in result.placements]
        sizes = [p.size_class for p in result.placements]
        proposal = LayoutProposal(
            placements=[Placement(specs[p.room_index].name,
                                  (p.row, p.col), p.size_class)
                        for p in result.placements],
            source="rl-rollout")

        reward = 0.0
        try:
            plan = GridPlan.from_feet(request.plot_w_ft, request.plot_h_ft)
            room_ids = carve_from_proposal(plan, request, proposal, cfg)
            SqueezeSettler(cfg).settle(plan, request, room_ids)
            StairFitter(cfg).fit(plan, request, room_ids)
            RuleConnector(cfg).connect(plan, request, room_ids)
            verdict = BasicValidator(cfg).check(plan, request, room_ids)
            if not verdict.hard:
                cand = Candidate(plan=plan, proposal=proposal, verdict=verdict,
                                 request=request, room_ids=room_ids)
                reward = _reward_for(cand, critic, critic_weight)
        except Exception:
            reward = 0.0

        trajectories.append({"cells": cells, "sizes": sizes, "base": base})
        rewards.append(float(reward))

    return trajectories, rewards


def main(argv=None) -> int:
    from modules.step4_generate.core.console import force_utf8_console
    from modules.step4_generate.critic.critic import LearnedCritic
    from modules.step4_generate.engine.contracts import EngineConfig
    from ml.placer_v3.selfplay import brief_at
    force_utf8_console()

    p = argparse.ArgumentParser(prog="ml.placer_v3.rl_finetune")
    p.add_argument("--init", required=True, help="stage (b) best checkpoint")
    p.add_argument("--out", default="runs/placer_v3_rl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--briefs-per-epoch", type=int, default=64)
    p.add_argument("--group", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta-kl", type=float, default=0.02)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--critic-weight", type=float, default=0.4)
    p.add_argument("--preset", choices=["small", "large"], default="small")
    p.add_argument("--eval-n", type=int, default=48)
    p.add_argument("--eval-seed", type=int, default=900000)
    p.add_argument("--brief-seed", type=int, default=20260905)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--strict-reward", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.dry_run:
        args.epochs, args.briefs_per_epoch, args.group, args.eval_n = 1, 3, 3, 3
        print(">>> DRY RUN\n")

    # RL optimises the reward directly — drift here is not a comparability
    # problem, it is training against a different objective than intended.
    from modules.step4_generate.engine import reward as reward_mod
    reward_mod.require(strict=args.strict_reward)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PRESET_LARGE if args.preset == "large" else PlacerV3Config()

    net = PlacerNetV3(cfg).to(device)
    payload = torch.load(args.init, map_location=device, weights_only=False)
    net.load_state_dict(payload["model"])
    print(f"policy initialised from {args.init}")

    # frozen reference — the anchor the KL term pulls back toward
    ref = PlacerNetV3(cfg).to(device)
    ref.load_state_dict(payload["model"])
    ref.eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)

    critic = LearnedCritic.load_if_available(config=EngineConfig())
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    manager = CheckpointManager(args.out, keep_last=3, milestone_every=5)
    key = eval_key_for("rl_eval", args.eval_n, args.eval_seed)
    state = manager.resume(model=net, optimizer=optimizer, eval_key=key)
    state.stage = "stage_c_rl"

    print(f"\n{'ep':>4} {'reward':>8} {'best-of-k':>10} {'adv|':>7} "
          f"{'KL':>7} {'ENGINE':>8} {'plan%':>6} {'min':>6}")
    print("-" * 68)

    for epoch in range(state.epoch + 1, args.epochs + 1):
        t0 = time.perf_counter()
        net.train()
        optimizer.zero_grad(set_to_none=True)

        ep_rewards, ep_best, ep_kl, ep_adv = [], [], [], []
        for b in range(args.briefs_per_epoch):
            request = brief_at(args.brief_seed + epoch * 10_000 + b)
            request.k = 1
            trajectories, rewards = rollout_group(
                net, request, args.group, args.temperature, device,
                critic, args.critic_weight, seed=epoch * 1013 + b)
            if not rewards or max(rewards) <= 0.0:
                continue

            r = np.asarray(rewards, dtype=np.float64)
            adv = (r - r.mean()) / (r.std() + 1e-6)
            ep_rewards.append(float(r.mean()))
            ep_best.append(float(r.max()))
            ep_adv.append(float(np.abs(adv).mean()))

            for traj, a in zip(trajectories, adv):
                if abs(a) < 1e-6:
                    continue
                tensors = _trajectory_arrays(traj["base"], traj["cells"],
                                             traj["sizes"], device)
                logp = _logprob(net, tensors)
                with torch.no_grad():
                    ref_logp = _logprob(ref, tensors)
                # k1 KL estimator on the trajectory log-ratio
                kl = (logp - ref_logp).abs()
                loss = (-float(a) * logp + args.beta_kl * kl) / \
                    max(1, args.briefs_per_epoch * args.group)
                loss.backward()
                ep_kl.append(float(kl))

        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            metrics = engine_eval(net, args.eval_n, args.eval_seed, device)
        score = metrics["engine_score"]
        is_best = score > state.best_score
        if is_best:
            state.best_score, state.best_epoch = score, epoch
        state.epoch = epoch

        row = {"epoch": epoch, "stage": state.stage,
               "mean_reward": round(float(np.mean(ep_rewards)), 3)
               if ep_rewards else 0.0,
               "best_of_k": round(float(np.mean(ep_best)), 3)
               if ep_best else 0.0,
               "mean_abs_adv": round(float(np.mean(ep_adv)), 3)
               if ep_adv else 0.0,
               "kl": round(float(np.mean(ep_kl)), 4) if ep_kl else 0.0,
               "minutes": round((time.perf_counter() - t0) / 60.0, 2),
               **metrics, "is_best": bool(is_best)}
        state.history.append(row)
        manager.append_log(row)
        manager.save(model=net, optimizer=optimizer, scheduler=None,
                     scaler=None, state=state, config=cfg.to_dict(),
                     is_best=is_best)

        print(f"{epoch:>4} {row['mean_reward']:>8.2f} {row['best_of_k']:>10.2f} "
              f"{row['mean_abs_adv']:>7.3f} {row['kl']:>7.3f} "
              f"{score:>8.2f} {metrics['plan_rate']:>6.2f} "
              f"{row['minutes']:>6.1f}" + ("  <- best" if is_best else ""))

    print("-" * 68)
    print(f"best engine score {state.best_score:.2f} at epoch "
          f"{state.best_epoch}\nbest weights -> {manager.best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
