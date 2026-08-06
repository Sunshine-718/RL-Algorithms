"""Vectorized Gymnasium trainer for QRSAC on BipedalWalker-v3.

Each vector environment owns an episode cache.  Completed episode caches are
fed to the existing ReplayBuffer one at a time, so n-step returns never cross
environment boundaries and SAC/replaybuffer.py does not need to change.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from qrsac_continuous import (
    Config,
    ContinuousSAC,
    ContinuousSACAgent,
    mirror_bipedal_action,
    mirror_bipedal_observation,
)
from train_qrsac_bipedalwalker import (
    append_jsonl,
    atomic_json,
    evaluate,
    initialize_from_checkpoint,
    load_checkpoint,
    make_env,
    save_checkpoint,
    write_episode_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qrsac_bipedalwalker_vector"))
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vector-mode", choices=("async", "sync"), default="async")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hardcore", action="store_true")
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument(
        "--cpg-period-steps",
        type=int,
        default=0,
        help="Append sin/cos gait phase with this period; zero disables CPG.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-quantiles", type=int, default=51)
    parser.add_argument("--capacity", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument(
        "--actor-learning-starts",
        type=int,
        help="Delay actor and temperature updates while the critic adapts to a fresh buffer.",
    )
    parser.add_argument("--random-steps", type=int)
    parser.add_argument("--deterministic-warmup", action="store_true")
    parser.add_argument(
        "--gradient-steps-per-vector-step",
        type=int,
        default=1,
        help="Gradient batches after one vector step. Use num-envs for UTD ~= 1.",
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--fall-penalty",
        type=float,
        default=-10.0,
        help="Training reward substituted for the environment's -100 fall event.",
    )
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument(
        "--actor-quantile-fraction",
        type=float,
        default=1.0,
        help="Fraction of the lowest critic quantiles optimized by the actor.",
    )
    parser.add_argument("--symmetry-loss-coef", type=float, default=0.0)
    parser.add_argument("--actor-rl-coef", type=float, default=1.0)
    parser.add_argument("--behavior-anchor-coef", type=float, default=0.0)
    parser.add_argument("--mirror-replay-augmentation", action="store_true")
    parser.add_argument("--energy-balance-coef", type=float, default=0.0)
    parser.add_argument("--energy-ema-decay", type=float, default=0.95)
    parser.add_argument("--contact-alternation-bonus", type=float, default=0.0)
    parser.add_argument("--flight-penalty", type=float, default=0.0)
    parser.add_argument("--double-support-penalty", type=float, default=0.0)
    parser.add_argument("--double-support-grace", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def make_vector_env(
    num_envs: int,
    seed: int,
    hardcore: bool,
    mode: str,
    frame_stack: int = 1,
    cpg_period_steps: int = 0,
):
    factories = [
        partial(make_env, seed + env_id, hardcore, frame_stack, cpg_period_steps)
        for env_id in range(num_envs)
    ]
    vector_type = gym.vector.AsyncVectorEnv if mode == "async" else gym.vector.SyncVectorEnv
    env = vector_type(factories)
    env.action_space.seed(seed)
    return env


def build_agent(args: argparse.Namespace, device: str) -> tuple[ContinuousSACAgent, Config]:
    probe = make_env(args.seed, args.hardcore, args.frame_stack, args.cpg_period_steps)
    obs_dim = int(np.prod(probe.observation_space.shape))
    action_dim = int(np.prod(probe.action_space.shape))
    probe.close()
    net = ContinuousSAC(
        args.actor_lr,
        args.critic_lr,
        obs_dim,
        args.hidden_dim,
        action_dim,
        action_limit=1.0,
        dropout=0.0,
        num_quantiles=args.num_quantiles,
        alpha=0.2,
        alpha_lr=args.alpha_lr,
        device=device,
    )
    config = Config(
        discount=args.discount,
        params=str(args.run_dir),
        tau=args.tau,
        capacity=args.capacity,
        epoch=1,
        reward_scale=args.reward_scale,
        n_step=args.n_step,
        actor_quantile_fraction=args.actor_quantile_fraction,
        symmetry_loss_coef=args.symmetry_loss_coef,
        actor_rl_coef=args.actor_rl_coef,
        behavior_anchor_coef=args.behavior_anchor_coef,
    )
    return ContinuousSACAgent("qrsac_bipedalwalker_vector", net, config), config


@torch.inference_mode()
def batched_actions(agent: ContinuousSACAgent, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=agent.net.device)
    agent.net.eval()
    actions, _ = agent.net.actor(state_tensor, deterministic=deterministic)
    agent.net.train()
    return actions.cpu().numpy()


def flush_episode(agent: ContinuousSACAgent, transitions: list[tuple]) -> int:
    """Process exactly one environment trajectory through the unchanged buffer."""
    if not transitions:
        return 0
    if agent.buffer.cache:
        raise RuntimeError("ReplayBuffer cache must be empty before flushing an environment")
    for transition in transitions:
        agent.cache(*transition)
    agent.process()
    return len(transitions)


def mirror_episode(transitions: list[tuple]) -> list[tuple]:
    """Create a physically equivalent trajectory with the leg identities exchanged."""
    return [
        (
            mirror_bipedal_observation(state),
            mirror_bipedal_action(action),
            reward,
            mirror_bipedal_observation(next_state),
            terminated,
            truncated,
        )
        for state, action, reward, next_state, terminated, truncated in transitions
    ]


def symmetry_reward(
    next_state: np.ndarray,
    action: np.ndarray,
    left_energy_avg: float,
    right_energy_avg: float,
    last_support: int,
    double_support_steps: int,
    args: argparse.Namespace,
) -> tuple[float, float, float, int, int]:
    """Return mild gait shaping and the updated per-environment gait state."""
    decay = args.energy_ema_decay
    left_energy = float(np.square(action[0:2]).sum())
    right_energy = float(np.square(action[2:4]).sum())
    left_energy_avg = decay * left_energy_avg + (1.0 - decay) * left_energy
    right_energy_avg = decay * right_energy_avg + (1.0 - decay) * right_energy
    shaped = -args.energy_balance_coef * (left_energy_avg - right_energy_avg) ** 2

    left_contact = bool(next_state[8] > 0.5)
    right_contact = bool(next_state[13] > 0.5)
    if left_contact ^ right_contact:
        current_support = 0 if left_contact else 1
        if last_support >= 0 and current_support != last_support:
            shaped += args.contact_alternation_bonus
        last_support = current_support
        double_support_steps = 0
    elif left_contact and right_contact:
        double_support_steps += 1
        if double_support_steps > args.double_support_grace:
            shaped -= args.double_support_penalty
    else:
        double_support_steps = 0
        shaped -= args.flight_penalty
    return shaped, left_energy_avg, right_energy_avg, last_support, double_support_steps


def train(args: argparse.Namespace) -> None:
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from cannot be used together")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.frame_stack < 1:
        raise ValueError("--frame-stack must be at least 1")
    if args.cpg_period_steps < 0:
        raise ValueError("--cpg-period-steps must be non-negative")
    if args.gradient_steps_per_vector_step < 0:
        raise ValueError("--gradient-steps-per-vector-step cannot be negative")
    if not 0.0 <= args.energy_ema_decay < 1.0:
        raise ValueError("--energy-ema-decay must be in [0, 1)")
    if min(
        args.symmetry_loss_coef,
        args.actor_rl_coef,
        args.behavior_anchor_coef,
        args.energy_balance_coef,
        args.contact_alternation_bonus,
        args.flight_penalty,
        args.double_support_penalty,
        args.double_support_grace,
    ) < 0:
        raise ValueError("symmetry coefficients and grace must be non-negative")
    if args.random_steps is None:
        args.random_steps = args.learning_starts
    if args.actor_learning_starts is None:
        args.actor_learning_starts = args.learning_starts
    if args.actor_learning_starts < args.learning_starts:
        raise ValueError("--actor-learning-starts must be at least --learning-starts")
    if args.smoke_test:
        args.total_steps = min(args.total_steps, 800)
        args.learning_starts = min(args.learning_starts, 128)
        args.random_steps = min(args.random_steps, 128)
        args.batch_size = min(args.batch_size, 64)
        args.capacity = min(args.capacity, 5_000)
        args.eval_every = args.total_steps + args.num_envs
        args.checkpoint_every = max(args.total_steps // 2, args.num_envs)

    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stop_file = args.run_dir / "STOP"
    if stop_file.exists():
        stop_file.unlink()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    agent, config = build_agent(args, device)
    global_step = 0
    episode = 0
    best_eval = -float("inf")
    if args.resume:
        restored = load_checkpoint(agent, args.resume.resolve())
        global_step = int(restored.get("global_step", 0))
        episode = int(restored.get("episode", 0))
        best_eval = float(restored.get("best_eval", best_eval))
    elif args.init_from:
        initialize_from_checkpoint(agent, args.init_from.resolve(), alpha=args.init_alpha)

    config_dump = vars(args).copy()
    for key in ("run_dir", "resume", "init_from"):
        value = config_dump[key]
        config_dump[key] = str(value.resolve()) if value else None
    config_dump["resolved_device"] = device
    config_dump["agent_config"] = asdict(config)
    atomic_json(args.run_dir / "config.json", config_dump)
    (args.run_dir / "pid.txt").write_text(str(os.getpid()), encoding="ascii")

    env = make_vector_env(
        args.num_envs,
        args.seed,
        args.hardcore,
        args.vector_mode,
        args.frame_stack,
        args.cpg_period_steps,
    )
    states, _ = env.reset(seed=args.seed)
    needs_reset = np.zeros(args.num_envs, dtype=bool)
    episode_caches: list[list[tuple]] = [[] for _ in range(args.num_envs)]
    raw_returns = np.zeros(args.num_envs, dtype=np.float64)
    train_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    symmetry_returns = np.zeros(args.num_envs, dtype=np.float64)
    left_energy_avgs = np.zeros(args.num_envs, dtype=np.float64)
    right_energy_avgs = np.zeros(args.num_envs, dtype=np.float64)
    last_support = np.full(args.num_envs, -1, dtype=np.int8)
    double_support_steps = np.zeros(args.num_envs, dtype=np.int32)
    recent_returns = deque(maxlen=20)
    start_time = time.time()
    run_start_step = global_step
    next_eval = ((global_step // args.eval_every) + 1) * args.eval_every
    next_checkpoint = ((global_step // args.checkpoint_every) + 1) * args.checkpoint_every
    last_losses: dict = {}

    def trainer_state() -> dict:
        return {"global_step": global_step, "episode": episode, "best_eval": best_eval}

    def log_episode(env_id: int) -> None:
        nonlocal episode
        episode += 1
        recent_returns.append(float(raw_returns[env_id]))
        row = {
            "episode": episode,
            "env_id": env_id,
            "global_step": global_step,
            "raw_return": round(float(raw_returns[env_id]), 6),
            "train_return": round(float(train_returns[env_id]), 6),
            "symmetry_return": round(float(symmetry_returns[env_id]), 6),
            "length": int(episode_lengths[env_id]),
            "return_mean_20": round(float(np.mean(recent_returns)), 6),
            "alpha": round(agent.alpha, 8),
        }
        write_episode_csv(args.run_dir / "episodes.csv", row)
        elapsed = max(time.time() - start_time, 1e-6)
        steps_per_second = max(global_step - run_start_step, 1) / elapsed
        status = {
            "state": "running",
            **row,
            **last_losses,
            "num_envs": args.num_envs,
            "vector_mode": args.vector_mode,
            "buffer_size": len(agent.buffer),
            "best_eval_return": None if best_eval == -float("inf") else best_eval,
            "steps_per_second": steps_per_second,
            "eta_seconds": max(args.total_steps - global_step, 0) / max(steps_per_second, 1e-6),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "device": device,
        }
        atomic_json(args.run_dir / "status.json", status)
        append_jsonl(args.run_dir / "metrics.jsonl", {"type": "episode", **status})
        print(
            f"step={global_step} episode={episode} env={env_id} "
            f"return={raw_returns[env_id]:.1f} mean20={np.mean(recent_returns):.1f} "
            f"buffer={len(agent.buffer)} sps={steps_per_second:.1f}",
            flush=True,
        )

    try:
        while global_step < args.total_steps and not stop_file.exists():
            reset_before_step = needs_reset.copy()
            if global_step < args.random_steps:
                actions = env.action_space.sample()
            elif args.deterministic_warmup and global_step < args.learning_starts:
                actions = batched_actions(agent, states, deterministic=True)
            else:
                actions = batched_actions(agent, states)

            next_states, raw_rewards, terminated, truncated, _ = env.step(actions)
            needs_reset.fill(False)
            valid_transitions = 0
            for env_id in range(args.num_envs):
                # NEXT_STEP autoreset emits a reset observation on this slot;
                # it is not an environment transition and must not enter replay.
                if reset_before_step[env_id]:
                    continue
                raw_reward = float(raw_rewards[env_id])
                reward = args.fall_penalty if raw_reward <= -99.0 else raw_reward
                (
                    symmetry_bonus,
                    left_energy_avgs[env_id],
                    right_energy_avgs[env_id],
                    last_support[env_id],
                    double_support_steps[env_id],
                ) = symmetry_reward(
                    next_states[env_id],
                    actions[env_id],
                    left_energy_avgs[env_id],
                    right_energy_avgs[env_id],
                    int(last_support[env_id]),
                    int(double_support_steps[env_id]),
                    args,
                )
                reward += symmetry_bonus
                done = bool(terminated[env_id] or truncated[env_id])
                episode_caches[env_id].append(
                    (
                        states[env_id].copy(),
                        actions[env_id].copy(),
                        reward,
                        next_states[env_id].copy(),
                        bool(terminated[env_id]),
                        bool(truncated[env_id]),
                    )
                )
                raw_returns[env_id] += raw_reward
                train_returns[env_id] += reward
                symmetry_returns[env_id] += symmetry_bonus
                episode_lengths[env_id] += 1
                valid_transitions += 1
                if done:
                    flush_episode(agent, episode_caches[env_id])
                    if args.mirror_replay_augmentation:
                        flush_episode(agent, mirror_episode(episode_caches[env_id]))
                    episode_caches[env_id].clear()
                    log_episode(env_id)
                    raw_returns[env_id] = 0.0
                    train_returns[env_id] = 0.0
                    symmetry_returns[env_id] = 0.0
                    episode_lengths[env_id] = 0
                    left_energy_avgs[env_id] = 0.0
                    right_energy_avgs[env_id] = 0.0
                    last_support[env_id] = -1
                    double_support_steps[env_id] = 0
                    needs_reset[env_id] = True

            states = next_states
            global_step += valid_transitions

            if global_step >= args.learning_starts and len(agent.buffer) >= args.batch_size:
                for _ in range(args.gradient_steps_per_vector_step):
                    metrics = agent.step(
                        args.batch_size,
                        update_actor=global_step >= args.actor_learning_starts,
                    )
                    if metrics:
                        last_losses = metrics

            while global_step >= next_checkpoint:
                save_checkpoint(args.run_dir / "latest.pt", agent, trainer_state())
                next_checkpoint += args.checkpoint_every

            while global_step >= next_eval:
                result = evaluate(
                    agent,
                    args.eval_episodes,
                    args.seed + 100_000 + next_eval,
                    args.hardcore,
                    args.frame_stack,
                    args.cpg_period_steps,
                )
                result.update({"type": "evaluation", "global_step": global_step, "episode": episode})
                append_jsonl(args.run_dir / "metrics.jsonl", result)
                if result["eval_return_mean"] > best_eval:
                    best_eval = result["eval_return_mean"]
                    save_checkpoint(args.run_dir / "best.pt", agent, trainer_state())
                next_eval += args.eval_every
    except Exception as exc:
        atomic_json(
            args.run_dir / "status.json",
            {"state": "failed", "global_step": global_step, "episode": episode, "error": repr(exc), "pid": os.getpid()},
        )
        raise
    finally:
        for transitions in episode_caches:
            flush_episode(agent, transitions)
            if args.mirror_replay_augmentation:
                flush_episode(agent, mirror_episode(transitions))
            transitions.clear()
        env.close()

    save_checkpoint(args.run_dir / "latest.pt", agent, trainer_state())
    final_status = {
        "state": "stopped" if stop_file.exists() else "completed",
        "global_step": global_step,
        "episode": episode,
        "buffer_size": len(agent.buffer),
        "best_eval_return": None if best_eval == -float("inf") else best_eval,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(args.run_dir / "status.json", final_status)
    print(json.dumps(final_status), flush=True)


if __name__ == "__main__":
    train(parse_args())
