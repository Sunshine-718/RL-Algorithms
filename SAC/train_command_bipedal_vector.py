"""Train command-conditioned BipedalWalker with vectorized QRSAC collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from command_bipedal_env import CommandBipedalConfig, FPS, make_command_env
from qrsac_continuous import Config, ContinuousSAC, ContinuousSACAgent


Transition = tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool]


def prepare_training_args(args: argparse.Namespace) -> argparse.Namespace:
    """Apply coherent defaults after optional smoke-test limits."""
    if args.smoke_test:
        args.total_steps = min(args.total_steps, 800)
        smoke_learning_starts = min(128, max(1, args.total_steps // 4))
        args.learning_starts = min(args.learning_starts, smoke_learning_starts)
        args.batch_size = min(args.batch_size, 64)
        args.capacity = min(args.capacity, 5_000)
        args.num_envs = min(args.num_envs, 2)
        args.eval_every = max(args.eval_every, args.total_steps + 1)
        args.checkpoint_every = min(
            args.checkpoint_every, max(1, args.total_steps // 2)
        )

    if args.random_steps is None:
        args.random_steps = args.learning_starts
    if args.actor_learning_starts is None:
        args.actor_learning_starts = args.learning_starts
    if args.actor_learning_starts < args.learning_starts:
        raise ValueError("actor_learning_starts must be >= learning_starts")
    if min(args.learning_starts, args.actor_learning_starts, args.random_steps) < 0:
        raise ValueError("learning and random step thresholds cannot be negative")
    if args.total_steps < 1 or args.num_envs < 1:
        raise ValueError("total_steps and num_envs must be positive")
    if args.batch_size < 1 or args.capacity < args.batch_size:
        raise ValueError("capacity must be at least batch_size")
    if (
        args.eval_every < 1
        or args.checkpoint_every < 1
        or getattr(args, "status_every", 1) < 1
        or getattr(args, "eval_episodes", 1) < 1
    ):
        raise ValueError("evaluation and checkpoint intervals must be positive")
    if getattr(args, "keep_checkpoints", 1) < 0:
        raise ValueError("keep_checkpoints cannot be negative")
    if getattr(args, "gradient_steps_per_vector_step", 0) < 0:
        raise ValueError("gradient_steps_per_vector_step cannot be negative")
    return args


def build_env_config(args: argparse.Namespace) -> CommandBipedalConfig:
    return CommandBipedalConfig(
        command_speed=args.command_speed,
        minimum_command_speed=args.minimum_command_speed,
        command_hold_min_steps=max(1, round(args.command_hold_min_seconds * FPS)),
        command_hold_max_steps=max(1, round(args.command_hold_max_seconds * FPS)),
        standing_probability=args.standing_probability,
        settling_time=args.settling_time,
        damping=args.reference_damping,
        acceleration_limit=args.acceleration_limit,
        jerk_limit=args.jerk_limit,
        acceleration_filter=args.acceleration_filter,
        action_penalty_weight=args.action_penalty_weight,
        action_rate_penalty_weight=args.action_rate_penalty_weight,
        gait_reward_weight=args.gait_reward_weight,
        target_stride_length=args.target_stride_length,
        target_swing_clearance=args.target_swing_clearance,
        max_support_steps=max(1, round(args.max_support_seconds * FPS)),
        alternating_step_reward_weight=args.alternating_step_reward_weight,
        support_stall_penalty_weight=args.support_stall_penalty_weight,
        airborne_penalty_weight=args.airborne_penalty_weight,
        max_episode_steps=args.max_episode_steps,
    )


def make_vector_env(
    num_envs: int,
    seed: int,
    config: CommandBipedalConfig,
    vector_mode: str,
    command_mode: str = "random",
):
    """Construct vector workers with explicit reset-step semantics."""
    factories = [
        partial(
            make_command_env,
            seed=seed + index,
            config=config,
            command_mode=command_mode,
        )
        for index in range(num_envs)
    ]
    kwargs = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}
    if vector_mode == "async":
        env = gym.vector.AsyncVectorEnv(factories, **kwargs)
    elif vector_mode == "sync":
        env = gym.vector.SyncVectorEnv(factories, **kwargs)
    else:
        raise ValueError("vector_mode must be async or sync")
    env.action_space.seed(seed)
    env.single_action_space.seed(seed)
    return env


def flush_episode(agent: ContinuousSACAgent, transitions: list[Transition]) -> int:
    """Insert one environment's episode without crossing n-step boundaries."""
    if not transitions:
        return 0
    if agent.buffer.cache:
        raise RuntimeError("shared n-step cache must be empty before an episode flush")
    for transition in transitions:
        agent.cache(*transition)
    agent.process()
    if agent.buffer.cache:
        raise RuntimeError("n-step cache was not fully processed")
    return len(transitions)


def _model_config(args: argparse.Namespace, obs_dim: int, action_dim: int) -> dict[str, Any]:
    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "num_quantiles": args.num_quantiles,
        "dropout": args.dropout,
        "action_limit": 1.0,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "alpha": args.alpha,
        "alpha_lr": args.alpha_lr,
        "discount": args.discount,
        "tau": args.tau,
        "reward_scale": args.reward_scale,
        "n_step": args.n_step,
        "actor_quantile_fraction": args.actor_quantile_fraction,
    }


def build_agent(
    args: argparse.Namespace,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> tuple[ContinuousSACAgent, dict[str, Any]]:
    obs_dim = int(np.prod(observation_space.shape))
    action_dim = int(np.prod(action_space.shape))
    model_config = _model_config(args, obs_dim, action_dim)
    network = ContinuousSAC(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        obs_dim=obs_dim,
        h_dim=args.hidden_dim,
        action_dim=action_dim,
        action_limit=1.0,
        dropout=args.dropout,
        num_quantiles=args.num_quantiles,
        alpha=args.alpha,
        alpha_lr=args.alpha_lr,
        device=args.device,
    )
    agent_config = Config(
        discount=args.discount,
        params=str(args.run_dir),
        tau=args.tau,
        capacity=args.capacity,
        epoch=1,
        reward_scale=args.reward_scale,
        n_step=args.n_step,
        critic_update_factor=1,
        actor_quantile_fraction=args.actor_quantile_fraction,
    )
    return ContinuousSACAgent("command_bipedal", network, agent_config), model_config


@torch.no_grad()
def batched_actions(
    agent: ContinuousSACAgent,
    observations: np.ndarray,
    deterministic: bool = False,
) -> np.ndarray:
    states = torch.as_tensor(observations, dtype=torch.float32, device=agent.net.device)
    agent.net.eval()
    actions, _ = agent.net.actor(states, deterministic=deterministic)
    agent.net.train()
    return actions.cpu().numpy().astype(np.float32, copy=False)


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(
    path: Path,
    agent: ContinuousSACAgent,
    trainer_state: dict[str, Any],
    model_config: dict[str, Any],
    env_config: CommandBipedalConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": agent.net.state_dict(),
        "target_model": agent.target_net.state_dict(),
        "actor_opt": agent.net.actor_opt.state_dict(),
        "critic_opt": agent.net.critic_opt.state_dict(),
        "alpha_opt": agent.net.alpha_opt.state_dict(),
        "trainer_state": trainer_state,
        "model_config": model_config,
        "env_config": asdict(env_config),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    agent: ContinuousSACAgent,
    expected_model_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = _torch_load(path, agent.net.device)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format_version')}")
    model_config = payload["model_config"]
    if expected_model_config is not None and model_config != expected_model_config:
        raise ValueError("checkpoint model configuration does not match training arguments")
    agent.net.load_state_dict(payload["model"], strict=True)
    agent.target_net.load_state_dict(payload["target_model"], strict=True)
    agent.net.actor_opt.load_state_dict(payload["actor_opt"])
    agent.net.critic_opt.load_state_dict(payload["critic_opt"])
    agent.net.alpha_opt.load_state_dict(payload["alpha_opt"])
    return payload.get("trainer_state", {}), model_config, payload["env_config"]


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def _append_csv(path: Path, data: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data))
        if write_header:
            writer.writeheader()
        writer.writerow(data)


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    """Bound disk use while preserving latest.pt and the newest numbered saves."""
    if keep == 0:
        return
    checkpoints = sorted(run_dir.glob("checkpoint_*.pt"))
    for checkpoint in checkpoints[:-keep]:
        checkpoint.unlink()


def _vector_info_value(
    infos: dict[str, Any], key: str, index: int, default: float = 0.0
) -> float:
    if key not in infos:
        return default
    mask = infos.get(f"_{key}")
    if mask is not None and not bool(mask[index]):
        return default
    return float(infos[key][index])


def evaluate(
    agent: ContinuousSACAgent,
    env_config: CommandBipedalConfig,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    env = make_command_env(seed, env_config, command_mode="cycle")
    returns: list[float] = []
    velocity_errors: list[float] = []
    acceleration_errors: list[float] = []
    standing_heights: list[float] = []
    stride_lengths: list[float] = []
    swing_clearances: list[float] = []
    single_support: list[float] = []
    alternating_steps: list[float] = []
    support_legs: list[float] = []
    airborne: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            while True:
                action = agent.action(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                velocity_errors.append(float(info["velocity_error"]) ** 2)
                acceleration_errors.append(float(info["acceleration_error"]) ** 2)
                if float(info["stand_gate"]) > 0.8:
                    standing_heights.append(float(info["height"]))
                if float(info["movement_gate"]) > 0.8:
                    support = float(info["single_support"])
                    single_support.append(support)
                    alternating_steps.append(float(info["alternating_step"]))
                    airborne.append(float(info["airborne"]))
                    if support > 0.5:
                        stride_lengths.append(float(info["stride_length"]))
                        swing_clearances.append(float(info["swing_clearance"]))
                        support_legs.append(float(info["support_leg"]))
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_velocity_rmse": float(np.sqrt(np.mean(velocity_errors))),
        "eval_acceleration_rmse": float(np.sqrt(np.mean(acceleration_errors))),
        "eval_standing_height": float(np.mean(standing_heights))
        if standing_heights
        else 0.0,
        "eval_stride_length": float(np.mean(stride_lengths))
        if stride_lengths
        else 0.0,
        "eval_swing_clearance": float(np.mean(swing_clearances))
        if swing_clearances
        else 0.0,
        "eval_single_support_rate": float(np.mean(single_support))
        if single_support
        else 0.0,
        "eval_alternating_step_rate_hz": float(FPS * np.mean(alternating_steps))
        if alternating_steps
        else 0.0,
        "eval_support_balance_error": float(abs(np.mean(support_legs)))
        if support_legs
        else 1.0,
        "eval_airborne_rate": float(np.mean(airborne)) if airborne else 0.0,
    }


def _trainer_state(
    global_step: int,
    completed_episodes: int,
    gradient_updates: int,
    started_at: float,
) -> dict[str, Any]:
    return {
        "global_step": global_step,
        "completed_episodes": completed_episodes,
        "gradient_updates": gradient_updates,
        "elapsed_seconds": time.time() - started_at,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    args = prepare_training_args(args)
    args.run_dir = Path(args.run_dir).resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_config = build_env_config(args)
    vector_env = make_vector_env(
        args.num_envs, args.seed, env_config, args.vector_mode, command_mode="random"
    )
    agent, model_config = build_agent(
        args, vector_env.single_observation_space, vector_env.single_action_space
    )
    global_step = 0
    completed_episodes = 0
    gradient_updates = 0
    elapsed_before_resume = 0.0
    restored_recent_returns: list[float] = []
    restored_last_metrics: dict[str, Any] = {}
    if args.resume is not None:
        restored, _, restored_env_config = load_checkpoint(
            Path(args.resume), agent, expected_model_config=model_config
        )
        if restored_env_config != asdict(env_config):
            raise ValueError("checkpoint environment configuration does not match")
        global_step = int(restored.get("global_step", 0))
        completed_episodes = int(restored.get("completed_episodes", 0))
        gradient_updates = int(restored.get("gradient_updates", 0))
        elapsed_before_resume = float(restored.get("elapsed_seconds", 0.0))
        restored_recent_returns = [
            float(value) for value in restored.get("recent_returns", [])
        ][-100:]
        restored_last_metrics = dict(restored.get("last_metrics", {}))

    config_record = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": model_config,
        "environment": asdict(env_config),
    }
    _atomic_json(args.run_dir / "config.json", config_record)

    observations, _ = vector_env.reset(seed=args.seed)
    episode_caches: list[list[Transition]] = [[] for _ in range(args.num_envs)]
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    episode_velocity_sse = np.zeros(args.num_envs, dtype=np.float64)
    episode_acceleration_sse = np.zeros(args.num_envs, dtype=np.float64)
    episode_standing_height_sum = np.zeros(args.num_envs, dtype=np.float64)
    episode_standing_height_count = np.zeros(args.num_envs, dtype=np.int64)
    needs_reset = np.zeros(args.num_envs, dtype=bool)
    recent_returns = restored_recent_returns
    last_metrics = restored_last_metrics
    started_at = time.time() - elapsed_before_resume
    next_checkpoint = (
        (global_step // args.checkpoint_every) + 1
    ) * args.checkpoint_every
    next_evaluation = ((global_step // args.eval_every) + 1) * args.eval_every
    next_status = ((global_step // args.status_every) + 1) * args.status_every

    try:
        while global_step < args.total_steps:
            reset_before_step = needs_reset.copy()
            if global_step < args.random_steps:
                actions = np.asarray(
                    [vector_env.single_action_space.sample() for _ in range(args.num_envs)],
                    dtype=np.float32,
                )
            else:
                actions = batched_actions(agent, observations)

            next_observations, rewards, terminated, truncated, infos = vector_env.step(
                actions
            )
            valid_transitions = 0
            for index in range(args.num_envs):
                if reset_before_step[index]:
                    needs_reset[index] = False
                    continue
                done = bool(terminated[index] or truncated[index])
                episode_caches[index].append(
                    (
                        np.asarray(observations[index], dtype=np.float32).copy(),
                        np.asarray(actions[index], dtype=np.float32).copy(),
                        float(rewards[index]),
                        np.asarray(next_observations[index], dtype=np.float32).copy(),
                        bool(terminated[index]),
                        bool(truncated[index]),
                    )
                )
                episode_returns[index] += float(rewards[index])
                episode_lengths[index] += 1
                velocity_error = _vector_info_value(infos, "velocity_error", index)
                acceleration_error = _vector_info_value(
                    infos, "acceleration_error", index
                )
                episode_velocity_sse[index] += velocity_error**2
                episode_acceleration_sse[index] += acceleration_error**2
                if _vector_info_value(infos, "stand_gate", index) > 0.8:
                    episode_standing_height_sum[index] += _vector_info_value(
                        infos, "height", index
                    )
                    episode_standing_height_count[index] += 1
                valid_transitions += 1
                if done:
                    flush_episode(agent, episode_caches[index])
                    episode_caches[index].clear()
                    completed_episodes += 1
                    recent_returns.append(float(episode_returns[index]))
                    recent_returns = recent_returns[-100:]
                    episode_record = {
                        "type": "episode",
                        "global_step": global_step + valid_transitions,
                        "episode": completed_episodes,
                        "return": float(episode_returns[index]),
                        "length": int(episode_lengths[index]),
                        "velocity_rmse": float(
                            np.sqrt(
                                episode_velocity_sse[index]
                                / max(1, episode_lengths[index])
                            )
                        ),
                        "acceleration_rmse": float(
                            np.sqrt(
                                episode_acceleration_sse[index]
                                / max(1, episode_lengths[index])
                            )
                        ),
                        "standing_height": float(
                            episode_standing_height_sum[index]
                            / max(1, episode_standing_height_count[index])
                        ),
                    }
                    _append_jsonl(args.run_dir / "metrics.jsonl", episode_record)
                    _append_csv(args.run_dir / "episodes.csv", episode_record)
                    episode_returns[index] = 0.0
                    episode_lengths[index] = 0
                    episode_velocity_sse[index] = 0.0
                    episode_acceleration_sse[index] = 0.0
                    episode_standing_height_sum[index] = 0.0
                    episode_standing_height_count[index] = 0
                    needs_reset[index] = True

            observations = next_observations
            global_step += valid_transitions

            if global_step >= args.learning_starts:
                for _ in range(args.gradient_steps_per_vector_step):
                    metrics = agent.step(
                        args.batch_size,
                        update_actor=global_step >= args.actor_learning_starts,
                    )
                    if metrics is not None:
                        gradient_updates += 1
                        last_metrics = metrics

            while global_step >= next_evaluation:
                evaluation = evaluate(
                    agent, env_config, args.eval_episodes, args.seed + 100_000
                )
                evaluation.update(
                    {"type": "evaluation", "global_step": global_step}
                )
                _append_jsonl(args.run_dir / "metrics.jsonl", evaluation)
                _append_csv(args.run_dir / "evaluations.csv", evaluation)
                next_evaluation += args.eval_every

            while global_step >= next_checkpoint:
                state = _trainer_state(
                    global_step, completed_episodes, gradient_updates, started_at
                )
                checkpoint_state = {
                    **state,
                    "recent_returns": recent_returns[-100:],
                    "last_metrics": last_metrics,
                }
                checkpoint = args.run_dir / f"checkpoint_{global_step:09d}.pt"
                save_checkpoint(
                    checkpoint, agent, checkpoint_state, model_config, env_config
                )
                save_checkpoint(
                    args.run_dir / "latest.pt",
                    agent,
                    checkpoint_state,
                    model_config,
                    env_config,
                )
                prune_checkpoints(args.run_dir, args.keep_checkpoints)
                next_checkpoint += args.checkpoint_every

            if global_step >= next_status:
                status = _trainer_state(
                    global_step, completed_episodes, gradient_updates, started_at
                )
                status.update(
                    {
                        "buffer_size": len(agent.buffer),
                        "recent_return_mean": float(np.mean(recent_returns))
                        if recent_returns
                        else 0.0,
                        **last_metrics,
                    }
                )
                _atomic_json(args.run_dir / "status.json", status)
                next_status += args.status_every
    finally:
        for cache in episode_caches:
            flush_episode(agent, cache)
        vector_env.close()

    state = _trainer_state(
        global_step, completed_episodes, gradient_updates, started_at
    )
    checkpoint_state = {
        **state,
        "recent_returns": recent_returns[-100:],
        "last_metrics": last_metrics,
    }
    save_checkpoint(
        args.run_dir / "latest.pt",
        agent,
        checkpoint_state,
        model_config,
        env_config,
    )
    final_status = {
        **state,
        "buffer_size": len(agent.buffer),
        "recent_return_mean": float(np.mean(recent_returns))
        if recent_returns
        else 0.0,
        **last_metrics,
        "completed": True,
    }
    _atomic_json(args.run_dir / "status.json", final_status)
    return final_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QRSAC to follow bidirectional velocity commands."
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qrsac_command_bipedal"))
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--vector-mode", choices=("async", "sync"), default="async")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-quantiles", type=int, default=51)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--capacity", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--actor-learning-starts", type=int)
    parser.add_argument("--random-steps", type=int)
    parser.add_argument(
        "--gradient-steps-per-vector-step",
        type=int,
        default=1,
        help="Use num-envs for an update-to-data ratio near 1.",
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--actor-quantile-fraction", type=float, default=1.0)
    parser.add_argument("--command-speed", type=float, default=3.0)
    parser.add_argument("--minimum-command-speed", type=float, default=1.0)
    parser.add_argument("--command-hold-min-seconds", type=float, default=2.0)
    parser.add_argument("--command-hold-max-seconds", type=float, default=5.0)
    parser.add_argument("--standing-probability", type=float, default=0.30)
    parser.add_argument("--settling-time", type=float, default=1.5)
    parser.add_argument("--reference-damping", type=float, default=1.0)
    parser.add_argument("--acceleration-limit", type=float, default=3.0)
    parser.add_argument("--jerk-limit", type=float, default=12.0)
    parser.add_argument("--acceleration-filter", type=float, default=0.85)
    parser.add_argument("--action-penalty-weight", type=float, default=0.004)
    parser.add_argument("--action-rate-penalty-weight", type=float, default=0.008)
    parser.add_argument("--gait-reward-weight", type=float, default=0.5)
    parser.add_argument("--target-stride-length", type=float, default=1.0)
    parser.add_argument("--target-swing-clearance", type=float, default=0.28)
    parser.add_argument("--max-support-seconds", type=float, default=0.7)
    parser.add_argument("--alternating-step-reward-weight", type=float, default=0.5)
    parser.add_argument("--support-stall-penalty-weight", type=float, default=0.5)
    parser.add_argument("--airborne-penalty-weight", type=float, default=0.25)
    parser.add_argument("--max-episode-steps", type=int, default=1600)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=20,
        help="Keep this many numbered checkpoints; use 0 to retain all.",
    )
    parser.add_argument("--status-every", type=int, default=1_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
