"""Train QRSAC joint targets with low-level PD control on BipedalWalker."""

from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm

from pd_bipedal_env import DEFAULT_KD, DEFAULT_KP, make_pd_bipedal_env
from qrsac_continuous import Config, ContinuousSAC, ContinuousSACAgent


Transition = tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, bool]


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke_test:
        if args.total_steps == 0:
            args.total_steps = 800
        else:
            args.total_steps = min(args.total_steps, 800)
        args.num_envs = min(args.num_envs, 2)
        args.learning_starts = min(args.learning_starts, 128)
        args.random_steps = min(args.random_steps, args.learning_starts)
        args.batch_size = min(args.batch_size, 64)
        args.capacity = min(args.capacity, 5_000)
        args.checkpoint_every = min(args.checkpoint_every, 400)

    if args.total_steps < 0:
        raise ValueError("total_steps cannot be negative")
    if args.num_envs < 1 or args.batch_size < 1:
        raise ValueError("num_envs and batch_size must be positive")
    if args.capacity < args.batch_size:
        raise ValueError("capacity must be at least batch_size")
    if min(args.learning_starts, args.random_steps) < 0:
        raise ValueError("learning_starts and random_steps cannot be negative")
    if args.gradient_steps < 0 or args.checkpoint_every < 1:
        raise ValueError("gradient_steps cannot be negative and checkpoint_every must be positive")
    return args


def make_vector_env(args: argparse.Namespace):
    factories = [
        partial(
            make_pd_bipedal_env,
            hardcore=args.hardcore,
            kp=tuple(args.kp),
            kd=tuple(args.kd),
            max_episode_steps=args.max_episode_steps,
        )
        for _ in range(args.num_envs)
    ]
    kwargs = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
    if args.vector_mode == "async":
        env = gym.vector.AsyncVectorEnv(factories, **kwargs)
    else:
        env = gym.vector.SyncVectorEnv(factories, **kwargs)
    env.action_space.seed(args.seed)
    env.single_action_space.seed(args.seed)
    return env


def build_agent(
    args: argparse.Namespace,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> ContinuousSACAgent:
    network = ContinuousSAC(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        obs_dim=int(np.prod(observation_space.shape)),
        h_dim=args.hidden_dim,
        action_dim=int(np.prod(action_space.shape)),
        action_limit=1.0,
        dropout=args.dropout,
        num_quantiles=args.num_quantiles,
        alpha=args.alpha,
        alpha_lr=args.alpha_lr,
        device=args.device,
    )
    config = Config(
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
    return ContinuousSACAgent("qrsac_pd_bipedalwalker", network, config)


def flush_episode(
    agent: ContinuousSACAgent, transitions: list[Transition]
) -> None:
    if not transitions:
        return
    if agent.buffer.cache:
        raise RuntimeError("shared n-step cache must be empty before flushing")
    for transition in transitions:
        agent.cache(*transition)
    agent.process()
    transitions.clear()


def _json_arguments(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def train(args: argparse.Namespace) -> dict[str, float | int | bool]:
    args = prepare_args(args)
    args.run_dir = Path(args.run_dir).resolve()
    checkpoint = args.run_dir / "qrsac_pd_bipedalwalker_last.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(
            f"{checkpoint} already exists; use --resume or another --run-dir"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(
        json.dumps(_json_arguments(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = make_vector_env(args)
    agent = build_agent(
        args, env.single_observation_space, env.single_action_space
    )
    if args.resume:
        agent.load(required=True)

    observations, _ = env.reset(seed=args.seed)
    episode_caches: list[list[Transition]] = [
        [] for _ in range(args.num_envs)
    ]
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    completed_episodes = 0
    global_step = 0
    gradient_updates = 0
    recent_returns: list[float] = []
    best_average_return = -float("inf")
    last_metrics = None
    next_checkpoint = args.checkpoint_every
    progress = tqdm(
        total=None if args.total_steps == 0 else args.total_steps,
        unit="step",
        dynamic_ncols=True,
    )

    try:
        while args.total_steps == 0 or global_step < args.total_steps:
            if global_step < args.random_steps:
                actions = np.asarray(
                    [
                        env.single_action_space.sample()
                        for _ in range(args.num_envs)
                    ],
                    dtype=np.float32,
                )
            else:
                actions = np.asarray(
                    agent.action(observations), dtype=np.float32
                )

            next_observations, rewards, terminated, truncated, _ = env.step(
                actions
            )
            rewards = np.where(rewards == -100.0, -10.0, rewards)
            done = np.logical_or(terminated, truncated)
            for index in range(args.num_envs):
                episode_caches[index].append(
                    (
                        np.asarray(observations[index], dtype=np.float32).copy(),
                        np.asarray(actions[index], dtype=np.float32).copy(),
                        float(rewards[index]),
                        np.asarray(
                            next_observations[index], dtype=np.float32
                        ).copy(),
                        bool(terminated[index]),
                        bool(truncated[index]),
                    )
                )
                episode_returns[index] += float(rewards[index])
                episode_lengths[index] += 1
                if not done[index]:
                    continue

                flush_episode(agent, episode_caches[index])
                completed_episodes += 1
                episode_return = float(episode_returns[index])
                recent_returns.append(episode_return)
                recent_returns = recent_returns[-10:]
                average_return = float(np.mean(recent_returns))
                progress.set_postfix(
                    episode=completed_episodes,
                    reward=f"{episode_return:.1f}",
                    average=f"{average_return:.1f}",
                    alpha=f"{agent.alpha:.3f}",
                )
                if len(recent_returns) == 10 and average_return > best_average_return:
                    best_average_return = average_return
                    agent.save("best")
                episode_returns[index] = 0.0
                episode_lengths[index] = 0

            global_step += args.num_envs
            progress.update(args.num_envs)

            if global_step >= args.learning_starts:
                for _ in range(args.gradient_steps):
                    metrics = agent.step(args.batch_size)
                    if metrics is not None:
                        last_metrics = metrics
                        gradient_updates += 1

            while global_step >= next_checkpoint:
                agent.save()
                next_checkpoint += args.checkpoint_every

            if np.any(done):
                next_observations, _ = env.reset(
                    options={"reset_mask": done.astype(np.bool_)}
                )
            observations = next_observations
    except KeyboardInterrupt:
        progress.write("Training interrupted; saving the latest checkpoint.")
    finally:
        for cache in episode_caches:
            flush_episode(agent, cache)
        agent.save()
        env.close()
        progress.close()

    result: dict[str, float | int | bool] = {
        "global_step": global_step,
        "completed_episodes": completed_episodes,
        "gradient_updates": gradient_updates,
        "buffer_size": len(agent.buffer),
        "recent_return_mean": float(np.mean(recent_returns))
        if recent_returns
        else 0.0,
        "checkpoint_saved": checkpoint.exists(),
    }
    if last_metrics is not None:
        result.update(last_metrics)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    args.run_dir = Path(args.run_dir).resolve()
    env = make_pd_bipedal_env(
        render_mode=None if args.no_render else "human",
        hardcore=args.hardcore,
        kp=args.kp,
        kd=args.kd,
        max_episode_steps=args.max_episode_steps,
    )
    agent = build_agent(args, env.observation_space, env.action_space)
    agent.load(required=True)
    returns = []
    try:
        for episode in range(args.eval_episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_return = 0.0
            while True:
                action = agent.action(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return {
        "episodes": len(returns),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train QRSAC joint targets with low-level PD control."
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/qrsac_pd_bipedalwalker")
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=0,
        help="Environment transitions; 0 trains until manually interrupted.",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--vector-mode", choices=("async", "sync"), default="async"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-quantiles", type=int, default=51)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--capacity", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--random-steps", type=int, default=10_000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--actor-quantile-fraction", type=float, default=1.0)
    parser.add_argument("--kp", type=float, nargs=4, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, nargs=4, default=DEFAULT_KD)
    parser.add_argument("--max-episode-steps", type=int, default=1600)
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--hardcore", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(prepare_args(args)) if args.evaluate else train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
