"""Headless, resumable QRSAC training for Gymnasium BipedalWalker-v3.

The process exposes its progress through files in --run-dir, so it is safe to
launch it in the background and inspect it from another terminal or process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RescaleAction

from qrsac_continuous import Config, ContinuousSAC, ContinuousSACAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/qrsac_bipedalwalker"))
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hardcore", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-quantiles", type=int, default=51)
    parser.add_argument("--capacity", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument(
        "--random-steps",
        type=int,
        help="Random-action steps; defaults to --learning-starts for fresh training.",
    )
    parser.add_argument(
        "--deterministic-warmup",
        action="store_true",
        help="Use deterministic policy actions before --learning-starts.",
    )
    parser.add_argument("--updates-per-step", type=int, default=1)
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
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--eval-seed",
        type=int,
        help="Initial environment seed for --eval-only; defaults to --seed + 100000.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-from",
        type=Path,
        help="Transfer model weights but start a fresh run (useful for hardcore fine-tuning).",
    )
    parser.add_argument("--init-alpha", type=float, default=0.1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(source: Path, destination: Path, attempts: int = 10) -> None:
    """Handle short-lived Windows reader/antivirus locks on atomic replacements."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.01 * (2**attempt), 0.5))


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_checkpoint(path: Path, agent: ContinuousSACAgent, trainer_state: dict) -> None:
    net = agent.net
    payload = {
        "model": net.state_dict(),
        "actor_opt": net.actor_opt.state_dict(),
        "critic_opt": net.critic_opt.state_dict(),
        "alpha_opt": net.alpha_opt.state_dict(),
        "alpha": net.alpha.detach().clone(),
        "trainer_state": trainer_state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    replace_with_retry(temporary, path)


class BipedalHistoryPhase(gym.Wrapper):
    """Flatten recent observations and append an optional cyclic gait phase."""

    def __init__(self, env, frame_stack: int = 1, cpg_period_steps: int = 0):
        super().__init__(env)
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least 1")
        if cpg_period_steps < 0:
            raise ValueError("cpg_period_steps must be non-negative")
        self.frame_stack = frame_stack
        self.cpg_period_steps = cpg_period_steps
        self.phase = 0.0
        self.frames = deque(maxlen=frame_stack)
        low = np.tile(env.observation_space.low, frame_stack)
        high = np.tile(env.observation_space.high, frame_stack)
        if cpg_period_steps:
            low = np.concatenate([low, np.array([-1.0, -1.0], dtype=np.float32)])
            high = np.concatenate([high, np.array([1.0, 1.0], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _augmented_observation(self) -> np.ndarray:
        observation = np.concatenate(tuple(self.frames)).astype(np.float32, copy=False)
        if self.cpg_period_steps:
            angle = 2.0 * np.pi * self.phase
            phase = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
            observation = np.concatenate([observation, phase])
        return observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.phase = 0.0
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(np.array(observation, copy=True))
        return self._augmented_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.array(observation, copy=True))
        if self.cpg_period_steps:
            self.phase = (self.phase + 1.0 / self.cpg_period_steps) % 1.0
        return self._augmented_observation(), reward, terminated, truncated, info


def make_env(
    seed: int,
    hardcore: bool = False,
    frame_stack: int = 1,
    cpg_period_steps: int = 0,
):
    env = gym.make("BipedalWalker-v3", hardcore=hardcore)
    env = RescaleAction(env, -1.0, 1.0)
    if frame_stack > 1 or cpg_period_steps:
        env = BipedalHistoryPhase(env, frame_stack, cpg_period_steps)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


@torch.inference_mode()
def evaluate(
    agent: ContinuousSACAgent,
    episodes: int,
    seed: int,
    hardcore: bool,
    frame_stack: int = 1,
    cpg_period_steps: int = 0,
) -> dict:
    env = make_env(seed, hardcore, frame_stack, cpg_period_steps)
    returns, lengths = [], []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            while True:
                action = agent.action(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)
    finally:
        env.close()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_returns": returns,
    }


def build_agent(args: argparse.Namespace, device: str) -> tuple[ContinuousSACAgent, Config]:
    probe = make_env(args.seed, args.hardcore)
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
    )
    return ContinuousSACAgent("qrsac_bipedalwalker", net, config), config


def load_checkpoint(agent: ContinuousSACAgent, path: Path) -> dict:
    payload = torch.load(path, map_location=agent.net.device, weights_only=False)
    agent.net.load(str(path))
    agent.soft_update(tau=1.0)
    return payload.get("trainer_state", {})


def initialize_from_checkpoint(agent: ContinuousSACAgent, path: Path, alpha: float = 0.1) -> None:
    """Transfer actor/critic weights while resetting optimizers and exploration."""
    payload = torch.load(path, map_location=agent.net.device, weights_only=False)
    model_state = dict(payload["model"])
    model_state.pop("alpha", None)
    old_obs_dim = int(model_state["hidden.0.norm.weight"].shape[0])
    if old_obs_dim != agent.net.obs_dim:
        if old_obs_dim != 24 or agent.net.action_dim != 4:
            raise RuntimeError(
                f"Unsupported observation transfer: {old_obs_dim} -> {agent.net.obs_dim}"
            )
        target_state = agent.net.state_dict()
        frame_dims = agent.net.obs_dim - (2 if agent.net.obs_dim % 24 == 2 else 0)
        if frame_dims < 24 or frame_dims % 24:
            raise RuntimeError(f"Invalid augmented BipedalWalker observation size: {agent.net.obs_dim}")
        current_frame_start = frame_dims - 24
        actor_input_weights = {
            "hidden.0.glu.gate.weight",
            "hidden.0.glu.proj.weight",
        }
        critic_prefixes = ("q1.0", "q2.0")
        adapted_state = {}
        for key, target in target_state.items():
            if key == "alpha":
                continue
            source = model_state[key]
            if source.shape == target.shape:
                adapted_state[key] = source
            elif key in actor_input_weights:
                adapted = torch.zeros_like(target)
                adapted[:, current_frame_start : current_frame_start + 24] = source
                adapted_state[key] = adapted
            elif key == "hidden.0.norm.weight":
                adapted = target.clone()
                adapted[current_frame_start : current_frame_start + 24] = source
                adapted_state[key] = adapted
            elif key.startswith(critic_prefixes) and key.endswith(("glu.gate.weight", "glu.proj.weight")):
                adapted = torch.zeros_like(target)
                adapted[:, current_frame_start : current_frame_start + 24] = source[:, :24]
                adapted[:, agent.net.obs_dim :] = source[:, 24:]
                adapted_state[key] = adapted
            elif key.startswith(critic_prefixes) and key.endswith("norm.weight"):
                adapted = target.clone()
                adapted[current_frame_start : current_frame_start + 24] = source[:24]
                adapted[agent.net.obs_dim :] = source[24:]
                adapted_state[key] = adapted
            else:
                raise RuntimeError(
                    f"Unsupported checkpoint tensor expansion for {key}: "
                    f"{tuple(source.shape)} -> {tuple(target.shape)}"
                )
        model_state = adapted_state
    missing, unexpected = agent.net.load_state_dict(model_state, strict=False)
    if set(missing) != {"alpha"} or unexpected:
        raise RuntimeError(f"Incompatible transfer checkpoint: missing={missing}, unexpected={unexpected}")
    agent.net.alpha.data.fill_(float(np.log(alpha)))
    agent.soft_update(tau=1.0)


def write_episode_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> None:
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from cannot be used together")
    if args.smoke_test:
        args.total_steps = min(args.total_steps, 600)
        args.learning_starts = min(args.learning_starts, 128)
        args.batch_size = min(args.batch_size, 64)
        args.eval_every = min(args.eval_every, 300)
        args.eval_episodes = 1
        args.checkpoint_every = min(args.checkpoint_every, 300)
        args.capacity = min(args.capacity, 5_000)
    if args.random_steps is None:
        args.random_steps = args.learning_starts

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
    config_dump["run_dir"] = str(args.run_dir)
    config_dump["resume"] = str(args.resume.resolve()) if args.resume else None
    config_dump["init_from"] = str(args.init_from.resolve()) if args.init_from else None
    config_dump["resolved_device"] = device
    config_dump["agent_config"] = asdict(config)
    atomic_json(args.run_dir / "config.json", config_dump)
    (args.run_dir / "pid.txt").write_text(str(os.getpid()), encoding="ascii")

    if args.eval_only:
        eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 100_000
        result = evaluate(agent, args.eval_episodes, eval_seed, args.hardcore)
        print(json.dumps(result, ensure_ascii=False))
        return

    env = make_env(args.seed, args.hardcore)
    recent_returns = deque(maxlen=20)
    start_time = time.time()
    run_start_step = global_step
    next_eval = ((global_step // args.eval_every) + 1) * args.eval_every
    next_checkpoint = ((global_step // args.checkpoint_every) + 1) * args.checkpoint_every
    last_losses = {}

    def trainer_state() -> dict:
        return {"global_step": global_step, "episode": episode, "best_eval": best_eval}

    try:
        while global_step < args.total_steps and not stop_file.exists():
            state, _ = env.reset(seed=args.seed + episode)
            train_return = 0.0
            raw_return = 0.0
            episode_length = 0
            while global_step < args.total_steps and not stop_file.exists():
                if global_step < args.random_steps:
                    action = env.action_space.sample()
                elif args.deterministic_warmup and global_step < args.learning_starts:
                    action = agent.action(state, deterministic=True)
                else:
                    action = agent.action(state)
                next_state, raw_reward, terminated, truncated, _ = env.step(action)
                # The original QRSAC example softens the fall penalty for learning.
                reward = args.fall_penalty if raw_reward <= -99.0 else float(raw_reward)
                agent.cache(state, action, reward, next_state, terminated, truncated)
                state = next_state
                global_step += 1
                episode_length += 1
                train_return += reward
                raw_return += float(raw_reward)

                if global_step >= args.learning_starts and len(agent.buffer) >= args.batch_size:
                    for _ in range(args.updates_per_step):
                        update_metrics = agent.step(args.batch_size)
                        if update_metrics:
                            last_losses = update_metrics

                if global_step >= next_checkpoint:
                    save_checkpoint(args.run_dir / "latest.pt", agent, trainer_state())
                    next_checkpoint += args.checkpoint_every

                if global_step >= next_eval:
                    result = evaluate(agent, args.eval_episodes, args.seed + 100_000 + global_step, args.hardcore)
                    result.update({"type": "evaluation", "global_step": global_step, "episode": episode})
                    append_jsonl(args.run_dir / "metrics.jsonl", result)
                    if result["eval_return_mean"] > best_eval:
                        best_eval = result["eval_return_mean"]
                        save_checkpoint(args.run_dir / "best.pt", agent, trainer_state())
                    next_eval += args.eval_every

                if terminated or truncated:
                    break

            if agent.buffer.cache:
                agent.process()
            episode += 1
            recent_returns.append(raw_return)
            elapsed = max(time.time() - start_time, 1e-6)
            steps_per_second = max(global_step - run_start_step, 1) / elapsed
            eta_seconds = max(args.total_steps - global_step, 0) / max(steps_per_second, 1e-6)
            row = {
                "episode": episode,
                "global_step": global_step,
                "raw_return": round(raw_return, 6),
                "train_return": round(train_return, 6),
                "length": episode_length,
                "return_mean_20": round(float(np.mean(recent_returns)), 6),
                "alpha": round(agent.alpha, 8),
            }
            write_episode_csv(args.run_dir / "episodes.csv", row)
            status = {
                "state": "running",
                **row,
                **last_losses,
                "best_eval_return": None if best_eval == -float("inf") else best_eval,
                "steps_per_second": steps_per_second,
                "eta_seconds": eta_seconds,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "pid": os.getpid(),
                "device": device,
            }
            atomic_json(args.run_dir / "status.json", status)
            append_jsonl(args.run_dir / "metrics.jsonl", {"type": "episode", **status})
            print(
                f"step={global_step} episode={episode} return={raw_return:.1f} "
                f"mean20={np.mean(recent_returns):.1f} best_eval={best_eval:.1f} "
                f"alpha={agent.alpha:.4f} sps={steps_per_second:.1f}",
                flush=True,
            )
    except Exception as exc:
        atomic_json(
            args.run_dir / "status.json",
            {"state": "failed", "global_step": global_step, "episode": episode, "error": repr(exc), "pid": os.getpid()},
        )
        raise
    finally:
        env.close()

    save_checkpoint(args.run_dir / "latest.pt", agent, trainer_state())
    final_state = "stopped" if stop_file.exists() else "completed"
    final_status = {
        "state": final_state,
        "global_step": global_step,
        "episode": episode,
        "best_eval_return": None if best_eval == -float("inf") else best_eval,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(args.run_dir / "status.json", final_status)
    print(json.dumps(final_status), flush=True)


if __name__ == "__main__":
    train(parse_args())
