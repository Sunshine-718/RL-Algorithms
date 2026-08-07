"""Control a trained command-conditioned BipedalWalker with the keyboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from command_bipedal_env import CommandBipedalConfig, make_command_env
from qrsac_continuous import Config, ContinuousSAC, ContinuousSACAgent
from train_command_bipedal_vector import load_checkpoint


def load_agent(
    checkpoint: Path, device: str
) -> tuple[ContinuousSACAgent, CommandBipedalConfig]:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = payload["model_config"]
    network = ContinuousSAC(
        actor_lr=float(model["actor_lr"]),
        critic_lr=float(model["critic_lr"]),
        obs_dim=int(model["obs_dim"]),
        h_dim=int(model["hidden_dim"]),
        action_dim=int(model["action_dim"]),
        action_limit=float(model["action_limit"]),
        dropout=float(model["dropout"]),
        num_quantiles=int(model["num_quantiles"]),
        alpha=float(model["alpha"]),
        alpha_lr=float(model["alpha_lr"]),
        device=device,
    )
    agent = ContinuousSACAgent(
        "command_bipedal",
        network,
        Config(params=None, capacity=1, epoch=1, n_step=1, reward_scale=1.0),
    )
    load_checkpoint(checkpoint, agent)
    env_values = dict(payload["env_config"])
    if "include_support_phase" not in env_values:
        env_values["include_support_phase"] = int(model["obs_dim"]) >= 41
    return agent, CommandBipedalConfig(**env_values)


def play(checkpoint: Path, device: str, seed: int) -> None:
    import pygame

    agent, env_config = load_agent(checkpoint, device)
    env = make_command_env(
        seed, env_config, command_mode="external", render_mode="human"
    )
    observation, _ = env.reset(seed=seed)
    try:
        running = True
        while running:
            reset_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_requested = True
            if not running:
                break
            if reset_requested:
                observation, _ = env.reset()

            keys = pygame.key.get_pressed()
            left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            direction = float(right) - float(left)
            env.unwrapped.set_command(direction * env_config.command_speed)
            observation = env.unwrapped.command_observation()
            action = agent.action(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(
                np.asarray(action, dtype=np.float32)
            )
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard control for a trained agent.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    play(args.checkpoint.resolve(), args.device, args.seed)


if __name__ == "__main__":
    main()
