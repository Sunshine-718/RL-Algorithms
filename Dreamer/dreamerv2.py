import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from carracing_env import (
    wrap_carracing_observation,
    wrap_continuous_carracing_observation,
)
from Dreamer.agent import DreamerV2Agent
from Dreamer.config import Config
from Dreamer.replaybuffer import as_chw_uint8


def make_carracing_env(update, continuous):
    env = gym.make(
        "CarRacing-v3",
        continuous=continuous,
        render_mode=None if update else "human",
    )
    wrapper = (
        wrap_continuous_carracing_observation
        if continuous
        else wrap_carracing_observation
    )
    return wrapper(env)


def build_agent(env, config, device):
    observation, _ = env.reset(seed=config.seed)
    as_chw_uint8(observation)
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete = True
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete = False
        action_dim = int(np.prod(env.action_space.shape))
        if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
            env.action_space.high, 1.0
        ):
            raise ValueError("continuous actions must be rescaled to [-1, 1]")
    else:
        raise TypeError("Dreamer supports only Discrete and Box actions")
    return DreamerV2Agent(action_dim, discrete, config, device)


def prefill_buffer(env, agent, config):
    iterator = tqdm(total=config.prefill, desc="random prefill")
    while len(agent.buffer) < config.prefill:
        observation, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )
            agent.cache(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                truncated,
            )
            observation = next_observation
            done = terminated or truncated
        before = len(agent.buffer)
        agent.process()
        progress = min(
            len(agent.buffer) - before,
            iterator.total - iterator.n,
        )
        iterator.update(progress)
    iterator.close()


def train(env, agent, config):
    if len(agent.buffer) < config.prefill:
        prefill_buffer(env, agent, config)
    observation, _ = env.reset(seed=config.seed)
    agent.reset()
    episode_reward = 0.0
    episode_length = 0
    best_reward = -math.inf
    metrics = None
    iterator = tqdm(range(config.total_steps))
    for step in iterator:
        action = agent.action(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        agent.cache(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )
        episode_reward += float(reward)
        episode_length += 1
        observation = next_observation

        if step % config.train_every == 0:
            for _ in range(config.train_steps):
                metrics = agent.step()

        if terminated or truncated:
            agent.process()
            agent.save("last")
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save("best")
            description = (
                f"reward: {episode_reward:.1f}, best: {best_reward:.1f}, "
                f"length: {episode_length}"
            )
            if metrics is not None:
                description += f", model: {metrics['model_loss']:.3f}"
            iterator.set_description(description)
            observation, _ = env.reset()
            agent.reset()
            episode_reward = 0.0
            episode_length = 0


def evaluate(env, agent, episodes):
    rewards = []
    for _ in range(episodes):
        observation, _ = env.reset()
        agent.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
        print(f"episode reward: {total:.1f}")
    return rewards


if __name__ == "__main__":
    update = 1
    continuous = True
    config = Config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_carracing_env(update, continuous)
    agent = build_agent(env, config, device)
    if bool(update):
        agent.load("last")
        train(env, agent, config)
    else:
        if not agent.load("best") and not agent.load("last"):
            raise FileNotFoundError("no Dreamer V2 checkpoint found")
        evaluate(env, agent, config.eval_episodes)
    env.close()
