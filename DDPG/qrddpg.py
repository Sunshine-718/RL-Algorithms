import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import NAdam, SGD
from dataclasses import dataclass
from replaybuffer import ReplayBuffer
from copy import deepcopy

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import (
    NetworkBase, AgentBase, quantile_huber_loss, ResidualBlock,
    make_train_test_env, single_spaces, reset_env, step_env,
    reset_done_envs, flush_episode,
)


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 1_000_000
    epoch: int = 30
    reward_scale: float = 1.
    n_step: int = 5
    noise: float = 0.2
    min_noise: float = 0.1
    decay: float = 0.999


class DDPG(NetworkBase):
    def __init__(self, lr, obs_dim, h_dim, action_dim, action_limit=1., dropout=0., num_quantiles=51, computes_grad=True, device='cpu'):
        super().__init__()
        self.pi = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                ResidualBlock(h_dim, h_dim, dropout),
                                ResidualBlock(h_dim, action_dim),
                                nn.Tanh())
        self.q = nn.Sequential(ResidualBlock(obs_dim + action_dim, h_dim, dropout),
                               ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, num_quantiles))
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_limit = action_limit
        self.num_quantiles = num_quantiles
        self.apply(self.init_weights)

        self.actor_opt = self.configure_optimizer(self.pi, 0.01, lr)
        self.critic_opt = self.configure_optimizer(self.q, 0.01, lr)
        self.computes_grad(computes_grad)
        self.device = device
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def actor(self, state):
        return self.pi(state) * self.action_limit

    def critic(self, state, action):
        x = torch.concat([state, action], dim=1)
        return self.q(x)

    def forward(self, state):
        action = self.actor(state)
        return action, self.critic(state, action)


class DDPGAgent(AgentBase):
    def __init__(self, name, ac, config):
        self.net = ac
        self.target_net = deepcopy(ac)
        self.target_net.computes_grad(False)
        self.buffer = ReplayBuffer(ac.obs_dim, config.capacity, self.net.action_dim,
                                   config.discount, config.n_step, ac.device)

        self.name = name
        self.n_actions = ac.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.noise = config.noise
        self.min_noise = config.min_noise
        self.decay = config.decay
        self.qr_tau = torch.linspace(0.5 / self.net.num_quantiles, 1 - 0.5 / self.net.num_quantiles,
                                     self.net.num_quantiles).to(ac.device).view(1, -1)
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == 1
        state = torch.from_numpy(state).float().to(self.net.device)
        if single_state:
            state = state.unsqueeze(0)
        self.net.eval()
        mu = self.net.actor(state).detach()
        if deterministic:
            actions = mu
        else:
            actions = mu + torch.normal(0, self.noise, mu.shape).to(mu.device)
            actions = torch.clamp(
                actions, -self.net.action_limit, self.net.action_limit
            )
        self.net.train()
        actions = actions.cpu().numpy()
        return actions[0] if single_state else actions

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        next_action = self.target_net.actor(next_state)
        next_value = self.target_net.critic(next_state, next_action)
        return reward + torch.pow(self.discount, n) * next_value * (1 - terminated)

    def step(self, batch_size=128):
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                state, action, reward, next_state, terminated, truncated, n = self.buffer.sample(batch_size)
                self.target_net.eval()
                self.net.train()

                td_target = self.td_target(reward, next_state, terminated, n)
                value = self.net.critic(state, action)
                self.net.critic_opt.zero_grad()
                critic_loss = quantile_huber_loss(value, td_target, self.qr_tau)
                critic_loss.backward()
                self.net.critic_opt.step()

                self.net.q.eval()
                self.net.actor_opt.zero_grad()
                pi = self.net.actor(state)
                actor_loss = -self.net.critic(state, pi).mean()
                actor_loss.backward()
                self.net.actor_opt.step()

                self.soft_update()
            self.decay_noise()


if __name__ == "__main__":
    update = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 16 if bool(update) else 1
    env = make_train_test_env(
        "Pendulum-v1", update, num_envs, rescale_action=True
    )
    observation_space, action_space = single_spaces(env, update)
    ac = DDPG(1e-3, observation_space.shape[0], 128,
              action_space.shape[0], 1, 0, 51, True, device)
    config = Config()
    agent = DDPGAgent('test', ac, config)
    agent.load()
    agent.n_step = 5
    reward_container = []
    Loss = []
    td_error = []
    max_steps = 1000
    interval = 10
    avg = np.zeros(interval)
    best_avg = -float('inf')
    res = 0
    total_episodes = 10000
    iterator = tqdm(total=total_episodes)
    plt.ion()
    states = reset_env(env, update)
    episode_caches = [[] for _ in range(num_envs)]
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    completed_episodes = 0
    while completed_episodes < total_episodes:
        actions = agent.action(states, not update)
        next_states, rewards, terminated, truncated, _ = step_env(
            env, actions, update
        )
        episode_lengths += 1
        truncated = np.logical_or(truncated, episode_lengths > max_steps)
        done = np.logical_or(terminated, truncated)

        for env_id in range(num_envs):
            if completed_episodes >= total_episodes:
                break
            if bool(update):
                episode_caches[env_id].append((
                    np.asarray(states[env_id]).copy(),
                    np.asarray(actions[env_id]).copy(),
                    float(rewards[env_id]),
                    np.asarray(next_states[env_id]).copy(),
                    bool(terminated[env_id]),
                    bool(truncated[env_id]),
                ))
            episode_rewards[env_id] += float(rewards[env_id])
            if not done[env_id]:
                continue

            if bool(update):
                flush_episode(agent, episode_caches[env_id])
                if completed_episodes != 0:
                    agent.step()

            i = completed_episodes
            episode_reward_sum = float(episode_rewards[env_id])
            j = int(episode_lengths[env_id])
            reward_container.append(episode_reward_sum)
            avg[i % interval] = episode_reward_sum
            agent.save()
            if i % interval == 0 and i != 0:
                plt.clf()
                plt.plot(reward_container, label='Reward')
                plt.title(f'Reward: {reward_container[-1]}')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.1)
                res = np.mean(avg)
                if res > best_avg:
                    best_avg = res
            iterator.set_description(
                f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}, avg step reward: {episode_reward_sum / j: .3f}')
            iterator.update(1)
            completed_episodes += 1
            episode_rewards[env_id] = 0
            episode_lengths[env_id] = 0

        if completed_episodes >= total_episodes:
            break
        next_states = reset_done_envs(env, next_states, done, update)
        states = next_states
    iterator.close()
    env.close()
