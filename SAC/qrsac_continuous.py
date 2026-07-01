import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import NAdam, SGD
from dataclasses import dataclass
from replaybuffer import ReplayBuffer
from copy import deepcopy
from torch.distributions import Beta

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import NetworkBase, AgentBase, quantile_huber_loss, ResidualBlock


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 100000
    epoch: int = 10
    reward_scale: float = 5
    n_step: int = 5
    critic_update_factor: int = 1


class ContinuousSAC(NetworkBase):
    def __init__(self, actor_lr, critic_lr, obs_dim, h_dim, action_dim, action_limit=1., dropout=0., num_quantiles=51, alpha=0.2, alpha_lr=1e-2, computes_grad=True, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                    ResidualBlock(h_dim, h_dim, dropout),)
        self.b_alpha = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                                     ResidualBlock(h_dim, action_dim))
        self.b_beta = deepcopy(self.b_alpha)
        self.q1 = nn.Sequential(ResidualBlock(obs_dim + action_dim, h_dim),
                                ResidualBlock(h_dim, h_dim),
                                ResidualBlock(h_dim, num_quantiles))
        self.q2 = deepcopy(self.q1)
        self.alpha = nn.Parameter(torch.tensor([[math.log(alpha)]]), requires_grad=True)
        self.alpha_opt = SGD([self.alpha], lr=alpha_lr)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_limit = action_limit
        self.num_quantiles = num_quantiles
        self.apply(self.init_weights)

        self.actor_opt = NAdam([{'params': self.hidden.parameters()},
                                {'params': self.b_alpha.parameters()},
                                {'params': self.b_beta.parameters()}], lr=actor_lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.critic_opt = NAdam([{'params': self.q1.parameters()}, {'params': self.q2.parameters()}],
                                lr=critic_lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.computes_grad(computes_grad)
        self.device = device
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def actor(self, state, deterministic=False):
        hidden = self.hidden(state)
        alpha = torch.exp(self.b_alpha(hidden)) + 1
        beta = torch.exp(self.b_beta(hidden)) + 1
        dist = Beta(alpha, beta)
        if bool(deterministic):
            raw_action = alpha / (alpha + beta)
        else:
            raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action) - math.log(2 * self.action_limit)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = (raw_action - 0.5) * 2
        return action * self.action_limit, log_prob

    def critic(self, state, action):
        x = torch.concat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def forward(self, state):
        action, entropy = self.actor(state)
        return (action, entropy), self.critic(state, action)


class ContinuousSACAgent(AgentBase):
    def __init__(self, name, ac, config):
        self.net = ac
        self.target_net = deepcopy(ac)
        self.target_net.computes_grad(False)
        self.buffer = ReplayBuffer(ac.obs_dim, config.capacity, self.net.action_dim,
                                   config.discount, config.n_step, ac.device)

        self.name = name
        self.action_dim = ac.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.critic_update_factor = config.critic_update_factor
        self.target_entropy = -self.action_dim
        self.qr_tau = torch.linspace(0.5 / self.net.num_quantiles, 1 - 0.5 / self.net.num_quantiles,
                                     self.net.num_quantiles).to(ac.device).view(1, -1)
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(self.net.device).reshape(1, -1)
        self.net.eval()
        action, _ = self.net.actor(state, deterministic)
        self.net.train()
        if action.numel() == 1:
            return float(action.cpu())
        return action.cpu().numpy().squeeze(0)

    @property
    @torch.no_grad()
    def alpha(self):
        return min(float(self.net.alpha.exp().item()), 1)

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        next_pi, next_log_prob = self.net.actor(next_state)
        next_q1, next_q2 = self.target_net.critic(next_state, next_pi)
        next_q = torch.minimum(next_q1, next_q2)
        return reward + (self.discount ** n) * (next_q - self.alpha * next_log_prob) * (1 - terminated)

    def step(self, batch_size=128):
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                state, action, reward0, next_state, terminated, truncated, n = self.buffer.sample(batch_size)
                reward = reward0 * self.reward_scale
                self.target_net.eval()
                self.net.train()

                td_target = self.td_target(reward, next_state, terminated, n)
                q1, q2 = self.net.critic(state, action)
                self.net.critic_opt.zero_grad()
                critic_loss = quantile_huber_loss(q1, td_target, self.qr_tau) + \
                    quantile_huber_loss(q2, td_target, self.qr_tau)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(list(self.net.q1.parameters()) + list(self.net.q2.parameters()), 0.5)
                self.net.critic_opt.step()

                self.net.actor_opt.zero_grad()
                pi, log_prob = self.net.actor(state)
                q1, q2 = self.net.critic(state, pi)
                q_pi = torch.minimum(q1, q2)
                actor_loss = (self.alpha * log_prob - q_pi.mean(dim=-1, keepdim=True)).mean()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(self.net.hidden.parameters()) + list(self.net.b_alpha.parameters()) + list(self.net.b_beta.parameters()), 0.5)
                self.net.actor_opt.step()

                alpha_loss = -(self.net.alpha * (log_prob.detach() + self.target_entropy)).mean()
                self.net.alpha_opt.zero_grad()
                alpha_loss.backward()
                nn.utils.clip_grad_norm_(self.net.alpha, 0.1)
                self.net.alpha_opt.step()
                self.soft_update()


if __name__ == "__main__":
    update = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode='human' if not update else None)
    env = RescaleAction(env, -1, 1)
    ac = ContinuousSAC(1e-3, 3e-3, env.observation_space.shape[0],
                       256, env.action_space.shape[0], 1, 0, 51, 0.2, 1e-2, device=device)
    config = Config()
    agent = ContinuousSACAgent('test', ac, config)
    agent.load()
    agent.n_step = 10
    reward_container = []
    Loss = []
    td_error = []
    max_steps = 1600
    interval = 10
    avg = np.zeros(interval)
    best_avg = -float('inf')
    res = 0
    iterator = tqdm(range(10000))
    plt.ion()
    for i in iterator:
        state = env.reset()[0]
        episode_reward_sum = 0
        j = 0
        while True:
            j += 1
            action = agent.action(state, not update)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if reward == -100:
                reward = -10
            if bool(update):
                agent.cache(state, action, reward, next_state, terminated, truncated)
            episode_reward_sum += reward
            state = next_state
            if terminated or truncated or j > max_steps:
                if bool(update):
                    agent.process()
                break
        if bool(update) and i != 0:
            agent.step()
        reward_container.append(episode_reward_sum)
        avg[i % interval] = episode_reward_sum
        agent.save() if update else None
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
            f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}, alpha: {agent.alpha: .4f}, avg step reward: {episode_reward_sum / j: .3f}')
    env.close()
