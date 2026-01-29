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
from common import NetworkBase, AgentBase


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 100000
    epoch: int = 30
    reward_scale: float = 1.
    n_step: int = 5
    noise: float = 0.2
    min_noise: float = 0.1
    decay: float = 0.999
    update_actor_interval: int = 2
    warmup: int = 0


class TD3(NetworkBase):
    def __init__(self, lr, obs_dim, h_dim, action_dim, action_limit=1., dropout=0., computes_grad=True, device='cpu'):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, h_dim),
                                nn.LayerNorm(h_dim),
                                nn.Dropout(dropout),
                                nn.SiLU(True),
                                nn.Linear(h_dim, h_dim),
                                nn.LayerNorm(h_dim),
                                nn.Dropout(dropout),
                                nn.SiLU(True),
                                nn.Linear(h_dim, action_dim),
                                nn.Tanh())
        self.q1 = nn.Sequential(nn.Linear(obs_dim + action_dim, h_dim),
                                nn.LayerNorm(h_dim),
                                nn.Dropout(dropout),
                                nn.SiLU(True),
                                nn.Linear(h_dim, h_dim),
                                nn.LayerNorm(h_dim),
                                nn.Dropout(dropout),
                                nn.SiLU(True),
                                nn.Linear(h_dim, 1))
        self.q2 = deepcopy(self.q1)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.action_limit = action_limit
        self.apply(self.init_weights)

        nn.init.constant_(self.pi[-2].weight, 0)

        self.actor_opt = self.configure_optimizer(self.pi, 0.01, lr)
        self.critic_opt = NAdam([{'params': self.q1.parameters()}, {'params': self.q2.parameters()}],
                                lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.computes_grad(computes_grad)
        self.device = device
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def actor(self, state):
        return self.pi(state) * self.action_limit

    def critic(self, state, action):
        x = torch.concat([state, action], dim=1)
        return self.q1(x), self.q2(x)

    def forward(self, state):
        action = self.actor(state)
        return action, self.critic(state, action)


class TD3Agent(AgentBase):
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
        self.warmup = config.warmup
        self.update_actor_interval = config.update_actor_interval
        self.time_step = 0
        self.update_step = 0
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(self.net.device).unsqueeze(0)
        self.net.eval()
        mu = self.net.actor(state).detach()
        if deterministic:
            return mu.cpu().numpy().squeeze(0)
        mu_prime = mu + torch.normal(0, self.noise, mu.shape).to(mu.device)
        mu_prime = torch.clamp(mu_prime, -self.net.action_limit, self.net.action_limit).cpu()
        self.net.train()
        mu_prime = mu_prime.numpy().squeeze(0)
        return mu_prime

    def step(self, batch_size=128, action_noise=0.2, noise_clip=0.5):
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                state, action, reward, next_state, terminated, truncated, n = self.buffer.sample(batch_size)
                self.target_net.eval()
                self.net.train()

                with torch.no_grad():
                    target_action = self.target_net.actor(next_state)
                    noise = torch.clamp(
                        torch.normal(mean=0, std=action_noise, size=target_action.shape),
                        -noise_clip,
                        noise_clip
                    ).to(self.net.device)
                    target_action = torch.clamp(
                        target_action + noise,
                        -self.net.action_limit,
                        self.net.action_limit
                    )
                    next_q1, next_q2 = self.target_net.critic(next_state, target_action)
                    next_q = torch.minimum(next_q1, next_q2)
                    td_target = reward + torch.pow(self.discount, n) * next_q * (1 - terminated)
                q1, q2 = self.net.critic(state, action)
                self.net.critic_opt.zero_grad()
                critic_loss = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)
                critic_loss.backward()
                self.net.critic_opt.step()
                self.update_step += 1
                if self.update_step % self.update_actor_interval == 0:
                    self.net.actor_opt.zero_grad()
                    q1, q2 = self.net.critic(state, self.net.actor(state))
                    q = torch.minimum(q1, q2)
                    actor_loss = -torch.mean(q)
                    actor_loss.backward()
                    self.net.actor_opt.step()
                self.soft_update()
            self.decay_noise()


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("Pendulum-v1", render_mode='human' if not update else None)
    env = RescaleAction(env, -1, 1)
    ac = TD3(1e-3, env.observation_space.shape[0], 128, env.action_space.shape[0], 1, 0, True, device)
    config = Config()
    agent = TD3Agent('test', ac, config)
    # agent.load()
    agent.n_step = 5
    reward_container = []
    Loss = []
    td_error = []
    max_steps = 1000
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
    env.close()
