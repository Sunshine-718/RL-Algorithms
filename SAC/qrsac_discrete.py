import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import NAdam, SGD
from dataclasses import dataclass
from replaybuffer import ReplayBuffer
from copy import deepcopy
from torch.distributions import Categorical

import gymnasium as gym
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import NetworkBase, AgentBase, quantile_huber_loss, ResidualBlock


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 100000
    epoch: int = 30
    reward_scale: float = 5
    n_step: int = 5
    critic_update_factor: int = 1


class DiscreteSAC(NetworkBase):
    def __init__(self, actor_lr, critic_lr, obs_dim, h_dim, action_dim, dropout=0., alpha=0.2, alpha_lr=1e-2, num_quantiles=51, computes_grad=True, device='cpu'):
        super().__init__()
        self.pi = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                ResidualBlock(h_dim, h_dim, dropout),
                                ResidualBlock(h_dim, action_dim))
        self.q1 = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout=dropout),
                                ResidualBlock(h_dim, h_dim, dropout=dropout),
                                ResidualBlock(h_dim, action_dim * num_quantiles))
        self.q2 = deepcopy(self.q1)
        self.alpha = nn.Parameter(torch.tensor([[math.log(alpha)]]), requires_grad=True)
        self.alpha_opt = SGD([self.alpha], lr=alpha_lr)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_quantiles = num_quantiles
        self.apply(self.init_weights)

        nn.init.constant_(self.pi[-1].linear.weight, 0)

        self.actor_opt = NAdam(self.pi.parameters(), lr=actor_lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.critic_opt = NAdam([{'params': self.q1.parameters()}, {'params': self.q2.parameters()}],
                                lr=critic_lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.computes_grad(computes_grad)
        self.device = device
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def actor(self, state, deterministic=False):
        logits = self.pi(state)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        if bool(deterministic):
            action = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            action = Categorical(probs).sample().unsqueeze(-1)
        return action, log_probs, probs

    def critic(self, state):
        batch_size, _ = state.shape
        return self.q1(state).reshape(batch_size, -1, self.num_quantiles), self.q2(state).reshape(batch_size, -1, self.num_quantiles)

    def forward(self, state):
        action, log_probs, probs = self.actor(state)
        return (action, log_probs, probs), self.critic(state)


class DiscreteSACAgent(AgentBase):
    def __init__(self, name, ac, config):
        self.net = ac
        self.target_net = deepcopy(ac)
        self.target_net.computes_grad(False)
        self.buffer = ReplayBuffer(ac.obs_dim, config.capacity, 1, config.discount, config.n_step, ac.device)

        self.name = name
        self.n_actions = ac.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.critic_update_factor = config.critic_update_factor
        self.target_entropy = math.log(self.n_actions) * 0.8
        self.qr_tau = torch.linspace(0.5 / self.net.num_quantiles, 1 - 0.5 / self.net.num_quantiles,
                                     self.net.num_quantiles).to(ac.device).view(1, -1)
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(self.net.device).unsqueeze(0)
        self.net.eval()
        action, _, prob = self.net.actor(state, deterministic)
        self.net.train()
        return action.item(), prob

    @property
    @torch.no_grad()
    def alpha(self):
        return min(float(self.net.alpha.exp().item()), 1)

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        _, next_log_probs, next_prob = self.net.actor(next_state)
        next_q1, next_q2 = self.target_net.critic(next_state)
        next_q = torch.minimum(next_q1, next_q2).mean(dim=-1)
        next_v = (next_prob * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
        return reward + (self.discount ** n) * next_v * (1 - terminated)

    def step(self, batch_size=128):
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                state, action, reward0, next_state, terminated, truncated, n = self.buffer.sample(batch_size)
                action = action.long()
                reward = reward0 * self.reward_scale
                self.target_net.eval()
                self.net.train()

                td_target = self.td_target(reward, next_state, terminated, n)
                q1, q2 = self.net.critic(state)
                self.net.critic_opt.zero_grad()
                action = action[:, :, None].expand(batch_size, 1, q1.shape[-1])
                critic_loss = quantile_huber_loss(q1.gather(1, action).squeeze(1), td_target, self.qr_tau) + \
                    quantile_huber_loss(q2.gather(1, action).squeeze(1), td_target, self.qr_tau)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(list(self.net.q1.parameters()) + list(self.net.q2.parameters()), 0.5)
                self.net.critic_opt.step()

                self.net.q1.eval()
                self.net.q2.eval()
                self.net.actor_opt.zero_grad()
                _, log_probs, prob = self.net.actor(state)
                q1, q2 = self.net.critic(state)
                q_pi = torch.minimum(q1, q2)
                actor_loss = (prob * (self.alpha * log_probs - q_pi.mean(dim=-1).detach())).sum(dim=1).mean()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.net.pi.parameters(), 0.5)
                self.net.actor_opt.step()

                alpha_loss = -(prob.detach() * (self.net.alpha.exp() * (log_probs.detach() + self.target_entropy))).sum(dim=1).mean()
                self.net.alpha_opt.zero_grad()
                alpha_loss.backward()
                nn.utils.clip_grad_norm_(self.net.alpha, 0.1)
                self.net.alpha_opt.step()
                self.soft_update()


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = env = gym.make("CartPole-v1", render_mode='human' if not update else None).unwrapped
    # env = RescaleAction(env, -1, 1)
    ac = DiscreteSAC(1e-3, 3e-3, env.observation_space.shape[0],
                     128, env.action_space.n, 0, 0.2, 1e-2, 51, device=device)
    config = Config()
    agent = DiscreteSACAgent('test', ac, config)
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
            action, prob = agent.action(state, not update)
            next_state, reward, terminated, truncated, _ = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = 2 * r1 + r2
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
            f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}, alpha: {agent.alpha: .4f}')
    env.close()
