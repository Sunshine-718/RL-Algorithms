import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import NAdam, SGD
from dataclasses import dataclass, asdict
from replaybuffer import ReplayBuffer
from copy import deepcopy

import gymnasium as gym
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import ResidualBlock, NNBase, DQNAgentBase, quantile_huber_loss, QuantileEmbedding


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
    decay: float = 0.99


class DueilingIQN(NNBase):
    def __init__(self, lr, obs_dim, h_dim, num_actions, dropout=0., computes_grad=True, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                    ResidualBlock(h_dim, h_dim, dropout))
        self.v = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, 1))
        self.a = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, num_actions))
        self.quantile_layer = QuantileEmbedding(h_dim, h_dim)
        self.norm = nn.LayerNorm(h_dim)
        self.action_dim = num_actions
        self.obs_dim = obs_dim
        self.apply(self.init_weights)

        self.opt = self.configure_optimizer(0.01, lr)
        self.computes_grad(computes_grad)
        self.device = device
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            # if hasattr(m, 'bias'):
            #     nn.init.constant_(m.bias, 0)

    def forward(self, state, tau):
        hidden = self.hidden(state)

        quantile_embedding = self.quantile_layer(tau)
        hidden = F.silu(self.norm(hidden.unsqueeze(1).expand(state.shape[0], tau.shape[1], -1) * quantile_embedding))

        v = self.v(hidden).permute(0, 2, 1)
        a = self.a(hidden).permute(0, 2, 1)
        q = v + (a - torch.mean(a, 1, keepdim=True))
        return q


class IQNAgent(DQNAgentBase):
    def __init__(self, name, Q, config):
        self.net = Q
        self.target_net = deepcopy(Q)
        self.target_net.computes_grad(False)
        self.buffer = ReplayBuffer(Q.obs_dim, config.capacity, 1, config.discount, config.n_step, Q.device)

        self.name = name
        self.n_actions = Q.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.noise = config.noise
        self.min_noise = config.min_noise
        self.decay = config.decay
        self.soft_update(tau=1)

    def qr_tau(self, batch_size):
        return torch.rand(batch_size, 51).to(self.net.device).view(batch_size, -1)

    @torch.no_grad()
    def action(self, state, deterministic=False, max_tau=1.):
        if not deterministic and np.random.random() < self.noise:
            action = np.random.randint(0, self.n_actions)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.net.device)
            self.net.eval()
            q_value = self.net(state, self.qr_tau(1) * max_tau).mean(dim=-1, keepdim=True)
            action = torch.argmax(q_value, dim=1).item()
            self.net.train()
        return action

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n, tau):
        batch_size = next_state.shape[0]
        next_action = self.net(next_state, tau).argmax(dim=1).reshape(next_state.shape[0], -1).unsqueeze(1)
        next_q = self.target_net(next_state, tau).gather(1, next_action).squeeze(1)
        return reward + torch.pow(self.discount, n) * next_q * (1 - terminated)

    def loss(self, state, action, reward, next_state, terminated, truncated, n):
        # Q(St, At) <- Q(St, At) + alpha * [R_{t+1} + gamma * max_a(Q_St+1, a)} - Q(S_t, A_t)]
        batch_size = state.shape[0]
        tau = self.qr_tau(state.shape[0])
        value = self.net(state, tau)
        action = action.view(batch_size, 1, 1).expand(
            batch_size, 1, tau.shape[1]).type(torch.int64)
        q = value.gather(1, action.long()).squeeze(1)
        td_target = self.td_target(reward, next_state, terminated, n, tau)
        return quantile_huber_loss(q, td_target, tau)

    def step(self, batch_size=128):
        loss = None
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                self.net.opt.zero_grad()
                self.target_net.eval()
                self.net.train()
                loss = self.loss(*self.buffer.sample(batch_size))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.net.opt.step()
                self.soft_update()
            self.decay_noise()
        return loss.item() if loss is not None else None


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("CartPole-v1", render_mode='human' if not update else None).unwrapped
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    Q = DueilingIQN(1e-3, obs_dim, 128, action_dim, 0., True, device)
    config = Config()
    agent = IQNAgent('test', Q, config)
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
            next_state, _, terminated, truncated, _ = env.step(action)
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
            f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}, avg step reward: {episode_reward_sum / j: .3f}')
    env.close()
