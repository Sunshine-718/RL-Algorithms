import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import NAdam, SGD
from dataclasses import dataclass, asdict
from torch.distributions import Categorical
from replaybuffer import ReplayBuffer
from copy import deepcopy

import gymnasium as gym
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import (
    ResidualBlock, NNBase, DQNAgentBase, quantile_huber_loss,
    QuantileEmbedding, make_train_test_env, single_spaces, reset_env,
    step_env, reset_done_envs, flush_episode,
)


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 100000
    epoch: int = 30
    reward_scale: float = 1.
    n_step: int = 5
    alpha: float = 0.1


class DuelingIQN(NNBase):
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


class SoftIQNAgent(DQNAgentBase):
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
        self._alpha = torch.FloatTensor([np.log(config.alpha)]).requires_grad_(True)
        self.alpha_opt = SGD([self._alpha], 0.1)
        self.target_entropy = float(np.log(Q.action_dim))
        self.soft_update(tau=1)

    def qr_tau(self, batch_size):
        return torch.rand(batch_size, 51).to(self.net.device).view(batch_size, -1)
    
    @property
    def alpha(self):
        return max(min(float(self._alpha.exp().item()), 1), 0.05)

    @alpha.setter
    def alpha(self, val):
        self._alpha = torch.FloatTensor([np.log(val)]).requires_grad_(True)
        self.alpha_opt = SGD([self._alpha], 0.1)
        return self.alpha

    @torch.no_grad()
    def action(self, state, deterministic=False, max_tau=1.):
        state = np.asarray(state)
        single_state = state.ndim == 1
        state = torch.from_numpy(state).float().to(self.net.device)
        if single_state:
            state = state.unsqueeze(0)
        self.net.eval()
        q_value = self.net(
            state, self.qr_tau(len(state)) * max_tau
        ).mean(dim=-1)
        probs = torch.softmax(q_value / (1e-3 if deterministic else self.alpha), dim=-1)
        dist = Categorical(probs)
        actions = dist.sample().cpu().numpy()
        self.net.train()
        return int(actions[0]) if single_state else actions

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n, tau):
        next_q1 = self.net(next_state, tau)
        next_q2 = self.target_net(next_state, tau)
        next_q = torch.minimum(next_q1, next_q2)
        logsumexp = self.alpha * torch.logsumexp(next_q / self.alpha, dim=1)
        return reward + torch.pow(self.discount, n) * logsumexp * (1 - terminated)

    def loss(self, state, action, reward, next_state, terminated, truncated, n):
        # Q(St, At) <- Q(St, At) + alpha * [R_{t+1} + gamma * max_a(Q_St+1, a)} - Q(S_t, A_t)]
        batch_size = state.shape[0]
        tau = self.qr_tau(state.shape[0])
        value = self.net(state, tau)
        action = action.view(batch_size, 1, 1).expand(
            batch_size, 1, tau.shape[1]).type(torch.int64)
        q = value.gather(1, action.long()).squeeze(1)
        td_target = self.td_target(reward, next_state, terminated, n, tau)
        return quantile_huber_loss(q, td_target, tau), value.detach().mean(dim=-1)

    def step(self, batch_size=128):
        loss = None
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                self.net.opt.zero_grad()
                self.target_net.eval()
                self.net.train()
                loss, q = self.loss(*self.buffer.sample(batch_size))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.net.opt.step()
                self.alpha_opt.zero_grad()
                probs = torch.softmax(q / self.alpha, dim=-1).cpu()
                dist = Categorical(probs)
                alpha_loss = torch.exp(self._alpha).clamp_max(1.) * (dist.entropy().mean() - self.target_entropy)
                alpha_loss.backward()
                nn.utils.clip_grad_norm_([self._alpha], 0.1)
                self.alpha_opt.step()
                self.soft_update()
        return loss.item() if loss is not None else None


if __name__ == "__main__":
    update = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 8 if bool(update) else 1
    env = make_train_test_env("CartPole-v1", update, num_envs, unwrap=True)
    observation_space, action_space = single_spaces(env, update)
    action_dim = action_space.n
    obs_dim = observation_space.shape[0]
    x_threshold = env.get_attr("x_threshold")[0] if bool(update) else env.x_threshold
    theta_threshold = (env.get_attr("theta_threshold_radians")[0]
                       if bool(update) else env.theta_threshold_radians)
    Q = DuelingIQN(1e-3, obs_dim, 128, action_dim, 0., True, device)
    config = Config()
    agent = SoftIQNAgent('test', Q, config)
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
        next_states, rewards, terminated, truncated, _ = step_env(env, actions, update)
        x, theta = next_states[:, 0], next_states[:, 2]
        rewards = (2 * ((x_threshold - np.abs(x)) / x_threshold - 0.8)
                   + (theta_threshold - np.abs(theta)) / theta_threshold - 0.5)
        episode_lengths += 1
        truncated = np.logical_or(truncated, episode_lengths > max_steps)
        done = np.logical_or(terminated, truncated)
        for env_id in range(num_envs):
            if completed_episodes >= total_episodes:
                break
            if bool(update):
                episode_caches[env_id].append((
                    np.asarray(states[env_id]).copy(), int(actions[env_id]),
                    float(rewards[env_id]), np.asarray(next_states[env_id]).copy(),
                    bool(terminated[env_id]), bool(truncated[env_id]),
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
        states = reset_done_envs(env, next_states, done, update)
    iterator.close()
    env.close()
