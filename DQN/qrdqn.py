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
from common import (
    ResidualBlock, NNBase, DQNAgentBase, quantile_huber_loss,
    make_train_test_env, single_spaces, reset_env, step_env,
    reset_done_envs, flush_episode,
)


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 5e-3
    capacity: int = 1_000_000
    epoch: int = 30
    reward_scale: float = 1.
    n_step: int = 5
    noise: float = 0.2
    min_noise: float = 0.01
    decay: float = 0.998


class QRDuelingDQN(NNBase):
    def __init__(self, lr, obs_dim, h_dim, num_actions, num_quantiles=51, dropout=0., computes_grad=True, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                    ResidualBlock(h_dim, h_dim, dropout))
        self.v = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, num_quantiles))
        self.a = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, num_actions * num_quantiles))
        self.action_dim = num_actions
        self.obs_dim = obs_dim
        self.num_quantiles = num_quantiles
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

    def forward(self, state):
        batch_size = state.shape[0]
        hidden = self.hidden(state)
        v = self.v(hidden).view(batch_size, 1, self.num_quantiles)
        a = self.a(hidden).view(batch_size, self.action_dim, self.num_quantiles)
        q = v + (a - torch.mean(a, 1, keepdim=True))
        return q


class QRDoubleDQNAgent(DQNAgentBase):
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
        self.qr_tau = torch.linspace(0.5 / Q.num_quantiles, 1 - 0.5 / Q.num_quantiles,
                                     Q.num_quantiles).to(Q.device).view(1, -1)
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == 1
        state = torch.from_numpy(state).float().to(self.net.device)
        if single_state:
            state = state.unsqueeze(0)
        self.net.eval()
        greedy_actions = self.net(state).mean(dim=2).argmax(dim=1).cpu().numpy()
        self.net.train()
        if deterministic:
            actions = greedy_actions
        else:
            explore = np.random.random(len(greedy_actions)) < self.noise
            random_actions = np.random.randint(
                0, self.n_actions, size=len(greedy_actions)
            )
            actions = np.where(explore, random_actions, greedy_actions)
        return int(actions[0]) if single_state else actions

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        next_q = self.net(next_state).mean(dim=-1)
        next_action = next_q.argmax(dim=1, keepdim=True).unsqueeze(-1).expand(-1, 1, self.net.num_quantiles)
        target_q = self.target_net(next_state).gather(1, next_action).squeeze(1)
        return reward + torch.pow(self.discount, n) * target_q * (1 - terminated)

    def loss(self, state, action, reward, next_state, terminated, truncated, n):
        # Q(St, At) <- Q(St, At) + alpha * [R_{t+1} + gamma * max_a(Q_St+1, a)} - Q(S_t, A_t)]
        batch_size = state.shape[0]
        value = self.net(state)
        action = action.view(batch_size, 1, 1).expand(batch_size, 1, self.net.num_quantiles)
        q = value.gather(1, action.long()).squeeze(1)
        td_target = self.td_target(reward, next_state, terminated, n)
        return quantile_huber_loss(q, td_target, self.qr_tau)

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
    num_envs = 16 if bool(update) else 1
    env = make_train_test_env("CartPole-v1", update, num_envs, unwrap=True)
    observation_space, action_space = single_spaces(env, update)
    action_dim = action_space.n
    obs_dim = observation_space.shape[0]
    x_threshold = env.get_attr("x_threshold")[0] if bool(update) else env.x_threshold
    theta_threshold = (env.get_attr("theta_threshold_radians")[0]
                       if bool(update) else env.theta_threshold_radians)
    Q = QRDuelingDQN(1e-3, obs_dim, 128, action_dim, 51, 0., True, device)
    config = Config()
    agent = QRDoubleDQNAgent('cartpole_qrdqn', Q, config)
    agent.load(required=not bool(update))
    agent.n_step = 5
    reward_container = []
    Loss = []
    td_error = []
    max_steps = 1000
    interval = 10
    avg = np.zeros(interval)
    best_avg = -float('inf')
    res = 0
    total_episodes = float('inf') if bool(update) else 10_000
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
        truncated = np.logical_or(truncated, episode_lengths >= max_steps)
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
            if bool(update):
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
