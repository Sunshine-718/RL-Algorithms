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
    ResidualBlock, NNBase, DQNAgentBase, make_train_test_env,
    single_spaces, reset_env, step_env, reset_done_envs, flush_episode,
)
import flappy_bird_gymnasium


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 1_000_000
    epoch: int = 30
    reward_scale: float = 1.
    n_step: int = 5
    noise: float = 0.02
    min_noise: float = 0.02
    decay: float = 0.99


class DuelingDQN(NNBase):
    def __init__(self, lr, obs_dim, h_dim, num_actions, dropout=0., computes_grad=True, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                    ResidualBlock(h_dim, h_dim, dropout))
        self.action_embed = nn.Embedding(num_actions, h_dim)
        self.next_latent = nn.Sequential(ResidualBlock(h_dim * 2, h_dim, dropout))
        self.v = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, 1))
        self.a = nn.Sequential(ResidualBlock(h_dim, h_dim, dropout),
                               ResidualBlock(h_dim, num_actions))
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

    def forward(self, state, action=None):
        hidden = self.hidden(state)
        if action is not None:
            action_embed = self.action_embed(action.view(-1).long())
            next_latent = self.next_latent(torch.cat([hidden, action_embed], dim=-1))
        
        v = self.v(hidden)
        a = self.a(hidden)
        q = v + (a - torch.mean(a, 1, keepdim=True))
        return (q, next_latent) if action is not None else (q, hidden)


class DoubleDQNAgent(DQNAgentBase):
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

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == 1
        state = torch.from_numpy(state).float().to(self.net.device)
        if single_state:
            state = state.unsqueeze(0)
        self.net.eval()
        greedy_actions = self.net(state)[0].argmax(dim=1).cpu().numpy()
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
        next_action = self.net(next_state)[0].argmax(dim=1).reshape(next_state.shape[0], -1)
        next_q = self.target_net(next_state)[0].gather(1, next_action)
        return reward + torch.pow(self.discount, n) * next_q * (1 - terminated)

    def loss(self, state, action, reward, next_state, terminated, truncated, n):
        # Q(St, At) <- Q(St, At) + alpha * [R_{t+1} + gamma * max_a(Q_St+1, a)} - Q(S_t, A_t)]
        value, next_latent = self.net(state, action)
        with torch.no_grad():
            _, next_current_latent = self.net(next_state)
        q = value.gather(1, action.long())
        td_target = self.td_target(reward, next_state, terminated, n)
        return F.smooth_l1_loss(q, td_target) - 5 * F.cosine_similarity(next_latent, next_current_latent).mean()

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
    update = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 16 if bool(update) else 1
    env = make_train_test_env(
        "FlappyBird-v0", update, num_envs, unwrap=True, use_lidar=True
    )
    observation_space, action_space = single_spaces(env, update)
    action_dim = action_space.n
    obs_dim = observation_space.shape[0]
    Q = DuelingDQN(1e-3, obs_dim, 128, action_dim, 0., True, device)
    config = Config()
    agent = DoubleDQNAgent('test', Q, config)
    agent.load()
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
