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
from common import (
    NetworkBase, AgentBase, ResidualBlock, discrete_temperature_loss,
    make_train_test_env,
    single_spaces, reset_env, step_env, reset_done_envs, flush_episode,
)


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 5e-3
    capacity: int = 1_000_000
    epoch: int = 30
    reward_scale: float = 5
    n_step: int = 5
    critic_update_factor: int = 1


class DiscreteSAC(NetworkBase):
    def __init__(self, actor_lr, critic_lr, obs_dim, h_dim, action_dim, dropout=0., alpha=0.2, alpha_lr=1e-2, computes_grad=True, device='cpu'):
        super().__init__()
        self.pi = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                ResidualBlock(h_dim, h_dim, dropout),
                                ResidualBlock(h_dim, action_dim))
        self.q1 = nn.Sequential(ResidualBlock(obs_dim, h_dim, dropout),
                                ResidualBlock(h_dim, h_dim, dropout),
                                ResidualBlock(h_dim, action_dim))
        self.q2 = deepcopy(self.q1)
        self.alpha = nn.Parameter(torch.tensor([[math.log(alpha)]]), requires_grad=True)
        self.alpha_opt = SGD([self.alpha], lr=alpha_lr)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.apply(self.init_weights)

        self.actor_opt = NAdam(self.pi.parameters(), lr=actor_lr, weight_decay=0.01, decoupled_weight_decay=True)
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
        logits = self.pi(state)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        if bool(deterministic):
            action = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            action = Categorical(probs).sample().unsqueeze(-1)
        return action, log_probs, probs

    def critic(self, state):
        return self.q1(state), self.q2(state)

    def forward(self, state):
        action, log_probs, probs = self.actor(state)
        return (action, log_probs, probs), self.critic(state)


class DiscreteSACAgent(AgentBase):
    def __init__(self, name, ac, config):
        critic_update_factor = config.critic_update_factor
        if isinstance(critic_update_factor, bool) or \
                not isinstance(critic_update_factor, int) or \
                critic_update_factor <= 0:
            raise ValueError("critic_update_factor must be a positive integer")
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
        self.critic_update_factor = critic_update_factor
        self.target_entropy = math.log(self.n_actions) * 0.45
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == 1
        state = torch.from_numpy(state).float().to(self.net.device)
        if single_state:
            state = state.unsqueeze(0)
        self.net.eval()
        action, _, prob = self.net.actor(state, deterministic)
        self.net.train()
        actions = action.squeeze(-1).cpu().numpy()
        if single_state:
            return int(actions[0]), prob
        return actions, prob

    @property
    @torch.no_grad()
    def alpha(self):
        return float(self.net.alpha.exp().item())

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        _, next_log_probs, next_prob = self.net.actor(next_state)
        next_q1, next_q2 = self.target_net.critic(next_state)
        next_q = torch.minimum(next_q1, next_q2)
        next_v = (next_prob * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
        return reward + (self.discount ** n) * next_v * (1 - terminated)

    def step(self, batch_size=128):
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                for _ in range(self.critic_update_factor):
                    state, action, reward0, next_state, terminated, truncated, n = self.buffer.sample(batch_size)
                    reward = reward0 * self.reward_scale
                    self.target_net.eval()
                    self.net.train()

                    td_target = self.td_target(reward, next_state, terminated, n)
                    q1, q2 = self.net.critic(state)
                    self.net.critic_opt.zero_grad()
                    critic_loss = F.smooth_l1_loss(q1.gather(1, action.long()), td_target) + \
                        F.smooth_l1_loss(q2.gather(1, action.long()), td_target)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.net.q1.parameters()) + list(self.net.q2.parameters()), 0.5)
                    self.net.critic_opt.step()

                self.net.q1.eval()
                self.net.q2.eval()
                self.net.actor_opt.zero_grad()
                _, log_probs, prob = self.net.actor(state)
                q1, q2 = self.net.critic(state)
                q_pi = torch.minimum(q1, q2)
                actor_loss = (prob * (self.alpha * log_probs - q_pi.detach())).sum(dim=1).mean()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.net.pi.parameters(), 0.5)
                self.net.actor_opt.step()

                alpha_loss = discrete_temperature_loss(
                    self.net.alpha, log_probs, prob, self.target_entropy
                )
                self.net.alpha_opt.zero_grad()
                alpha_loss.backward()
                nn.utils.clip_grad_norm_(self.net.alpha, 0.1)
                self.net.alpha_opt.step()
                self.soft_update()


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 16 if bool(update) else 1
    env = make_train_test_env(
        "CartPole-v1", update, num_envs, unwrap=True
    )
    observation_space, action_space = single_spaces(env, update)
    x_threshold = (
        env.get_attr("x_threshold")[0] if bool(update) else env.x_threshold
    )
    theta_threshold = (
        env.get_attr("theta_threshold_radians")[0]
        if bool(update) else env.theta_threshold_radians
    )
    ac = DiscreteSAC(1e-3, 3e-3, observation_space.shape[0], 128,
                     action_space.n, 0, 0.2, 1e-2, device=device)
    config = Config()
    agent = DiscreteSACAgent('cartpole_sac_discrete', ac, config)
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
        actions, _ = agent.action(states, not update)
        next_states, rewards, terminated, truncated, _ = step_env(
            env, actions, update
        )
        x = next_states[:, 0]
        theta = next_states[:, 2]
        r1 = (x_threshold - np.abs(x)) / x_threshold - 0.8
        r2 = (theta_threshold - np.abs(theta)) / theta_threshold - 0.5
        rewards = 2 * r1 + r2
        episode_lengths += 1
        truncated = np.logical_or(truncated, episode_lengths >= max_steps)
        done = np.logical_or(terminated, truncated)
        for env_id in range(num_envs):
            if completed_episodes >= total_episodes:
                break
            if bool(update):
                episode_caches[env_id].append((
                    np.asarray(states[env_id]).copy(),
                    int(actions[env_id]),
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
                f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}, alpha: {agent.alpha: .4f}')
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
