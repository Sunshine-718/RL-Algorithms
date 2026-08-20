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
from common import (
    NetworkBase, AgentBase, quantile_huber_loss, ResidualBlock,
    continuous_temperature_loss, make_train_test_env,
    single_spaces, reset_env, step_env,
    reset_done_envs, flush_episode,
)


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 5e-3
    capacity: int = 1_000_000
    epoch: int = 10
    reward_scale: float = 5
    n_step: int = 5
    critic_update_factor: int = 1
    actor_quantile_fraction: float = 1.0


def lower_tail_quantile_mean(q_values, fraction):
    """Mean the lower critic quantiles for optional risk-sensitive control."""
    if not 0 < fraction <= 1:
        raise ValueError("actor_quantile_fraction must be in (0, 1]")
    if fraction == 1:
        return q_values.mean(dim=-1, keepdim=True)
    count = max(1, math.ceil(q_values.shape[-1] * fraction))
    lower_tail = torch.topk(q_values, count, dim=-1, largest=False).values
    return lower_tail.mean(dim=-1, keepdim=True)


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
        alpha_logits = torch.nan_to_num(
            self.b_alpha(hidden), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        beta_logits = torch.nan_to_num(
            self.b_beta(hidden), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        alpha = torch.exp(alpha_logits) + 1
        beta = torch.exp(beta_logits) + 1
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
        critic_update_factor = config.critic_update_factor
        if isinstance(critic_update_factor, bool) or \
                not isinstance(critic_update_factor, int) or \
                critic_update_factor <= 0:
            raise ValueError("critic_update_factor must be a positive integer")
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
        self.critic_update_factor = critic_update_factor
        self.actor_quantile_fraction = config.actor_quantile_fraction
        if not 0 < self.actor_quantile_fraction <= 1:
            raise ValueError("actor_quantile_fraction must be in (0, 1]")
        self.target_entropy = -self.action_dim
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
        action, _ = self.net.actor(state, deterministic)
        self.net.train()
        actions = action.cpu().numpy()
        if not single_state:
            return actions
        if actions.shape[1] == 1:
            return float(actions[0, 0])
        return actions[0]

    @property
    @torch.no_grad()
    def alpha(self):
        return float(self.net.alpha.exp().item())

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        next_pi, next_log_prob = self.net.actor(next_state)
        next_q1, next_q2 = self.target_net.critic(next_state, next_pi)
        next_q = torch.minimum(next_q1, next_q2)
        return reward + (self.discount ** n) * (next_q - self.alpha * next_log_prob) * (1 - terminated)

    def step(self, batch_size=128, update_actor=True):
        metrics = None
        if batch_size <= len(self.buffer):
            for _ in range(self.epoch):
                critic_step_ok = True
                for _ in range(self.critic_update_factor):
                    state, action, reward0, next_state, terminated, truncated, n = self.buffer.sample(batch_size)
                    reward = reward0 * self.reward_scale
                    self.target_net.eval()
                    self.net.train()

                    td_target = self.td_target(reward, next_state, terminated, n)
                    q1, q2 = self.net.critic(state, action)
                    self.net.critic_opt.zero_grad()
                    critic_loss = quantile_huber_loss(q1, td_target, self.qr_tau) + \
                        quantile_huber_loss(q2, td_target, self.qr_tau)
                    current_critic_step_ok = bool(torch.isfinite(critic_loss))
                    if current_critic_step_ok:
                        critic_loss.backward()
                        critic_grad_norm = nn.utils.clip_grad_norm_(
                            list(self.net.q1.parameters()) + list(self.net.q2.parameters()), 0.5
                        )
                        current_critic_step_ok = bool(torch.isfinite(critic_grad_norm))
                        if current_critic_step_ok:
                            self.net.critic_opt.step()
                    self.net.critic_opt.zero_grad()
                    critic_step_ok = critic_step_ok and current_critic_step_ok

                actor_loss = torch.zeros((), device=state.device)
                alpha_loss = torch.zeros((), device=state.device)
                actor_step_ok = True
                alpha_step_ok = True
                if update_actor:
                    self.net.actor_opt.zero_grad()
                    pi, log_prob = self.net.actor(state)
                    q1, q2 = self.net.critic(state, pi)
                    q_pi = torch.minimum(q1, q2)
                    actor_value = lower_tail_quantile_mean(
                        q_pi, self.actor_quantile_fraction
                    )
                    actor_loss = (self.alpha * log_prob - actor_value).mean()
                    actor_step_ok = bool(torch.isfinite(actor_loss))
                    if actor_step_ok:
                        actor_loss.backward()
                        actor_grad_norm = nn.utils.clip_grad_norm_(
                            list(self.net.hidden.parameters())
                            + list(self.net.b_alpha.parameters())
                            + list(self.net.b_beta.parameters()),
                            0.5,
                        )
                        actor_step_ok = bool(torch.isfinite(actor_grad_norm))
                        if actor_step_ok:
                            self.net.actor_opt.step()
                    self.net.actor_opt.zero_grad()

                    alpha_loss = continuous_temperature_loss(
                        self.net.alpha, log_prob, self.target_entropy
                    )
                    self.net.alpha_opt.zero_grad()
                    alpha_step_ok = bool(torch.isfinite(alpha_loss))
                    if alpha_step_ok:
                        alpha_loss.backward()
                        alpha_grad_norm = nn.utils.clip_grad_norm_(self.net.alpha, 0.1)
                        alpha_step_ok = bool(torch.isfinite(alpha_grad_norm))
                        if alpha_step_ok:
                            self.net.alpha_opt.step()
                    self.net.alpha_opt.zero_grad()
                self.soft_update()
                metrics = {
                    "critic_loss": float(critic_loss.detach().item()),
                    "actor_loss": float(actor_loss.detach().item()),
                    "alpha_loss": float(alpha_loss.detach().item()),
                    "alpha": self.alpha,
                    "actor_updated": bool(update_actor),
                    "skipped_nonfinite_update": not (
                        critic_step_ok and actor_step_ok and alpha_step_ok
                    ),
                }
        return metrics


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 16 if bool(update) else 1
    env = make_train_test_env(
        "BipedalWalker-v3", update, num_envs,
        rescale_action=True, hardcore=False,
    )
    observation_space, action_space = single_spaces(env, update)
    ac = ContinuousSAC(1e-3, 3e-3, observation_space.shape[0],
                       256, action_space.shape[0], 1, 0, 51, 0.2, 1e-2,
                       device=device)
    config = Config()
    agent = ContinuousSACAgent('bipedalwalker_qrsac_continuous', ac, config)
    agent.load(required=not bool(update))
    agent.n_step = 10
    reward_container = []
    Loss = []
    td_error = []
    max_steps = 1600
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
        next_states, rewards, terminated, truncated, _ = step_env(
            env, actions, update
        )
        rewards = np.where(rewards == -100, -10, rewards)
        episode_lengths += 1
        truncated = np.logical_or(truncated, episode_lengths >= max_steps)
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
                f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}, alpha: {agent.alpha: .4f}, avg step reward: {episode_reward_sum / j: .3f}')
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
