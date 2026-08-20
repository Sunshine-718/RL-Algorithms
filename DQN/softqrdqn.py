import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from dataclasses import dataclass
from replaybuffer import ReplayBuffer
from copy import deepcopy

import gymnasium as gym
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import (
    ResidualBlock, NNBase, SoftDQNAgentBase,
    weighted_quantile_huber_loss,
    make_train_test_env, single_spaces, reset_env, step_env,
    reset_done_envs, flush_episode,
)
import flappy_bird_gymnasium


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 5e-3
    capacity: int = 1_000_000
    epoch: int = 30
    reward_scale: float = 1.
    n_step: int = 5
    alpha: float = 0.1
    alpha_lr: float = 1e-2


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


class SoftQRDQNAgent(SoftDQNAgentBase):
    def __init__(self, name, Q, config):
        self.net = Q
        self.target_net = deepcopy(Q)
        self.target_net.computes_grad(False)
        self.target_net.eval()
        self.buffer = ReplayBuffer(Q.obs_dim, config.capacity, 1, config.discount, config.n_step, Q.device)

        self.name = name
        self.n_actions = Q.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.configure_alpha(config.alpha, lr=config.alpha_lr)
        self.target_entropy = float(np.log(Q.action_dim)) * 0.45
        self.qr_tau = torch.linspace(0.5 / Q.num_quantiles, 1 - 0.5 / Q.num_quantiles,
                                     Q.num_quantiles).to(Q.device).view(1, -1)
        self.soft_update(tau=1)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == 1
        self.net.eval()
        state = torch.from_numpy(state).float().reshape(
            -1, self.net.obs_dim
        ).to(self.net.device)
        q_value = self.net(state).mean(dim=-1).cpu()
        probs = torch.softmax(q_value / (1e-3 if deterministic else self.alpha), dim=-1)
        dist = Categorical(probs)
        actions = dist.sample().numpy()
        self.net.train()
        return int(actions[0]) if single_state else actions

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        training = self.net.training
        self.net.eval()
        online_quantiles = self.net(next_state)
        self.net.train(training)
        target_quantiles = self.target_net(next_state)

        log_probabilities = torch.log_softmax(
            online_quantiles.mean(dim=-1) / self.alpha,
            dim=1,
        )
        probabilities = log_probabilities.exp()
        soft_target_atoms = (
            target_quantiles
            - self.alpha * log_probabilities.unsqueeze(-1)
        )
        target_atoms = (
            reward.unsqueeze(-1)
            + torch.pow(self.discount, n).unsqueeze(-1)
            * soft_target_atoms
            * (1.0 - terminated).unsqueeze(-1)
        )
        atom_weights = probabilities.unsqueeze(-1).expand_as(
            target_quantiles
        ) / self.net.num_quantiles
        return target_atoms.flatten(1), atom_weights.flatten(1)

    def loss(self, state, action, reward, next_state, terminated, truncated, n):
        # Q(St, At) <- Q(St, At) + alpha * [R_{t+1} + gamma * max_a(Q_St+1, a)} - Q(S_t, A_t)]
        batch_size = state.shape[0]
        value = self.net(state)
        action = action.view(batch_size, 1, 1).expand(batch_size, 1, self.net.num_quantiles).long()
        q = value.gather(1, action).squeeze(1)
        target_atoms, target_weights = self.td_target(
            reward, next_state, terminated, n
        )
        loss = weighted_quantile_huber_loss(
            q, target_atoms, target_weights, self.qr_tau
        )
        return loss, value.detach()

    def step(self, batch_size=128):
        if len(self.buffer) >= batch_size:
            for _ in range(self.epoch):
                self.net.opt.zero_grad()
                self.target_net.eval()
                self.net.train()
                loss, q = self.loss(*self.buffer.sample(batch_size))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.net.opt.step()
                entropy = self._update_alpha(q)
                self.soft_update()
            metrics = self.training_metrics(loss, entropy)
            return {"loss": metrics["loss"]}
        self.training_metrics()
        return {}


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_envs = 16 if bool(update) else 1
    env = make_train_test_env("FlappyBird-v0", update, num_envs, unwrap=True)
    observation_space, action_space = single_spaces(env, update)
    action_dim = action_space.n
    obs_dim = observation_space.shape[0]
    Q = QRDuelingDQN(1e-3, obs_dim, 128, action_dim, 51, 0., True, device)
    config = Config()
    agent = SoftQRDQNAgent('flappybird_softqrdqn_v3', Q, config)
    agent.load(required=not bool(update))
    agent.n_step = 5
    training_metrics = agent.training_metrics()
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
                    training_metrics = agent.last_training_metrics
                else:
                    training_metrics = agent.training_metrics()
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
            training_status = (
                f", {agent.format_training_metrics(training_metrics)}"
                if bool(update) else ""
            )
            iterator.set_description(
                f'episode reward: {episode_reward_sum: .0f}, '
                f'avg: {res: .0f}, best avg: {best_avg: .0f}, '
                f'episode_length: {j}, '
                f'avg step reward: {episode_reward_sum / j: .3f}'
                f'{training_status}'
            )
            iterator.update(1)
            completed_episodes += 1
            episode_rewards[env_id] = 0
            episode_lengths[env_id] = 0
        if completed_episodes >= total_episodes:
            break
        states = reset_done_envs(env, next_states, done, update)
    iterator.close()
    env.close()
