import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta
from torch.optim import NAdam, SGD
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from carracing_env import OBSERVATION_SHAPE, make_carracing_env
from DQN.image_replaybuffer import ImageReplayBuffer
from common import (
    AgentBase,
    NetworkBase,
    continuous_temperature_loss,
    flush_episode,
    quantile_huber_loss,
    reset_done_envs,
    reset_env,
    single_spaces,
    step_env,
)


@dataclass
class Config:
    discount: float = 0.99
    params: str = "./params"
    tau: float = 5e-3
    capacity: int = 1_000_000
    epoch: int = 10
    reward_scale: float = 5.0
    n_step: int = 5
    actor_quantile_fraction: float = 1.0


def lower_tail_quantile_mean(q_values, fraction):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("actor_quantile_fraction must be in (0, 1]")
    if fraction == 1.0:
        return q_values.mean(dim=-1, keepdim=True)
    count = max(1, math.ceil(q_values.shape[-1] * fraction))
    lower_tail = torch.topk(
        q_values, count, dim=-1, largest=False
    ).values
    return lower_tail.mean(dim=-1, keepdim=True)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            encoded = self.features(torch.zeros(1, *OBSERVATION_SHAPE))
        self.output_dim = encoded.shape[1]

    def forward(self, state):
        if state.ndim == len(OBSERVATION_SHAPE):
            state = state.unsqueeze(0)
        if tuple(state.shape[1:]) != OBSERVATION_SHAPE:
            raise ValueError(
                f"expected state shape [B, {OBSERVATION_SHAPE}], "
                f"got {tuple(state.shape)}"
            )
        if state.dtype == torch.uint8:
            state = state.to(dtype=torch.float32).div_(255.0)
        else:
            state = state.to(dtype=torch.float32)
        return self.features(state)


class CarRacingActor(nn.Module):
    def __init__(self, action_dim, latent_dim=512, action_limit=1.0):
        super().__init__()
        if action_dim != 3:
            raise ValueError("CarRacing actor expects three actions")
        self.encoder = ImageEncoder()
        self.state_projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.alpha_head = nn.Linear(latent_dim, action_dim)
        self.beta_head = nn.Linear(latent_dim, action_dim)
        self.action_dim = action_dim
        self.action_limit = action_limit

    def initialize_distribution(self):
        nn.init.orthogonal_(self.alpha_head.weight, gain=0.01)
        nn.init.orthogonal_(self.beta_head.weight, gain=0.01)
        nn.init.zeros_(self.alpha_head.bias)
        initial_beta = torch.tensor(
            [0.0, math.log(5.0), math.log(37.0)],
            dtype=self.beta_head.bias.dtype,
            device=self.beta_head.bias.device,
        )
        with torch.no_grad():
            self.beta_head.bias.copy_(initial_beta)

    def forward(self, state, deterministic=False):
        hidden = self.state_projector(self.encoder(state))
        alpha_logits = torch.nan_to_num(
            self.alpha_head(hidden), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        beta_logits = torch.nan_to_num(
            self.beta_head(hidden), nan=0.0, posinf=10.0, neginf=-10.0
        ).clamp(-10.0, 10.0)
        alpha = torch.exp(alpha_logits) + 1.0
        beta = torch.exp(beta_logits) + 1.0
        distribution = Beta(alpha, beta)
        if deterministic:
            raw_action = alpha / (alpha + beta)
        else:
            raw_action = distribution.rsample()
        log_prob = distribution.log_prob(raw_action)
        log_prob = log_prob - math.log(2.0 * self.action_limit)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = (raw_action - 0.5) * 2.0 * self.action_limit
        return action, log_prob


class QuantileCritic(nn.Module):
    """Fuse equally wide state and action embeddings into quantile values."""

    def __init__(self, action_dim, num_quantiles=51, latent_dim=512):
        super().__init__()
        self.state_encoder = ImageEncoder()
        self.state_projector = nn.Sequential(
            nn.Linear(self.state_encoder.output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, num_quantiles),
        )
        self.state_feature_dim = self.state_encoder.output_dim
        self.state_latent_dim = latent_dim
        self.action_latent_dim = latent_dim
        self.fused_dim = latent_dim * 2
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

    def forward(self, state, action):
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action dimension {self.action_dim}, "
                f"got {action.shape[-1]}"
            )
        state_features = self.state_projector(self.state_encoder(state))
        action_features = self.action_projector(action.to(torch.float32))
        fused = torch.cat([state_features, action_features], dim=-1)
        return self.quantile_head(fused)


class CarRacingQRSAC(NetworkBase):
    def __init__(
        self,
        actor_lr,
        critic_lr,
        action_dim,
        action_limit=1.0,
        num_quantiles=51,
        latent_dim=512,
        alpha=0.2,
        alpha_lr=1e-2,
        computes_grad=True,
        device="cpu",
    ):
        super().__init__()
        self.actor_network = CarRacingActor(
            action_dim, latent_dim, action_limit
        )
        self.q1 = QuantileCritic(action_dim, num_quantiles, latent_dim)
        self.q2 = QuantileCritic(action_dim, num_quantiles, latent_dim)
        self.alpha = nn.Parameter(
            torch.tensor([[math.log(alpha)]], dtype=torch.float32),
            requires_grad=True,
        )
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.num_quantiles = num_quantiles
        self.obs_shape = OBSERVATION_SHAPE
        self.device = torch.device(device)

        self.apply(self.init_weights)
        self.actor_network.initialize_distribution()
        self.actor_opt = NAdam(
            self.actor_network.parameters(),
            lr=actor_lr,
            weight_decay=0.01,
            decoupled_weight_decay=True,
        )
        self.critic_opt = NAdam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=critic_lr,
            weight_decay=0.01,
            decoupled_weight_decay=True,
        )
        self.alpha_opt = SGD([self.alpha], lr=alpha_lr)
        self.computes_grad(computes_grad)
        self.to(self.device)

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def actor(self, state, deterministic=False):
        return self.actor_network(state, deterministic)

    def critic(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def set_critic_grad(self, requires_grad):
        for parameter in list(self.q1.parameters()) + list(
            self.q2.parameters()
        ):
            parameter.requires_grad_(requires_grad)

    def forward(self, state):
        action, log_prob = self.actor(state)
        return (action, log_prob), self.critic(state, action)


class CarRacingQRSACAgent(AgentBase):
    def __init__(self, name, network, config):
        self.net = network
        self.target_net = deepcopy(network)
        self.target_net.computes_grad(False)
        self.buffer = ImageReplayBuffer(
            network.obs_shape,
            config.capacity,
            network.action_dim,
            config.discount,
            config.n_step,
            network.device,
            action_dtype=np.float32,
        )

        self.name = name
        self.action_dim = network.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.actor_quantile_fraction = config.actor_quantile_fraction
        if not 0.0 < self.actor_quantile_fraction <= 1.0:
            raise ValueError("actor_quantile_fraction must be in (0, 1]")
        self.target_entropy = -self.action_dim
        self.qr_tau = torch.linspace(
            0.5 / network.num_quantiles,
            1.0 - 0.5 / network.num_quantiles,
            network.num_quantiles,
            device=network.device,
        ).view(1, -1)
        self.soft_update(tau=1.0)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == len(self.net.obs_shape)
        state_tensor = torch.as_tensor(state, device=self.net.device)
        if single_state:
            state_tensor = state_tensor.unsqueeze(0)
        self.net.eval()
        action, _ = self.net.actor(state_tensor, deterministic)
        self.net.train()
        actions = action.cpu().numpy()
        return actions[0] if single_state else actions

    @property
    @torch.no_grad()
    def alpha(self):
        return float(self.net.alpha.exp().item())

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        next_action, next_log_prob = self.net.actor(next_state)
        next_q1, next_q2 = self.target_net.critic(
            next_state, next_action
        )
        next_q = torch.minimum(next_q1, next_q2)
        discount = torch.pow(self.discount, n)
        return reward + discount * (
            next_q - self.alpha * next_log_prob
        ) * (1.0 - terminated)

    def step(self, batch_size=128, update_actor=True):
        if len(self.buffer) < batch_size:
            return None

        metrics = None
        for _ in range(self.epoch):
            batch = self.buffer.sample(batch_size)
            state, action, reward, next_state, terminated, truncated, n = batch
            reward = reward * self.reward_scale
            self.target_net.eval()
            self.net.train()

            td_target = self.td_target(
                reward, next_state, terminated, n
            )
            q1, q2 = self.net.critic(state, action)
            critic_loss = quantile_huber_loss(
                q1, td_target, self.qr_tau
            ) + quantile_huber_loss(q2, td_target, self.qr_tau)
            self.net.critic_opt.zero_grad()
            critic_step_ok = bool(torch.isfinite(critic_loss))
            if critic_step_ok:
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    list(self.net.q1.parameters())
                    + list(self.net.q2.parameters()),
                    0.5,
                )
                critic_step_ok = bool(torch.isfinite(critic_grad_norm))
                if critic_step_ok:
                    self.net.critic_opt.step()
            self.net.critic_opt.zero_grad()

            actor_loss = torch.zeros((), device=state.device)
            alpha_loss = torch.zeros((), device=state.device)
            actor_step_ok = True
            alpha_step_ok = True
            if update_actor:
                self.net.actor_opt.zero_grad()
                self.net.set_critic_grad(False)
                try:
                    policy_action, log_prob = self.net.actor(state)
                    policy_q1, policy_q2 = self.net.critic(
                        state, policy_action
                    )
                    policy_q = torch.minimum(policy_q1, policy_q2)
                    actor_value = lower_tail_quantile_mean(
                        policy_q, self.actor_quantile_fraction
                    )
                    actor_loss = (
                        self.alpha * log_prob - actor_value
                    ).mean()
                    actor_step_ok = bool(torch.isfinite(actor_loss))
                    if actor_step_ok:
                        actor_loss.backward()
                        actor_grad_norm = nn.utils.clip_grad_norm_(
                            self.net.actor_network.parameters(), 0.5
                        )
                        actor_step_ok = bool(
                            torch.isfinite(actor_grad_norm)
                        )
                        if actor_step_ok:
                            self.net.actor_opt.step()
                finally:
                    self.net.set_critic_grad(True)
                self.net.actor_opt.zero_grad()

                alpha_loss = continuous_temperature_loss(
                    self.net.alpha, log_prob, self.target_entropy
                )
                self.net.alpha_opt.zero_grad()
                alpha_step_ok = bool(torch.isfinite(alpha_loss))
                if alpha_step_ok:
                    alpha_loss.backward()
                    alpha_grad_norm = nn.utils.clip_grad_norm_(
                        self.net.alpha, 0.1
                    )
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
                "skipped_nonfinite_update": not (
                    critic_step_ok and actor_step_ok and alpha_step_ok
                ),
            }
        return metrics


if __name__ == "__main__":
    update = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4 if bool(update) else 1
    env = make_carracing_env(
        update, num_envs=num_envs, continuous=True
    )
    observation_space, action_space = single_spaces(env, update)
    if observation_space.shape != OBSERVATION_SHAPE:
        raise RuntimeError(
            f"unexpected observation shape: {observation_space.shape}"
        )
    if not (
        np.allclose(action_space.low, -1.0)
        and np.allclose(action_space.high, 1.0)
    ):
        raise RuntimeError(f"unexpected action space: {action_space}")

    network = CarRacingQRSAC(
        actor_lr=1e-4,
        critic_lr=3e-4,
        action_dim=action_space.shape[0],
        action_limit=1.0,
        num_quantiles=51,
        latent_dim=512,
        alpha=0.2,
        alpha_lr=1e-2,
        device=device,
    )
    config = Config()
    agent = CarRacingQRSACAgent(
        "qrsac_carracing", network, config
    )
    agent.load(required=not bool(update))

    reward_container = []
    max_steps = 1_000
    interval = 10
    recent_rewards = np.zeros(interval)
    best_average = -float("inf")
    average_reward = 0.0
    total_episodes = float("inf") if bool(update) else 10_000
    iterator = tqdm(total=total_episodes)
    plt.ion()

    states = reset_env(env, update)
    episode_caches = [[] for _ in range(num_envs)]
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    completed_episodes = 0

    while completed_episodes < total_episodes:
        actions = agent.action(states, deterministic=not bool(update))
        next_states, rewards, terminated, truncated, _ = step_env(
            env, actions, update
        )
        episode_lengths += 1
        truncated = np.logical_or(truncated, episode_lengths >= max_steps)
        done = np.logical_or(terminated, truncated)

        for env_id in range(num_envs):
            if completed_episodes >= total_episodes:
                break
            if bool(update):
                episode_caches[env_id].append(
                    (
                        np.asarray(states[env_id]).copy(),
                        np.asarray(actions[env_id], dtype=np.float32).copy(),
                        float(rewards[env_id]),
                        np.asarray(next_states[env_id]).copy(),
                        bool(terminated[env_id]),
                        bool(truncated[env_id]),
                    )
                )
            episode_rewards[env_id] += float(rewards[env_id])
            if not done[env_id]:
                continue

            if bool(update):
                flush_episode(agent, episode_caches[env_id])
                agent.step()

            episode_index = completed_episodes
            episode_reward = float(episode_rewards[env_id])
            episode_length = int(episode_lengths[env_id])
            reward_container.append(episode_reward)
            recent_rewards[episode_index % interval] = episode_reward
            if bool(update):
                agent.save()

            if episode_index % interval == 0 and episode_index != 0:
                average_reward = float(np.mean(recent_rewards))
                if bool(update) and average_reward > best_average:
                    best_average = average_reward
                    agent.save("best")
                plt.clf()
                plt.plot(reward_container, label="Reward")
                plt.title(f"Reward: {episode_reward:.1f}")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.1)

            iterator.set_description(
                f"episode reward: {episode_reward: .0f}, "
                f"avg: {average_reward: .0f}, "
                f"best avg: {best_average: .0f}, "
                f"episode length: {episode_length}, "
                f"alpha: {agent.alpha:.4f}"
            )
            iterator.update(1)
            completed_episodes += 1
            episode_rewards[env_id] = 0.0
            episode_lengths[env_id] = 0

        if completed_episodes >= total_episodes:
            break
        states = reset_done_envs(env, next_states, done, update)

    iterator.close()
    env.close()
