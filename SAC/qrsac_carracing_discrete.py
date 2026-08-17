import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import NAdam, SGD
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from carracing_env import OBSERVATION_SHAPE, make_carracing_env
from DQN.image_replaybuffer import ImageReplayBuffer
from qrsac_carracing import ImageEncoder
from common import (
    AgentBase,
    NetworkBase,
    discrete_temperature_loss,
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
    tau: float = 3e-2
    # State and next_state together use about 3.43 GiB at this capacity.
    capacity: int = 100_000
    epoch: int = 10
    reward_scale: float = 1.0
    n_step: int = 5


class CarRacingDiscreteActor(nn.Module):
    def __init__(self, action_dim, latent_dim=512):
        super().__init__()
        self.encoder = ImageEncoder()
        self.state_projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.policy_head = nn.Linear(latent_dim, action_dim)
        self.action_dim = action_dim

    def initialize_policy(self):
        # Start from a uniform policy instead of an arbitrary steering bias.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, state):
        hidden = self.state_projector(self.encoder(state))
        return torch.nan_to_num(
            self.policy_head(hidden), nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp(-30.0, 30.0)


class CarRacingDiscreteQuantileCritic(nn.Module):
    def __init__(
        self,
        action_dim,
        num_quantiles=51,
        latent_dim=512,
        action_embedding_dim=64,
    ):
        super().__init__()
        if action_embedding_dim < 1:
            raise ValueError("action_embedding_dim must be positive")
        self.encoder = ImageEncoder()
        self.state_projector = nn.Sequential(
            nn.Linear(self.encoder.output_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        self.action_embedding = nn.Embedding(
            action_dim, action_embedding_dim
        )
        self.action_normalizer = nn.Sequential(
            nn.LayerNorm(action_embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, num_quantiles),
        )
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.action_embedding_dim = action_embedding_dim

    def forward(self, state, action=None):
        state_features = self.state_projector(self.encoder(state))
        batch_size = state_features.shape[0]
        all_actions = action is None
        if all_actions:
            action_indices = torch.arange(
                self.action_dim, device=state_features.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            action_indices = torch.as_tensor(
                action, device=state_features.device
            ).long().reshape(batch_size, -1)
            if action_indices.shape[1] != 1:
                raise ValueError(
                    "critic expects one action per state or action=None"
                )

        action_features = self.action_normalizer(
            self.action_embedding(action_indices)
        )
        state_features = state_features.unsqueeze(1).expand(
            -1, action_indices.shape[1], -1
        )
        fused_features = torch.cat(
            [state_features, action_features], dim=-1
        )
        quantiles = self.quantile_head(fused_features)
        return quantiles if all_actions else quantiles.squeeze(1)


class CarRacingDiscreteQRSAC(NetworkBase):
    def __init__(
        self,
        actor_lr,
        critic_lr,
        action_dim,
        num_quantiles=51,
        latent_dim=512,
        action_embedding_dim=64,
        alpha=0.2,
        alpha_lr=1e-2,
        computes_grad=True,
        device="cpu",
    ):
        super().__init__()
        if action_dim < 2:
            raise ValueError("discrete QRSAC requires at least two actions")
        if num_quantiles < 1:
            raise ValueError("num_quantiles must be positive")
        if action_embedding_dim < 1:
            raise ValueError("action_embedding_dim must be positive")

        self.actor_network = CarRacingDiscreteActor(
            action_dim, latent_dim
        )
        self.q1 = CarRacingDiscreteQuantileCritic(
            action_dim,
            num_quantiles,
            latent_dim,
            action_embedding_dim,
        )
        self.q2 = CarRacingDiscreteQuantileCritic(
            action_dim,
            num_quantiles,
            latent_dim,
            action_embedding_dim,
        )
        self.alpha = nn.Parameter(
            torch.tensor([[math.log(alpha)]], dtype=torch.float32),
            requires_grad=True,
        )
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.action_embedding_dim = action_embedding_dim
        self.obs_shape = OBSERVATION_SHAPE
        self.device = torch.device(device)

        self.apply(self.init_weights)
        self.actor_network.initialize_policy()
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
        elif isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight)

    def actor(self, state, deterministic=False):
        logits = self.actor_network(state)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log(probabilities.clamp_min(1e-8))
        if deterministic:
            action = probabilities.argmax(dim=-1, keepdim=True)
        else:
            action = Categorical(probabilities).sample().unsqueeze(-1)
        return action, log_probabilities, probabilities

    def critic(self, state, action=None):
        return self.q1(state, action), self.q2(state, action)

    def set_critic_grad(self, requires_grad):
        for parameter in list(self.q1.parameters()) + list(
            self.q2.parameters()
        ):
            parameter.requires_grad_(requires_grad)

    def forward(self, state):
        actor_output = self.actor(state)
        return actor_output, self.critic(state)


class CarRacingDiscreteQRSACAgent(AgentBase):
    def __init__(self, name, network, config):
        self.net = network
        self.target_net = deepcopy(network)
        self.target_net.computes_grad(False)
        self.buffer = ImageReplayBuffer(
            network.obs_shape,
            config.capacity,
            1,
            config.discount,
            config.n_step,
            network.device,
        )

        self.name = name
        self.n_actions = network.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.target_entropy = math.log(self.n_actions) * 0.8
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
        action, _, _ = self.net.actor(state_tensor, deterministic)
        self.net.train()
        actions = action.squeeze(-1).cpu().numpy()
        return int(actions[0]) if single_state else actions

    @property
    @torch.no_grad()
    def alpha(self):
        return float(self.net.alpha.exp().item())

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        _, next_log_probabilities, next_probabilities = self.net.actor(
            next_state
        )
        next_q1, next_q2 = self.target_net.critic(next_state)
        next_q = torch.minimum(next_q1, next_q2).mean(dim=-1)
        temperature = self.net.alpha.detach().exp()
        next_value = (
            next_probabilities
            * (next_q - temperature * next_log_probabilities)
        ).sum(dim=1, keepdim=True)
        discount = torch.pow(self.discount, n)
        return reward + discount * next_value * (1.0 - terminated)

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
            chosen_q1, chosen_q2 = self.net.critic(state, action)
            critic_loss = quantile_huber_loss(
                chosen_q1, td_target, self.qr_tau
            ) + quantile_huber_loss(
                chosen_q2, td_target, self.qr_tau
            )

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
                    _, log_probabilities, probabilities = self.net.actor(
                        state
                    )
                    policy_q1, policy_q2 = self.net.critic(state)
                    policy_q = torch.minimum(
                        policy_q1, policy_q2
                    ).mean(dim=-1).detach()
                    temperature = self.net.alpha.detach().exp()
                    actor_loss = (
                        probabilities
                        * (
                            temperature * log_probabilities
                            - policy_q
                        )
                    ).sum(dim=1).mean()
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

                alpha_loss = discrete_temperature_loss(
                    self.net.alpha,
                    log_probabilities,
                    probabilities,
                    self.target_entropy,
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
    update = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4 if bool(update) else 1
    env = make_carracing_env(
        update, num_envs=num_envs, continuous=False
    )
    observation_space, action_space = single_spaces(env, update)
    if observation_space.shape != OBSERVATION_SHAPE:
        raise RuntimeError(
            f"unexpected observation shape: {observation_space.shape}"
        )
    if not hasattr(action_space, "n"):
        raise RuntimeError(f"unexpected action space: {action_space}")

    network = CarRacingDiscreteQRSAC(
        actor_lr=1e-4,
        critic_lr=3e-4,
        action_dim=action_space.n,
        num_quantiles=51,
        latent_dim=512,
        action_embedding_dim=64,
        alpha=0.2,
        alpha_lr=1e-2,
        device=device,
    )
    config = Config()
    agent = CarRacingDiscreteQRSACAgent(
        "qrsac_carracing_discrete", network, config
    )
    if not bool(update):
        agent.load()

    reward_container = []
    max_steps = 1_000
    interval = 10
    recent_rewards = np.zeros(interval)
    best_average = -float("inf")
    average_reward = 0.0
    total_episodes = 10_000
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
                        int(actions[env_id]),
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
