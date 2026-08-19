import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import SGD
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from breakout_env import OBSERVATION_SHAPE, make_breakout_env
from common import (
    DQNAgentBase,
    NNBase,
    reset_done_envs,
    reset_env,
    single_spaces,
    step_env,
)
from image_replaybuffer import ImageReplayBuffer


def weighted_quantile_huber_loss(pred, target, target_weight, tau, kappa=1.0):
    """Quantile Huber loss for a weighted empirical target distribution."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("pred and target must have shape [batch, quantiles]")
    if target.shape != target_weight.shape:
        raise ValueError("target and target_weight must have the same shape")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target batch sizes must match")
    if tau.shape[-1] != pred.shape[-1]:
        raise ValueError("tau must contain one value per predicted quantile")

    td_error = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_error = td_error.abs()
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    quantile_weight = torch.abs(
        tau.unsqueeze(-1) - (td_error.detach() < 0).float()
    )
    return (
        quantile_weight * huber * target_weight.unsqueeze(1)
    ).sum(dim=-1).mean()


@dataclass
class Config:
    discount: float = 0.99
    params: str = "./params"
    tau: float = 3e-2
    # State and next-state frame stacks use about 26.29 GiB as uint8.
    capacity: int = 500_000
    epoch: int = 30
    learning_starts: int = 20_000
    reward_scale: float = 1.0
    n_step: int = 5
    alpha: float = 0.1
    alpha_min: float = 0.05
    alpha_max: float = 1.0


class BreakoutQRDuelingNetwork(NNBase):
    def __init__(self, lr, num_actions, num_quantiles=51,
                 computes_grad=True, device="cpu"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(OBSERVATION_SHAPE[0], 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Flatten(),
        )
        self.feature_dim = self._feature_dim()
        self.hidden = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.SiLU(inplace=True),
        )
        self.value = nn.Linear(512, num_quantiles)
        self.advantage = nn.Linear(512, num_actions * num_quantiles)
        self.action_dim = num_actions
        self.num_quantiles = num_quantiles
        self.obs_shape = OBSERVATION_SHAPE
        self.device = torch.device(device)

        self.apply(self.init_weights)
        self.opt = self.configure_optimizer(0.01, lr)
        self.computes_grad(computes_grad)
        self.to(self.device)

    def _feature_dim(self):
        training = self.features.training
        self.features.eval()
        with torch.no_grad():
            features = self.features(torch.zeros(1, *OBSERVATION_SHAPE))
        self.features.train(training)
        return features.shape[1]

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state):
        if state.ndim == len(self.obs_shape):
            state = state.unsqueeze(0)
        if tuple(state.shape[1:]) != self.obs_shape:
            raise ValueError(
                f"expected input shape [B, {self.obs_shape}], "
                f"got {tuple(state.shape)}"
            )
        if state.dtype == torch.uint8:
            state = state.to(dtype=torch.float32).div_(255.0)
        else:
            state = state.to(dtype=torch.float32)

        batch_size = state.shape[0]
        hidden = self.hidden(self.features(state))
        value = self.value(hidden).view(
            batch_size, 1, self.num_quantiles
        )
        advantage = self.advantage(hidden).view(
            batch_size, self.action_dim, self.num_quantiles
        )
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class BreakoutSoftQRDQNAgent(DQNAgentBase):
    def __init__(self, name, q_network, config):
        self.net = q_network
        self.target_net = deepcopy(q_network)
        self.target_net.computes_grad(False)
        self.target_net.eval()
        self.buffer = ImageReplayBuffer(
            q_network.obs_shape,
            config.capacity,
            1,
            config.discount,
            config.n_step,
            q_network.device,
        )

        self.name = name
        self.n_actions = q_network.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.learning_starts = config.learning_starts
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.alpha_min = float(config.alpha_min)
        self.alpha_max = float(config.alpha_max)
        if not (
            np.isfinite(self.alpha_min)
            and np.isfinite(self.alpha_max)
            and 0.0 < self.alpha_min <= self.alpha_max
        ):
            raise ValueError("alpha bounds must satisfy 0 < min <= max")
        initial_alpha = float(config.alpha)
        if not np.isfinite(initial_alpha) or initial_alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        self._alpha = torch.tensor(
            [np.log(initial_alpha)], dtype=torch.float32, requires_grad=True
        )
        self._project_alpha_()
        self.alpha_opt = SGD([self._alpha], lr=0.1)
        self.target_entropy = float(np.log(q_network.action_dim)) * 0.45
        self.qr_tau = torch.linspace(
            0.5 / q_network.num_quantiles,
            1.0 - 0.5 / q_network.num_quantiles,
            q_network.num_quantiles,
            device=q_network.device,
        ).view(1, -1)
        self.soft_update(tau=1.0)

    @property
    def alpha(self):
        value = float(self._alpha.detach().exp().item())
        return max(min(value, self.alpha_max), self.alpha_min)

    @alpha.setter
    def alpha(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("alpha must be finite and positive")
        value = float(np.clip(value, self.alpha_min, self.alpha_max))
        with torch.no_grad():
            self._alpha.fill_(np.log(value))
        return self.alpha

    @torch.no_grad()
    def _project_alpha_(self):
        self._alpha.clamp_(
            min=np.log(self.alpha_min),
            max=np.log(self.alpha_max),
        )

    def load(self, model="last", required=False):
        loaded = super().load(model, required)
        if not bool(torch.isfinite(self._alpha).all().item()):
            raise ValueError("checkpoint log_alpha must be finite")
        self._project_alpha_()
        return loaded

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == len(self.net.obs_shape)
        state_tensor = torch.as_tensor(state, device=self.net.device)
        if single_state:
            state_tensor = state_tensor.unsqueeze(0)

        self.net.eval()
        q_value = self.net(state_tensor).mean(dim=-1).cpu()
        if deterministic:
            actions = q_value.argmax(dim=-1).numpy()
        else:
            probabilities = torch.softmax(q_value / self.alpha, dim=-1)
            actions = Categorical(probabilities).sample().numpy()
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

    def loss(self, state, action, reward, next_state, terminated, truncated,
             n):
        batch_size = state.shape[0]
        quantiles = self.net(state)
        action = action.view(batch_size, 1, 1).expand(
            batch_size, 1, self.net.num_quantiles
        )
        chosen_quantiles = quantiles.gather(
            1, action.long()
        ).squeeze(1)
        target_quantiles, target_weights = self.td_target(
            reward, next_state, terminated, n
        )
        loss = weighted_quantile_huber_loss(
            chosen_quantiles,
            target_quantiles,
            target_weights,
            self.qr_tau,
        )
        return loss, quantiles.detach()

    def _update_alpha(self, quantiles):
        self._project_alpha_()
        q_value = quantiles.detach().mean(dim=-1).cpu()
        probabilities = torch.softmax(
            q_value / self.alpha, dim=-1
        )
        entropy = Categorical(probabilities).entropy().mean()

        self.alpha_opt.zero_grad()
        alpha_loss = self._alpha.exp() * (
            entropy - self.target_entropy
        ).detach()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self._alpha], 0.1)
        self.alpha_opt.step()
        self._project_alpha_()
        return entropy

    def step(self, batch_size=128):
        if len(self.buffer) < max(batch_size, self.learning_starts):
            return {}

        for _ in range(self.epoch):
            self.net.opt.zero_grad()
            self.target_net.eval()
            self.net.train()
            loss, quantiles = self.loss(*self.buffer.sample(batch_size))
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.net.opt.step()

            self._update_alpha(quantiles)
            self.soft_update()

        return {"loss": loss.item(), "alpha": self.alpha}

    def soft_update(self, tau=None):
        super().soft_update(tau)
        for target_buffer, buffer in zip(
            self.target_net.buffers(), self.net.buffers()
        ):
            target_buffer.copy_(buffer)


def store_n_step_transition(agent, transition_cache, force=False):
    if not transition_cache:
        return False
    if len(transition_cache) < agent.n_step and not force:
        return False

    horizon = min(agent.n_step, len(transition_cache))
    reward = sum(
        transition_cache[index][2] * agent.discount ** index
        for index in range(horizon)
    )
    state, action = transition_cache[0][:2]
    next_state, terminated, truncated = transition_cache[horizon - 1][3:]
    agent.buffer.store(
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
        horizon,
    )
    transition_cache.pop(0)
    return True


if __name__ == "__main__":
    update = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4 if bool(update) else 1
    env = make_breakout_env(update, num_envs)
    observation_space, action_space = single_spaces(env, update)
    if observation_space.shape != OBSERVATION_SHAPE:
        raise RuntimeError(
            f"unexpected observation shape: {observation_space.shape}"
        )

    q_network = BreakoutQRDuelingNetwork(
        lr=1e-4,
        num_actions=action_space.n,
        num_quantiles=51,
        computes_grad=True,
        device=device,
    )
    config = Config()
    agent = BreakoutSoftQRDQNAgent(
        "softqrdqn_breakout_v2", q_network, config
    )
    agent.load(required=not bool(update))

    reward_container = []
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
        done = np.logical_or(terminated, truncated)

        for env_id in range(num_envs):
            if completed_episodes >= total_episodes:
                break
            if bool(update):
                episode_caches[env_id].append(
                    (
                        np.asarray(states[env_id]).copy(),
                        int(actions[env_id]),
                        float(rewards[env_id]) * agent.reward_scale,
                        np.asarray(next_states[env_id]).copy(),
                        bool(terminated[env_id]),
                        bool(truncated[env_id]),
                    )
                )
                store_n_step_transition(
                    agent, episode_caches[env_id]
                )
            episode_rewards[env_id] += float(rewards[env_id])
            if not done[env_id]:
                continue

            if bool(update):
                while episode_caches[env_id]:
                    store_n_step_transition(
                        agent, episode_caches[env_id], force=True
                    )
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
                f"avg: {average_reward: .1f}, "
                f"best avg: {best_average: .1f}, "
                f"episode length: {episode_length}"
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
