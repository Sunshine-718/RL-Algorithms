from copy import deepcopy
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import SGD
from tqdm.auto import tqdm

from common import (
    DQNAgentBase,
    NNBase,
    flush_episode,
    reset_done_envs,
    reset_env,
    single_spaces,
    step_env,
)
from image_replaybuffer import ImageReplayBuffer


OBSERVATION_SHAPE = (2, 96, 96)
FRAME_SKIP = 2


@dataclass
class Config:
    discount: float = 0.99
    params: str = "./params"
    tau: float = 3e-2
    # Two uint8 frame stacks (state and next_state) use about 3.43 GiB here.
    capacity: int = 100_000
    epoch: int = 30
    reward_scale: float = 1.0
    n_step: int = 5
    alpha: float = 0.1


@dataclass(frozen=True)
class RewardShapingConfig:
    progress_weight: float = 1.0
    offtrack_weight: float = 0.2
    idle_penalty: float = 0.05
    idle_speed_threshold: float = 1.0
    idle_step_threshold: int = 20
    max_progress_delta: float = 2.0


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=FRAME_SKIP):
        super().__init__(env)
        if not isinstance(skip, int) or skip < 1:
            raise ValueError("skip must be a positive integer")
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            total_reward += float(reward)
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


class CarRacingRewardShaping(gym.Wrapper):
    def __init__(self, env, config=None):
        super().__init__(env)
        self.config = config or RewardShapingConfig()
        self._track_points = None
        self._track_vectors = None
        self._track_vector_norms = None
        self._previous_progress = None
        self._idle_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._prepare_track()
        self._previous_progress = self._track_progress()
        self._idle_steps = 0
        return observation, info

    def step(self, action):
        observation, env_reward, terminated, truncated, info = self.env.step(
            action
        )
        progress = self._track_progress()
        progress_delta = self._cyclic_progress_delta(
            progress, self._previous_progress
        )
        self._previous_progress = progress

        road_wheel_ratio = self._road_wheel_ratio()
        offtrack_ratio = 1.0 - road_wheel_ratio
        # Gating progress by road contact prevents a jump across a hairpin from
        # becoming a large positive progress reward.
        progress_reward = progress_delta * road_wheel_ratio

        speed = self._speed()
        if speed < self.config.idle_speed_threshold:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        idle_penalty = (
            self.config.idle_penalty
            if self._idle_steps > self.config.idle_step_threshold
            else 0.0
        )

        shaped_reward = (
            float(env_reward)
            + self.config.progress_weight * progress_reward
            - self.config.offtrack_weight * offtrack_ratio
            - idle_penalty
        )
        info = dict(info)
        info.update(
            {
                "reward_env": float(env_reward),
                "reward_progress": (
                    self.config.progress_weight * progress_reward
                ),
                "reward_offtrack": (
                    -self.config.offtrack_weight * offtrack_ratio
                ),
                "reward_idle": -idle_penalty,
                "track_progress_delta": progress_delta,
                "road_wheel_ratio": road_wheel_ratio,
                "speed": speed,
            }
        )
        return observation, shaped_reward, terminated, truncated, info

    def _prepare_track(self):
        self._track_points = np.asarray(
            [(point[2], point[3]) for point in self.unwrapped.track],
            dtype=np.float32,
        )
        self._track_vectors = (
            np.roll(self._track_points, -1, axis=0) - self._track_points
        )
        self._track_vector_norms = np.maximum(
            np.einsum("ij,ij->i", self._track_vectors, self._track_vectors),
            1e-8,
        )

    def _track_progress(self):
        position = np.asarray(
            self.unwrapped.car.hull.position, dtype=np.float32
        )
        relative = position - self._track_points
        projection = np.clip(
            np.einsum("ij,ij->i", relative, self._track_vectors)
            / self._track_vector_norms,
            0.0,
            1.0,
        )
        closest = self._track_points + projection[:, None] * self._track_vectors
        distance_squared = np.einsum(
            "ij,ij->i", position - closest, position - closest
        )
        segment = int(np.argmin(distance_squared))
        return float(segment + projection[segment])

    def _cyclic_progress_delta(self, current, previous):
        track_length = len(self._track_points)
        delta = (current - previous + track_length / 2) % track_length
        delta -= track_length / 2
        return float(
            np.clip(
                delta,
                -self.config.max_progress_delta,
                self.config.max_progress_delta,
            )
        )

    def _road_wheel_ratio(self):
        wheels = self.unwrapped.car.wheels
        return sum(bool(wheel.tiles) for wheel in wheels) / len(wheels)

    def _speed(self):
        velocity = np.asarray(
            self.unwrapped.car.hull.linearVelocity, dtype=np.float32
        )
        return float(np.linalg.norm(velocity))


def wrap_carracing_observation(env):
    env = FrameSkip(env, skip=FRAME_SKIP)
    env = CarRacingRewardShaping(env)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    return gym.wrappers.FrameStackObservation(env, stack_size=2)


def make_carracing_env(update, num_envs=4):
    if bool(update):
        return gym.make_vec(
            "CarRacing-v3",
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            },
            wrappers=[wrap_carracing_observation],
            continuous=False,
        )

    env = gym.make(
        "CarRacing-v3", render_mode="human", continuous=False
    )
    return wrap_carracing_observation(env)


class CarRacingQNetwork(NNBase):
    def __init__(self, lr, num_actions, computes_grad=True, device="cpu"):
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
        self.feature_dim = self._feature_dim()
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, num_actions),
        )
        self.action_dim = num_actions
        self.obs_shape = OBSERVATION_SHAPE
        self.device = torch.device(device)

        self.apply(self.init_weights)
        self.opt = self.configure_optimizer(0.01, lr)
        self.computes_grad(computes_grad)
        self.to(self.device)

    def _feature_dim(self):
        with torch.no_grad():
            features = self.features(torch.zeros(1, *OBSERVATION_SHAPE))
        return features.shape[1]

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            if module.bias is not None:
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
        return self.head(self.features(state))


class CarRacingSoftDQNAgent(DQNAgentBase):
    def __init__(self, name, q_network, config):
        self.net = q_network
        self.target_net = deepcopy(q_network)
        self.target_net.computes_grad(False)
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
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self._alpha = torch.tensor(
            [np.log(config.alpha)], dtype=torch.float32, requires_grad=True
        )
        self.alpha_opt = SGD([self._alpha], lr=0.1)
        self.target_entropy = float(np.log(q_network.action_dim)) * 0.98
        self.soft_update(tau=1.0)

    @property
    def alpha(self):
        return max(min(float(self._alpha.exp().item()), 1.0), 0.05)

    @alpha.setter
    def alpha(self, value):
        self._alpha = torch.tensor(
            [np.log(value)], dtype=torch.float32, requires_grad=True
        )
        self.alpha_opt = SGD([self._alpha], lr=0.1)
        return self.alpha

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == len(self.net.obs_shape)
        state_tensor = torch.as_tensor(state, device=self.net.device)
        if single_state:
            state_tensor = state_tensor.unsqueeze(0)

        self.net.eval()
        q_value = self.net(state_tensor).cpu()
        if deterministic:
            actions = q_value.argmax(dim=-1).numpy()
        else:
            probabilities = torch.softmax(q_value / self.alpha, dim=-1)
            actions = Categorical(probabilities).sample().numpy()
        self.net.train()
        return int(actions[0]) if single_state else actions

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        online_q = self.net(next_state)
        target_q = self.target_net(next_state)
        next_q = torch.minimum(online_q, target_q)
        soft_value = self.alpha * torch.logsumexp(
            next_q / self.alpha, dim=1, keepdim=True
        )
        return (
            reward
            + torch.pow(self.discount, n) * soft_value * (1.0 - terminated)
        )

    def loss(self, state, action, reward, next_state, terminated, truncated,
             n):
        values = self.net(state)
        q_value = values.gather(1, action.long())
        target = self.td_target(reward, next_state, terminated, n)
        return F.smooth_l1_loss(q_value, target), values.detach()

    def step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return {}

        for _ in range(self.epoch):
            self.net.opt.zero_grad()
            self.target_net.eval()
            self.net.train()
            loss, q_value = self.loss(*self.buffer.sample(batch_size))
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.net.opt.step()

            self.alpha_opt.zero_grad()
            probabilities = torch.softmax(
                q_value / self.alpha, dim=-1
            ).cpu()
            entropy = Categorical(probabilities).entropy().mean()
            alpha_loss = torch.exp(self._alpha).clamp_max(1.0) * (
                entropy - self.target_entropy
            )
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self._alpha], 0.1)
            self.alpha_opt.step()
            self.soft_update()

        return {"loss": loss.item(), "alpha": self.alpha}


if __name__ == "__main__":
    update = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4 if bool(update) else 1
    env = make_carracing_env(update, num_envs)
    observation_space, action_space = single_spaces(env, update)
    if observation_space.shape != OBSERVATION_SHAPE:
        raise RuntimeError(
            f"unexpected observation shape: {observation_space.shape}"
        )

    q_network = CarRacingQNetwork(
        lr=1e-4,
        num_actions=action_space.n,
        computes_grad=True,
        device=device,
    )
    config = Config()
    agent = CarRacingSoftDQNAgent(
        "softdqn_carracing", q_network, config
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
                        float(rewards[env_id]) * agent.reward_scale,
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
