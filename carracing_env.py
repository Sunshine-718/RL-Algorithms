from dataclasses import dataclass

import gymnasium as gym
import numpy as np


OBSERVATION_SHAPE = (2, 96, 96)
FRAME_SKIP = 2


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


def wrap_continuous_carracing_observation(env):
    action_shape = env.action_space.shape
    minimum_action = np.full(action_shape, -1.0, dtype=np.float32)
    maximum_action = np.full(action_shape, 1.0, dtype=np.float32)
    env = gym.wrappers.RescaleAction(
        env, minimum_action, maximum_action
    )
    return wrap_carracing_observation(env)


def make_carracing_env(update, num_envs=4, continuous=False):
    wrapper = (
        wrap_continuous_carracing_observation
        if continuous
        else wrap_carracing_observation
    )
    if bool(update):
        return gym.make_vec(
            "CarRacing-v3",
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            },
            wrappers=[wrapper],
            continuous=continuous,
        )

    env = gym.make(
        "CarRacing-v3", render_mode="human", continuous=continuous
    )
    return wrapper(env)
