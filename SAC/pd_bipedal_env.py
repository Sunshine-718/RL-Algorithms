"""BipedalWalker environment controlled by RL joint targets and a PD loop."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker


JOINT_LOWER_LIMITS = np.asarray([-0.8, -1.6, -0.8, -1.6], dtype=np.float32)
JOINT_UPPER_LIMITS = np.asarray([1.1, -0.1, 1.1, -0.1], dtype=np.float32)
DEFAULT_KP = (2.0, 2.0, 2.0, 2.0)
DEFAULT_KD = (0.15, 0.10, 0.15, 0.10)


def _joint_vector(value: float | Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim == 0:
        vector = np.repeat(vector, 4)
    if vector.shape != (4,):
        raise ValueError(f"{name} must be a scalar or contain four values")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


class PDController:
    """Vectorized proportional-derivative controller with output clipping."""

    def __init__(
        self,
        kp: float | Sequence[float],
        kd: float | Sequence[float],
        output_low: float = -1.0,
        output_high: float = 1.0,
    ) -> None:
        self.kp = _joint_vector(kp, "kp")
        self.kd = _joint_vector(kd, "kd")
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError("kp and kd must be non-negative")
        if not np.isfinite((output_low, output_high)).all():
            raise ValueError("output limits must be finite")
        if output_low >= output_high:
            raise ValueError("output_low must be smaller than output_high")
        self.output_low = float(output_low)
        self.output_high = float(output_high)

    @staticmethod
    def clamp(value: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.clip(value, low, high)

    def update(
        self,
        target: Sequence[float] | np.ndarray,
        value: Sequence[float] | np.ndarray,
        velocity: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        target = _joint_vector(target, "target")
        value = _joint_vector(value, "value")
        velocity = _joint_vector(velocity, "velocity")
        error = target - value
        proportional = self.kp * error
        # 目标角由策略逐步改变，直接对测量角速度做微分，避免 setpoint kick。
        derivative = -self.kd * velocity
        output = self.clamp(
            proportional + derivative, self.output_low, self.output_high
        )
        return output.astype(np.float32, copy=False)


def normalized_action_to_target(action: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map the policy's normalized action to physical joint-angle limits."""
    action = _joint_vector(action, "action")
    action = np.clip(action, -1.0, 1.0)
    center = (JOINT_LOWER_LIMITS + JOINT_UPPER_LIMITS) * 0.5
    half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) * 0.5
    return (center + action * half_range).astype(np.float32, copy=False)


class PDBipedalWalker(BipedalWalker):
    """Interpret actions as joint targets and execute them through PD control."""

    def __init__(
        self,
        render_mode: str | None = None,
        hardcore: bool = False,
        kp: float | Sequence[float] = DEFAULT_KP,
        kd: float | Sequence[float] = DEFAULT_KD,
    ) -> None:
        super().__init__(render_mode=render_mode, hardcore=hardcore)
        self.pd_controller = PDController(kp=kp, kd=kd)
        self._resetting = False

    def reset(self, *, seed=None, options=None):
        # BipedalWalker.reset() 内部会调用一次 step(0)。初始化帧保持原环境语义。
        self._resetting = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self._resetting = False

    def step(self, action: np.ndarray):
        if self._resetting:
            return BipedalWalker.step(self, action)

        target_angles = normalized_action_to_target(action)
        joint_angles = np.asarray(
            [joint.angle for joint in self.joints], dtype=np.float32
        )
        joint_velocities = np.asarray(
            [joint.speed for joint in self.joints], dtype=np.float32
        )
        motor_actions = self.pd_controller.update(
            target_angles, joint_angles, joint_velocities
        )
        observation, reward, terminated, truncated, info = super().step(
            motor_actions
        )
        info = dict(info)
        info.update(
            {
                "pd_target_angles": target_angles,
                "pd_joint_angles": joint_angles,
                "pd_motor_actions": motor_actions,
                "pd_mean_absolute_error": float(
                    np.mean(np.abs(target_angles - joint_angles))
                ),
                "pd_saturation_rate": float(
                    np.mean(np.abs(motor_actions) >= 1.0 - 1e-6)
                ),
            }
        )
        return observation, reward, terminated, truncated, info


def make_pd_bipedal_env(
    *,
    render_mode: str | None = None,
    hardcore: bool = False,
    kp: float | Sequence[float] = DEFAULT_KP,
    kd: float | Sequence[float] = DEFAULT_KD,
    max_episode_steps: int = 1600,
) -> gym.Env:
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    env = PDBipedalWalker(
        render_mode=render_mode,
        hardcore=hardcore,
        kp=kp,
        kd=kd,
    )
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
