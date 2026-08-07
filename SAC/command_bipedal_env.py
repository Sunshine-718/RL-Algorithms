"""Command-conditioned, bidirectional BipedalWalker environment.

The policy receives an abrupt operator command together with a jerk-limited
reference velocity/acceleration.  Rewards track that reachable reference and
add a high, quiet standing objective once the reference has settled at zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.bipedal_walker import (
    BipedalWalker,
    FPS,
    LEG_H,
    LIDAR_RANGE,
    TERRAIN_LENGTH,
    TERRAIN_STEP,
)


LEFT_LIDAR_DIM = 10


@dataclass
class CommandBipedalConfig:
    command_speed: float = 3.0
    minimum_command_speed: float = 1.0
    command_hold_min_steps: int = 100
    command_hold_max_steps: int = 250
    standing_probability: float = 0.30
    settling_time: float = 1.5
    damping: float = 1.0
    acceleration_limit: float = 3.0
    jerk_limit: float = 12.0
    acceleration_filter: float = 0.85
    velocity_tolerance: float = 0.45
    acceleration_tolerance: float = 1.25
    stand_velocity_tolerance: float = 0.15
    stand_acceleration_tolerance: float = 0.45
    minimum_height: float = 1.0
    standing_height: float = 2.15
    standing_reward_weight: float = 1.0
    height_reward_weight: float = 0.5
    quiet_reward_weight: float = 0.3
    contact_reward_weight: float = 0.2
    orientation_penalty_weight: float = 0.35
    angular_velocity_penalty_weight: float = 0.015
    vertical_velocity_penalty_weight: float = 0.03
    action_penalty_weight: float = 0.004
    action_rate_penalty_weight: float = 0.008
    gait_reward_weight: float = 0.5
    target_stride_length: float = 1.0
    target_swing_clearance: float = 0.28
    gait_velocity_tolerance: float = 0.5
    max_support_steps: int = 35
    alternating_step_reward_weight: float = 0.5
    support_stall_penalty_weight: float = 0.5
    airborne_penalty_weight: float = 0.25
    failure_penalty: float = -10.0
    boundary_margin: float = 4.0
    include_left_lidar: bool = True
    include_support_phase: bool = True
    max_episode_steps: int = 1600

    def validate(self) -> None:
        if self.command_speed <= 0:
            raise ValueError("command_speed must be positive")
        if not 0 < self.minimum_command_speed <= self.command_speed:
            raise ValueError(
                "minimum_command_speed must be positive and no greater than command_speed"
            )
        if self.command_hold_min_steps < 1:
            raise ValueError("command_hold_min_steps must be at least 1")
        if self.command_hold_max_steps < self.command_hold_min_steps:
            raise ValueError("command_hold_max_steps must be >= command_hold_min_steps")
        if not 0 <= self.standing_probability <= 1:
            raise ValueError("standing_probability must be in [0, 1]")
        if self.settling_time <= 0 or self.damping <= 0:
            raise ValueError("reference dynamics must be positive")
        if self.acceleration_limit <= 0 or self.jerk_limit <= 0:
            raise ValueError("acceleration and jerk limits must be positive")
        if not 0 <= self.acceleration_filter < 1:
            raise ValueError("acceleration_filter must be in [0, 1)")
        if self.minimum_height >= self.standing_height:
            raise ValueError("standing_height must exceed minimum_height")
        if self.gait_reward_weight < 0:
            raise ValueError("gait_reward_weight cannot be negative")
        if self.max_support_steps < 1:
            raise ValueError("max_support_steps must be positive")
        if min(
            self.alternating_step_reward_weight,
            self.support_stall_penalty_weight,
            self.airborne_penalty_weight,
        ) < 0:
            raise ValueError("gait reward and penalty weights cannot be negative")
        if min(
            self.target_stride_length,
            self.target_swing_clearance,
            self.gait_velocity_tolerance,
        ) <= 0:
            raise ValueError("gait targets and tolerance must be positive")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")


class VelocityReference:
    """Critically damped, acceleration- and jerk-limited velocity reference."""

    def __init__(
        self,
        dt: float,
        settling_time: float,
        damping: float,
        acceleration_limit: float,
        jerk_limit: float,
    ):
        if min(dt, settling_time, damping, acceleration_limit, jerk_limit) <= 0:
            raise ValueError("reference parameters must be positive")
        self.dt = float(dt)
        self.omega = 4.0 / float(settling_time)
        self.damping = float(damping)
        self.acceleration_limit = float(acceleration_limit)
        self.jerk_limit = float(jerk_limit)
        self.velocity = 0.0
        self.acceleration = 0.0

    def reset(self, velocity: float = 0.0, acceleration: float = 0.0) -> None:
        self.velocity = float(velocity)
        self.acceleration = float(acceleration)

    def step(self, command: float) -> tuple[float, float]:
        jerk = (
            self.omega**2 * (float(command) - self.velocity)
            - 2.0 * self.damping * self.omega * self.acceleration
        )
        jerk = float(np.clip(jerk, -self.jerk_limit, self.jerk_limit))
        self.acceleration = float(
            np.clip(
                self.acceleration + jerk * self.dt,
                -self.acceleration_limit,
                self.acceleration_limit,
            )
        )
        self.velocity += self.acceleration * self.dt
        if (
            abs(float(command) - self.velocity) < 1e-5
            and abs(self.acceleration) < 1e-4
        ):
            self.velocity = float(command)
            self.acceleration = 0.0
        return self.velocity, self.acceleration


def compute_command_reward(
    *,
    velocity: float,
    acceleration: float,
    reference_velocity: float,
    reference_acceleration: float,
    height: float,
    torso_angle: float,
    angular_velocity: float,
    vertical_velocity: float,
    left_contact: bool,
    right_contact: bool,
    action: np.ndarray,
    previous_action: np.ndarray,
    terminated: bool,
    config: CommandBipedalConfig,
    stride_length: float = 0.0,
    swing_clearance: float = 0.0,
    single_support: bool = False,
    support_steps: int = 0,
    alternating_step: bool = False,
) -> tuple[float, dict[str, float]]:
    """Compute bounded task rewards and smooth auxiliary penalties."""
    velocity_error = (velocity - reference_velocity) / config.velocity_tolerance
    acceleration_error = (
        acceleration - reference_acceleration
    ) / config.acceleration_tolerance
    velocity_reward = math.exp(-(velocity_error**2))
    acceleration_reward = math.exp(-(acceleration_error**2))

    transient = float(
        np.clip(abs(reference_acceleration) / config.acceleration_limit, 0.0, 1.0)
    )
    velocity_weight = 0.8 - 0.3 * transient
    acceleration_weight = 0.2 + 0.3 * transient
    tracking_reward = (
        velocity_weight * velocity_reward
        + acceleration_weight * acceleration_reward
    )

    stand_gate = math.exp(
        -((reference_velocity / config.stand_velocity_tolerance) ** 2)
        -((reference_acceleration / config.stand_acceleration_tolerance) ** 2)
    )
    both_feet = float(left_contact and right_contact)
    height_score = float(
        np.clip(
            (height - config.minimum_height)
            / (config.standing_height - config.minimum_height),
            0.0,
            1.0,
        )
    )
    quiet_score = math.exp(
        -((velocity / config.stand_velocity_tolerance) ** 2)
        -((vertical_velocity / 0.35) ** 2)
        -((angular_velocity / 1.0) ** 2)
    )
    height_reward = stand_gate * both_feet * height_score
    standing_reward = stand_gate * both_feet * (
        config.height_reward_weight * height_score
        + config.quiet_reward_weight * quiet_score
        + config.contact_reward_weight
    )

    movement_gate = 1.0 - math.exp(
        -((reference_velocity / config.gait_velocity_tolerance) ** 2)
    )
    stride_scale = max(0.5 * config.target_stride_length, 1e-6)
    stride_score = math.exp(
        -((stride_length - config.target_stride_length) / stride_scale) ** 2
    )
    clearance_scale = max(0.5 * config.target_swing_clearance, 1e-6)
    clearance_score = math.exp(
        -((swing_clearance - config.target_swing_clearance) / clearance_scale) ** 2
    )
    single_support_score = float(single_support)
    support_phase_score = math.exp(
        -((float(support_steps) / config.max_support_steps) ** 2)
    )
    gait_score = single_support_score * support_phase_score * (
        0.60 * stride_score + 0.40 * clearance_score
    )
    alternating_step_score = float(alternating_step) * (
        0.5 + 0.5 * stride_score
    )
    gait_reward = movement_gate * (
        gait_score
        + config.alternating_step_reward_weight * alternating_step_score
    )
    support_stall = movement_gate * (1.0 - support_phase_score)
    airborne = float(not left_contact and not right_contact)

    action = np.asarray(action, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)
    auxiliary_penalty = (
        config.orientation_penalty_weight * torso_angle**2
        + config.angular_velocity_penalty_weight * angular_velocity**2
        + config.vertical_velocity_penalty_weight * vertical_velocity**2
        + config.action_penalty_weight * float(np.square(action).sum())
        + config.action_rate_penalty_weight
        * float(np.square(action - previous_action).sum())
        + config.support_stall_penalty_weight * support_stall
        + config.airborne_penalty_weight * movement_gate * airborne
    )
    task_reward = (
        tracking_reward
        + config.standing_reward_weight * standing_reward
        + config.gait_reward_weight * gait_reward
    )
    reward = task_reward * math.exp(-min(auxiliary_penalty, 50.0))
    if terminated:
        reward += config.failure_penalty

    components = {
        "velocity_reward": float(velocity_reward),
        "acceleration_reward": float(acceleration_reward),
        "tracking_reward": float(tracking_reward),
        "stand_gate": float(stand_gate),
        "height_reward": float(height_reward),
        "standing_reward": float(standing_reward),
        "movement_gate": float(movement_gate),
        "gait_reward": float(gait_reward),
        "stride_length": float(stride_length),
        "swing_clearance": float(swing_clearance),
        "single_support": float(single_support_score),
        "support_steps": float(support_steps),
        "support_phase_score": float(support_phase_score),
        "alternating_step": float(alternating_step),
        "support_stall": float(support_stall),
        "airborne": float(airborne),
        "auxiliary_penalty": float(auxiliary_penalty),
        "velocity_error": float(velocity - reference_velocity),
        "acceleration_error": float(acceleration - reference_acceleration),
        "task_reward": float(task_reward),
    }
    return float(reward), components


class CommandBipedalWalker(BipedalWalker):
    """BipedalWalker with symmetric terrain access and velocity commands."""

    metadata = BipedalWalker.metadata.copy()

    def __init__(
        self,
        config: CommandBipedalConfig | None = None,
        command_mode: str = "random",
        fixed_command: float = 0.0,
        render_mode: str | None = None,
    ):
        self.command_config = config or CommandBipedalConfig()
        self.command_config.validate()
        if command_mode not in {"random", "cycle", "fixed", "external"}:
            raise ValueError("command_mode must be random, cycle, fixed, or external")
        self.command_mode = command_mode
        self.fixed_command = float(fixed_command)
        self.raw_command = 0.0
        self.command_steps_remaining = 0
        self.cycle_index = 0
        self.reference = VelocityReference(
            dt=1.0 / FPS,
            settling_time=self.command_config.settling_time,
            damping=self.command_config.damping,
            acceleration_limit=self.command_config.acceleration_limit,
            jerk_limit=self.command_config.jerk_limit,
        )
        self.actual_acceleration = 0.0
        self.previous_velocity = 0.0
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.last_support_leg = 0
        self.support_steps = 0
        self._last_base_observation: np.ndarray | None = None
        self._last_left_lidar = np.ones(LEFT_LIDAR_DIM, dtype=np.float32)
        self._during_base_reset = False
        super().__init__(render_mode=render_mode, hardcore=False)

        extra_low = []
        extra_high = []
        if self.command_config.include_left_lidar:
            extra_low.extend([0.0] * LEFT_LIDAR_DIM)
            extra_high.extend([1.0] * LEFT_LIDAR_DIM)
        extra_low.extend([-1.0, -2.0, -1.0, -2.0, 0.0])
        extra_high.extend([1.0, 2.0, 1.0, 2.0, 2.0])
        if self.command_config.include_support_phase:
            extra_low.extend([-1.0, 0.0])
            extra_high.extend([1.0, 2.0])
        low = np.concatenate(
            [self.observation_space.low, np.asarray(extra_low, dtype=np.float32)]
        )
        high = np.concatenate(
            [self.observation_space.high, np.asarray(extra_high, dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _reset_control_state(self) -> None:
        self.reference.reset()
        self.actual_acceleration = 0.0
        self.previous_velocity = 0.0
        self.previous_action.fill(0.0)
        self.last_support_leg = 0
        self.support_steps = 0
        self.command_steps_remaining = 0
        self.cycle_index = 0
        self._sample_next_command(initial=True)

    def _sample_next_command(self, initial: bool = False) -> None:
        config = self.command_config
        if self.command_mode == "external":
            self.raw_command = float(
                np.clip(self.raw_command, -config.command_speed, config.command_speed)
            )
            self.command_steps_remaining = config.command_hold_max_steps
            return
        if self.command_mode == "fixed":
            self.raw_command = float(
                np.clip(self.fixed_command, -config.command_speed, config.command_speed)
            )
            self.command_steps_remaining = config.command_hold_max_steps
            return
        if self.command_mode == "cycle":
            middle_speed = 0.5 * (
                config.minimum_command_speed + config.command_speed
            )
            commands = (
                0.0,
                config.minimum_command_speed,
                middle_speed,
                config.command_speed,
                0.0,
                -config.minimum_command_speed,
                -middle_speed,
                -config.command_speed,
            )
            if not initial:
                self.cycle_index = (self.cycle_index + 1) % len(commands)
            self.raw_command = commands[self.cycle_index]
            self.command_steps_remaining = (
                config.command_hold_min_steps + config.command_hold_max_steps
            ) // 2
            return

        if self.np_random.random() < config.standing_probability:
            self.raw_command = 0.0
        else:
            direction = -1.0 if self.np_random.random() < 0.5 else 1.0
            magnitude = self.np_random.uniform(
                config.minimum_command_speed, config.command_speed
            )
            self.raw_command = direction * float(magnitude)
        self.command_steps_remaining = int(
            self.np_random.integers(
                config.command_hold_min_steps,
                config.command_hold_max_steps + 1,
            )
        )

    def _advance_command_schedule(self) -> None:
        if self.command_mode in {"external", "fixed"}:
            return
        self.command_steps_remaining -= 1
        if self.command_steps_remaining <= 0:
            self._sample_next_command()

    def set_command(self, command: float) -> None:
        if self.command_mode != "external":
            raise RuntimeError("set_command is only available in external command mode")
        self.raw_command = float(
            np.clip(
                command,
                -self.command_config.command_speed,
                self.command_config.command_speed,
            )
        )

    def _translate_robot_to_terrain_center(self) -> None:
        center_index = TERRAIN_LENGTH // 2
        target_x = float(self.terrain_x[center_index])
        target_y = float(self.terrain_y[center_index] + 2.0 * LEG_H)
        delta_x = target_x - float(self.hull.position.x)
        delta_y = target_y - float(self.hull.position.y)
        for body in [self.hull, *self.legs]:
            body.position = (
                float(body.position.x + delta_x),
                float(body.position.y + delta_y),
            )
            body.linearVelocity = (0.0, 0.0)
            body.angularVelocity = 0.0
            body.awake = True
        self.scroll = target_x - 600.0 / 30.0 / 2.0
        self.prev_shaping = None

    def _ground_height(self) -> float:
        return float(
            np.interp(
                float(self.hull.position.x),
                np.asarray(self.terrain_x),
                np.asarray(self.terrain_y),
            )
        )

    def _base_height(self) -> float:
        return float(self.hull.position.y) - self._ground_height()

    def _foot_position_and_clearance(self, leg_index: int) -> tuple[float, float]:
        foot = self.legs[leg_index].GetWorldPoint((0.0, -LEG_H / 2.0))
        foot_x = float(foot[0])
        ground_y = float(
            np.interp(
                foot_x,
                np.asarray(self.terrain_x),
                np.asarray(self.terrain_y),
            )
        )
        return foot_x, max(0.0, float(foot[1]) - ground_y)

    def _update_support_phase(
        self, left_contact: bool, right_contact: bool
    ) -> tuple[int, bool]:
        support_leg = 0
        if left_contact and not right_contact:
            support_leg = 1
        elif right_contact and not left_contact:
            support_leg = -1

        alternating_step = False
        if support_leg != 0:
            if self.last_support_leg == 0:
                self.last_support_leg = support_leg
                self.support_steps = 0
            elif support_leg != self.last_support_leg:
                self.last_support_leg = support_leg
                self.support_steps = 0
                alternating_step = True
            else:
                self.support_steps += 1
        else:
            self.support_steps += 1
        return support_leg, alternating_step

    def _scan_left_lidar(self) -> np.ndarray:
        if not self.command_config.include_left_lidar:
            return np.empty(0, dtype=np.float32)
        position = self.hull.position
        fractions = np.ones(LEFT_LIDAR_DIM, dtype=np.float32)
        for index, lidar in enumerate(self.lidar):
            lidar.fraction = 1.0
            lidar.p1 = position
            lidar.p2 = (
                position[0] - math.sin(1.5 * index / 10.0) * LIDAR_RANGE,
                position[1] - math.cos(1.5 * index / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(lidar, lidar.p1, lidar.p2)
            fractions[index] = lidar.fraction
        return fractions

    def _augment_observation(self, base_observation: np.ndarray) -> np.ndarray:
        config = self.command_config
        features = []
        if config.include_left_lidar:
            features.extend(self._last_left_lidar.tolist())
        features.extend(
            [
                float(np.clip(self.raw_command / config.command_speed, -1.0, 1.0)),
                float(
                    np.clip(
                        self.reference.velocity / config.command_speed, -2.0, 2.0
                    )
                ),
                float(
                    np.clip(
                        self.reference.acceleration / config.acceleration_limit,
                        -1.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        self.actual_acceleration / config.acceleration_limit,
                        -2.0,
                        2.0,
                    )
                ),
                float(
                    np.clip(
                        self._base_height() / config.standing_height, 0.0, 2.0
                    )
                ),
            ]
        )
        if config.include_support_phase:
            features.extend(
                [
                    float(self.last_support_leg),
                    float(
                        np.clip(
                            self.support_steps / config.max_support_steps, 0.0, 2.0
                        )
                    ),
                ]
            )
        return np.concatenate(
            [
                np.asarray(base_observation, dtype=np.float32),
                np.asarray(features, dtype=np.float32),
            ]
        )

    def command_observation(self) -> np.ndarray:
        if self._last_base_observation is None:
            raise RuntimeError("environment must be reset before requesting an observation")
        return self._augment_observation(self._last_base_observation)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.raw_command = 0.0
        self.reference.reset()
        self.actual_acceleration = 0.0
        self.previous_velocity = 0.0
        self.previous_action.fill(0.0)
        self.last_support_leg = 0
        self.support_steps = 0
        self._during_base_reset = True
        try:
            base_observation, info = super().reset(seed=seed, options=options)
        finally:
            self._during_base_reset = False
        self._translate_robot_to_terrain_center()
        self._reset_control_state()
        base_observation, _, terminated, truncated, info = super().step(
            np.zeros(4, dtype=np.float32)
        )
        if terminated or truncated:
            raise RuntimeError("centered BipedalWalker failed during reset")
        self._last_base_observation = np.asarray(base_observation, dtype=np.float32)
        self._last_left_lidar = self._scan_left_lidar()
        self.previous_velocity = float(self.hull.linearVelocity.x)
        return self._augment_observation(self._last_base_observation), info

    def step(self, action: np.ndarray):
        if self._during_base_reset:
            return super().step(action)

        action = np.asarray(action, dtype=np.float32)
        self.reference.step(self.raw_command)
        base_observation, _, _, truncated, info = super().step(action)
        velocity = float(self.hull.linearVelocity.x)
        raw_acceleration = (velocity - self.previous_velocity) * FPS
        beta = self.command_config.acceleration_filter
        self.actual_acceleration = (
            beta * self.actual_acceleration + (1.0 - beta) * raw_acceleration
        )
        self.previous_velocity = velocity

        minimum_x = self.command_config.boundary_margin
        maximum_x = TERRAIN_LENGTH * TERRAIN_STEP - self.command_config.boundary_margin
        terminated = bool(
            self.game_over
            or float(self.hull.position.x) <= minimum_x
            or float(self.hull.position.x) >= maximum_x
        )
        height = self._base_height()
        left_contact = bool(base_observation[8] > 0.5)
        right_contact = bool(base_observation[13] > 0.5)
        left_foot_x, left_clearance = self._foot_position_and_clearance(1)
        right_foot_x, right_clearance = self._foot_position_and_clearance(3)
        single_support = left_contact != right_contact
        support_leg, alternating_step = self._update_support_phase(
            left_contact, right_contact
        )
        if left_contact and not right_contact:
            swing_clearance = right_clearance
        elif right_contact and not left_contact:
            swing_clearance = left_clearance
        else:
            swing_clearance = 0.0
        stride_length = abs(left_foot_x - right_foot_x)
        reward, reward_info = compute_command_reward(
            velocity=velocity,
            acceleration=self.actual_acceleration,
            reference_velocity=self.reference.velocity,
            reference_acceleration=self.reference.acceleration,
            height=height,
            torso_angle=float(self.hull.angle),
            angular_velocity=float(self.hull.angularVelocity),
            vertical_velocity=float(self.hull.linearVelocity.y),
            left_contact=left_contact,
            right_contact=right_contact,
            action=action,
            previous_action=self.previous_action,
            terminated=terminated,
            config=self.command_config,
            stride_length=stride_length,
            swing_clearance=swing_clearance,
            single_support=single_support,
            support_steps=self.support_steps,
            alternating_step=alternating_step,
        )
        self.previous_action = action.copy()
        self._last_base_observation = np.asarray(base_observation, dtype=np.float32)
        self._last_left_lidar = self._scan_left_lidar()

        info = dict(info)
        info.update(reward_info)
        info.update(
            {
                "command_velocity": float(self.raw_command),
                "reference_velocity": float(self.reference.velocity),
                "reference_acceleration": float(self.reference.acceleration),
                "actual_velocity": velocity,
                "actual_acceleration": float(self.actual_acceleration),
                "height": height,
                "support_leg": float(support_leg),
            }
        )
        self._advance_command_schedule()
        return self._augment_observation(self._last_base_observation), reward, terminated, truncated, info


def make_command_env(
    seed: int,
    config: CommandBipedalConfig | None = None,
    command_mode: str = "random",
    fixed_command: float = 0.0,
    render_mode: str | None = None,
):
    config = config or CommandBipedalConfig()
    env = CommandBipedalWalker(
        config=config,
        command_mode=command_mode,
        fixed_command=fixed_command,
        render_mode=render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    env.action_space.seed(seed)
    return env
