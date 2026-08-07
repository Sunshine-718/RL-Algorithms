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
    gait_reward_version: int = 2
    gait_reward_weight: float = 0.25
    target_stride_length: float = 1.0
    target_swing_clearance: float = 0.28
    gait_velocity_tolerance: float = 0.5
    max_support_steps: int = 35
    contact_debounce_steps: int = 2
    min_swing_steps: int = 8
    max_swing_steps: int = 40
    min_support_steps: int = 6
    min_step_interval_steps: int = 10
    minimum_step_displacement: float = 0.20
    minimum_com_progress: float = 0.10
    minimum_swing_clearance: float = 0.08
    maximum_stance_slip: float = 0.15
    minimum_step_frequency: float = 0.75
    maximum_step_frequency: float = 4.0
    cadence_tolerance: float = 0.75
    step_event_scale: float = 0.5
    # Kept only so v1 checkpoint environment dictionaries remain loadable.
    alternating_step_reward_weight: float = 0.0
    support_stall_penalty_weight: float = 0.5
    airborne_penalty_weight: float = 0.25
    invalid_touchdown_penalty_weight: float = 0.25
    cadence_penalty_weight: float = 0.15
    stance_slip_penalty_weight: float = 0.15
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
        if self.gait_reward_version != 2:
            raise ValueError("unsupported gait_reward_version")
        if self.gait_reward_weight < 0:
            raise ValueError("gait_reward_weight cannot be negative")
        if min(
            self.max_support_steps,
            self.contact_debounce_steps,
            self.min_swing_steps,
            self.max_swing_steps,
            self.min_support_steps,
            self.min_step_interval_steps,
        ) < 1:
            raise ValueError("gait timing values must be positive")
        if self.max_swing_steps < self.min_swing_steps:
            raise ValueError("max_swing_steps must be >= min_swing_steps")
        if min(
            self.alternating_step_reward_weight,
            self.support_stall_penalty_weight,
            self.airborne_penalty_weight,
            self.invalid_touchdown_penalty_weight,
            self.cadence_penalty_weight,
            self.stance_slip_penalty_weight,
        ) < 0:
            raise ValueError("gait reward and penalty weights cannot be negative")
        if min(
            self.target_stride_length,
            self.target_swing_clearance,
            self.gait_velocity_tolerance,
            self.minimum_step_displacement,
            self.minimum_com_progress,
            self.minimum_swing_clearance,
            self.maximum_stance_slip,
            self.minimum_step_frequency,
            self.maximum_step_frequency,
            self.cadence_tolerance,
            self.step_event_scale,
        ) <= 0:
            raise ValueError("gait targets and tolerance must be positive")
        if self.maximum_step_frequency <= self.minimum_step_frequency:
            raise ValueError(
                "maximum_step_frequency must exceed minimum_step_frequency"
            )
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


@dataclass(frozen=True)
class GaitEvent:
    support_leg: int = 0
    phase_steps: int = 0
    support_steps: int = 0
    single_support: bool = False
    support_switch: bool = False
    valid_step: bool = False
    invalid_touchdown: bool = False
    step_displacement: float = 0.0
    com_progress: float = 0.0
    swing_clearance: float = 0.0
    swing_steps: int = 0
    step_frequency: float = 0.0
    stance_slip: float = 0.0


class GaitTracker:
    """Debounce contacts and validate a complete swing trajectory at touchdown."""

    def __init__(self, config: CommandBipedalConfig):
        self.config = config
        self.reset()

    @property
    def stable_contacts(self) -> tuple[bool, bool]:
        return bool(self._stable_contacts[0]), bool(self._stable_contacts[1])

    def reset(
        self,
        contacts: tuple[bool, bool] = (False, False),
        foot_x: tuple[float, float] = (0.0, 0.0),
        com_x: float = 0.0,
    ) -> None:
        self._stable_contacts = [bool(contacts[0]), bool(contacts[1])]
        self._transition_counts = [0, 0]
        self._transition_start_x = [float(foot_x[0]), float(foot_x[1])]
        self._transition_start_com = [float(com_x), float(com_x)]
        self._contact_steps = [int(contacts[0]), int(contacts[1])]
        self._air_steps = [int(not contacts[0]), int(not contacts[1])]
        self._swinging = [False, False]
        self._liftoff_x = [float(foot_x[0]), float(foot_x[1])]
        self._liftoff_com_x = [float(com_x), float(com_x)]
        self._swing_steps = [0, 0]
        self._max_clearance = [0.0, 0.0]
        self._opposite_supported = [False, False]
        self._stance_start_x = [float(foot_x[1]), float(foot_x[0])]
        self._stance_max_slip = [0.0, 0.0]
        self._last_valid_foot: int | None = None
        self._last_valid_step_index: int | None = None
        self._last_support_leg = self._support_leg()
        self._step_index = 0

    def _support_leg(self) -> int:
        left, right = self._stable_contacts
        if left and not right:
            return 1
        if right and not left:
            return -1
        return 0

    def _update_debounced_contacts(
        self,
        raw_contacts: tuple[bool, bool],
        foot_x: tuple[float, float],
        com_x: float,
    ) -> list[int]:
        transitions = [0, 0]
        for foot in range(2):
            raw_contact = bool(raw_contacts[foot])
            if raw_contact == self._stable_contacts[foot]:
                self._transition_counts[foot] = 0
            else:
                if self._transition_counts[foot] == 0:
                    self._transition_start_x[foot] = float(foot_x[foot])
                    self._transition_start_com[foot] = float(com_x)
                self._transition_counts[foot] += 1
                if (
                    self._transition_counts[foot]
                    >= self.config.contact_debounce_steps
                ):
                    self._stable_contacts[foot] = raw_contact
                    transitions[foot] = 1 if raw_contact else -1
                    self._transition_counts[foot] = 0
                    self._contact_steps[foot] = int(raw_contact)
                    self._air_steps[foot] = int(not raw_contact)
                    continue

            if self._stable_contacts[foot]:
                self._contact_steps[foot] += 1
                self._air_steps[foot] = 0
            else:
                self._air_steps[foot] += 1
                self._contact_steps[foot] = 0
        return transitions

    def update(
        self,
        *,
        raw_contacts: tuple[bool, bool],
        foot_x: tuple[float, float],
        clearances: tuple[float, float],
        com_x: float,
        reference_velocity: float,
    ) -> GaitEvent:
        self._step_index += 1
        for foot in range(2):
            if self._swinging[foot]:
                self._swing_steps[foot] += 1
                self._max_clearance[foot] = max(
                    self._max_clearance[foot], float(clearances[foot])
                )

        transitions = self._update_debounced_contacts(
            raw_contacts, foot_x, com_x
        )
        for foot, transition in enumerate(transitions):
            if transition != -1:
                continue
            other = 1 - foot
            self._swinging[foot] = True
            self._liftoff_x[foot] = self._transition_start_x[foot]
            self._liftoff_com_x[foot] = self._transition_start_com[foot]
            self._swing_steps[foot] = self.config.contact_debounce_steps
            self._max_clearance[foot] = float(clearances[foot])
            self._opposite_supported[foot] = self._stable_contacts[other]
            self._stance_start_x[foot] = float(foot_x[other])
            self._stance_max_slip[foot] = 0.0

        for foot in range(2):
            if not self._swinging[foot]:
                continue
            other = 1 - foot
            self._opposite_supported[foot] = (
                self._opposite_supported[foot]
                and self._stable_contacts[other]
            )
            self._stance_max_slip[foot] = max(
                self._stance_max_slip[foot],
                abs(float(foot_x[other]) - self._stance_start_x[foot]),
            )

        valid_step = False
        invalid_touchdown = False
        step_displacement = 0.0
        com_progress = 0.0
        swing_clearance = 0.0
        swing_steps = 0
        step_frequency = 0.0
        touchdown_stance_slip = 0.0
        direction = float(np.sign(reference_velocity))

        for foot, transition in enumerate(transitions):
            if transition != 1 or not self._swinging[foot]:
                continue
            other = 1 - foot
            swing_steps = self._swing_steps[foot]
            swing_clearance = self._max_clearance[foot]
            touchdown_stance_slip = self._stance_max_slip[foot]
            step_displacement = direction * (
                float(foot_x[foot]) - self._liftoff_x[foot]
            )
            com_progress = direction * (
                float(com_x) - self._liftoff_com_x[foot]
            )
            interval = (
                self._step_index - self._last_valid_step_index
                if self._last_valid_step_index is not None
                else swing_steps
            )
            step_frequency = FPS / max(1, interval)
            alternates = (
                self._last_valid_foot is None or self._last_valid_foot != foot
            )
            valid_step = bool(
                direction != 0.0
                and self.config.min_swing_steps
                <= swing_steps
                <= self.config.max_swing_steps
                and self._contact_steps[other] >= self.config.min_support_steps
                and interval >= self.config.min_step_interval_steps
                and self.config.minimum_step_frequency
                <= step_frequency
                <= self.config.maximum_step_frequency
                and step_displacement >= self.config.minimum_step_displacement
                and com_progress >= self.config.minimum_com_progress
                and swing_clearance >= self.config.minimum_swing_clearance
                and self._opposite_supported[foot]
                and touchdown_stance_slip <= self.config.maximum_stance_slip
                and alternates
            )
            invalid_touchdown = not valid_step
            if valid_step:
                self._last_valid_foot = foot
                self._last_valid_step_index = self._step_index
            self._swinging[foot] = False
            self._swing_steps[foot] = 0
            self._max_clearance[foot] = 0.0

        support_leg = self._support_leg()
        support_switch = bool(
            support_leg != 0
            and self._last_support_leg != 0
            and support_leg != self._last_support_leg
        )
        if support_leg != 0:
            self._last_support_leg = support_leg
        single_support = support_leg != 0
        support_foot = 0 if support_leg == 1 else 1
        support_steps = self._contact_steps[support_foot] if single_support else 0
        active_swing_steps = [
            self._swing_steps[foot]
            for foot in range(2)
            if self._swinging[foot]
        ]
        phase_steps = max(active_swing_steps, default=support_steps)
        active_stance_slip = max(
            [
                self._stance_max_slip[foot]
                for foot in range(2)
                if self._swinging[foot]
            ],
            default=0.0,
        )
        return GaitEvent(
            support_leg=support_leg,
            phase_steps=phase_steps,
            support_steps=support_steps,
            single_support=single_support,
            support_switch=support_switch,
            valid_step=valid_step,
            invalid_touchdown=invalid_touchdown,
            step_displacement=float(step_displacement),
            com_progress=float(com_progress),
            swing_clearance=float(swing_clearance),
            swing_steps=swing_steps,
            step_frequency=float(step_frequency),
            stance_slip=float(max(touchdown_stance_slip, active_stance_slip)),
        )


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
    step_displacement: float = 0.0,
    swing_clearance: float = 0.0,
    com_progress: float = 0.0,
    step_frequency: float = 0.0,
    stance_slip: float = 0.0,
    single_support: bool = False,
    support_steps: int = 0,
    support_switch: bool = False,
    valid_step: bool = False,
    invalid_touchdown: bool = False,
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
        -((step_displacement - config.target_stride_length) / stride_scale) ** 2
    )
    clearance_scale = max(0.5 * config.target_swing_clearance, 1e-6)
    clearance_score = math.exp(
        -((swing_clearance - config.target_swing_clearance) / clearance_scale) ** 2
    )
    single_support_score = float(single_support)
    progress_target = max(0.5 * config.target_stride_length, 1e-6)
    progress_score = float(np.clip(com_progress / progress_target, 0.0, 1.0))
    target_step_frequency = float(
        np.clip(
            abs(reference_velocity) / config.target_stride_length,
            config.minimum_step_frequency,
            config.maximum_step_frequency,
        )
    )
    cadence_score = math.exp(
        -(
            (step_frequency - target_step_frequency)
            / config.cadence_tolerance
        )
        ** 2
    )
    physical_step_score = (
        0.40 * stride_score
        + 0.25 * clearance_score
        + 0.35 * progress_score
    )
    step_quality = cadence_score * physical_step_score
    step_event_scale = config.step_event_scale * min(
        FPS / max(target_step_frequency, 1e-6),
        float(config.max_support_steps),
    )
    gait_reward = (
        movement_gate
        * velocity_reward
        * float(valid_step)
        * step_quality
        * step_event_scale
    )
    support_stall = movement_gate * float(
        np.clip(
            (support_steps - config.max_support_steps)
            / config.max_support_steps,
            0.0,
            2.0,
        )
    )
    invalid_touchdown_score = movement_gate * float(invalid_touchdown)
    cadence_excess = movement_gate * float(valid_step) * float(
        np.clip(
            (
                step_frequency
                - target_step_frequency
                - config.cadence_tolerance
            )
            / config.cadence_tolerance,
            0.0,
            2.0,
        )
    )
    stance_slip_score = movement_gate * float(
        np.clip(
            stance_slip / config.maximum_stance_slip - 1.0,
            0.0,
            2.0,
        )
    )
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
        + config.invalid_touchdown_penalty_weight * invalid_touchdown_score
        + config.cadence_penalty_weight * cadence_excess
        + config.stance_slip_penalty_weight * stance_slip_score
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
        "step_displacement": float(step_displacement),
        "swing_clearance": float(swing_clearance),
        "com_progress": float(com_progress),
        "step_frequency": float(step_frequency),
        "target_step_frequency": float(target_step_frequency),
        "cadence_score": float(cadence_score),
        "step_event_scale": float(step_event_scale),
        "stance_slip": float(stance_slip),
        "single_support": float(single_support_score),
        "support_steps": float(support_steps),
        "support_switch": float(support_switch),
        "valid_step": float(valid_step),
        "invalid_touchdown": float(invalid_touchdown),
        "support_stall": float(support_stall),
        "cadence_excess": float(cadence_excess),
        "stance_slip_penalty": float(stance_slip_score),
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
        self.gait_tracker = GaitTracker(self.command_config)
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
        self.gait_tracker.reset()
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
        left_foot_x, _ = self._foot_position_and_clearance(1)
        right_foot_x, _ = self._foot_position_and_clearance(3)
        self.gait_tracker.reset(
            contacts=(
                bool(base_observation[8] > 0.5),
                bool(base_observation[13] > 0.5),
            ),
            foot_x=(left_foot_x, right_foot_x),
            com_x=float(self.hull.position.x),
        )
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
        raw_left_contact = bool(base_observation[8] > 0.5)
        raw_right_contact = bool(base_observation[13] > 0.5)
        left_foot_x, left_clearance = self._foot_position_and_clearance(1)
        right_foot_x, right_clearance = self._foot_position_and_clearance(3)
        gait_event = self.gait_tracker.update(
            raw_contacts=(raw_left_contact, raw_right_contact),
            foot_x=(left_foot_x, right_foot_x),
            clearances=(left_clearance, right_clearance),
            com_x=float(self.hull.position.x),
            reference_velocity=self.reference.velocity,
        )
        left_contact, right_contact = self.gait_tracker.stable_contacts
        self.last_support_leg = gait_event.support_leg
        self.support_steps = gait_event.phase_steps
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
            step_displacement=gait_event.step_displacement,
            swing_clearance=gait_event.swing_clearance,
            com_progress=gait_event.com_progress,
            step_frequency=gait_event.step_frequency,
            stance_slip=gait_event.stance_slip,
            single_support=gait_event.single_support,
            support_steps=gait_event.support_steps,
            support_switch=gait_event.support_switch,
            valid_step=gait_event.valid_step,
            invalid_touchdown=gait_event.invalid_touchdown,
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
                "support_leg": float(gait_event.support_leg),
                "raw_left_contact": float(raw_left_contact),
                "raw_right_contact": float(raw_right_contact),
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
