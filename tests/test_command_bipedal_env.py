import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAC = ROOT / "SAC"
sys.path.insert(0, str(SAC))

from command_bipedal_env import (  # noqa: E402
    CommandBipedalConfig,
    VelocityReference,
    compute_command_reward,
    make_command_env,
)


class VelocityReferenceTests(unittest.TestCase):
    def test_step_command_produces_continuous_velocity_and_settles_acceleration(self):
        reference = VelocityReference(
            dt=1 / 50,
            settling_time=1.5,
            damping=1.0,
            acceleration_limit=3.0,
            jerk_limit=12.0,
        )

        for _ in range(300):
            velocity, acceleration = reference.step(1.5)

        self.assertAlmostEqual(velocity, 1.5, delta=0.03)
        self.assertAlmostEqual(acceleration, 0.0, delta=0.08)

        previous_velocity = velocity
        velocity, acceleration = reference.step(-1.5)

        self.assertGreater(velocity, 0.0)
        self.assertLess(velocity, previous_velocity)
        self.assertLess(acceleration, 0.0)

        for _ in range(400):
            velocity, acceleration = reference.step(-1.5)

        self.assertAlmostEqual(velocity, -1.5, delta=0.03)
        self.assertAlmostEqual(acceleration, 0.0, delta=0.08)

    def test_jerk_limit_bounds_acceleration_change(self):
        reference = VelocityReference(
            dt=0.02,
            settling_time=1.0,
            damping=1.0,
            acceleration_limit=4.0,
            jerk_limit=10.0,
        )
        previous_acceleration = 0.0
        for command in (2.0, -2.0, 2.0, -2.0):
            _, acceleration = reference.step(command)
            self.assertLessEqual(abs(acceleration - previous_acceleration), 0.2 + 1e-7)
            previous_acceleration = acceleration


class CommandRewardTests(unittest.TestCase):
    def test_stale_support_is_penalized_and_alternating_step_is_rewarded(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=config.command_speed,
            acceleration=0.0,
            reference_velocity=config.command_speed,
            reference_acceleration=0.0,
            height=config.standing_height,
            torso_angle=0.0,
            angular_velocity=0.0,
            vertical_velocity=0.0,
            left_contact=True,
            right_contact=False,
            action=np.zeros(4, dtype=np.float32),
            previous_action=np.zeros(4, dtype=np.float32),
            terminated=False,
            config=config,
            stride_length=config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            single_support=True,
        )

        fresh_reward, _ = compute_command_reward(support_steps=0, **common)
        stale_reward, stale = compute_command_reward(
            support_steps=3 * config.max_support_steps,
            **common,
        )
        alternating_reward, alternating = compute_command_reward(
            support_steps=0,
            alternating_step=True,
            **common,
        )

        self.assertGreater(fresh_reward, stale_reward)
        self.assertGreater(stale["support_stall"], 0.9)
        self.assertGreater(alternating_reward, fresh_reward)
        self.assertEqual(alternating["alternating_step"], 1.0)

    def test_airborne_motion_is_penalized(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=config.command_speed,
            acceleration=0.0,
            reference_velocity=config.command_speed,
            reference_acceleration=0.0,
            height=config.standing_height,
            torso_angle=0.0,
            angular_velocity=0.0,
            vertical_velocity=0.0,
            action=np.zeros(4, dtype=np.float32),
            previous_action=np.zeros(4, dtype=np.float32),
            terminated=False,
            config=config,
        )

        grounded_reward, _ = compute_command_reward(
            left_contact=True,
            right_contact=True,
            **common,
        )
        airborne_reward, airborne = compute_command_reward(
            left_contact=False,
            right_contact=False,
            **common,
        )

        self.assertGreater(grounded_reward, airborne_reward)
        self.assertEqual(airborne["airborne"], 1.0)

    def test_moving_reward_prefers_stride_and_swing_clearance(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=config.command_speed,
            acceleration=0.0,
            reference_velocity=config.command_speed,
            reference_acceleration=0.0,
            height=config.standing_height,
            torso_angle=0.0,
            angular_velocity=0.0,
            vertical_velocity=0.0,
            left_contact=True,
            right_contact=False,
            action=np.zeros(4, dtype=np.float32),
            previous_action=np.zeros(4, dtype=np.float32),
            terminated=False,
            config=config,
            single_support=True,
        )

        short_reward, short = compute_command_reward(
            stride_length=0.15,
            swing_clearance=0.02,
            **common,
        )
        full_reward, full = compute_command_reward(
            stride_length=config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            **common,
        )
        overextended_reward, overextended = compute_command_reward(
            stride_length=3.5 * config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            **common,
        )

        self.assertGreater(full_reward, short_reward)
        self.assertGreater(full["gait_reward"], short["gait_reward"])
        self.assertGreater(full_reward, overextended_reward)
        self.assertGreater(full["gait_reward"], overextended["gait_reward"])

    def test_zero_speed_rewards_higher_stable_stance(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=0.0,
            acceleration=0.0,
            reference_velocity=0.0,
            reference_acceleration=0.0,
            torso_angle=0.0,
            angular_velocity=0.0,
            vertical_velocity=0.0,
            left_contact=True,
            right_contact=True,
            action=np.zeros(4, dtype=np.float32),
            previous_action=np.zeros(4, dtype=np.float32),
            terminated=False,
            config=config,
        )

        low_reward, low = compute_command_reward(height=1.3, **common)
        high_reward, high = compute_command_reward(height=config.standing_height, **common)

        self.assertGreater(high_reward, low_reward)
        self.assertGreater(high["height_reward"], low["height_reward"])
        self.assertGreater(high["standing_reward"], low["standing_reward"])

    def test_height_reward_requires_standing_reference_and_ground_contact(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=0.0,
            acceleration=0.0,
            torso_angle=0.0,
            angular_velocity=0.0,
            vertical_velocity=0.0,
            height=config.standing_height,
            action=np.zeros(4, dtype=np.float32),
            previous_action=np.zeros(4, dtype=np.float32),
            terminated=False,
            config=config,
        )
        _, standing = compute_command_reward(
            reference_velocity=0.0,
            reference_acceleration=0.0,
            left_contact=True,
            right_contact=True,
            **common,
        )
        _, moving = compute_command_reward(
            reference_velocity=1.0,
            reference_acceleration=1.0,
            left_contact=True,
            right_contact=True,
            **common,
        )
        _, airborne = compute_command_reward(
            reference_velocity=0.0,
            reference_acceleration=0.0,
            left_contact=False,
            right_contact=False,
            **common,
        )

        self.assertGreater(standing["standing_reward"], moving["standing_reward"])
        self.assertGreater(standing["standing_reward"], airborne["standing_reward"])
        self.assertEqual(airborne["standing_reward"], 0.0)


class CommandBipedalEnvironmentTests(unittest.TestCase):
    def test_support_phase_detects_alternating_feet(self):
        env = make_command_env(19, CommandBipedalConfig(), command_mode="external")
        try:
            env.reset(seed=19)
            support_leg, alternating = env.unwrapped._update_support_phase(True, False)
            self.assertEqual(support_leg, 1)
            self.assertFalse(alternating)

            env.unwrapped._update_support_phase(True, False)
            self.assertGreater(env.unwrapped.support_steps, 0)

            support_leg, alternating = env.unwrapped._update_support_phase(False, True)
            self.assertEqual(support_leg, -1)
            self.assertTrue(alternating)
            self.assertEqual(env.unwrapped.support_steps, 0)
        finally:
            env.close()

    def test_random_commands_cover_multiple_moving_speeds(self):
        config = CommandBipedalConfig(
            minimum_command_speed=1.0,
            command_speed=3.0,
            standing_probability=0.0,
            max_episode_steps=50,
        )
        env = make_command_env(13, config, command_mode="random")
        try:
            env.reset(seed=13)
            magnitudes = []
            for _ in range(40):
                env.unwrapped._sample_next_command()
                magnitudes.append(abs(env.unwrapped.raw_command))

            self.assertGreater(max(magnitudes) - min(magnitudes), 1.0)
            self.assertGreaterEqual(min(magnitudes), config.minimum_command_speed)
            self.assertLessEqual(max(magnitudes), config.command_speed)
        finally:
            env.close()

    def test_external_command_is_exposed_and_environment_steps(self):
        config = CommandBipedalConfig(max_episode_steps=50)
        env = make_command_env(7, config, command_mode="external")
        try:
            observation, _ = env.reset(seed=7)
            self.assertEqual(observation.shape, env.observation_space.shape)
            env.unwrapped.set_command(-config.command_speed)
            observation = env.unwrapped.command_observation()
            self.assertEqual(observation.shape[0], 41)
            self.assertAlmostEqual(observation[-7], -1.0)

            next_observation, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )

            self.assertEqual(next_observation.shape, observation.shape)
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            for key in (
                "velocity_reward",
                "acceleration_reward",
                "standing_reward",
                "gait_reward",
                "stride_length",
                "swing_clearance",
                "height",
            ):
                self.assertIn(key, info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
