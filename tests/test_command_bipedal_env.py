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
    def test_external_command_is_exposed_and_environment_steps(self):
        config = CommandBipedalConfig(max_episode_steps=50)
        env = make_command_env(7, config, command_mode="external")
        try:
            observation, _ = env.reset(seed=7)
            self.assertEqual(observation.shape, env.observation_space.shape)
            env.unwrapped.set_command(-config.command_speed)
            observation = env.unwrapped.command_observation()
            self.assertAlmostEqual(observation[-5], -1.0)

            next_observation, reward, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )

            self.assertEqual(next_observation.shape, observation.shape)
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            for key in ("velocity_reward", "acceleration_reward", "standing_reward", "height"):
                self.assertIn(key, info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
