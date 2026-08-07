import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAC = ROOT / "SAC"
sys.path.insert(0, str(SAC))

from command_bipedal_env import (  # noqa: E402
    CommandBipedalConfig,
    GaitTracker,
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
    def test_valid_step_is_rewarded_and_invalid_touchdown_is_penalized(self):
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

        neutral_reward, _ = compute_command_reward(**common)
        valid_reward, valid = compute_command_reward(
            valid_step=True,
            step_displacement=config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            com_progress=0.5 * config.target_stride_length,
            step_frequency=config.command_speed / config.target_stride_length,
            **common,
        )
        invalid_reward, invalid = compute_command_reward(
            invalid_touchdown=True,
            **common,
        )

        self.assertGreater(valid_reward, neutral_reward)
        self.assertGreater(neutral_reward, invalid_reward)
        self.assertEqual(valid["valid_step"], 1.0)
        self.assertEqual(invalid["invalid_touchdown"], 1.0)

    def test_support_stall_starts_only_after_maximum_duration(self):
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

        _, normal = compute_command_reward(
            support_steps=config.max_support_steps, **common
        )
        _, stale = compute_command_reward(
            support_steps=2 * config.max_support_steps, **common
        )

        self.assertEqual(normal["support_stall"], 0.0)
        self.assertGreater(stale["support_stall"], 0.0)

    def test_target_cadence_outscores_excessive_valid_cadence(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=1.0,
            acceleration=0.0,
            reference_velocity=1.0,
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
            valid_step=True,
            single_support=True,
            step_displacement=config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            com_progress=0.5 * config.target_stride_length,
        )

        target_reward, target = compute_command_reward(
            step_frequency=1.0, **common
        )
        fast_reward, fast = compute_command_reward(
            step_frequency=3.0, **common
        )

        self.assertGreater(target_reward, fast_reward)
        self.assertGreater(target["cadence_score"], fast["cadence_score"])

    def test_one_second_valid_gait_outscores_static_contact_spam(self):
        config = CommandBipedalConfig()
        common = dict(
            velocity=1.0,
            acceleration=0.0,
            reference_velocity=1.0,
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
        neutral_reward, _ = compute_command_reward(**common)
        invalid_reward, _ = compute_command_reward(
            invalid_touchdown=True, **common
        )
        valid_reward, _ = compute_command_reward(
            valid_step=True,
            step_displacement=config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            com_progress=0.5 * config.target_stride_length,
            step_frequency=1.0,
            **common,
        )

        valid_gait_total = valid_reward + 49 * neutral_reward
        contact_spam_total = 11 * invalid_reward + 39 * neutral_reward

        self.assertGreater(valid_gait_total, contact_spam_total)

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
            valid_step=True,
            com_progress=0.5 * config.target_stride_length,
            step_frequency=config.command_speed / config.target_stride_length,
        )

        short_reward, short = compute_command_reward(
            step_displacement=0.15,
            swing_clearance=0.02,
            **common,
        )
        full_reward, full = compute_command_reward(
            step_displacement=config.target_stride_length,
            swing_clearance=config.target_swing_clearance,
            **common,
        )
        overextended_reward, overextended = compute_command_reward(
            step_displacement=3.5 * config.target_stride_length,
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


class GaitTrackerTests(unittest.TestCase):
    def setUp(self):
        self.config = CommandBipedalConfig(
            contact_debounce_steps=2,
            min_swing_steps=3,
            max_swing_steps=20,
            min_support_steps=3,
            min_step_interval_steps=3,
            minimum_step_displacement=0.2,
            minimum_com_progress=0.1,
            minimum_swing_clearance=0.08,
            maximum_stance_slip=0.2,
        )
        self.tracker = GaitTracker(self.config)
        self.tracker.reset((True, True), (-0.5, 0.0), 0.0)

    def update(
        self,
        contacts,
        left_x=-0.5,
        right_x=0.0,
        left_clearance=0.0,
        right_clearance=0.0,
        com_x=0.0,
    ):
        return self.tracker.update(
            raw_contacts=contacts,
            foot_x=(left_x, right_x),
            clearances=(left_clearance, right_clearance),
            com_x=com_x,
            reference_velocity=1.0,
        )

    def test_one_frame_contact_bounce_does_not_create_event(self):
        event = self.update((False, True), left_clearance=0.02)
        event = self.update((True, True))

        self.assertFalse(event.valid_step)
        self.assertFalse(event.invalid_touchdown)
        self.assertEqual(self.tracker.stable_contacts, (True, True))

    def test_fixed_foot_position_contact_cycle_is_not_a_step(self):
        events = [self.update((False, True), left_clearance=0.02)]
        events.append(self.update((False, True), left_clearance=0.10))
        for _ in range(3):
            events.append(self.update((False, True), left_clearance=0.15))
        events.append(self.update((True, True)))
        event = self.update((True, True))
        events.append(event)

        events.append(self.update((True, False), right_clearance=0.02))
        events.append(self.update((True, False), right_clearance=0.10))
        for _ in range(3):
            events.append(self.update((True, False), right_clearance=0.15))
        events.append(self.update((True, True)))
        second_touchdown = self.update((True, True))
        events.append(second_touchdown)

        self.assertFalse(event.valid_step)
        self.assertTrue(event.invalid_touchdown)
        self.assertEqual(event.step_displacement, 0.0)
        self.assertTrue(any(item.support_switch for item in events))
        self.assertFalse(any(item.valid_step for item in events))
        self.assertTrue(second_touchdown.invalid_touchdown)

    def test_swing_with_foot_and_com_progress_is_a_valid_step(self):
        self.update((False, True), left_clearance=0.02)
        self.update((False, True), left_clearance=0.10)
        self.update(
            (False, True), left_x=-0.2, left_clearance=0.15, com_x=0.05
        )
        self.update((False, True), left_x=0.1, left_clearance=0.20, com_x=0.12)
        self.update((False, True), left_x=0.4, left_clearance=0.16, com_x=0.20)
        for _ in range(6):
            self.update(
                (False, True), left_x=0.4, left_clearance=0.16, com_x=0.20
            )
        self.update((True, True), left_x=0.5, com_x=0.25)
        event = self.update((True, True), left_x=0.5, com_x=0.25)

        self.assertTrue(event.valid_step)
        self.assertFalse(event.invalid_touchdown)
        self.assertGreaterEqual(event.step_displacement, 1.0)
        self.assertGreaterEqual(event.com_progress, 0.25)
        self.assertGreaterEqual(event.swing_clearance, 0.20)

    def test_geometrically_valid_but_excessive_cadence_is_rejected(self):
        config = CommandBipedalConfig(
            contact_debounce_steps=1,
            min_swing_steps=1,
            max_swing_steps=20,
            min_support_steps=1,
            min_step_interval_steps=1,
            minimum_step_displacement=0.2,
            minimum_com_progress=0.1,
            minimum_swing_clearance=0.08,
            maximum_stance_slip=0.2,
            maximum_step_frequency=4.0,
        )
        tracker = GaitTracker(config)
        tracker.reset((True, True), (-0.5, 0.0), 0.0)
        tracker.update(
            raw_contacts=(False, True),
            foot_x=(-0.5, 0.0),
            clearances=(0.1, 0.0),
            com_x=0.0,
            reference_velocity=1.0,
        )
        tracker.update(
            raw_contacts=(False, True),
            foot_x=(0.5, 0.0),
            clearances=(0.2, 0.0),
            com_x=0.2,
            reference_velocity=1.0,
        )
        event = tracker.update(
            raw_contacts=(True, True),
            foot_x=(0.5, 0.0),
            clearances=(0.0, 0.0),
            com_x=0.2,
            reference_velocity=1.0,
        )

        self.assertGreater(event.step_frequency, config.maximum_step_frequency)
        self.assertFalse(event.valid_step)
        self.assertTrue(event.invalid_touchdown)


class CommandBipedalEnvironmentTests(unittest.TestCase):

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
                "valid_step",
                "invalid_touchdown",
                "step_displacement",
                "swing_clearance",
                "height",
            ):
                self.assertIn(key, info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
