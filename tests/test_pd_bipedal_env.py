from functools import partial
from pathlib import Path
import sys
import unittest

import gymnasium as gym
import numpy as np


SAC_ROOT = Path(__file__).resolve().parents[1] / "SAC"
if str(SAC_ROOT) not in sys.path:
    sys.path.insert(0, str(SAC_ROOT))

from pd_bipedal_env import (
    JOINT_LOWER_LIMITS,
    JOINT_UPPER_LIMITS,
    PDController,
    make_pd_bipedal_env,
    normalized_action_to_target,
)


class PDControllerTest(unittest.TestCase):
    def test_pd_formula_and_clipping(self):
        controller = PDController(
            kp=(2.0, 1.0, 2.0, 1.0),
            kd=(0.5, 0.2, 0.5, 0.2),
        )
        output = controller.update(
            target=(1.0, 0.0, -1.0, 0.5),
            value=(0.0, 1.0, 0.0, 0.0),
            velocity=(1.0, -1.0, -1.0, 1.0),
        )
        np.testing.assert_allclose(output, (1.0, -0.8, -1.0, 0.3))

    def test_normalized_action_uses_joint_limits(self):
        np.testing.assert_allclose(
            normalized_action_to_target((-1.0, -1.0, -1.0, -1.0)),
            JOINT_LOWER_LIMITS,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            normalized_action_to_target((1.0, 1.0, 1.0, 1.0)),
            JOINT_UPPER_LIMITS,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            normalized_action_to_target((0.0, 0.0, 0.0, 0.0)),
            (JOINT_LOWER_LIMITS + JOINT_UPPER_LIMITS) * 0.5,
        )

    def test_environment_reports_pd_control_values(self):
        env = make_pd_bipedal_env(max_episode_steps=20)
        try:
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (24,))
            next_observation, reward, _, _, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (24,))
            self.assertTrue(np.isfinite(reward))
            self.assertEqual(info["pd_target_angles"].shape, (4,))
            self.assertEqual(info["pd_motor_actions"].shape, (4,))
            self.assertTrue(np.all(np.abs(info["pd_motor_actions"]) <= 1.0))
        finally:
            env.close()

    def test_synchronous_vector_environment(self):
        factories = [
            partial(make_pd_bipedal_env, max_episode_steps=1)
            for _ in range(2)
        ]
        env = gym.vector.SyncVectorEnv(
            factories, autoreset_mode=gym.vector.AutoresetMode.DISABLED
        )
        try:
            observations, _ = env.reset(seed=7)
            self.assertEqual(observations.shape, (2, 24))
            next_observations, rewards, terminated, truncated, infos = env.step(
                np.zeros((2, 4), dtype=np.float32)
            )
            self.assertEqual(next_observations.shape, (2, 24))
            self.assertTrue(np.isfinite(rewards).all())
            self.assertEqual(infos["pd_motor_actions"].shape, (2, 4))
            done = np.logical_or(terminated, truncated)
            self.assertTrue(done.all())
            reset_observations, _ = env.reset(
                options={"reset_mask": done}
            )
            self.assertEqual(reset_observations.shape, (2, 24))
            self.assertTrue(np.isfinite(reset_observations).all())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
