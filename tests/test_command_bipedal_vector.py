import argparse
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAC = ROOT / "SAC"
sys.path.insert(0, str(SAC))

from command_bipedal_env import CommandBipedalConfig  # noqa: E402
from replaybuffer import ReplayBuffer  # noqa: E402
from train_command_bipedal_vector import (  # noqa: E402
    flush_episode,
    make_vector_env,
    prepare_training_args,
)


class FakeAgent:
    def __init__(self, buffer):
        self.buffer = buffer

    def cache(self, *transition):
        self.buffer.cache_transition(*transition)

    def process(self):
        self.buffer.process()


class CommandVectorTrainerTests(unittest.TestCase):
    def test_completed_environment_caches_do_not_mix_n_step_returns(self):
        buffer = ReplayBuffer(1, capacity=20, action_dim=1, discount=0.9, n_step=3)
        agent = FakeAgent(buffer)
        env0 = [
            (
                np.array([i], dtype=np.float32),
                np.array([0], dtype=np.float32),
                1.0,
                np.array([i + 1], dtype=np.float32),
                i == 2,
                False,
            )
            for i in range(3)
        ]
        env1 = [
            (
                np.array([100 + i], dtype=np.float32),
                np.array([1], dtype=np.float32),
                10.0,
                np.array([101 + i], dtype=np.float32),
                i == 2,
                False,
            )
            for i in range(3)
        ]

        flush_episode(agent, env0)
        flush_episode(agent, env1)

        np.testing.assert_array_equal(
            buffer.state[:6, 0].cpu().numpy(),
            np.array([0, 1, 2, 100, 101, 102]),
        )
        np.testing.assert_allclose(
            buffer.reward[:6, 0].cpu().numpy(),
            np.array([1 + 0.9 + 0.81, 1 + 0.9, 1, 10 + 9 + 8.1, 10 + 9, 10]),
            rtol=1e-6,
        )
        self.assertEqual(buffer.cache, [])

    def test_smoke_defaults_update_actor(self):
        args = argparse.Namespace(
            smoke_test=True,
            total_steps=1_000_000,
            learning_starts=10_000,
            actor_learning_starts=None,
            random_steps=None,
            batch_size=256,
            capacity=500_000,
            eval_every=20_000,
            checkpoint_every=20_000,
            num_envs=4,
        )

        prepared = prepare_training_args(args)

        self.assertEqual(prepared.learning_starts, 128)
        self.assertEqual(prepared.actor_learning_starts, 128)
        self.assertLess(prepared.actor_learning_starts, prepared.total_steps)

    def test_sync_vector_env_uses_next_step_autoreset(self):
        config = CommandBipedalConfig(max_episode_steps=20)
        env = make_vector_env(2, 11, config, "sync")
        try:
            observations, _ = env.reset(seed=11)
            self.assertEqual(observations.shape[0], 2)
            self.assertEqual(
                env.metadata["autoreset_mode"].value,
                "NextStep",
            )
            observations, rewards, terminated, truncated, _ = env.step(
                np.zeros((2, 4), dtype=np.float32)
            )
            self.assertEqual(observations.shape[0], 2)
            self.assertEqual(rewards.shape, (2,))
            self.assertFalse(np.any(terminated))
            self.assertFalse(np.any(truncated))
        finally:
            env.close()

    def test_async_vector_env_steps_in_subprocesses(self):
        config = CommandBipedalConfig(max_episode_steps=20)
        env = make_vector_env(2, 17, config, "async")
        try:
            observations, _ = env.reset(seed=17)
            observations, rewards, terminated, truncated, _ = env.step(
                np.zeros((2, 4), dtype=np.float32)
            )

            self.assertEqual(observations.shape[0], 2)
            self.assertEqual(rewards.shape, (2,))
            self.assertFalse(np.any(terminated))
            self.assertFalse(np.any(truncated))
            self.assertEqual(env.metadata["autoreset_mode"].value, "NextStep")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
