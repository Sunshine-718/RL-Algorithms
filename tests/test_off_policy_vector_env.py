import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DQN"))

from common import (  # noqa: E402
    flush_episode,
    make_train_test_env,
    reset_done_envs,
    reset_env,
    single_spaces,
    step_env,
)


class FakeAgent:
    def __init__(self):
        self.cache_batches = []
        self.current_cache = []

    def cache(self, *transition):
        self.current_cache.append(transition)

    def process(self):
        self.cache_batches.append(self.current_cache.copy())
        self.current_cache.clear()


class OffPolicyVectorEnvTests(unittest.TestCase):
    def test_async_vector_env_steps_and_resets_done_workers(self):
        env = make_train_test_env(
            "CartPole-v1",
            update=True,
            num_envs=2,
            max_episode_steps=1,
        )
        try:
            observation_space, action_space = single_spaces(env, update=True)
            self.assertEqual(observation_space.shape, (4,))
            self.assertEqual(action_space.n, 2)
            self.assertEqual(len(env.get_attr("x_threshold")), 2)

            states = reset_env(env, update=True)
            self.assertEqual(states.shape, (2, 4))
            next_states, rewards, terminated, truncated, _ = step_env(
                env, np.zeros(2, dtype=np.int64), update=True
            )
            done = terminated | truncated
            self.assertTrue(np.all(done))
            self.assertEqual(rewards.shape, (2,))

            reset_states = reset_done_envs(
                env, next_states, done, update=True
            )
            self.assertEqual(reset_states.shape, (2, 4))
        finally:
            env.close()

    def test_vector_rescale_action_keeps_single_action_space(self):
        env = make_train_test_env(
            "Pendulum-v1",
            update=True,
            num_envs=2,
            rescale_action=True,
        )
        try:
            _, action_space = single_spaces(env, update=True)
            np.testing.assert_allclose(action_space.low, -1)
            np.testing.assert_allclose(action_space.high, 1)
            states = reset_env(env, update=True)
            next_states, rewards, _, _, _ = step_env(
                env, np.zeros((2, 1), dtype=np.float32), update=True
            )
            self.assertEqual(states.shape, next_states.shape)
            self.assertEqual(rewards.shape, (2,))
        finally:
            env.close()

    def test_episode_flushes_remain_separate(self):
        agent = FakeAgent()
        episode0 = [("env0-step0",), ("env0-step1",)]
        episode1 = [("env1-step0",)]

        flush_episode(agent, episode0)
        flush_episode(agent, episode1)

        self.assertEqual(
            agent.cache_batches,
            [[("env0-step0",), ("env0-step1",)], [("env1-step0",)]],
        )
        self.assertEqual(episode0, [])
        self.assertEqual(episode1, [])


if __name__ == "__main__":
    unittest.main()
