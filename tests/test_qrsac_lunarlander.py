import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SAC"))

from common import (  # noqa: E402
    make_train_test_env,
    reset_env,
    single_spaces,
    step_env,
)
from qrsac_lunarlander import (  # noqa: E402
    Config,
    DiscreteSAC,
    DiscreteSACAgent,
    flush_episode_with_mirror,
    mirror_actions,
    mirror_observations,
)


class QRSACLunarLanderTests(unittest.TestCase):
    def make_agent(self, n_step=1):
        network = DiscreteSAC(
            1e-3,
            3e-3,
            obs_dim=8,
            h_dim=32,
            action_dim=4,
            num_quantiles=11,
            device="cpu",
        )
        config = Config(params=None, capacity=128, epoch=1, n_step=n_step)
        return DiscreteSACAgent("test", network, config)

    def test_mirror_swaps_horizontal_state_actions_and_legs(self):
        observations = np.asarray([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0],
            [-2.0, 3.0, -4.0, 5.0, -6.0, -7.0, 1.0, 0.0],
        ], dtype=np.float32)
        expected = np.asarray([
            [-1.0, 2.0, -3.0, 4.0, -5.0, -6.0, 1.0, 0.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0],
        ], dtype=np.float32)

        mirrored = mirror_observations(observations)
        np.testing.assert_array_equal(mirrored, expected)
        np.testing.assert_array_equal(
            mirror_observations(mirrored), observations
        )
        np.testing.assert_array_equal(
            mirror_actions(np.asarray([0, 1, 2, 3])),
            np.asarray([0, 3, 2, 1]),
        )

    def test_mirrored_episode_is_processed_as_a_separate_trajectory(self):
        agent = self.make_agent(n_step=2)
        state0 = np.arange(8, dtype=np.float32)
        state1 = state0 + 1
        state2 = state0 + 2
        transitions = [
            (state0, 1, 1.0, state1, False, False),
            (state1, 3, 2.0, state2, True, False),
        ]

        flush_episode_with_mirror(agent, transitions)
        states, actions, rewards, next_states, terminated, truncated = (
            agent.buffer.retrive_all()
        )

        self.assertEqual(len(agent.buffer), 4)
        self.assertEqual(transitions, [])
        np.testing.assert_array_equal(states[0].numpy(), state0)
        np.testing.assert_array_equal(states[1].numpy(), state1)
        np.testing.assert_array_equal(
            states[2:].numpy(), mirror_observations(np.stack([state0, state1]))
        )
        np.testing.assert_array_equal(actions.squeeze(1).numpy(), [1, 3, 3, 1])
        np.testing.assert_allclose(
            rewards.squeeze(1).numpy(),
            [1.0 + 0.99 * 2.0, 2.0, 1.0 + 0.99 * 2.0, 2.0],
        )
        np.testing.assert_array_equal(
            next_states[2:].numpy(),
            mirror_observations(np.stack([state2, state2])),
        )
        np.testing.assert_array_equal(
            terminated.squeeze(1).numpy(), [1, 1, 1, 1]
        )
        np.testing.assert_array_equal(
            truncated.squeeze(1).numpy(), [0, 0, 0, 0]
        )

    def test_vector_environment_and_batch_action_shapes(self):
        env = make_train_test_env("LunarLander-v3", True, num_envs=2)
        try:
            observation_space, action_space = single_spaces(env, True)
            self.assertEqual(observation_space.shape, (8,))
            self.assertEqual(action_space.n, 4)

            states = reset_env(env, True)
            agent = self.make_agent()
            actions, probabilities = agent.action(states)
            self.assertEqual(actions.shape, (2,))
            self.assertEqual(probabilities.shape, (2, 4))

            next_states, rewards, terminated, truncated, _ = step_env(
                env, actions, True
            )
            self.assertEqual(next_states.shape, (2, 8))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
        finally:
            env.close()

    def test_agent_can_complete_one_update(self):
        agent = self.make_agent()
        rng = np.random.default_rng(0)
        for index in range(32):
            state = rng.normal(size=8).astype(np.float32)
            next_state = rng.normal(size=8).astype(np.float32)
            agent.cache(
                state,
                index % 4,
                float(rng.normal()),
                next_state,
                index == 31,
                False,
            )
        agent.process()

        before = [parameter.detach().clone() for parameter in agent.net.parameters()]
        agent.step(batch_size=16)

        self.assertTrue(all(
            torch.isfinite(parameter).all()
            for parameter in agent.net.parameters()
        ))
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, agent.net.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
