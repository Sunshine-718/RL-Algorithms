import tempfile
import unittest

import numpy as np
import torch

from Dreamer.agent import DreamerV2Agent
from Dreamer.config import Config
from Dreamer.dreamerv2 import make_carracing_env
from Dreamer.replaybuffer import SequenceReplayBuffer, as_chw_uint8


def tiny_config(params=None):
    return Config(
        capacity=64,
        batch_size=2,
        sequence_length=5,
        imagination_horizon=3,
        deter_dim=16,
        stoch_dim=4,
        stoch_classes=4,
        hidden_dim=32,
        cnn_depth=4,
        free_nats=0.0,
        params=params,
    )


class SequenceReplayBufferTest(unittest.TestCase):
    def test_episode_boundaries_and_timeout_bootstrap(self):
        buffer = SequenceReplayBuffer(20, 3, device="cpu")
        for index in range(4):
            state = np.full((2, 96, 96), index, dtype=np.uint8)
            next_state = np.full((2, 96, 96), index + 1, dtype=np.uint8)
            buffer.cache_transition(
                state,
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                index + 1,
                next_state,
                terminated=index == 2,
                truncated=index == 3,
            )
        episode = buffer.process()

        self.assertEqual(len(buffer), 4)
        self.assertEqual(episode["observation"].shape, (5, 2, 96, 96))
        self.assertEqual(float(episode["continue"][3, 0]), 0.0)
        self.assertEqual(float(episode["continue"][4, 0]), 1.0)
        batch = buffer.sample(2, 5)
        self.assertEqual(batch["observation"].shape, (2, 5, 2, 96, 96))
        self.assertTrue(torch.all(batch["is_first"][:, 0] == 1))
        self.assertTrue(torch.all(batch["action"][:, 0] == 0))

    def test_discrete_action_encoding(self):
        buffer = SequenceReplayBuffer(10, 5, discrete=True)
        image = np.zeros((2, 96, 96), dtype=np.uint8)
        buffer.cache_transition(image, 3, 1.0, image, False, True)
        episode = buffer.process()
        np.testing.assert_array_equal(
            episode["action"][1], np.array([0, 0, 0, 1, 0])
        )

    def test_capacity_evicts_complete_oldest_episode(self):
        buffer = SequenceReplayBuffer(5, 1)
        image = np.zeros((2, 96, 96), dtype=np.uint8)
        for episode_id in range(2):
            for step in range(3):
                buffer.cache_transition(
                    image,
                    [0.0],
                    episode_id,
                    image,
                    False,
                    step == 2,
                )
            buffer.process()
        self.assertEqual(len(buffer), 3)
        self.assertEqual(len(buffer.episodes), 1)

    def test_observation_contract_is_strict(self):
        with self.assertRaises(ValueError):
            as_chw_uint8(np.zeros((96, 96, 2), dtype=np.uint8))
        with self.assertRaises(TypeError):
            as_chw_uint8(np.zeros((2, 96, 96), dtype=np.float32))


class DreamerV2AgentTest(unittest.TestCase):
    def _fill_and_step(self, discrete, params=None):
        action_dim = 5 if discrete else 3
        agent = DreamerV2Agent(
            action_dim,
            discrete,
            tiny_config(params),
            device="cpu",
        )
        rng = np.random.default_rng(7)
        for index in range(7):
            state = rng.integers(0, 256, (2, 96, 96), dtype=np.uint8)
            next_state = rng.integers(
                0, 256, (2, 96, 96), dtype=np.uint8
            )
            action = (
                int(index % action_dim)
                if discrete
                else rng.uniform(-1, 1, action_dim).astype(np.float32)
            )
            agent.cache(
                state,
                action,
                float(index) / 10.0,
                next_state,
                terminated=index == 6,
                truncated=False,
            )
        agent.process()
        metrics = agent.step()
        self.assertIsNotNone(metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        observation = rng.integers(0, 256, (2, 96, 96), dtype=np.uint8)
        action = agent.action(observation, deterministic=True)
        if discrete:
            self.assertIsInstance(action, int)
            self.assertTrue(0 <= action < action_dim)
        else:
            self.assertEqual(action.shape, (action_dim,))
            self.assertTrue(np.all(action >= -1.0))
            self.assertTrue(np.all(action <= 1.0))
        return agent

    def test_discrete_and_continuous_training_steps(self):
        self._fill_and_step(discrete=True)
        self._fill_and_step(discrete=False)

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self._fill_and_step(False, directory)
            path = agent.save("smoke")
            clone = DreamerV2Agent(
                3, False, tiny_config(directory), device="cpu"
            )
            self.assertTrue(clone.load("smoke"))
            self.assertTrue(path.exists())
            for left, right in zip(
                agent.actor.parameters(), clone.actor.parameters()
            ):
                torch.testing.assert_close(left, right)


class CarRacingIntegrationTest(unittest.TestCase):
    def test_both_action_spaces_reset_and_step(self):
        for continuous in (False, True):
            env = make_carracing_env(update=True, continuous=continuous)
            try:
                observation, _ = env.reset(seed=1)
                next_observation, reward, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                self.assertEqual(as_chw_uint8(observation).shape, (2, 96, 96))
                self.assertEqual(
                    as_chw_uint8(next_observation).shape, (2, 96, 96)
                )
                self.assertTrue(np.isfinite(reward))
                self.assertIsInstance(bool(terminated), bool)
                self.assertIsInstance(bool(truncated), bool)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
