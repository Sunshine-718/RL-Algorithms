import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAC = ROOT / "SAC"


def load_vector_module():
    for name in ("common", "replaybuffer", "qrsac_continuous", "train_qrsac_bipedalwalker"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(SAC))
    try:
        spec = importlib.util.spec_from_file_location("vector_trainer_test", SAC / "train_qrsac_bipedalwalker_vector.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class FakeAgent:
    def __init__(self, buffer):
        self.buffer = buffer

    def cache(self, *transition):
        self.buffer.cache_transition(*transition)

    def process(self):
        self.buffer.process()


class VectorEpisodeCacheTests(unittest.TestCase):
    def test_completed_environment_caches_do_not_mix_n_step_returns(self):
        module = load_vector_module()
        from replaybuffer import ReplayBuffer

        buffer = ReplayBuffer(1, capacity=20, action_dim=1, discount=0.9, n_step=3)
        agent = FakeAgent(buffer)

        env0 = [
            (np.array([i], dtype=np.float32), np.array([0], dtype=np.float32), 1.0,
             np.array([i + 1], dtype=np.float32), i == 2, False)
            for i in range(3)
        ]
        env1 = [
            (np.array([100 + i], dtype=np.float32), np.array([1], dtype=np.float32), 10.0,
             np.array([101 + i], dtype=np.float32), i == 2, False)
            for i in range(3)
        ]

        module.flush_episode(agent, env0)
        module.flush_episode(agent, env1)

        states = buffer.state[:6, 0].cpu().numpy()
        rewards = buffer.reward[:6, 0].cpu().numpy()
        np.testing.assert_array_equal(states, np.array([0, 1, 2, 100, 101, 102]))
        np.testing.assert_allclose(
            rewards,
            np.array([1 + 0.9 + 0.81, 1 + 0.9, 1, 10 + 9 + 8.1, 10 + 9, 10]),
            rtol=1e-6,
        )
        self.assertEqual(buffer.cache, [])


if __name__ == "__main__":
    unittest.main()
