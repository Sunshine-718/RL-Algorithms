import argparse
import importlib.util
import sys
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAC = ROOT / "SAC"
sys.path.insert(0, str(SAC))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BipedalSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qrsac = load_module("qrsac_symmetry_test", SAC / "qrsac_continuous.py")
        cls.vector = load_module("vector_symmetry_test", SAC / "train_qrsac_bipedalwalker_vector.py")

    def test_mirror_is_an_involution_for_numpy_and_torch(self):
        observation = np.arange(24, dtype=np.float32)
        action = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mirrored_observation = self.qrsac.mirror_bipedal_observation(observation)
        mirrored_action = self.qrsac.mirror_bipedal_action(action)
        np.testing.assert_array_equal(mirrored_observation[4:9], observation[9:14])
        np.testing.assert_array_equal(mirrored_observation[9:14], observation[4:9])
        np.testing.assert_array_equal(mirrored_action, [3.0, 4.0, 1.0, 2.0])
        np.testing.assert_array_equal(
            self.qrsac.mirror_bipedal_observation(mirrored_observation), observation
        )
        stacked_cpg = np.concatenate([observation, observation + 24, [0.5, -0.5]]).astype(np.float32)
        mirrored_stacked_cpg = self.qrsac.mirror_bipedal_observation(stacked_cpg)
        np.testing.assert_array_equal(mirrored_stacked_cpg[-2:], [-0.5, 0.5])
        np.testing.assert_array_equal(
            self.qrsac.mirror_bipedal_observation(mirrored_stacked_cpg), stacked_cpg
        )
        torch_action = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        torch.testing.assert_close(
            self.qrsac.mirror_bipedal_action(self.qrsac.mirror_bipedal_action(torch_action)),
            torch_action,
        )

    def test_contact_reward_only_bonuses_alternation(self):
        args = argparse.Namespace(
            energy_ema_decay=0.95,
            energy_balance_coef=0.0,
            contact_alternation_bonus=0.02,
            flight_penalty=0.003,
            double_support_penalty=0.01,
            double_support_grace=1,
        )
        state = np.zeros(24, dtype=np.float32)
        action = np.zeros(4, dtype=np.float32)
        state[8] = 1.0
        first = self.vector.symmetry_reward(state, action, 0.0, 0.0, -1, 0, args)
        self.assertEqual(first[0], 0.0)
        repeated = self.vector.symmetry_reward(state, action, first[1], first[2], first[3], first[4], args)
        self.assertEqual(repeated[0], 0.0)
        state[8], state[13] = 0.0, 1.0
        alternated = self.vector.symmetry_reward(
            state, action, repeated[1], repeated[2], repeated[3], repeated[4], args
        )
        self.assertAlmostEqual(alternated[0], 0.02)


if __name__ == "__main__":
    unittest.main()
