import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAC = ROOT / "SAC"
sys.path.insert(0, str(SAC))

from common import (  # noqa: E402
    make_train_test_env,
    reset_env,
    single_spaces,
    step_env,
)
from qrsac_bipedalwalker import (  # noqa: E402
    Config,
    ContinuousSAC,
    ContinuousSACAgent,
)


class QRSACBipedalWalkerTests(unittest.TestCase):
    def make_agent(self):
        network = ContinuousSAC(
            1e-3,
            3e-3,
            obs_dim=24,
            h_dim=32,
            action_dim=4,
            num_quantiles=11,
            device="cpu",
        )
        config = Config(
            params=None,
            capacity=128,
            epoch=1,
            reward_scale=1.0,
            n_step=1,
        )
        return ContinuousSACAgent("test", network, config)

    def test_uses_plain_bipedalwalker_without_command_control(self):
        source = (SAC / "qrsac_bipedalwalker.py").read_text()
        self.assertIn('"BipedalWalker-v3"', source)
        self.assertNotIn("command_bipedal", source)
        self.assertNotIn("GaitTracker", source)
        self.assertNotIn("pygame", source)

    def test_vector_environment_and_batch_action_shapes(self):
        env = make_train_test_env(
            "BipedalWalker-v3",
            update=True,
            num_envs=2,
            rescale_action=True,
            hardcore=False,
        )
        try:
            observation_space, action_space = single_spaces(env, True)
            self.assertEqual(observation_space.shape, (24,))
            self.assertEqual(action_space.shape, (4,))

            states = reset_env(env, True)
            agent = self.make_agent()
            actions = agent.action(states)
            self.assertEqual(actions.shape, (2, 4))
            self.assertTrue(np.all(actions >= -1))
            self.assertTrue(np.all(actions <= 1))

            next_states, rewards, terminated, truncated, _ = step_env(
                env, actions, True
            )
            self.assertEqual(next_states.shape, (2, 24))
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(terminated.shape, (2,))
            self.assertEqual(truncated.shape, (2,))
        finally:
            env.close()

    def test_agent_can_complete_one_update(self):
        agent = self.make_agent()
        rng = np.random.default_rng(0)
        for index in range(32):
            state = rng.normal(size=24).astype(np.float32)
            action = rng.uniform(-1, 1, size=4).astype(np.float32)
            next_state = rng.normal(size=24).astype(np.float32)
            agent.cache(
                state,
                action,
                float(rng.normal()),
                next_state,
                index == 31,
                False,
            )
        agent.process()
        agent.step(batch_size=16)

        self.assertTrue(all(
            torch.isfinite(parameter).all()
            for parameter in agent.net.parameters()
        ))
        self.assertTrue(np.isfinite(agent.alpha))


if __name__ == "__main__":
    unittest.main()
