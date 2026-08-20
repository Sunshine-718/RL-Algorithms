import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DQN_ROOT = REPOSITORY_ROOT / "DQN"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(DQN_ROOT) not in sys.path:
    sys.path.insert(0, str(DQN_ROOT))

from breakout_network import BreakoutQRDuelingNetwork
from common import NNBase
from qrdqn_breakout import BreakoutQRDQNAgent, Config


class TinyQuantileNetwork(NNBase):
    def __init__(self, quantiles):
        super().__init__()
        quantiles = torch.as_tensor(quantiles, dtype=torch.float32)
        self.quantiles = nn.Parameter(quantiles)
        self.action_dim = quantiles.shape[0]
        self.num_quantiles = quantiles.shape[1]
        self.obs_shape = (1,)
        self.device = torch.device("cpu")
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, state):
        batch_size = state.shape[0] if state.ndim > 1 else 1
        return self.quantiles.unsqueeze(0).expand(batch_size, -1, -1)


def make_agent(
    quantiles=((0.0, 10.0), (6.0, 6.0)),
    *,
    epoch=1,
    discount=1.0,
    n_step=1,
    noise=0.0,
    min_noise=0.0,
    decay=0.99,
    params=None,
):
    return BreakoutQRDQNAgent(
        "qrdqn_breakout_test",
        TinyQuantileNetwork(quantiles),
        Config(
            params=params,
            capacity=16,
            epoch=epoch,
            learning_starts=1,
            discount=discount,
            n_step=n_step,
            noise=noise,
            min_noise=min_noise,
            decay=decay,
        ),
    )


class BreakoutQRDQNTest(unittest.TestCase):
    def test_default_exploration_and_target_update_config(self):
        config = Config()
        self.assertEqual(config.noise, 0.5)
        self.assertEqual(config.min_noise, 0.01)
        self.assertEqual(config.decay, 0.998)
        self.assertEqual(config.tau, 0.005)

    def test_double_q_target_selects_online_action_and_target_quantiles(self):
        agent = make_agent(discount=0.5, n_step=2)
        with torch.no_grad():
            agent.target_net.quantiles.copy_(
                torch.tensor([[100.0, 100.0], [20.0, 40.0]])
            )

        reward = torch.tensor([[1.0]])
        next_state = torch.zeros(1, 1)
        horizon = torch.tensor([[2]], dtype=torch.int64)
        target = agent.td_target(
            reward, next_state, torch.zeros(1, 1), horizon
        )
        torch.testing.assert_close(target, torch.tensor([[6.0, 11.0]]))

        terminal_target = agent.td_target(
            torch.tensor([[7.0]]),
            next_state,
            torch.ones(1, 1),
            horizon,
        )
        torch.testing.assert_close(
            terminal_target, torch.tensor([[7.0, 7.0]])
        )

    def test_noise_does_not_decay_when_selecting_actions(self):
        agent = make_agent(
            noise=0.5,
            min_noise=0.01,
            decay=0.998,
        )
        states = np.zeros((4, 1), dtype=np.uint8)

        agent.action(states)
        self.assertAlmostEqual(agent.noise, 0.5)

        agent.action(states, deterministic=True)
        self.assertAlmostEqual(agent.noise, 0.5)

    def test_step_updates_once_per_epoch_and_reports_metrics(self):
        agent = make_agent(
            epoch=3, noise=0.5, min_noise=0.01, decay=0.998
        )
        for index in range(2):
            state = np.asarray([index], dtype=np.uint8)
            agent.buffer.store(
                state, index, 1.0, state, False, False, 1
            )

        soft_update_calls = 0

        def count_soft_update(tau=None):
            nonlocal soft_update_calls
            soft_update_calls += 1

        agent.soft_update = count_soft_update
        loss = agent.step(batch_size=2)

        self.assertEqual(soft_update_calls, 3)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(agent.last_training_metrics["updated"])
        self.assertEqual(agent.last_training_metrics["buffer_size"], 2)
        self.assertAlmostEqual(agent.noise, 0.499)

    def test_cnn_outputs_quantiles_for_each_action(self):
        network = BreakoutQRDuelingNetwork(
            lr=1e-4,
            num_actions=3,
            num_quantiles=51,
            device="cpu",
        )
        single = network(torch.zeros(4, 84, 84, dtype=torch.uint8))
        batch = network(torch.zeros(2, 4, 84, 84))

        self.assertEqual(single.shape, (1, 3, 51))
        self.assertEqual(batch.shape, (2, 3, 51))
        self.assertTrue(torch.isfinite(single).all())
        self.assertTrue(torch.isfinite(batch).all())


if __name__ == "__main__":
    unittest.main()
