import sys
import tempfile
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
    epsilon_start=0.0,
    epsilon_end=0.0,
    epsilon_decay_steps=10,
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
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        ),
    )


class BreakoutQRDQNTest(unittest.TestCase):
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

    def test_epsilon_decays_by_environment_transition(self):
        agent = make_agent(
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=10,
        )
        states = np.zeros((4, 1), dtype=np.uint8)

        agent.action(states)
        self.assertEqual(agent.environment_steps, 4)
        self.assertAlmostEqual(agent.epsilon, 0.6)

        agent.action(states, deterministic=True)
        self.assertEqual(agent.environment_steps, 4)
        self.assertAlmostEqual(agent.epsilon, 0.6)

    def test_step_updates_once_per_epoch_and_reports_metrics(self):
        agent = make_agent(epoch=3)
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

    def test_checkpoint_restores_exploration_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = make_agent(params=temp_dir)
            agent.environment_steps = 4_321
            agent.save()

            restored = make_agent(params=temp_dir)
            self.assertTrue(restored.load(required=True))
            self.assertEqual(restored.environment_steps, 4_321)
            self.assertAlmostEqual(restored.epsilon, agent.epsilon)

    def test_cnn_outputs_quantiles_for_each_action(self):
        network = BreakoutQRDuelingNetwork(
            lr=1e-4,
            num_actions=4,
            num_quantiles=51,
            device="cpu",
        )
        single = network(torch.zeros(4, 84, 84, dtype=torch.uint8))
        batch = network(torch.zeros(2, 4, 84, 84))

        self.assertEqual(single.shape, (1, 4, 51))
        self.assertEqual(batch.shape, (2, 4, 51))
        self.assertTrue(torch.isfinite(single).all())
        self.assertTrue(torch.isfinite(batch).all())


if __name__ == "__main__":
    unittest.main()
