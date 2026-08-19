import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DQN_ROOT = Path(__file__).resolve().parents[1] / "DQN"
if str(DQN_ROOT) not in sys.path:
    sys.path.insert(0, str(DQN_ROOT))

from common import NNBase, quantile_huber_loss
from softqrdqn_breakout import (
    BreakoutSoftQRDQNAgent,
    Config,
    weighted_quantile_huber_loss,
)


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


def make_agent(quantiles, epoch=1):
    return BreakoutSoftQRDQNAgent(
        "softqrdqn_breakout_test",
        TinyQuantileNetwork(quantiles),
        Config(
            capacity=4,
            epoch=epoch,
            learning_starts=1,
            discount=1.0,
            n_step=1,
            alpha=0.1,
        ),
    )


class BreakoutSoftQRDQNTest(unittest.TestCase):
    def test_soft_target_uses_shared_action_distribution(self):
        quantiles = [[0.0, 100.0], [50.0, 50.0]]
        agent = make_agent(quantiles)
        reward = torch.zeros(1, 1)
        next_state = torch.zeros(1, 1)
        terminated = torch.zeros(1, 1)
        horizon = torch.ones(1, 1, dtype=torch.int64)

        atoms, weights = agent.td_target(
            reward, next_state, terminated, horizon
        )

        self.assertEqual(atoms.shape, (1, 4))
        self.assertEqual(weights.shape, (1, 4))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))
        expected_mean = 50.0 + agent.alpha * math.log(2.0)
        actual_mean = float((atoms * weights).sum())
        self.assertAlmostEqual(actual_mean, expected_mean, places=5)

        with torch.no_grad():
            agent.net.quantiles.copy_(
                torch.tensor([[100.0, 0.0], [50.0, 50.0]])
            )
        reordered_atoms, reordered_weights = agent.td_target(
            reward, next_state, terminated, horizon
        )
        torch.testing.assert_close(reordered_atoms, atoms)
        torch.testing.assert_close(reordered_weights, weights)

        terminal_atoms, _ = agent.td_target(
            torch.tensor([[7.0]]),
            next_state,
            torch.ones(1, 1),
            horizon,
        )
        torch.testing.assert_close(
            terminal_atoms, torch.full_like(terminal_atoms, 7.0)
        )

    def test_weighted_loss_matches_uniform_quantile_huber(self):
        pred = torch.tensor([[0.0, 1.0]], requires_grad=True)
        target = torch.tensor([[2.0, 3.0]])
        target_weight = torch.full_like(target, 0.5)
        tau = torch.tensor([[0.25, 0.75]])

        weighted_loss = weighted_quantile_huber_loss(
            pred, target, target_weight, tau
        )
        ordinary_loss = quantile_huber_loss(pred, target, tau)

        torch.testing.assert_close(weighted_loss, ordinary_loss)
        weighted_loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_alpha_recovers_at_both_bounds_without_changing_schedule(self):
        agent = make_agent(torch.zeros(4, 2), epoch=3)
        alpha_parameter = agent._alpha

        with torch.no_grad():
            agent._alpha.fill_(math.log(2.0))
        agent._update_alpha(torch.zeros(8, 4, 2))
        self.assertGreaterEqual(agent.alpha, agent.alpha_min)
        self.assertLess(agent.alpha, agent.alpha_max)

        with torch.no_grad():
            agent._alpha.fill_(math.log(0.005))
        peaked = torch.tensor(
            [[[20.0, 20.0], [-20.0, -20.0],
              [-20.0, -20.0], [-20.0, -20.0]]]
        ).expand(8, -1, -1)
        agent._update_alpha(peaked)
        self.assertGreater(agent.alpha, agent.alpha_min)
        self.assertLessEqual(agent.alpha, agent.alpha_max)
        self.assertIs(agent._alpha, alpha_parameter)
        self.assertIs(
            agent.alpha_opt.param_groups[0]["params"][0],
            alpha_parameter,
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
        metrics = agent.step(batch_size=2)
        self.assertEqual(soft_update_calls, 3)
        self.assertTrue(np.isfinite(metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
