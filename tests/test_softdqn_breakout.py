import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DQN_ROOT = Path(__file__).resolve().parents[1] / "DQN"
if str(DQN_ROOT) not in sys.path:
    sys.path.insert(0, str(DQN_ROOT))

from common import NNBase, store_n_step_transition
from softdqn_breakout import (
    BreakoutDuelingNetwork,
    BreakoutSoftDQNAgent,
    Config,
)


class TinyScalarNetwork(NNBase):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))
        self.action_dim = len(values)
        self.obs_shape = (1,)
        self.device = torch.device("cpu")
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, state):
        batch_size = state.shape[0] if state.ndim > 1 else 1
        return self.values.unsqueeze(0).expand(batch_size, -1)


def make_agent(values=(0.0, 2.0), epoch=1, discount=1.0, n_step=1):
    return BreakoutSoftDQNAgent(
        "softdqn_breakout_test",
        TinyScalarNetwork(values),
        Config(
            params=None,
            capacity=16,
            epoch=epoch,
            learning_starts=1,
            discount=discount,
            n_step=n_step,
            alpha=0.5,
        ),
    )


class BreakoutSoftDQNTest(unittest.TestCase):
    def test_cnn_outputs_one_scalar_per_action(self):
        network = BreakoutDuelingNetwork(
            lr=1e-4,
            num_actions=4,
            device="cpu",
        )
        image = torch.randint(
            0,
            256,
            (4, 84, 84),
            dtype=torch.uint8,
        )

        uint8_output = network(image)
        float_output = network(
            image.unsqueeze(0).repeat(2, 1, 1, 1).float() / 255.0
        )

        self.assertEqual(uint8_output.shape, (1, 4))
        self.assertEqual(float_output.shape, (2, 4))
        torch.testing.assert_close(uint8_output[0], float_output[0])
        self.assertTrue(torch.isfinite(float_output).all())

    def test_target_uses_online_policy_and_target_values(self):
        agent = make_agent()
        with torch.no_grad():
            agent.target_net.values.copy_(torch.tensor([10.0, -5.0]))

        reward = torch.zeros(1, 1)
        next_state = torch.zeros(1, 1)
        terminated = torch.zeros(1, 1)
        horizon = torch.ones(1, 1, dtype=torch.int64)
        target = agent.td_target(
            reward,
            next_state,
            terminated,
            horizon,
        )

        online_q = agent.net.values.detach()
        target_q = agent.target_net.values.detach()
        log_probabilities = torch.log_softmax(
            online_q / agent.alpha,
            dim=0,
        )
        expected = (
            log_probabilities.exp()
            * (target_q - agent.alpha * log_probabilities)
        ).sum()
        torch.testing.assert_close(target, expected.reshape(1, 1))
        self.assertAlmostEqual(float(target), -4.685159, places=5)

        terminal_target = agent.td_target(
            torch.tensor([[7.0]]),
            next_state,
            torch.ones(1, 1),
            horizon,
        )
        torch.testing.assert_close(terminal_target, torch.tensor([[7.0]]))
        self.assertFalse(target.requires_grad)

    def test_loss_updates_online_network_only(self):
        agent = make_agent()
        with torch.no_grad():
            agent.target_net.values.copy_(torch.tensor([1.0, -1.0]))

        loss, _ = agent.loss(
            torch.zeros(2, 1),
            torch.tensor([[0], [1]]),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.ones(2, 1),
            torch.ones(2, 1, dtype=torch.int64),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(agent.net.values.grad).all())
        self.assertTrue(
            all(param.grad is None for param in agent.target_net.parameters())
        )

    def test_n_step_target_keeps_bootstrap_for_truncation(self):
        agent = make_agent(values=(0.0, 0.0), discount=0.5)
        self.assertAlmostEqual(
            agent.target_entropy,
            0.45 * math.log(agent.n_actions),
        )
        reward = torch.tensor([[1.0], [2.0], [3.0]])
        next_state = torch.zeros(3, 1)
        terminated = torch.tensor([[0.0], [1.0], [0.0]])
        truncated = torch.ones(3, 1)
        horizon = torch.tensor([[1], [3], [2]])

        target = agent.td_target(
            reward,
            next_state,
            terminated,
            horizon,
        )
        soft_value = agent.alpha * math.log(agent.n_actions)
        expected = reward + torch.pow(0.5, horizon) * (
            1.0 - terminated
        ) * soft_value
        torch.testing.assert_close(target, expected)

        loss, q_values = agent.loss(
            torch.zeros(3, 1),
            torch.zeros(3, 1, dtype=torch.int64),
            reward,
            next_state,
            terminated,
            truncated,
            horizon,
        )
        self.assertEqual(q_values.shape, (3, agent.n_actions))
        torch.testing.assert_close(
            loss,
            F.smooth_l1_loss(torch.zeros(3, 1), expected),
        )

    def test_action_is_argmax_and_preserves_network_mode(self):
        agent = make_agent(values=(1.0, 3.0, 2.0))
        state = np.zeros((1,), dtype=np.uint8)

        agent.net.eval()
        action = agent.action(state, deterministic=True)
        self.assertEqual(action, 1)
        self.assertFalse(agent.net.training)

        batch_actions = agent.action(
            np.zeros((2, 1), dtype=np.uint8),
            deterministic=True,
        )
        np.testing.assert_array_equal(batch_actions, np.asarray([1, 1]))

    def test_step_updates_target_once_per_epoch(self):
        agent = make_agent(epoch=3)
        self.assertAlmostEqual(
            agent.alpha_opt.param_groups[0]["lr"], 1e-2
        )
        for index in range(2):
            state = np.asarray([index], dtype=np.uint8)
            agent.buffer.store(
                state,
                index,
                1.0,
                state,
                False,
                False,
                1,
            )

        target_before = agent.target_net.values.detach().clone()
        events = []
        original_critic_step = agent.net.opt.step
        original_alpha_update = agent._update_alpha
        original_soft_update = agent.soft_update

        def count_critic_step(*args, **kwargs):
            events.append("critic")
            return original_critic_step(*args, **kwargs)

        def count_alpha_update(q_values):
            events.append("alpha")
            return original_alpha_update(q_values)

        def count_soft_update(tau=None):
            events.append("target")
            return original_soft_update(tau)

        agent.net.opt.step = count_critic_step
        agent._update_alpha = count_alpha_update
        agent.soft_update = count_soft_update
        result = agent.step(batch_size=2)
        metrics = agent.last_training_metrics

        self.assertEqual(events, ["critic", "alpha", "target"] * 3)
        self.assertFalse(torch.equal(agent.target_net.values, target_before))
        self.assertEqual(set(result), {"loss", "alpha"})
        self.assertTrue(metrics["updated"])
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(np.isfinite(result["alpha"]))
        self.assertTrue(np.isfinite(metrics["entropy"]))

    def test_parallel_n_step_caches_do_not_mix(self):
        agent = make_agent(discount=0.5, n_step=2)
        cache_a = [
            (
                np.asarray([1], dtype=np.uint8),
                0,
                1.0,
                np.asarray([2], dtype=np.uint8),
                False,
                False,
            ),
            (
                np.asarray([2], dtype=np.uint8),
                1,
                2.0,
                np.asarray([3], dtype=np.uint8),
                False,
                False,
            ),
        ]
        cache_b = [
            (
                np.asarray([10], dtype=np.uint8),
                1,
                10.0,
                np.asarray([11], dtype=np.uint8),
                False,
                False,
            ),
            (
                np.asarray([11], dtype=np.uint8),
                0,
                20.0,
                np.asarray([12], dtype=np.uint8),
                False,
                False,
            ),
        ]

        self.assertTrue(store_n_step_transition(agent, cache_a))
        self.assertTrue(store_n_step_transition(agent, cache_b))

        np.testing.assert_allclose(
            agent.buffer.reward[:2, 0],
            np.asarray([2.0, 20.0]),
        )
        np.testing.assert_array_equal(
            agent.buffer.state[:2, 0],
            np.asarray([1, 10], dtype=np.uint8),
        )
        np.testing.assert_array_equal(agent.buffer.n[:2, 0], [2, 2])


if __name__ == "__main__":
    unittest.main()
