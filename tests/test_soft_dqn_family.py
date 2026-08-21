import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DQN_ROOT = Path(__file__).resolve().parents[1] / "DQN"
if str(DQN_ROOT) not in sys.path:
    sys.path.insert(0, str(DQN_ROOT))

from common import NNBase, SoftDQNAgentBase, discrete_temperature_loss
from softdqn import Config as SoftDQNConfig
from softdqn import SoftDQNAgent
from softdqn_carracing import CarRacingSoftDQNAgent
from softdqn_carracing import Config as CarRacingConfig
from softiqn import Config as SoftIQNConfig
from softiqn import SoftIQNAgent
from softqrdqn import Config as SoftQRDQNConfig
from softqrdqn import SoftQRDQNAgent


class TinyScalarNetwork(NNBase):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))
        self.action_dim = len(values)
        self.obs_dim = 1
        self.obs_shape = (1,)
        self.device = torch.device("cpu")
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, state):
        return self.values.unsqueeze(0).expand(state.shape[0], -1)


class TinyQuantileNetwork(NNBase):
    def __init__(self, quantiles):
        super().__init__()
        quantiles = torch.as_tensor(quantiles, dtype=torch.float32)
        self.quantiles = nn.Parameter(quantiles)
        self.action_dim = quantiles.shape[0]
        self.num_quantiles = quantiles.shape[1]
        self.obs_dim = 1
        self.device = torch.device("cpu")
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, state):
        return self.quantiles.unsqueeze(0).expand(state.shape[0], -1, -1)


class TinyIQNNetwork(NNBase):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))
        self.action_dim = len(values)
        self.obs_dim = 1
        self.device = torch.device("cpu")
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, state, tau):
        return self.values.view(1, -1, 1) + tau.unsqueeze(1)


def scalar_agent_cases():
    softdqn = SoftDQNAgent(
        "softdqn_test",
        TinyScalarNetwork([0.0, 2.0]),
        SoftDQNConfig(
            params=None,
            capacity=4,
            epoch=1,
            discount=1.0,
            n_step=1,
            alpha=0.5,
        ),
    )
    carracing = CarRacingSoftDQNAgent(
        "carracing_softdqn_test",
        TinyScalarNetwork([0.0, 2.0]),
        CarRacingConfig(
            params=None,
            capacity=4,
            epoch=1,
            discount=1.0,
            n_step=1,
            alpha=0.5,
        ),
    )
    return softdqn, carracing


def make_qr_agent():
    return SoftQRDQNAgent(
        "softqrdqn_test",
        TinyQuantileNetwork([[0.0, 100.0], [50.0, 50.0]]),
        SoftQRDQNConfig(
            params=None,
            capacity=4,
            epoch=1,
            discount=1.0,
            n_step=1,
            alpha=0.1,
        ),
    )


def make_iqn_agent():
    return SoftIQNAgent(
        "softiqn_test",
        TinyIQNNetwork([0.0, 2.0]),
        SoftIQNConfig(
            params=None,
            capacity=4,
            epoch=1,
            discount=1.0,
            n_step=1,
            alpha=0.5,
        ),
    )


class SoftDQNFamilyTest(unittest.TestCase):
    def test_scalar_targets_use_online_policy_and_target_values(self):
        reward = torch.zeros(1, 1)
        next_state = torch.zeros(1, 1)
        terminated = torch.zeros(1, 1)
        horizon = torch.ones(1, 1, dtype=torch.int64)
        target_values = torch.tensor([10.0, -5.0])

        for agent in scalar_agent_cases():
            with self.subTest(agent=type(agent).__name__):
                with torch.no_grad():
                    agent.target_net.values.copy_(target_values)
                target = agent.td_target(
                    reward, next_state, terminated, horizon
                )
                online_values = agent.net.values.detach()
                log_probabilities = torch.log_softmax(
                    online_values / agent.alpha, dim=0
                )
                probabilities = log_probabilities.exp()
                expected = (
                    probabilities
                    * (target_values - agent.alpha * log_probabilities)
                ).sum()
                torch.testing.assert_close(target, expected.reshape(1, 1))

                terminal_target = agent.td_target(
                    torch.tensor([[7.0]]),
                    next_state,
                    torch.ones(1, 1),
                    horizon,
                )
                torch.testing.assert_close(
                    terminal_target, torch.tensor([[7.0]])
                )

    def test_qr_target_is_action_weighted_atom_distribution(self):
        agent = make_qr_agent()
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
        self.assertAlmostEqual(
            float((atoms * weights).sum()), expected_mean, places=5
        )

        with torch.no_grad():
            agent.net.quantiles.copy_(
                torch.tensor([[100.0, 0.0], [50.0, 50.0]])
            )
        reordered_atoms, reordered_weights = agent.td_target(
            reward, next_state, terminated, horizon
        )
        torch.testing.assert_close(reordered_atoms, atoms)
        torch.testing.assert_close(reordered_weights, weights)

    def test_iqn_uses_independent_tau_and_weighted_target_atoms(self):
        agent = make_iqn_agent()
        self.assertAlmostEqual(
            agent.target_entropy, math.log(agent.n_actions) * 0.45
        )
        with torch.no_grad():
            agent.target_net.values.copy_(torch.tensor([1.0, -1.0]))

        policy_tau = torch.tensor([[0.2, 0.4, 0.6]])
        target_tau = torch.tensor([[0.1, 0.5, 0.9]])
        atoms, weights = agent.td_target(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1, 1, dtype=torch.int64),
            policy_tau,
            target_tau,
        )
        self.assertEqual(atoms.shape, (1, 6))
        self.assertEqual(weights.shape, (1, 6))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))

        tau_calls = []

        def independent_tau(batch_size):
            value = 0.1 + 0.3 * len(tau_calls)
            tau = torch.full((batch_size, 3), value)
            tau_calls.append(tau)
            return tau

        agent.qr_tau = independent_tau
        loss, _ = agent.loss(
            torch.zeros(2, 1),
            torch.tensor([[0], [1]]),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.ones(2, 1, dtype=torch.int64),
        )
        self.assertEqual(len(tau_calls), 3)
        self.assertFalse(torch.equal(tau_calls[0], tau_calls[1]))
        self.assertFalse(torch.equal(tau_calls[1], tau_calls[2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(agent.net.values.grad).all())
        self.assertTrue(
            all(param.grad is None for param in agent.target_net.parameters())
        )

    def test_temperature_gradient_is_independent_of_alpha_scale(self):
        probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        log_probabilities = probabilities.log()
        target_entropy = math.log(4) * 0.45

        def gradient(initial_alpha):
            log_alpha = torch.nn.Parameter(
                torch.tensor([math.log(initial_alpha)])
            )
            loss = discrete_temperature_loss(
                log_alpha,
                log_probabilities,
                probabilities,
                target_entropy,
            )
            loss.backward()
            return float(log_alpha.grad.item())

        small_gradient = gradient(0.2)
        large_gradient = gradient(200.0)
        self.assertGreater(small_gradient, 0.0)
        self.assertAlmostEqual(small_gradient, large_gradient, places=6)

        peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
        log_alpha = torch.nn.Parameter(torch.tensor([math.log(0.2)]))
        loss = discrete_temperature_loss(
            log_alpha,
            peaked.log(),
            peaked,
            target_entropy,
        )
        loss.backward()
        self.assertLess(float(log_alpha.grad.item()), 0.0)

    def test_shared_alpha_is_uncapped_and_self_correcting(self):
        agents = [*scalar_agent_cases(), make_qr_agent(), make_iqn_agent()]

        for agent in agents:
            with self.subTest(agent=type(agent).__name__):
                self.assertIsInstance(agent, SoftDQNAgentBase)
                alpha_parameter = agent._alpha
                self.assertAlmostEqual(
                    agent.alpha_opt.param_groups[0]["lr"], 1e-2
                )

                agent.alpha = 2.5
                self.assertAlmostEqual(agent.alpha, 2.5, places=5)
                high_entropy_alpha = agent.alpha
                log_alpha_before = float(agent._alpha.detach().item())
                agent._update_alpha(torch.zeros(8, agent.n_actions))
                self.assertLess(agent.alpha, high_entropy_alpha)
                self.assertGreater(agent.alpha, 1.0)
                self.assertLessEqual(
                    abs(float(agent._alpha.detach().item()) - log_alpha_before),
                    1e-3 + 1e-7,
                )

                agent.alpha = 1e-8
                peaked = torch.full((8, agent.n_actions), -20.0)
                peaked[:, 0] = 20.0
                low_entropy_alpha = agent.alpha
                agent._update_alpha(peaked)
                self.assertGreater(agent.alpha, low_entropy_alpha)

                agent.alpha = 0.2
                self.assertIs(agent._alpha, alpha_parameter)
                self.assertIs(
                    agent.alpha_opt.param_groups[0]["params"][0],
                    alpha_parameter,
                )

    def test_shared_alpha_checkpoint_is_uncapped_and_validated(self):
        def checkpoint_agent(directory):
            return SoftDQNAgent(
                "softdqn_alpha_test",
                TinyScalarNetwork([0.0, 0.0]),
                SoftDQNConfig(
                    params=directory,
                    capacity=2,
                    epoch=1,
                    n_step=1,
                    alpha=0.2,
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            writer = checkpoint_agent(directory)
            writer.alpha = 2.5
            writer.save()

            restored = checkpoint_agent(directory)
            alpha_parameter = restored._alpha
            self.assertTrue(restored.load(required=True))
            self.assertAlmostEqual(restored.alpha, 2.5, places=5)
            self.assertIs(restored._alpha, alpha_parameter)
            self.assertIs(
                restored.alpha_opt.param_groups[0]["params"][0],
                alpha_parameter,
            )

            with torch.no_grad():
                writer._alpha.fill_(float("nan"))
            writer.save("nan")
            invalid = checkpoint_agent(directory)
            with self.assertRaisesRegex(ValueError, "log_alpha must be finite"):
                invalid.load("nan", required=True)
            self.assertAlmostEqual(invalid.alpha, 0.2)

            with self.assertRaisesRegex(
                ValueError, "alpha must fit its parameter dtype"
            ):
                invalid.alpha = 1e-300
            self.assertAlmostEqual(invalid.alpha, 0.2)

            legacy_path = (
                Path(directory) / "softdqn_alpha_test_legacy.pt"
            )
            writer.net.save(legacy_path)
            legacy = checkpoint_agent(directory)
            self.assertTrue(legacy.load("legacy", required=True))
            self.assertAlmostEqual(legacy.alpha, 0.2)

    def test_soft_update_option_keeps_one_target_update_per_epoch(self):
        agents = [*scalar_agent_cases(), make_qr_agent(), make_iqn_agent()]

        for agent in agents:
            with self.subTest(agent=type(agent).__name__):
                agent.hard_update = False
                agent.epoch = 2
                for index in range(2):
                    state = np.asarray([index], dtype=np.uint8)
                    agent.buffer.store(
                        state,
                        index % agent.n_actions,
                        1.0,
                        state,
                        False,
                        False,
                        1,
                    )

                soft_update_calls = 0

                def count_soft_update(tau=None):
                    nonlocal soft_update_calls
                    soft_update_calls += 1

                agent.soft_update = count_soft_update
                result = agent.step(batch_size=2)
                metrics = agent.last_training_metrics
                self.assertEqual(soft_update_calls, agent.epoch)
                self.assertEqual(
                    set(metrics),
                    {
                        "updated",
                        "loss",
                        "alpha",
                        "entropy",
                        "target_entropy",
                        "buffer_size",
                        "buffer_capacity",
                    },
                )
                self.assertTrue(metrics["updated"])
                self.assertTrue(np.isfinite(metrics["loss"]))
                self.assertTrue(np.isfinite(metrics["alpha"]))
                self.assertTrue(np.isfinite(metrics["entropy"]))
                self.assertEqual(metrics["buffer_size"], 2)
                if isinstance(agent, SoftIQNAgent):
                    self.assertIsInstance(result, float)
                elif isinstance(agent, CarRacingSoftDQNAgent):
                    self.assertEqual(set(result), {"loss", "alpha"})
                else:
                    self.assertEqual(set(result), {"loss"})

    def test_warmup_metrics_report_replay_state(self):
        agent = make_qr_agent()
        result = agent.step(batch_size=2)
        metrics = agent.last_training_metrics
        self.assertEqual(result, {})
        self.assertFalse(metrics["updated"])
        self.assertIsNone(metrics["loss"])
        self.assertIsNone(metrics["entropy"])
        self.assertEqual(metrics["buffer_size"], 0)
        self.assertEqual(metrics["buffer_capacity"], 4)
        description = agent.format_training_metrics(metrics)
        self.assertIn("loss: n/a", description)
        self.assertIn("H: n/a/", description)
        self.assertIn("replay: 0/4", description)

    def test_numerical_alpha_range_keeps_soft_targets_finite(self):
        agent = scalar_agent_cases()[0]
        reward = torch.zeros(1, 1)
        next_state = torch.zeros(1, 1)
        terminated = torch.zeros(1, 1)
        horizon = torch.ones(1, 1, dtype=torch.int64)

        for log_alpha in (-20.0, 20.0):
            with self.subTest(log_alpha=log_alpha):
                agent.alpha = math.exp(log_alpha)
                target = agent.td_target(
                    reward, next_state, terminated, horizon
                )
                self.assertTrue(torch.isfinite(target).all())

        previous_alpha = agent.alpha
        for invalid_alpha in (1e-12, 1e10):
            with self.subTest(invalid_alpha=invalid_alpha):
                with self.assertRaisesRegex(
                    ValueError, "numerical safety range"
                ):
                    agent.alpha = invalid_alpha
                self.assertEqual(agent.alpha, previous_alpha)

    def test_nonfinite_temperature_update_keeps_last_valid_alpha(self):
        agent = make_qr_agent()
        previous_alpha = agent.alpha
        with self.assertRaisesRegex(
            FloatingPointError, "alpha loss must be finite"
        ):
            agent._update_alpha(
                torch.full((2, agent.n_actions), float("nan"))
            )
        self.assertEqual(agent.alpha, previous_alpha)

        def corrupt_temperature():
            with torch.no_grad():
                agent._alpha.fill_(float("inf"))

        agent.alpha_opt.step = corrupt_temperature
        with self.assertRaisesRegex(ValueError, "log_alpha must be finite"):
            agent._update_alpha(torch.zeros(2, agent.n_actions))
        self.assertEqual(agent.alpha, previous_alpha)


if __name__ == "__main__":
    unittest.main()
