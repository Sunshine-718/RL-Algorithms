import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path):
    path = ROOT / relative_path
    for name in ("common", "replaybuffer"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "test_" + relative_path.replace("/", "_").replace(".", "_"), path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class ContinuousSACLogProbTests(unittest.TestCase):
    def test_qrsac_step_reports_finite_metrics(self):
        module = load_module("SAC/qrsac_continuous.py")
        net = module.ContinuousSAC(
            1e-3,
            1e-3,
            obs_dim=8,
            h_dim=16,
            action_dim=4,
            num_quantiles=5,
            device="cpu",
        )
        config = module.Config(capacity=16, epoch=1, reward_scale=1.0, n_step=1)
        agent = module.ContinuousSACAgent("metrics_test", net, config)
        state = torch.zeros(8)
        action = torch.zeros(4)
        for _ in range(8):
            agent.buffer.store(state, action, 0.0, state, False, False, 1)

        metrics = agent.step(batch_size=4)

        self.assertTrue(metrics["actor_updated"])
        self.assertTrue(math.isfinite(metrics["critic_loss"]))
        self.assertTrue(math.isfinite(metrics["actor_loss"]))

    def test_quantile_huber_loss_uses_target_minus_prediction_sign(self):
        module = load_module("SAC/qrsac_continuous.py")
        tau = torch.tensor([[0.2]])
        pred = torch.tensor([[0.0]])

        positive_error = module.quantile_huber_loss(pred, torch.tensor([[1.0]]), tau)
        negative_error = module.quantile_huber_loss(pred, torch.tensor([[-1.0]]), tau)

        torch.testing.assert_close(positive_error, torch.tensor(0.1))
        torch.testing.assert_close(negative_error, torch.tensor(0.4))

    def assert_actor_returns_transformed_log_prob(self, module):
        torch.manual_seed(0)
        net = module.ContinuousSAC(
            1e-3,
            1e-3,
            obs_dim=3,
            h_dim=8,
            action_dim=2,
            action_limit=1.5,
            device="cpu",
        )
        state = torch.randn(4, 3)

        action, log_prob = net.actor(state)

        alpha, beta = net.b_alpha(net.hidden(state)), net.b_beta(net.hidden(state))
        dist = torch.distributions.Beta(torch.exp(alpha) + 1, torch.exp(beta) + 1)
        raw_action = action / (2 * net.action_limit) + 0.5
        expected = (
            dist.log_prob(raw_action) - math.log(2 * net.action_limit)
        ).sum(dim=-1, keepdim=True)

        self.assertEqual(action.shape, (4, 2))
        self.assertEqual(log_prob.shape, (4, 1))
        torch.testing.assert_close(log_prob, expected)

    def assert_td_target_uses_minus_alpha_log_prob(self, module):
        class FakeNet:
            alpha = torch.nn.Parameter(torch.tensor([[math.log(0.5)]]))

            def actor(self, next_state):
                return torch.ones(next_state.shape[0], 1), torch.full((next_state.shape[0], 1), -2.0)

        class FakeTargetNet:
            def critic(self, next_state, action):
                q1 = torch.full((next_state.shape[0], 1), 3.0)
                q2 = torch.full((next_state.shape[0], 1), 5.0)
                return q1, q2

        agent = object.__new__(module.ContinuousSACAgent)
        agent.net = FakeNet()
        agent.target_net = FakeTargetNet()
        agent.discount = 0.9

        reward = torch.ones(2, 1)
        next_state = torch.zeros(2, 3)
        terminated = torch.zeros(2, 1)
        n = torch.ones(2, 1, dtype=torch.long)

        target = module.ContinuousSACAgent.td_target(agent, reward, next_state, terminated, n)

        expected = reward + 0.9 * (torch.full((2, 1), 3.0) - 0.5 * torch.full((2, 1), -2.0))
        torch.testing.assert_close(target, expected)

    def assert_quantile_td_target_uses_minus_alpha_log_prob(self, module):
        class FakeNet:
            alpha = torch.nn.Parameter(torch.tensor([[math.log(0.5)]]))

            def actor(self, next_state):
                return torch.ones(next_state.shape[0], 1), torch.full((next_state.shape[0], 1), -2.0)

        class FakeTargetNet:
            def critic(self, next_state, action):
                q1 = torch.full((next_state.shape[0], 3), 3.0)
                q2 = torch.full((next_state.shape[0], 3), 5.0)
                return q1, q2

        agent = object.__new__(module.ContinuousSACAgent)
        agent.net = FakeNet()
        agent.target_net = FakeTargetNet()
        agent.discount = 0.9

        reward = torch.ones(2, 1)
        next_state = torch.zeros(2, 3)
        terminated = torch.zeros(2, 1)
        n = torch.ones(2, 1, dtype=torch.long)

        target = module.ContinuousSACAgent.td_target(agent, reward, next_state, terminated, n)

        expected = reward + 0.9 * (torch.full((2, 3), 3.0) - 0.5 * torch.full((2, 1), -2.0))
        torch.testing.assert_close(target, expected)

    def test_continuous_sac_actor_returns_sampled_action_log_prob(self):
        self.assert_actor_returns_transformed_log_prob(load_module("SAC/sac_continuous.py"))

    def test_qr_continuous_sac_actor_returns_sampled_action_log_prob(self):
        self.assert_actor_returns_transformed_log_prob(load_module("SAC/qrsac_continuous.py"))

    def test_continuous_sac_td_target_uses_soft_value_log_prob_term(self):
        self.assert_td_target_uses_minus_alpha_log_prob(load_module("SAC/sac_continuous.py"))

    def test_qr_continuous_sac_td_target_uses_soft_value_log_prob_term(self):
        self.assert_quantile_td_target_uses_minus_alpha_log_prob(load_module("SAC/qrsac_continuous.py"))


if __name__ == "__main__":
    unittest.main()
