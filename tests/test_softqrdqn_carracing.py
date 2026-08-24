import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


DQN_ROOT = Path(__file__).resolve().parents[1] / "DQN"
if str(DQN_ROOT) not in sys.path:
    sys.path.insert(0, str(DQN_ROOT))

from common import NNBase
from image_replaybuffer import ImageReplayBuffer
from softqrdqn_carracing import (
    CarRacingQRDuelingNetwork,
    CarRacingSoftQRDQNAgent,
    Config,
)


class TinyImageQuantileNetwork(NNBase):
    def __init__(self, quantiles):
        super().__init__()
        quantiles = torch.as_tensor(quantiles, dtype=torch.float32)
        self.quantiles = torch.nn.Parameter(quantiles)
        self.action_dim = quantiles.shape[0]
        self.num_quantiles = quantiles.shape[1]
        self.obs_shape = (2, 4, 4)
        self.device = torch.device("cpu")
        self.opt = torch.optim.SGD(self.parameters(), lr=1e-2)

    def forward(self, state):
        if state.ndim == len(self.obs_shape):
            state = state.unsqueeze(0)
        return self.quantiles.unsqueeze(0).expand(state.shape[0], -1, -1)


def make_agent(quantiles, epoch=1):
    return CarRacingSoftQRDQNAgent(
        "softqrdqn_carracing_test",
        TinyImageQuantileNetwork(quantiles),
        Config(
            params=None,
            capacity=4,
            epoch=epoch,
            discount=1.0,
            n_step=1,
            alpha=0.2,
        ),
    )


class CarRacingSoftQRDQNTest(unittest.TestCase):
    def test_convolution_and_linear_layers_use_kaiming_normal(self):
        initializer = torch.nn.init.kaiming_normal_
        with patch(
            "softqrdqn_carracing.nn.init.kaiming_normal_",
            wraps=initializer,
        ) as kaiming_normal:
            CarRacingQRDuelingNetwork(
                lr=1e-4,
                num_actions=5,
                num_quantiles=7,
                device="cpu",
            )

        # 3 个卷积层，加上 Value/Advantage 分支各 2 个线性层。
        self.assertEqual(kaiming_normal.call_count, 7)

    def test_cnn_outputs_one_distribution_per_action(self):
        network = CarRacingQRDuelingNetwork(
            lr=1e-4,
            num_actions=5,
            num_quantiles=7,
            device="cpu",
        )
        uint8_state = torch.randint(
            0, 256, (2, 2, 96, 96), dtype=torch.uint8
        )

        network.eval()
        uint8_output = network(uint8_state)
        float_output = network(uint8_state.float() / 255.0)

        self.assertEqual(uint8_output.shape, (2, 5, 7))
        torch.testing.assert_close(uint8_output, float_output)
        with self.assertRaisesRegex(ValueError, "expected input shape"):
            network(torch.zeros(2, 3, 96, 96))

    def test_cnn_and_dueling_heads_follow_requested_structure(self):
        network = CarRacingQRDuelingNetwork(
            lr=1e-4,
            num_actions=5,
            num_quantiles=7,
            device="cpu",
        )
        feature_types = [type(module) for module in network.features]
        self.assertEqual(
            feature_types,
            [
                torch.nn.Conv2d,
                torch.nn.BatchNorm2d,
                torch.nn.SiLU,
                torch.nn.Conv2d,
                torch.nn.BatchNorm2d,
                torch.nn.SiLU,
                torch.nn.Conv2d,
                torch.nn.BatchNorm2d,
                torch.nn.SiLU,
                torch.nn.Flatten,
            ],
        )
        self.assertFalse(hasattr(network, "hidden"))
        self.assertEqual(
            [type(module) for module in network.value],
            [
                torch.nn.Linear,
                torch.nn.RMSNorm,
                torch.nn.SiLU,
                torch.nn.Linear,
            ],
        )
        self.assertEqual(
            [type(module) for module in network.advantage],
            [
                torch.nn.Linear,
                torch.nn.RMSNorm,
                torch.nn.SiLU,
                torch.nn.Linear,
            ],
        )
        self.assertEqual(network.value[0].in_features, network.feature_dim)
        self.assertEqual(network.value[0].out_features, 512)
        self.assertEqual(network.value[-1].out_features, 7)
        self.assertEqual(network.advantage[0].out_features, 512)
        self.assertEqual(network.advantage[-1].out_features, 5 * 7)

    def test_target_update_copies_batch_norm_buffers(self):
        network = CarRacingQRDuelingNetwork(
            lr=1e-4,
            num_actions=5,
            num_quantiles=7,
            device="cpu",
        )
        agent = CarRacingSoftQRDQNAgent(
            "softqrdqn_carracing_batch_norm_test",
            network,
            Config(params=None, capacity=1),
        )
        online_norm = agent.net.features[1]
        target_norm = agent.target_net.features[1]
        online_norm.running_mean.fill_(3.0)
        online_norm.running_var.fill_(4.0)
        online_norm.num_batches_tracked.fill_(5)
        target_norm.running_mean.zero_()
        target_norm.running_var.fill_(1.0)
        target_norm.num_batches_tracked.zero_()

        agent.soft_update(tau=0.5)

        torch.testing.assert_close(
            target_norm.running_mean, online_norm.running_mean
        )
        torch.testing.assert_close(
            target_norm.running_var, online_norm.running_var
        )
        torch.testing.assert_close(
            target_norm.num_batches_tracked,
            online_norm.num_batches_tracked,
        )

    def test_target_keeps_complete_action_quantile_distributions(self):
        agent = make_agent([[0.0, 10.0], [5.0, 5.0]])
        with torch.no_grad():
            agent.target_net.quantiles.copy_(
                torch.tensor([[0.0, 2.0], [4.0, 6.0]])
            )

        reward = torch.zeros(1, 1)
        next_state = torch.zeros(1, 2, 4, 4)
        terminated = torch.zeros(1, 1)
        horizon = torch.ones(1, 1, dtype=torch.int64)
        atoms, weights = agent.td_target(
            reward, next_state, terminated, horizon
        )

        self.assertEqual(atoms.shape, (1, 4))
        self.assertEqual(weights.shape, (1, 4))
        torch.testing.assert_close(weights, torch.full_like(weights, 0.25))
        entropy_bonus = agent.alpha * math.log(2.0)
        expected_atoms = torch.tensor(
            [[0.0, 2.0, 4.0, 6.0]]
        ) + entropy_bonus
        torch.testing.assert_close(atoms, expected_atoms)

        terminal_atoms, _ = agent.td_target(
            torch.tensor([[7.0]]),
            next_state,
            torch.ones(1, 1),
            horizon,
        )
        torch.testing.assert_close(
            terminal_atoms, torch.full_like(terminal_atoms, 7.0)
        )

    def test_action_uses_mean_quantiles_and_preserves_network_mode(self):
        agent = make_agent([[0.0, 10.0], [4.0, 4.0]])
        state = np.zeros((2, 4, 4), dtype=np.uint8)
        batch = np.stack([state, state])

        agent.net.eval()
        self.assertEqual(agent.action(state, deterministic=True), 0)
        np.testing.assert_array_equal(
            agent.action(batch, deterministic=True), np.asarray([0, 0])
        )
        self.assertFalse(agent.net.training)

    def test_image_replay_supports_a_complete_optimizer_step(self):
        agent = make_agent([[0.0, 1.0], [1.0, 2.0]], epoch=2)
        self.assertIsInstance(agent.buffer, ImageReplayBuffer)
        self.assertAlmostEqual(
            agent.target_entropy, math.log(agent.n_actions) * 0.45
        )

        for index in range(2):
            state = np.full(
                agent.net.obs_shape, index * 255, dtype=np.uint8
            )
            agent.buffer.store(
                state,
                index,
                1.0,
                state,
                False,
                False,
                1,
            )

        agent.hard_update = False
        update_calls = 0

        def count_soft_update(tau=None):
            nonlocal update_calls
            update_calls += 1

        agent.soft_update = count_soft_update
        result = agent.step(batch_size=2)

        self.assertEqual(update_calls, agent.epoch)
        self.assertEqual(set(result), {"loss", "alpha"})
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(np.isfinite(result["alpha"]))
        self.assertTrue(agent.last_training_metrics["updated"])
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in agent.target_net.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
