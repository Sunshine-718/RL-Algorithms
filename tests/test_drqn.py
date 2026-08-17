import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_dqn(source):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'DQN')
    subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


class DRQNTests(unittest.TestCase):
    def test_network_shapes_and_masked_burn_in(self):
        run_dqn(
            """
            import torch
            from drqn import RecurrentDuelingDQN

            torch.manual_seed(0)
            net = RecurrentDuelingDQN(
                1e-3, obs_dim=2, h_dim=16, recurrent_dim=8,
                num_actions=2, recurrent_layers=2, device='cpu',
            )
            net.eval()
            q, hidden = net(torch.zeros(3, 5, 2))
            assert q.shape == (3, 5, 2)
            assert hidden.shape == (2, 3, 8)

            observation = torch.randn(3, 4, 2)
            mask = torch.tensor([
                [False, False, True, True],
                [False, True, True, True],
                [True, True, True, True],
            ]).unsqueeze(-1)
            changed_padding = observation.clone()
            changed_padding[~mask.expand_as(observation)] = 1000.
            hidden_a = net.burn_in(observation, mask)
            hidden_b = net.burn_in(changed_padding, mask)
            torch.testing.assert_close(hidden_a, hidden_b)
            """
        )

    def test_partial_cartpole_wrapper_removes_both_velocities(self):
        run_dqn(
            """
            import gymnasium as gym
            import numpy as np
            from drqn import make_partial_cartpole_env

            raw_env = gym.make('CartPole-v1')
            partial_env = make_partial_cartpole_env()
            try:
                raw_observation, _ = raw_env.reset(seed=7)
                partial_observation, _ = partial_env.reset(seed=7)
                assert partial_env.observation_space.shape == (2,)
                np.testing.assert_allclose(
                    partial_observation, raw_observation[[0, 2]]
                )

                raw_next, _, _, _, _ = raw_env.step(1)
                partial_next, _, _, _, _ = partial_env.step(1)
                np.testing.assert_allclose(
                    partial_next, raw_next[[0, 2]]
                )
                assert partial_env.observation_space.contains(partial_next)
            finally:
                raw_env.close()
                partial_env.close()
            """
        )

    def test_agent_runs_recurrent_replay_update_on_cpu(self):
        run_dqn(
            """
            import math
            import numpy as np
            import torch
            from drqn import Config, DRQNAgent, RecurrentDuelingDQN

            torch.manual_seed(0)
            np.random.seed(0)
            net = RecurrentDuelingDQN(
                1e-3, obs_dim=2, h_dim=16, recurrent_dim=8,
                num_actions=2, device='cpu',
            )
            config = Config(
                params=None, capacity=64, epoch=1, burn_in=2,
                sequence_length=4, batch_size=2, noise=0., min_noise=0.,
            )
            agent = DRQNAgent('test', net, config)
            for step in range(8):
                state = np.asarray([step / 10, -step / 20], dtype=np.float32)
                next_state = np.asarray(
                    [(step + 1) / 10, -(step + 1) / 20],
                    dtype=np.float32,
                )
                agent.cache(
                    state, step % 2, 1., next_state, step == 7, False
                )
            agent.process()

            before = [parameter.detach().clone() for parameter in net.parameters()]
            loss = agent.step()
            assert math.isfinite(loss)
            assert any(
                not torch.equal(old, new)
                for old, new in zip(before, net.parameters())
            )
            assert all(
                parameter.grad is None
                for parameter in agent.target_net.parameters()
            )
            """
        )

    def test_hidden_reset_and_terminal_bootstrap_semantics(self):
        run_dqn(
            """
            import torch
            from drqn import Config, DRQNAgent, RecurrentDuelingDQN

            net = RecurrentDuelingDQN(
                1e-3, obs_dim=2, h_dim=8, recurrent_dim=4,
                num_actions=2, device='cpu',
            )
            agent = DRQNAgent(
                'test', net,
                Config(params=None, capacity=16, epoch=1),
            )
            agent.reset_hidden(batch_size=2)
            agent._action_hidden.fill_(1.)
            agent.reset_hidden(done=[True, False])
            assert torch.count_nonzero(agent._action_hidden[:, 0]) == 0
            assert torch.all(agent._action_hidden[:, 1] == 1)

            reward = torch.tensor([[[1.]], [[1.]]])
            next_q = torch.tensor([[[2.]], [[2.]]])
            terminated = torch.tensor([[[True]], [[False]]])
            target = agent.td_target(reward, next_q, terminated)
            torch.testing.assert_close(target[0], torch.tensor([[1.]]))
            torch.testing.assert_close(
                target[1], torch.tensor([[1. + agent.discount * 2.]])
            )

            try:
                agent.n_step = 2
            except ValueError:
                pass
            else:
                raise AssertionError('multi-step target must be rejected')
            """
        )


if __name__ == '__main__':
    unittest.main()
