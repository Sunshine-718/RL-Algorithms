import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_family(family: str, source: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / family)
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


class OffPolicyBatchActionTests(unittest.TestCase):
    def test_dqn_agents_accept_batch_states(self):
        run_family(
            "DQN",
            """
            import importlib.util
            import sys
            import types
            import numpy as np

            sys.modules['flappy_bird_gymnasium'] = types.ModuleType(
                'flappy_bird_gymnasium'
            )
            from dqn import DuelingDQN, DoubleDQNAgent, Config as DQNConfig
            from iqn import DuelingIQN, IQNAgent, Config as IQNConfig
            from qrdqn import QRDuelingDQN, QRDoubleDQNAgent, Config as QRConfig
            from softdqn import (
                DuelingDQN as SoftNet, SoftDQNAgent, Config as SoftConfig,
            )
            from softiqn import (
                DuelingIQN as SoftIQNNet, SoftIQNAgent,
                Config as SoftIQNConfig,
            )
            from softqrdqn import (
                QRDuelingDQN as SoftQRNet, SoftQRDQNAgent,
                Config as SoftQRConfig,
            )

            spec = importlib.util.spec_from_file_location(
                'spr_dqn', 'DQN/spr-dqn.py'
            )
            spr = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(spr)
            agents = [
                DoubleDQNAgent(
                    'x', DuelingDQN(1e-3, 4, 16, 2, device='cpu'),
                    DQNConfig(params=None),
                ),
                IQNAgent(
                    'x', DuelingIQN(1e-3, 4, 16, 2, device='cpu'),
                    IQNConfig(params=None),
                ),
                QRDoubleDQNAgent(
                    'x', QRDuelingDQN(1e-3, 4, 16, 2, 11, device='cpu'),
                    QRConfig(params=None),
                ),
                SoftDQNAgent(
                    'x', SoftNet(1e-3, 4, 16, 2, device='cpu'),
                    SoftConfig(params=None),
                ),
                SoftIQNAgent(
                    'x', SoftIQNNet(1e-3, 4, 16, 2, device='cpu'),
                    SoftIQNConfig(params=None),
                ),
                SoftQRDQNAgent(
                    'x', SoftQRNet(1e-3, 4, 16, 2, 11, device='cpu'),
                    SoftQRConfig(params=None),
                ),
                spr.DoubleDQNAgent(
                    'x', spr.DuelingDQN(1e-3, 4, 16, 2, device='cpu'),
                    spr.Config(params=None),
                ),
            ]
            states = np.zeros((4, 4), dtype=np.float32)
            for agent in agents:
                assert np.asarray(agent.action(states, True)).shape == (4,)
                assert np.asarray(agent.action(states[0], True)).shape == ()
            """,
        )

    def test_ddpg_agents_accept_batch_states(self):
        run_family(
            "DDPG",
            """
            import numpy as np
            import ddpg
            import qrddpg

            agents = [
                ddpg.DDPGAgent(
                    'x', ddpg.DDPG(1e-3, 3, 16, 2, device='cpu'),
                    ddpg.Config(params=None),
                ),
                qrddpg.DDPGAgent(
                    'x', qrddpg.DDPG(
                        1e-3, 3, 16, 2, num_quantiles=11, device='cpu'
                    ),
                    qrddpg.Config(params=None),
                ),
            ]
            states = np.zeros((4, 3), dtype=np.float32)
            for agent in agents:
                assert agent.action(states).shape == (4, 2)
                assert agent.action(states[0]).shape == (2,)
            """,
        )

    def test_td3_agents_accept_batch_states(self):
        run_family(
            "TD3",
            """
            import numpy as np
            import td3
            import qrtd3

            agents = [
                td3.TD3Agent(
                    'x', td3.TD3(1e-3, 3, 16, 2, device='cpu'),
                    td3.Config(params=None),
                ),
                qrtd3.TD3Agent(
                    'x', qrtd3.TD3(
                        1e-3, 3, 16, 2, num_quantiles=11, device='cpu'
                    ),
                    qrtd3.Config(params=None),
                ),
            ]
            states = np.zeros((4, 3), dtype=np.float32)
            for agent in agents:
                assert agent.action(states).shape == (4, 2)
                assert agent.action(states[0]).shape == (2,)
            """,
        )

    def test_sac_agents_accept_batch_states(self):
        run_family(
            "SAC",
            """
            import numpy as np
            import sac_continuous
            import qrsac_continuous
            import sac_discrete
            import qrsac_discrete

            continuous_agents = [
                sac_continuous.ContinuousSACAgent(
                    'x', sac_continuous.ContinuousSAC(
                        1e-3, 1e-3, 4, 16, 2, device='cpu'
                    ),
                    sac_continuous.Config(params=None),
                ),
                qrsac_continuous.ContinuousSACAgent(
                    'x', qrsac_continuous.ContinuousSAC(
                        1e-3, 1e-3, 4, 16, 2,
                        num_quantiles=11, device='cpu'
                    ),
                    qrsac_continuous.Config(params=None),
                ),
            ]
            states = np.zeros((4, 4), dtype=np.float32)
            for agent in continuous_agents:
                assert agent.action(states).shape == (4, 2)
                assert agent.action(states[0]).shape == (2,)

            discrete_agents = [
                sac_discrete.DiscreteSACAgent(
                    'x', sac_discrete.DiscreteSAC(
                        1e-3, 1e-3, 4, 16, 2, device='cpu'
                    ),
                    sac_discrete.Config(params=None),
                ),
                qrsac_discrete.DiscreteSACAgent(
                    'x', qrsac_discrete.DiscreteSAC(
                        1e-3, 1e-3, 4, 16, 2,
                        num_quantiles=11, device='cpu'
                    ),
                    qrsac_discrete.Config(params=None),
                ),
            ]
            for agent in discrete_agents:
                actions, probabilities = agent.action(states)
                action, single_probabilities = agent.action(states[0])
                assert actions.shape == (4,)
                assert probabilities.shape == (4, 2)
                assert isinstance(action, int)
                assert single_probabilities.shape == (1, 2)
            """,
        )


if __name__ == "__main__":
    unittest.main()
