from copy import deepcopy
from types import SimpleNamespace
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DQN_ROOT = Path(__file__).resolve().parents[1] / 'DQN'
if str(DQN_ROOT) not in sys.path:
    sys.path.insert(0, str(DQN_ROOT))

from common import (
    DQNAgentBase,
    flush_n_step_transitions,
    store_n_step_transition,
)


class TinyAgent(DQNAgentBase):
    def __init__(self, hard_update=True, target_interval=8,
                 update_interval=4, tau=0.25):
        self.net = nn.Linear(1, 1, bias=False)
        self.target_net = deepcopy(self.net)
        config = SimpleNamespace(
            tau=tau,
            hard_update=hard_update,
            target_update_interval=target_interval,
            update_interval=update_interval,
        )
        self.configure_updates(config)


class StoreRecorder:
    def __init__(self):
        self.transitions = []

    def store(self, *transition):
        self.transitions.append(transition)


class DQNUpdateScheduleTest(unittest.TestCase):
    def test_one_vector_env_step_counts_as_one_environment_step(self):
        agent = TinyAgent(update_interval=4)

        self.assertEqual(agent.record_environment_steps(), 0)
        self.assertEqual(agent.record_environment_steps(), 0)
        self.assertEqual(agent.record_environment_steps(), 0)
        self.assertEqual(agent.record_environment_steps(), 1)
        self.assertEqual(agent.environment_steps, 4)

    def test_hard_update_only_runs_at_environment_interval(self):
        agent = TinyAgent(hard_update=True, target_interval=8)
        with torch.no_grad():
            agent.net.weight.fill_(2.0)
            agent.target_net.weight.zero_()

        agent.record_environment_steps(7)
        self.assertFalse(agent.update_target_after_environment_step())
        torch.testing.assert_close(
            agent.target_net.weight, torch.zeros_like(agent.target_net.weight)
        )

        agent.record_environment_steps()
        self.assertTrue(agent.update_target_after_environment_step())
        torch.testing.assert_close(agent.target_net.weight, agent.net.weight)

        with torch.no_grad():
            agent.net.weight.fill_(3.0)
        self.assertFalse(agent.update_target_after_optimizer_step())
        torch.testing.assert_close(
            agent.target_net.weight,
            torch.full_like(agent.target_net.weight, 2.0),
        )

    def test_soft_update_option_updates_after_optimizer_step(self):
        agent = TinyAgent(hard_update=False, tau=0.25)
        with torch.no_grad():
            agent.net.weight.fill_(4.0)
            agent.target_net.weight.zero_()

        self.assertTrue(agent.update_target_after_optimizer_step())
        torch.testing.assert_close(
            agent.target_net.weight,
            torch.ones_like(agent.target_net.weight),
        )
        agent.record_environment_steps(10_000)
        self.assertFalse(agent.update_target_after_environment_step())

    def test_parallel_n_step_caches_can_be_stored_incrementally(self):
        agent = SimpleNamespace(
            n_step=3,
            discount=0.5,
            reward_scale=2.0,
            buffer=StoreRecorder(),
        )
        transitions = []
        for index, reward in enumerate((1.0, 2.0, 3.0, 4.0)):
            transitions.append((
                np.asarray([index], dtype=np.float32),
                index,
                reward,
                np.asarray([index + 1], dtype=np.float32),
                False,
                False,
            ))
            store_n_step_transition(agent, transitions)

        flush_n_step_transitions(agent, transitions)
        self.assertEqual([item[-1] for item in agent.buffer.transitions], [3, 3, 2, 1])
        np.testing.assert_allclose(
            [item[2] for item in agent.buffer.transitions],
            [5.5, 9.0, 10.0, 8.0],
        )


if __name__ == '__main__':
    unittest.main()
