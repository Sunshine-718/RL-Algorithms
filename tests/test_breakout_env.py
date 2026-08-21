import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DQN_ROOT = REPOSITORY_ROOT / "DQN"
for source_root in (REPOSITORY_ROOT, DQN_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from breakout_env import EpisodicLifeFire, LIFE_LOSS_REWARD
from qrdqn_breakout import BreakoutQRDQNAgent, Config


class FakeALE:
    def __init__(self, lives):
        self.current_lives = lives

    def lives(self):
        return self.current_lives


class FakeBreakoutEnv(gym.Env):
    def __init__(self, lives=3):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(2, 2), dtype=np.uint8
        )
        self.ale = FakeALE(lives)
        self.next_reward = 0.0
        self.next_terminated = False

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def step(self, action):
        observation = np.zeros((2, 2), dtype=np.uint8)
        return (
            observation,
            self.next_reward,
            self.next_terminated,
            False,
            {},
        )


class TinyQuantilePolicy(torch.nn.Module):
    obs_shape = (4, 84, 84)
    device = torch.device("cpu")

    def forward(self, state):
        quantiles = torch.tensor(
            [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]]
        )
        return quantiles.unsqueeze(0).expand(state.shape[0], -1, -1)


class EpisodicLifeRewardTest(unittest.TestCase):
    def test_life_loss_replaces_environment_reward_with_minus_one(self):
        env = FakeBreakoutEnv(lives=3)
        wrapper = EpisodicLifeFire(env)
        wrapper.lives = 3
        env.ale.current_lives = 2
        env.next_reward = 0.5

        _, reward, terminated, truncated, info = wrapper.step(0)

        self.assertEqual(reward, LIFE_LOSS_REWARD)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["life_lost"])
        self.assertTrue(info["life_terminal"])
        self.assertEqual(info["life_loss_reward"], LIFE_LOSS_REWARD)

    def test_final_life_loss_also_receives_minus_one(self):
        env = FakeBreakoutEnv(lives=1)
        wrapper = EpisodicLifeFire(env)
        wrapper.lives = 1
        env.ale.current_lives = 0
        env.next_reward = 1.0
        env.next_terminated = True

        _, reward, terminated, _, info = wrapper.step(0)

        self.assertEqual(reward, LIFE_LOSS_REWARD)
        self.assertTrue(terminated)
        self.assertTrue(info["life_lost"])
        self.assertFalse(info["life_terminal"])

    def test_reward_is_unchanged_without_life_loss(self):
        env = FakeBreakoutEnv(lives=3)
        wrapper = EpisodicLifeFire(env)
        wrapper.lives = 3
        env.next_reward = 0.5

        _, reward, terminated, _, info = wrapper.step(0)

        self.assertEqual(reward, 0.5)
        self.assertFalse(terminated)
        self.assertFalse(info["life_lost"])
        self.assertEqual(info["life_loss_reward"], 0.0)


class BreakoutEvaluationExplorationTest(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(
            net=TinyQuantilePolicy(),
            n_actions=3,
            noise=0.5,
        )
        self.state = np.zeros((4, 84, 84), dtype=np.uint8)

    def test_default_eval_epsilon_is_very_small(self):
        self.assertEqual(Config().eval_epsilon, 1e-3)

    def test_explicit_eval_epsilon_can_break_greedy_action(self):
        with (
            patch(
                "qrdqn_breakout.np.random.random",
                return_value=np.asarray([0.0005]),
            ),
            patch(
                "qrdqn_breakout.np.random.randint",
                return_value=np.asarray([2]),
            ),
        ):
            action = BreakoutQRDQNAgent.action(
                self.agent, self.state, epsilon=1e-3
            )
        self.assertEqual(action, 2)

    def test_eval_epsilon_normally_keeps_greedy_action(self):
        with (
            patch(
                "qrdqn_breakout.np.random.random",
                return_value=np.asarray([0.002]),
            ),
            patch(
                "qrdqn_breakout.np.random.randint",
                return_value=np.asarray([2]),
            ),
        ):
            action = BreakoutQRDQNAgent.action(
                self.agent, self.state, epsilon=1e-3
            )
        self.assertEqual(action, 1)


if __name__ == "__main__":
    unittest.main()
