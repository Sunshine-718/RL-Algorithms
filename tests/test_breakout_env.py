import sys
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from breakout_env import EpisodicLifeFire, LIFE_LOSS_REWARD


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


if __name__ == "__main__":
    unittest.main()
