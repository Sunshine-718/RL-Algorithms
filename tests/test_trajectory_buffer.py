import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DQN'))

from trajectorybuffer import TrajectoryBuffer  # noqa: E402


def add_trajectory(buffer, start, length, truncated=False):
    for step in range(length):
        state = np.asarray([start + step, -(start + step)], dtype=np.float32)
        next_state = np.asarray(
            [start + step + 1, -(start + step + 1)], dtype=np.float32
        )
        final = step == length - 1
        buffer.cache_transition(
            state,
            step % 2,
            float(step),
            next_state,
            final and not truncated,
            final and truncated,
        )
    buffer.process()


class TrajectoryBufferTests(unittest.TestCase):
    def test_sample_keeps_continuous_episode_history_and_masks_padding(self):
        buffer = TrajectoryBuffer(2, capacity=20, action_dim=1)
        add_trajectory(buffer, start=0, length=3)
        add_trajectory(buffer, start=100, length=5, truncated=True)

        with patch(
            'trajectorybuffer.np.random.randint',
            return_value=np.asarray([0, 2, 3, 7]),
        ):
            batch = buffer.sample(
                batch_size=4, burn_in=2, sequence_length=3
            )

        (burn_observation, burn_mask, observation, action, reward,
         terminated, truncated, loss_mask) = batch
        self.assertEqual(burn_observation.shape, (4, 2, 2))
        self.assertEqual(observation.shape, (4, 4, 2))
        self.assertEqual(action.shape, (4, 3, 1))
        self.assertEqual(reward.shape, (4, 3, 1))
        self.assertEqual(terminated.dtype, torch.bool)
        self.assertEqual(truncated.dtype, torch.bool)

        self.assertFalse(torch.any(burn_mask[0]))
        torch.testing.assert_close(
            observation[0, :4, 0], torch.tensor([0., 1., 2., 3.])
        )
        self.assertTrue(torch.all(loss_mask[0]))

        torch.testing.assert_close(
            burn_observation[1, :, 0], torch.tensor([0., 1.])
        )
        self.assertTrue(torch.all(burn_mask[1]))
        torch.testing.assert_close(
            observation[1, :2, 0], torch.tensor([2., 3.])
        )
        self.assertEqual(int(loss_mask[1].sum()), 1)
        self.assertTrue(bool(terminated[1, 0]))

        self.assertFalse(torch.any(burn_mask[2]))
        self.assertEqual(float(observation[2, 0, 0]), 100.)
        self.assertEqual(int(loss_mask[2].sum()), 3)

        torch.testing.assert_close(
            burn_observation[3, :, 0], torch.tensor([102., 103.])
        )
        torch.testing.assert_close(
            observation[3, :2, 0], torch.tensor([104., 105.])
        )
        self.assertEqual(int(loss_mask[3].sum()), 1)
        self.assertTrue(bool(truncated[3, 0]))

    def test_capacity_evicts_whole_old_trajectories(self):
        buffer = TrajectoryBuffer(2, capacity=5, action_dim=1)
        add_trajectory(buffer, start=0, length=3)
        add_trajectory(buffer, start=100, length=3)

        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.num_trajectories, 1)
        with patch(
            'trajectorybuffer.np.random.randint',
            return_value=np.asarray([0]),
        ):
            observation = buffer.sample(1, 0, 1)[2]
        self.assertEqual(float(observation[0, 0, 0]), 100.)

    def test_rejects_incomplete_or_discontinuous_trajectory(self):
        buffer = TrajectoryBuffer(2, capacity=10, action_dim=1)
        buffer.cache_transition(
            [0., 0.], 0, 1., [1., 1.], False, False
        )
        with self.assertRaisesRegex(ValueError, 'end exactly'):
            buffer.process()

        buffer.reset()
        buffer.cache_transition(
            [0., 0.], 0, 1., [1., 1.], False, False
        )
        buffer.cache_transition(
            [2., 2.], 1, 1., [3., 3.], True, False
        )
        with self.assertRaisesRegex(ValueError, 'not continuous'):
            buffer.process()

    def test_rejects_trajectory_larger_than_capacity(self):
        buffer = TrajectoryBuffer(2, capacity=2, action_dim=1)
        for step in range(3):
            buffer.cache_transition(
                [step, step], 0, 1., [step + 1, step + 1],
                step == 2, False,
            )
        with self.assertRaisesRegex(ValueError, 'full trajectory'):
            buffer.process()


if __name__ == '__main__':
    unittest.main()
