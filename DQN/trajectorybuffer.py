from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class _Trajectory:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    def __len__(self):
        return len(self.action)

    def to(self, device):
        return _Trajectory(
            self.observation.to(device),
            self.action.to(device),
            self.reward.to(device),
            self.terminated.to(device),
            self.truncated.to(device),
        )


class TrajectoryBuffer:
    """Replay buffer that stores complete episodes and samples sequences."""

    def __init__(self, state_dim, capacity, action_dim, device='cpu'):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        if state_dim < 1 or action_dim < 1:
            raise ValueError('state_dim and action_dim must be positive')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = torch.device(device)
        self.trajectories = deque()
        self.cache = []
        self.counter = 0

    def __len__(self):
        return self.counter

    @property
    def num_trajectories(self):
        return len(self.trajectories)

    def reset(self):
        self.trajectories.clear()
        self.cache.clear()
        self.counter = 0
        return self

    def to(self, device):
        self.device = torch.device(device)
        self.trajectories = deque(
            trajectory.to(self.device) for trajectory in self.trajectories
        )
        return self

    def cache_transition(self, state, action, reward, next_state,
                         terminated, truncated):
        self.cache.append((
            state, action, reward, next_state, terminated, truncated,
        ))

    def _stack_vector(self, values, width, dtype):
        vectors = []
        for value in values:
            vector = torch.as_tensor(
                value, dtype=dtype, device=self.device
            ).reshape(-1)
            if vector.numel() != width:
                raise ValueError(
                    f'expected width {width}, got {vector.numel()}'
                )
            vectors.append(vector)
        return torch.stack(vectors)

    def _make_trajectory(self):
        state = self._stack_vector(
            (transition[0] for transition in self.cache),
            self.state_dim,
            torch.float32,
        )
        action = self._stack_vector(
            (transition[1] for transition in self.cache),
            self.action_dim,
            torch.float32,
        )
        reward = self._stack_vector(
            (transition[2] for transition in self.cache),
            1,
            torch.float32,
        )
        next_state = self._stack_vector(
            (transition[3] for transition in self.cache),
            self.state_dim,
            torch.float32,
        )
        terminated = self._stack_vector(
            (transition[4] for transition in self.cache),
            1,
            torch.bool,
        )
        truncated = self._stack_vector(
            (transition[5] for transition in self.cache),
            1,
            torch.bool,
        )

        done = terminated | truncated
        if torch.any(done[:-1]) or not bool(done[-1].item()):
            raise ValueError(
                'a trajectory must end exactly at its final transition'
            )
        if len(state) > 1 and not torch.allclose(
                state[1:], next_state[:-1], rtol=1e-5, atol=1e-6):
            raise ValueError('trajectory states are not continuous')

        observation = torch.cat((state, next_state[-1:]), dim=0)
        return _Trajectory(
            observation, action, reward, terminated, truncated,
        )

    def process(self):
        if not self.cache:
            raise RuntimeError('cannot process an empty trajectory')

        trajectory = self._make_trajectory()
        if len(trajectory) > self.capacity:
            raise ValueError('capacity must fit at least one full trajectory')

        while self.counter + len(trajectory) > self.capacity:
            self.counter -= len(self.trajectories.popleft())

        self.trajectories.append(trajectory)
        self.counter += len(trajectory)
        self.cache.clear()

    def can_sample(self, batch_size=1):
        return batch_size > 0 and len(self) >= batch_size

    def sample(self, batch_size, burn_in, sequence_length):
        """Sample padded learning sequences from uniformly chosen transitions.

        Returns burn observations, burn mask, learning observations, actions,
        rewards, terminated, truncated and loss mask. Learning observations
        contain one extra step for the bootstrap value.
        """
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if burn_in < 0:
            raise ValueError('burn_in must be non-negative')
        if sequence_length < 1:
            raise ValueError('sequence_length must be positive')
        if not self.trajectories:
            raise RuntimeError('cannot sample an empty buffer')

        trajectories = list(self.trajectories)
        cumulative_lengths = np.cumsum(
            [len(trajectory) for trajectory in trajectories]
        )
        flat_indices = np.random.randint(
            0, cumulative_lengths[-1], size=batch_size
        )

        burn_observation = torch.zeros(
            batch_size, burn_in, self.state_dim,
            dtype=torch.float32, device=self.device,
        )
        burn_mask = torch.zeros(
            batch_size, burn_in, 1,
            dtype=torch.bool, device=self.device,
        )
        observation = torch.zeros(
            batch_size, sequence_length + 1, self.state_dim,
            dtype=torch.float32, device=self.device,
        )
        action = torch.zeros(
            batch_size, sequence_length, self.action_dim,
            dtype=torch.float32, device=self.device,
        )
        reward = torch.zeros(
            batch_size, sequence_length, 1,
            dtype=torch.float32, device=self.device,
        )
        terminated = torch.zeros(
            batch_size, sequence_length, 1,
            dtype=torch.bool, device=self.device,
        )
        truncated = torch.zeros_like(terminated)
        loss_mask = torch.zeros_like(terminated)

        for batch_idx, flat_idx in enumerate(flat_indices):
            trajectory_idx = int(np.searchsorted(
                cumulative_lengths, flat_idx, side='right'
            ))
            previous_length = (
                0 if trajectory_idx == 0
                else int(cumulative_lengths[trajectory_idx - 1])
            )
            start = int(flat_idx) - previous_length
            trajectory = trajectories[trajectory_idx]

            burn_start = max(0, start - burn_in)
            burn_length = start - burn_start
            if burn_length:
                burn_observation[
                    batch_idx, burn_in - burn_length:
                ] = trajectory.observation[burn_start:start]
                burn_mask[batch_idx, burn_in - burn_length:] = True

            learn_length = min(sequence_length, len(trajectory) - start)
            learn_slice = slice(start, start + learn_length)
            observation[
                batch_idx, :learn_length + 1
            ] = trajectory.observation[start:start + learn_length + 1]
            action[batch_idx, :learn_length] = trajectory.action[learn_slice]
            reward[batch_idx, :learn_length] = trajectory.reward[learn_slice]
            terminated[
                batch_idx, :learn_length
            ] = trajectory.terminated[learn_slice]
            truncated[
                batch_idx, :learn_length
            ] = trajectory.truncated[learn_slice]
            loss_mask[batch_idx, :learn_length] = True

        return (
            burn_observation, burn_mask, observation, action, reward,
            terminated, truncated, loss_mask,
        )
