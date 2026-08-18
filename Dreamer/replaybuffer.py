from collections import deque

import numpy as np
import torch

from carracing_env import OBSERVATION_SHAPE


def as_chw_uint8(observation):
    """Require the fixed CarRacing observation used by this implementation."""
    image = np.asarray(observation)
    if image.shape != OBSERVATION_SHAPE:
        raise ValueError(
            f"expected uint8 CHW observation {OBSERVATION_SHAPE}, "
            f"got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 observation, got {image.dtype}")
    return np.ascontiguousarray(image)


class SequenceReplayBuffer:
    """Episode replay that samples contiguous sequences for an RSSM."""

    def __init__(
        self,
        capacity,
        action_dim,
        discrete=False,
        device="cpu",
    ):
        if capacity < 1 or action_dim < 1:
            raise ValueError("capacity and action_dim must be positive")

        self.capacity = int(capacity)
        self.action_dim = int(action_dim)
        self.discrete = bool(discrete)
        self.device = torch.device(device)
        self.episodes = deque()
        self.cache = []
        self.transitions = 0

    def __len__(self):
        return self.transitions

    def reset(self):
        self.episodes.clear()
        self.cache.clear()
        self.transitions = 0
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def cache_transition(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
    ):
        state = as_chw_uint8(state)
        next_state = as_chw_uint8(next_state)
        self.cache.append(
            (
                state.copy(),
                action,
                float(reward),
                next_state.copy(),
                bool(terminated),
                bool(truncated),
            )
        )

    def process(self):
        if not self.cache:
            raise RuntimeError("cannot process an empty episode")

        transitions = len(self.cache)
        observation = np.stack(
            [self.cache[0][0]] + [item[3] for item in self.cache]
        )
        action = np.zeros(
            (transitions + 1, self.action_dim), dtype=np.float32
        )
        reward = np.zeros((transitions + 1, 1), dtype=np.float32)
        continued = np.ones((transitions + 1, 1), dtype=np.float32)
        is_first = np.zeros((transitions + 1, 1), dtype=np.float32)
        terminated = np.zeros((transitions + 1, 1), dtype=np.float32)
        truncated = np.zeros((transitions + 1, 1), dtype=np.float32)
        is_first[0] = 1.0

        for index, item in enumerate(self.cache, start=1):
            action[index] = self._encode_action(item[1])
            reward[index, 0] = item[2]
            terminated[index, 0] = float(item[4])
            truncated[index, 0] = float(item[5])
            # 只有真正终止才阻断 bootstrap；时间上限仍可继续估值。
            continued[index, 0] = 1.0 - float(item[4])

        if transitions > self.capacity:
            start = transitions - self.capacity
            observation = observation[start:]
            action = action[start:]
            reward = reward[start:]
            continued = continued[start:]
            is_first = is_first[start:]
            terminated = terminated[start:]
            truncated = truncated[start:]
            transitions = self.capacity
            action[0] = 0.0
            reward[0] = 0.0
            continued[0] = 1.0
            is_first[0] = 1.0
            terminated[0] = 0.0
            truncated[0] = 0.0

        episode = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "continue": continued,
            "is_first": is_first,
            "terminated": terminated,
            "truncated": truncated,
        }
        self.episodes.append(episode)
        self.transitions += transitions
        self.cache.clear()
        self._enforce_capacity()
        return episode

    def can_sample(self, batch_size, sequence_length):
        if batch_size < 1 or sequence_length < 2:
            return False
        return any(
            len(episode["observation"]) >= sequence_length
            for episode in self.episodes
        )

    def sample(self, batch_size, sequence_length):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        eligible = [
            episode
            for episode in self.episodes
            if len(episode["observation"]) >= sequence_length
        ]
        if not eligible:
            raise RuntimeError("no episode is long enough for this sequence")

        chunks = []
        for _ in range(batch_size):
            episode = eligible[np.random.randint(len(eligible))]
            upper = len(episode["observation"]) - sequence_length + 1
            start = np.random.randint(upper)
            chunk = {
                key: value[start : start + sequence_length].copy()
                for key, value in episode.items()
            }
            # 任意截取点都视作新的递归起点，首个动作和奖励不再依赖片段外历史。
            chunk["action"][0] = 0.0
            chunk["reward"][0] = 0.0
            chunk["continue"][0] = 1.0
            chunk["is_first"][0] = 1.0
            chunk["terminated"][0] = 0.0
            chunk["truncated"][0] = 0.0
            chunks.append(chunk)

        batch = {}
        for key in chunks[0]:
            values = np.stack([chunk[key] for chunk in chunks])
            tensor = torch.from_numpy(values).to(self.device)
            if key == "observation":
                tensor = tensor.to(torch.float32).div_(255.0).sub_(0.5)
            else:
                tensor = tensor.to(torch.float32)
            batch[key] = tensor
        return batch

    def _encode_action(self, action):
        if self.discrete:
            index = int(np.asarray(action).reshape(-1)[0])
            if not 0 <= index < self.action_dim:
                raise ValueError(
                    f"discrete action {index} is outside [0, {self.action_dim})"
                )
            encoded = np.zeros(self.action_dim, dtype=np.float32)
            encoded[index] = 1.0
            return encoded

        encoded = np.asarray(action, dtype=np.float32).reshape(-1)
        if encoded.size != self.action_dim:
            raise ValueError(
                f"expected {self.action_dim} action values, got {encoded.size}"
            )
        if not np.all(np.isfinite(encoded)):
            raise ValueError("action contains non-finite values")
        return encoded

    def _enforce_capacity(self):
        while self.episodes and self.transitions > self.capacity:
            oldest = self.episodes.popleft()
            self.transitions -= len(oldest["observation"]) - 1
