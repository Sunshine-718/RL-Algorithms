import numpy as np
import torch


class ImageReplayBuffer:
    """Replay buffer that stores image observations as uint8 on the CPU."""

    def __init__(self, observation_shape, capacity, action_dim, discount,
                 n_step=1, device="cpu"):
        self.observation_shape = tuple(observation_shape)
        if not self.observation_shape:
            raise ValueError("observation_shape must not be empty")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not isinstance(n_step, int) or n_step < 1:
            raise ValueError("n_step must be a positive integer")

        self.state = np.empty(
            (capacity, *self.observation_shape), dtype=np.uint8
        )
        self.next_state = np.empty_like(self.state)
        self.action = np.empty((capacity, action_dim), dtype=np.int64)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.bool_)
        self.truncated = np.empty_like(self.terminated)
        self.n = np.empty((capacity, 1), dtype=np.int64)

        self.counter = 0
        self.device = torch.device(device)
        self.discount = discount
        self.n_step = n_step
        self.capacity = capacity
        self.cache = []

    def __len__(self):
        return min(self.counter, self.capacity)

    def reset(self):
        self.counter = 0
        self.cache.clear()
        return self

    def to(self, device):
        # Images deliberately remain on the CPU and are copied only when sampled.
        self.device = torch.device(device)
        return self

    def cache_transition(self, state, action, reward, next_state, terminated,
                         truncated):
        self.cache.append(
            (state, action, reward, next_state, terminated, truncated)
        )

    def process(self):
        if not self.cache:
            raise RuntimeError("cannot process an empty transition cache")

        episode_length = len(self.cache)
        for start in range(episode_length):
            end = min(start + self.n_step, episode_length)
            reward = sum(
                self.cache[index][2] * self.discount ** (index - start)
                for index in range(start, end)
            )
            state, action = self.cache[start][:2]
            next_state, terminated, truncated = self.cache[end - 1][3:]
            self.store(
                state, action, reward, next_state, terminated, truncated,
                end - start,
            )
        self.cache.clear()

    def store(self, state, action, reward, next_state, terminated, truncated,
              n):
        index = self.counter % self.capacity
        self.counter += 1

        self.state[index] = self._as_uint8_image(state)
        self.next_state[index] = self._as_uint8_image(next_state)

        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_array.size != self.action.shape[1]:
            raise ValueError(
                f"expected {self.action.shape[1]} action values, "
                f"got {action_array.size}"
            )
        self.action[index] = action_array
        self.reward[index, 0] = float(reward)
        self.terminated[index, 0] = bool(terminated)
        self.truncated[index, 0] = bool(truncated)
        self.n[index, 0] = int(n)

    def sample(self, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self) == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")

        indices = np.random.randint(0, len(self), size=batch_size)
        state = self._image_tensor(self.state[indices])
        next_state = self._image_tensor(self.next_state[indices])
        action = torch.from_numpy(self.action[indices]).to(self.device)
        reward = torch.from_numpy(self.reward[indices]).to(self.device)
        terminated = torch.from_numpy(
            self.terminated[indices]
        ).to(self.device, dtype=torch.float32)
        truncated = torch.from_numpy(
            self.truncated[indices]
        ).to(self.device, dtype=torch.float32)
        n = torch.from_numpy(self.n[indices]).to(self.device)
        return state, action, reward, next_state, terminated, truncated, n

    def _as_uint8_image(self, observation):
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        image = np.asarray(observation)
        if image.shape != self.observation_shape:
            raise ValueError(
                f"expected observation shape {self.observation_shape}, "
                f"got {image.shape}"
            )
        if image.dtype == np.uint8:
            return image
        if not np.issubdtype(image.dtype, np.number):
            raise TypeError("image observation must be numeric")
        if not np.all(np.isfinite(image)):
            raise ValueError("image observation contains non-finite values")

        image = image.astype(np.float32, copy=False)
        if image.size and image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 255.0
        return np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)

    def _image_tensor(self, image):
        return torch.from_numpy(image).to(
            self.device, dtype=torch.float32
        ).div_(255.0)
