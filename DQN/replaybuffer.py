import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, discount, n_step=1, device='cpu'):
        self.state = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.next_state = torch.empty_like(self.state)
        self.terminated = torch.empty((capacity, 1), dtype=torch.bool, device=device)
        self.truncated = torch.empty_like(self.terminated)
        self.n = torch.empty_like(self.reward, dtype=torch.long, device=device)
        self.counter = 0
        self.device = device
        self.discount = discount
        self.n_step = n_step
        self.capacity = capacity
        self.cache = []

    def __len__(self):
        return min(self.counter, self.capacity)

    def reset(self):
        self.__init__(
            self.state.shape[1],
            self.capacity,
            self.action.shape[1],
            self.discount,
            self.n_step,
            self.device
        )
        return self

    def to(self, device):
        self.device = device
        for name in list(vars(self)):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def cache_transition(self, state, action, reward, next_state, terminated, truncated):
        self.cache.append((state, action, reward, next_state, terminated, truncated))

    def process(self):
        assert self.cache
        if self.n_step > 1:
            n = [min(i, self.n_step - 1) + 1 for i in range(len(self.cache))]
            while self.cache:
                state, action, _, _, _, _ = self.cache[0]
                i = 0
                reward = 0
                end = min(self.n_step, len(self.cache))
                for i in range(0, end):
                    reward += self.cache[i][2] * pow(self.discount, i)
                _, _, _, next_state, terminated, truncated = self.cache[end - 1]
                self.store(state, action, reward, next_state, terminated, truncated, n.pop())
                self.cache.pop(0)
        elif self.n_step == 1:
            for i in self.cache:
                self.store(*i, n=1)
            self.cache = []

    def store(self, state, action, reward, next_state, terminated, truncated, n):
        idx = self.counter % len(self.state)
        self.counter += 1
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        elif isinstance(action, np.float32):
            action = float(action)
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).to(self.device)
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = float(reward)
        self.next_state[idx] = next_state
        self.terminated[idx] = terminated
        self.truncated[idx] = truncated
        self.n[idx] = n

    def retrive_all(self):
        length = len(self)
        assert self.counter >= length
        return self.state[:length], self.action[:length, :], self.reward[:length, :], self.next_state[:length], \
            self.terminated[:length, :].int(), self.truncated[:length, :].int()

    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(0, len(self), batch_size)).long()
        return self.state[idx], self.action[idx, :], self.reward[idx, :], self.next_state[idx], \
            self.terminated[idx, :].int(), self.truncated[idx, :].int(), self.n[idx, :]
