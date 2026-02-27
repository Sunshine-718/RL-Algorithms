import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.clamp(torch.abs(x), max=20)) - 1)


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    # pred: [B, N], target: [B, N], tau: [1, N]
    error = pred.unsqueeze(2) - target.expand_as(pred).unsqueeze(1)  # [B, N, N]
    huber = torch.where(error.abs() <= kappa, 0.5 * error.pow(2), kappa * (error.abs() - 0.5 * kappa))
    loss = torch.abs(tau.unsqueeze(-1) - (error.detach() < 0).float()) * huber  # [B, N, N]
    return loss.mean()


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.proj(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        self.glu = GLU(in_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim)
        self.norm = nn.RMSNorm(in_dim, 1e-5)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.residual = in_dim == out_dim

    def forward(self, x):
        residual = 0
        if self.residual:
            residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.linear(x)
        return self.dropout(x) + residual


class NNBase(nn.Module):
    def computes_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def save(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)

    def load(self, path=None):
        try:
            if path is not None:
                self.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as _:
            print('Failed to load parameters.')
        finally:
            self.to(self.device)


class PPOAgentBase:
    def store(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.store(state, action, reward, next_state, terminated, truncated)

    def save(self, model='last'):
        if self.params is not None:
            self.net.save(f'{self.params}/{self.name}_{model}.pt')
        else:
            self.net.save(f'{self.name}_{model}.pt')

    def load(self, model='last'):
        if self.params is not None:
            self.net.load(f'{self.params}/{self.name}_{model}.pt')
        else:
            self.net.load(f'{self.name}_{model}.pt')

    @staticmethod
    def GAE(discount, gaeLambda, rewards, values, next_values, terminated, truncated):
        done = terminated | truncated
        adv = torch.zeros_like(values)
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + discount * next_values[t] * (1 - terminated[t]) - values[t]
            adv[t] = advantage = delta + discount * gaeLambda * (1 - done[t]) * advantage
        return adv

    @staticmethod
    def reward_to_go(reward, discount):
        discount = torch.FloatTensor([pow(discount, i) for i in range(len(reward))]
                                     ).view(reward.shape).to(reward.device)
        ret = torch.zeros_like(reward, dtype=torch.float32)
        for idx in range(len(reward)):
            ret[idx] = torch.sum(reward[idx:] * discount[:len(ret) - idx])  # 可优化时间复杂度
        return ret


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, device='cpu'):
        self.state = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.next_state = torch.empty_like(self.state)
        self.terminated = torch.empty((capacity, 1), dtype=torch.bool, device=device)
        self.truncated = torch.empty_like(self.terminated)
        self.counter = 0
        self.device = device
        self.capacity = capacity

    def __len__(self):
        return min(self.counter, self.capacity)

    def reset(self):
        self.__init__(
            self.state.shape[1],
            self.capacity,
            self.action.shape[1],
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

    def store(self, state, action, reward, next_state, terminated, truncated):
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

    def retrive_all(self):
        length = len(self)
        assert self.counter <= self.capacity
        return self.state[:length], self.action[:length, :], self.reward[:length, :], \
            self.next_state[:length, :], self.terminated[:length, :].int(), self.truncated[:length, :].int()
