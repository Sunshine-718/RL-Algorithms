import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    """Pairwise quantile Huber loss for prediction and target distributions."""
    td_error = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_error = td_error.abs()
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    weight = torch.abs(
        tau.unsqueeze(-1) - (td_error.detach() < 0).float()
    )
    return (weight * huber).mean()


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

    def save(self, path=None, extra_state=None):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            checkpoint = self.state_dict()
            if extra_state:
                checkpoint = {
                    'model_state_dict': checkpoint,
                    **extra_state,
                }
            torch.save(checkpoint, path)

    def load(self, path=None):
        try:
            if path is None:
                return False
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True
            )
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f'Checkpoint not found: {path}')
            return False
        finally:
            self.to(self.device)
        return checkpoint


class PPOAgentBase:
    def store(self, state, action, reward, next_state, terminated, truncated):
        reward = reward * getattr(self, 'reward_scale', 1.0)
        self.buffer.store(
            state, action, reward, next_state, terminated, truncated
        )

    def save(self, model='last'):
        path = f'{self.name}_{model}.pt'
        if self.params is not None:
            path = f'{self.params}/{path}'
        normalizer_state = {}
        for name, wrapper in getattr(self, 'normalizers', {}).items():
            rms = getattr(wrapper, 'obs_rms', None)
            if rms is None:
                rms = getattr(wrapper, 'return_rms', None)
            if rms is None:
                continue
            state = {
                'mean': torch.as_tensor(np.asarray(rms.mean)).clone(),
                'var': torch.as_tensor(np.asarray(rms.var)).clone(),
                'count': float(rms.count),
            }
            if hasattr(wrapper, 'discounted_reward'):
                state['discounted_reward'] = torch.as_tensor(
                    np.asarray(wrapper.discounted_reward)
                ).clone()
            normalizer_state[name] = state
        extra_state = (
            {'normalizers': normalizer_state} if normalizer_state else None
        )
        self.net.save(path, extra_state)

    def load(self, model='last', required=False):
        path = f'{self.name}_{model}.pt'
        if self.params is not None:
            path = f'{self.params}/{path}'
        checkpoint = self.net.load(path)
        if checkpoint is False:
            if required:
                raise FileNotFoundError(path)
            return False
        normalizers = getattr(self, 'normalizers', {})
        saved_normalizers = checkpoint.get('normalizers', {})
        missing_normalizers = set(normalizers) - set(saved_normalizers)
        if missing_normalizers:
            missing = ', '.join(sorted(missing_normalizers))
            raise RuntimeError(
                f'Checkpoint is missing normalizer state: {missing}'
            )
        for name, state in saved_normalizers.items():
            wrapper = normalizers.get(name)
            if wrapper is None:
                continue
            rms = getattr(wrapper, 'obs_rms', None)
            if rms is None:
                rms = getattr(wrapper, 'return_rms', None)
            if rms is None:
                continue
            mean = state['mean']
            var = state['var']
            if torch.is_tensor(mean):
                mean = mean.cpu().numpy()
            if torch.is_tensor(var):
                var = var.cpu().numpy()
            rms.mean = np.asarray(mean, dtype=rms.mean.dtype).copy()
            rms.var = np.asarray(var, dtype=rms.var.dtype).copy()
            rms.count = float(state['count'])
            if 'discounted_reward' in state and hasattr(
                    wrapper, 'discounted_reward'):
                discounted_reward = state['discounted_reward']
                if torch.is_tensor(discounted_reward):
                    discounted_reward = discounted_reward.cpu().numpy()
                wrapper.discounted_reward = np.asarray(
                    discounted_reward
                ).copy()
        return True

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
