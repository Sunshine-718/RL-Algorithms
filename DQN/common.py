import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam, SGD


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    # pred: [B, N], target: [B, N], tau: [1, N] or [B, N]
    error = pred.unsqueeze(2) - target.expand_as(pred).unsqueeze(1)  # [B, N, N]
    huber = torch.where(error.abs() <= kappa, 0.5 * error.pow(2), kappa * (error.abs() - 0.5 * kappa))
    loss = torch.abs(tau.unsqueeze(-1) - (error.detach() < 0).float()) * huber  # [B, N, N]
    return loss.mean()


class QuantileEmbedding(nn.Module):
    def __init__(self, quantile_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(quantile_dim, out_dim),
                                 nn.LayerNorm(out_dim),
                                 nn.SiLU(True),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LayerNorm(out_dim),
                                 nn.SiLU(True))
        self.quantile_dim = quantile_dim + 1

    def forward(self, tau):
        quantile_embedding = torch.arange(1, self.quantile_dim, device=tau.device) * \
            torch.pi * tau.unsqueeze(-1)  # [batch_size, num_quantiles, 64]
        quantile_embedding = torch.cos(quantile_embedding)
        return self.net(quantile_embedding)


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
    def configure_optimizer(self, weight_decay, learning_rate, betas=(0.9, 0.999)):
        return NAdam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay, decoupled_weight_decay=True)

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


class DQNAgentBase:
    @property
    def n_step(self):
        return self._n_step

    @n_step.setter
    def n_step(self, val):
        assert val >= 1 and isinstance(val, int)
        self._n_step = val
        self.buffer.n_step = val

    def cache(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.cache_transition(state, action, reward, next_state, terminated, truncated)

    def process(self):
        self.buffer.process()

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
        self.soft_update(tau=1)

    def decay_noise(self, zero_noise=False):
        self.noise = max(self.min_noise, self.noise * self.decay) * (1 - bool(zero_noise))

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
