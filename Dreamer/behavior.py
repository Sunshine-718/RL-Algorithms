import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class TruncatedNormal:
    def __init__(self, mean, std, low=-1.0, high=1.0):
        self.mean = mean
        self.std = std
        self.low = torch.as_tensor(low, dtype=mean.dtype, device=mean.device)
        self.high = torch.as_tensor(
            high, dtype=mean.dtype, device=mean.device
        )
        self.normal = Normal(mean, std)
        self.low_cdf = self.normal.cdf(self.low)
        self.high_cdf = self.normal.cdf(self.high)
        self.normalizer = (self.high_cdf - self.low_cdf).clamp_min(1e-6)

    def rsample(self):
        probability = self.low_cdf + torch.rand_like(self.mean) * self.normalizer
        sample = self.normal.icdf(probability.clamp(1e-6, 1.0 - 1e-6))
        clipped = sample.clamp(self.low + 1e-6, self.high - 1e-6)
        return sample + (clipped - sample).detach()

    def log_prob(self, value):
        return self.normal.log_prob(value) - self.normalizer.log()

    def entropy(self):
        alpha = (self.low - self.mean) / self.std
        beta = (self.high - self.mean) / self.std
        alpha_pdf = torch.exp(-0.5 * alpha.square()) / math.sqrt(
            2.0 * math.pi
        )
        beta_pdf = torch.exp(-0.5 * beta.square()) / math.sqrt(
            2.0 * math.pi
        )
        correction = (
            alpha * alpha_pdf - beta * beta_pdf
        ) / (2.0 * self.normalizer)
        return (
            self.std.log()
            + self.normalizer.log()
            + 0.5 * math.log(2.0 * math.pi * math.e)
            + correction
        )


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, discrete, hidden_dim=400):
        super().__init__()
        self.discrete = bool(discrete)
        self.action_dim = action_dim
        output_dim = action_dim if discrete else action_dim * 2
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def sample(self, feature, deterministic=False):
        output = self.network(feature)
        if self.discrete:
            distribution = Categorical(logits=output)
            index = output.argmax(-1) if deterministic else distribution.sample()
            action = F.one_hot(index, self.action_dim).to(feature.dtype)
            return {
                "action": action,
                "index": index,
                "log_prob": distribution.log_prob(index).unsqueeze(-1),
                "entropy": distribution.entropy().unsqueeze(-1),
            }

        mean, raw_std = output.chunk(2, dim=-1)
        mean = torch.tanh(mean)
        std = 2.0 * torch.sigmoid(raw_std / 2.0) + 0.1
        distribution = TruncatedNormal(mean, std)
        action = mean if deterministic else distribution.rsample()
        log_prob = distribution.log_prob(action)
        return {
            "action": action,
            "index": None,
            "log_prob": log_prob.sum(-1, keepdim=True),
            "entropy": distribution.entropy().sum(-1, keepdim=True),
        }


class Critic(nn.Module):
    def __init__(self, feature_dim, hidden_dim=400):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature):
        return self.network(feature)


def lambda_return(reward, value, discount, lambda_):
    """Compute returns for [H, B, 1] imagined trajectories."""
    returns = []
    next_return = value[-1]
    for index in reversed(range(reward.shape[0])):
        next_return = reward[index] + discount[index] * (
            (1.0 - lambda_) * value[index] + lambda_ * next_return
        )
        returns.append(next_return)
    return torch.stack(list(reversed(returns)))
