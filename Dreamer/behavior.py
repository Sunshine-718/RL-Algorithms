import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


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
        mean = 5.0 * torch.tanh(mean / 5.0)
        std = 2.0 * torch.sigmoid(raw_std / 2.0) + 0.1
        distribution = Normal(mean, std)
        raw_action = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
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
