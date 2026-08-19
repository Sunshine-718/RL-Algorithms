import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from carracing_env import OBSERVATION_SHAPE


def stack_states(states):
    return {
        key: torch.stack([state[key] for state in states], dim=1) for key in states[0]
    }


class DreamerGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_state_dim):
        super().__init__()
        self.linear = nn.Linear(
            input_dim + hidden_state_dim,
            3 * hidden_state_dim,
        )
        self.norm = nn.LayerNorm(3 * hidden_state_dim, eps=1e-3)

    def forward(self, inputs, state):
        reset, candidate, update = self.norm(self.linear(torch.cat([inputs, state], dim=-1))).chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update = torch.sigmoid(update - 1.0)
        return update * candidate + (1.0 - update) * state


class ImageEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        channels = OBSERVATION_SHAPE[0]
        self.network = nn.Sequential(
            nn.Conv2d(channels, depth, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(depth, depth * 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(depth * 4, depth * 8, 4, 2, 1),
            nn.SiLU(inplace=True),
        )
        self.output_dim = depth * 8 * 6 * 6

    def forward(self, observation):
        leading = observation.shape[:-3]
        observation = observation.reshape(-1, *OBSERVATION_SHAPE)
        encoded = self.network(observation).flatten(1)
        return encoded.reshape(*leading, self.output_dim)


class ImageDecoder(nn.Module):
    def __init__(self, feature_dim, depth=32):
        super().__init__()
        channels = OBSERVATION_SHAPE[0]
        self.depth = depth
        self.input = nn.Linear(feature_dim, depth * 8 * 6 * 6)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(depth, channels, 4, 2, 1),
        )

    def forward(self, feature):
        leading = feature.shape[:-1]
        hidden = self.input(feature.reshape(-1, feature.shape[-1]))
        reconstruction = self.network(
            hidden.reshape(-1, self.depth * 8, 6, 6)
        )
        return reconstruction.reshape(*leading, *OBSERVATION_SHAPE)


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim,
        embed_dim,
        hidden_state_dim=200,
        stochastic_dim=32,
        stochastic_classes=32,
        mlp_dim=400,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim
        self.stochastic_dim = stochastic_dim
        self.stochastic_classes = stochastic_classes
        self.stochastic_size = stochastic_dim * stochastic_classes
        self.img_in = nn.Sequential(
            nn.Linear(self.stochastic_size + action_dim, mlp_dim),
            nn.SiLU(inplace=True),
        )
        self.gru = DreamerGRUCell(mlp_dim, hidden_state_dim)
        self.prior = nn.Sequential(
            nn.Linear(hidden_state_dim, mlp_dim),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_dim, self.stochastic_size),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_state_dim + embed_dim, mlp_dim),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_dim, self.stochastic_size),
        )

    @property
    def feature_dim(self):
        return self.hidden_state_dim + self.stochastic_size

    def initial(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "hidden_state": torch.zeros(
                batch_size,
                self.hidden_state_dim,
                device=device,
            ),
            "stochastic_state": torch.zeros(
                batch_size,
                self.stochastic_dim,
                self.stochastic_classes,
                device=device,
            ),
            "logits": torch.zeros(
                batch_size,
                self.stochastic_dim,
                self.stochastic_classes,
                device=device,
            ),
        }

    def get_feature(self, state):
        stochastic_state = state["stochastic_state"].flatten(start_dim=-2)
        return torch.cat(
            [state["hidden_state"], stochastic_state],
            dim=-1,
        )

    def img_step(self, previous, previous_action, sample=True):
        stochastic_state = previous["stochastic_state"].flatten(
            start_dim=-2
        )
        transition = self.img_in(
            torch.cat([stochastic_state, previous_action], dim=-1)
        )
        hidden_state = self.gru(transition, previous["hidden_state"])
