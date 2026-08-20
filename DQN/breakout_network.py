import torch
import torch.nn as nn

from breakout_env import OBSERVATION_SHAPE
from common import NNBase


class BreakoutQRDuelingNetwork(NNBase):
    def __init__(self, lr, num_actions, num_quantiles=51,
                 computes_grad=True, device="cpu"):
        super().__init__()
        self.hidden = nn.Sequential(
            self._conv_block(
                OBSERVATION_SHAPE[0], 32, kernel_size=8, stride=4
            ),
            self._conv_block(32, 64, kernel_size=4, stride=2),
        )
        value_features = self._conv_block(
            64, 64, kernel_size=3, stride=1
        )
        advantage_features = self._conv_block(
            64, 64, kernel_size=3, stride=1
        )
        self.feature_dim = self._feature_dim(value_features)
        self.v = nn.Sequential(
            value_features,
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_quantiles),
        )
        self.a = nn.Sequential(
            advantage_features,
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_actions * num_quantiles),
        )
        self.action_dim = num_actions
        self.num_quantiles = num_quantiles
        self.obs_shape = OBSERVATION_SHAPE
        self.device = torch.device(device)

        self.apply(self.init_weights)
        self.opt = self.configure_optimizer(0.01, lr)
        self.computes_grad(computes_grad)
        self.to(self.device)

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def _feature_dim(self, branch):
        hidden_training = self.hidden.training
        branch_training = branch.training
        self.hidden.eval()
        branch.eval()
        with torch.no_grad():
            hidden = self.hidden(torch.zeros(1, *OBSERVATION_SHAPE))
            features = branch(hidden).flatten(start_dim=1)
        self.hidden.train(hidden_training)
        branch.train(branch_training)
        return features.shape[1]

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight)

    def forward(self, state):
        if state.ndim == len(self.obs_shape):
            state = state.unsqueeze(0)
        if tuple(state.shape[1:]) != self.obs_shape:
            raise ValueError(
                f"expected input shape [B, {self.obs_shape}], "
                f"got {tuple(state.shape)}"
            )

        if state.dtype == torch.uint8:
            state = state.to(dtype=torch.float32).div_(255.0)
        else:
            state = state.to(dtype=torch.float32)

        batch_size = state.shape[0]
        hidden = self.hidden(state)
        value = self.v(hidden).view(
            batch_size, 1, self.num_quantiles
        )
        advantage = self.a(hidden).view(
            batch_size, self.action_dim, self.num_quantiles
        )
        return value + advantage - advantage.mean(dim=1, keepdim=True)
