import torch
import torch.nn as nn

from breakout_env import OBSERVATION_SHAPE
from common import NNBase


class BreakoutQRDuelingNetwork(NNBase):
    def __init__(self, lr, num_actions, num_quantiles=51,
                 computes_grad=True, device="cpu"):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(OBSERVATION_SHAPE[0], 32, kernel_size=8, stride=4),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(inplace=True),
            nn.Flatten(),
        )
        self.feature_dim = self._feature_dim()
        self.hidden = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.SiLU(inplace=True),
        )
        self.value = nn.Linear(512, num_quantiles)
        self.advantage = nn.Linear(512, num_actions * num_quantiles)
        self.action_dim = num_actions
        self.num_quantiles = num_quantiles
        self.obs_shape = OBSERVATION_SHAPE
        self.device = torch.device(device)

        self.apply(self.init_weights)
        self.opt = self.configure_optimizer(0.01, lr)
        self.computes_grad(computes_grad)
        self.to(self.device)

    def _feature_dim(self):
        training = self.features.training
        self.features.eval()
        with torch.no_grad():
            features = self.features(torch.zeros(1, *OBSERVATION_SHAPE))
        self.features.train(training)
        return features.shape[1]

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        hidden = self.hidden(self.features(state))
        value = self.value(hidden).view(
            batch_size, 1, self.num_quantiles
        )
        advantage = self.advantage(hidden).view(
            batch_size, self.action_dim, self.num_quantiles
        )
        return value + advantage - advantage.mean(dim=1, keepdim=True)
