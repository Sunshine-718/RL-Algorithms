import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from carracing_env import OBSERVATION_SHAPE


def stack_states(states):
    return {
        key: torch.stack([state[key] for state in states], dim=1)
        for key in states[0]
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
        reset, candidate, update = self.norm(
            self.linear(torch.cat([inputs, state], dim=-1))
        ).chunk(3, dim=-1)
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
        logits = self.prior(hidden_state).reshape(
            -1,
            self.stochastic_dim,
            self.stochastic_classes,
        )
        return {
            "hidden_state": hidden_state,
            "stochastic_state": self._sample(logits, sample),
            "logits": logits,
        }

    def obs_step(
        self,
        previous,
        previous_action,
        embed,
        is_first=None,
        sample=True,
    ):
        # 第 t 个观测与 a_(t-1) 配对；回合首帧会清空递归状态和上一动作。
        if is_first is not None:
            keep = 1.0 - is_first.reshape(-1, 1)
            previous = {
                key: value * keep.reshape(
                    keep.shape[0], *([1] * (value.ndim - 1))
                )
                for key, value in previous.items()
            }
            previous_action = previous_action * keep

        prior = self.img_step(previous, previous_action, sample)
        logits = self.posterior(
            torch.cat([prior["hidden_state"], embed], dim=-1)
        ).reshape(
            -1,
            self.stochastic_dim,
            self.stochastic_classes,
        )
        posterior = {
            "hidden_state": prior["hidden_state"],
            "stochastic_state": self._sample(logits, sample),
            "logits": logits,
        }
        return posterior, prior

    def observe(self, embed, action, is_first, state=None, sample=True):
        batch_size, sequence_length = action.shape[:2]
        state = state or self.initial(batch_size, action.device)
        posts, priors = [], []
        for index in range(sequence_length):
            state, prior = self.obs_step(
                state,
                action[:, index],
                embed[:, index],
                is_first[:, index],
                sample,
            )
            posts.append(state)
            priors.append(prior)
        return stack_states(posts), stack_states(priors)

    def _sample(self, logits, sample):
        probabilities = torch.softmax(logits, dim=-1)
        if sample:
            index = Categorical(logits=logits).sample()
        else:
            index = logits.argmax(dim=-1)
        hard = F.one_hot(index, self.stochastic_classes).to(
            probabilities.dtype
        )
        # 前向使用 one-hot，反向使用类别概率的梯度。
        return hard + probabilities - probabilities.detach()


class WorldModel(nn.Module):
    def __init__(self, action_dim, config):
        super().__init__()
        self.config = config
        self.encoder = ImageEncoder(config.cnn_depth)
        self.rssm = RSSM(
            action_dim,
            self.encoder.output_dim,
            config.hidden_state_dim,
            config.stochastic_dim,
            config.stochastic_classes,
            config.rssm_mlp_dim,
        )
        feature_dim = self.rssm.feature_dim
        self.decoder = ImageDecoder(feature_dim, config.cnn_depth)
        self.reward = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, 1),
        )

    def loss(self, batch):
        observation = batch["observation"]
        embed = self.encoder(observation)
        posterior, prior = self.rssm.observe(
            embed, batch["action"], batch["is_first"]
        )
        feature = self.rssm.get_feature(posterior)
        reconstruction = self.decoder(feature)
        reward = self.reward(feature)
        continue_logits = self.continue_head(feature)
        transition_mask = 1.0 - batch["is_first"]
        denominator = transition_mask.sum().clamp_min(1.0)

        reconstruction_loss = (
            0.5
            * (reconstruction - observation)
            .square()
            .sum(dim=(-3, -2, -1))
        ).mean()
        reward_loss = (
            0.5
            * (reward - batch["reward"]).pow(2)
            * transition_mask
        ).sum() / denominator
        continue_loss = (
            F.binary_cross_entropy_with_logits(
                continue_logits, batch["continue"], reduction="none"
            )
            * transition_mask
        ).sum() / denominator
        kl_loss, kl_value = self._kl_loss(
            posterior["logits"], prior["logits"]
        )
        total = (
            self.config.reconstruction_scale * reconstruction_loss
            + self.config.reward_scale * reward_loss
            + self.config.continue_scale * continue_loss
            + self.config.kl_scale * kl_loss
        )
        outputs = {
            "posterior": posterior,
            "prior": prior,
            "feature": feature,
            "reconstruction": reconstruction,
        }
        metrics = {
            "model_loss": total.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "continue_loss": continue_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "kl_value": kl_value.detach(),
        }
        return total, outputs, metrics

    def _kl_loss(self, posterior_logits, prior_logits):
        posterior = torch.softmax(posterior_logits, dim=-1)
        prior = torch.softmax(prior_logits, dim=-1)

        def categorical_kl(left, right):
            return (
                left
                * (
                    torch.log(left.clamp_min(1e-8))
                    - torch.log(right.clamp_min(1e-8))
                )
            ).sum(dim=(-1, -2))

        # KL balance 让先验承担更多拟合责任，后验仍保留较弱的正则梯度。
        posterior_kl = categorical_kl(posterior, prior.detach()).mean()
        prior_kl = categorical_kl(posterior.detach(), prior).mean()
        free = posterior_kl.new_tensor(self.config.free_nats)
        loss = (
            (1.0 - self.config.kl_balance)
            * torch.maximum(posterior_kl, free)
            + self.config.kl_balance * torch.maximum(prior_kl, free)
        )
        value = categorical_kl(posterior.detach(), prior.detach()).mean()
        return loss, value
