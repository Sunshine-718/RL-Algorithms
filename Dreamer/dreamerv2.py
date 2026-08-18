import math
import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.optim import NAdam
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from carracing_env import (
    OBSERVATION_SHAPE,
    wrap_carracing_observation,
    wrap_continuous_carracing_observation,
)
from replaybuffer import SequenceReplayBuffer, as_chw_uint8


@dataclass
class Config:
    discount: float = 0.99
    discount_lambda: float = 0.95
    capacity: int = 100_000
    batch_size: int = 16
    sequence_length: int = 32
    imagination_horizon: int = 15
    prefill: int = 1_000
    train_every: int = 5
    train_steps: int = 1
    total_steps: int = 1_000_000
    model_lr: float = 3e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    grad_clip: float = 100.0
    kl_scale: float = 1.0
    kl_balance: float = 0.8
    free_nats: float = 1.0
    reconstruction_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    discrete_entropy: float = 1e-3
    continuous_entropy: float = 1e-4
    slow_target_update: int = 100
    deter_dim: int = 200
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 400
    cnn_depth: int = 32
    params: str = "./params"
    eval_episodes: int = 3
    seed: int = 0


def mlp(in_dim, hidden_dim, out_dim, layers=2):
    modules = []
    for _ in range(layers):
        modules.extend(
            [nn.Linear(in_dim, hidden_dim), nn.ELU(inplace=True)]
        )
        in_dim = hidden_dim
    modules.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*modules)


def detach_state(state):
    return {key: value.detach() for key, value in state.items()}


def stack_states(states):
    return {
        key: torch.stack([state[key] for state in states], dim=1)
        for key in states[0]
    }


@contextmanager
def freeze_modules(*modules):
    parameters = [
        parameter for module in modules for parameter in module.parameters()
    ]
    requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


class ImageEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        channels = OBSERVATION_SHAPE[0]
        self.network = nn.Sequential(
            nn.Conv2d(channels, depth, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(depth, depth * 2, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(depth * 4, depth * 8, 4, 2, 1),
            nn.ELU(inplace=True),
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
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1),
            nn.ELU(inplace=True),
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
        deter_dim=200,
        stoch_dim=32,
        stoch_classes=32,
        hidden_dim=400,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_size = stoch_dim * stoch_classes
        self.img_in = nn.Sequential(
            nn.Linear(self.stoch_size + action_dim, hidden_dim),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)
        self.prior = mlp(deter_dim, hidden_dim, self.stoch_size, layers=1)
        self.posterior = mlp(
            deter_dim + embed_dim,
            hidden_dim,
            self.stoch_size,
            layers=1,
        )

    @property
    def feature_dim(self):
        return self.deter_dim + self.stoch_size

    def initial(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(
                batch_size,
                self.stoch_dim,
                self.stoch_classes,
                device=device,
            ),
            "logits": torch.zeros(
                batch_size,
                self.stoch_dim,
                self.stoch_classes,
                device=device,
            ),
        }

    def get_feature(self, state):
        stochastic = state["stoch"].flatten(start_dim=-2)
        return torch.cat([state["deter"], stochastic], dim=-1)

    def img_step(self, previous, previous_action, sample=True):
        stochastic = previous["stoch"].flatten(start_dim=-2)
        hidden = self.img_in(torch.cat([stochastic, previous_action], -1))
        deter = self.gru(hidden, previous["deter"])
        logits = self.prior(deter).reshape(
            -1, self.stoch_dim, self.stoch_classes
        )
        return {
            "deter": deter,
            "stoch": self._sample(logits, sample),
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
            torch.cat([prior["deter"], embed], dim=-1)
        ).reshape(-1, self.stoch_dim, self.stoch_classes)
        posterior = {
            "deter": prior["deter"],
            "stoch": self._sample(logits, sample),
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
        hard = F.one_hot(index, self.stoch_classes).to(probabilities.dtype)
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
            config.deter_dim,
            config.stoch_dim,
            config.stoch_classes,
            config.hidden_dim,
        )
        feature_dim = self.rssm.feature_dim
        self.decoder = ImageDecoder(feature_dim, config.cnn_depth)
        self.reward = mlp(feature_dim, config.hidden_dim, 1)
        self.continue_head = mlp(feature_dim, config.hidden_dim, 1)

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

        reconstruction_loss = F.mse_loss(reconstruction, observation)
        reward_loss = (
            (reward - batch["reward"]).pow(2) * transition_mask
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


class Actor(nn.Module):
    def __init__(self, feature_dim, action_dim, discrete, hidden_dim=400):
        super().__init__()
        self.discrete = bool(discrete)
        self.action_dim = action_dim
        output_dim = action_dim if discrete else action_dim * 2
        self.network = mlp(feature_dim, hidden_dim, output_dim, layers=3)

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
        self.network = mlp(feature_dim, hidden_dim, 1, layers=3)

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


class DreamerV2Agent:
    def __init__(
        self,
        action_dim,
        discrete,
        config=None,
        device="cpu",
        name="dreamerv2",
    ):
        self.config = config or Config()
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.discrete = bool(discrete)
        self.name = name
        self.params = self.config.params

        self.world_model = WorldModel(self.action_dim, self.config).to(
            self.device
        )
        feature_dim = self.world_model.rssm.feature_dim
        self.actor = Actor(
            feature_dim,
            self.action_dim,
            self.discrete,
            self.config.hidden_dim,
        ).to(self.device)
        self.critic = Critic(feature_dim, self.config.hidden_dim).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

        self.model_opt = NAdam(
            self.world_model.parameters(), lr=self.config.model_lr
        )
        self.actor_opt = NAdam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = NAdam(
            self.critic.parameters(), lr=self.config.critic_lr
        )
        self.buffer = SequenceReplayBuffer(
            self.config.capacity,
            self.action_dim,
            self.discrete,
            self.device,
        )
        self.state = None
        self.previous_action = None
        self.is_first = True
        self.updates = 0

    @torch.no_grad()
    def action(self, observation, deterministic=False):
        image = as_chw_uint8(observation)
        image = (
            torch.from_numpy(image)
            .to(self.device, dtype=torch.float32)
            .div_(255.0)
            .sub_(0.5)
            .unsqueeze(0)
        )
        if self.state is None:
            self.state = self.world_model.rssm.initial(1, self.device)
            self.previous_action = torch.zeros(
                1, self.action_dim, device=self.device
            )
        embed = self.world_model.encoder(image)
        self.state, _ = self.world_model.rssm.obs_step(
            self.state,
            self.previous_action,
            embed,
            torch.tensor([[float(self.is_first)]], device=self.device),
            sample=not deterministic,
        )
        actor_output = self.actor.sample(
            self.world_model.rssm.get_feature(self.state), deterministic
        )
        self.previous_action = actor_output["action"].detach()
        self.state = detach_state(self.state)
        self.is_first = False
        if self.discrete:
            return int(actor_output["index"].item())
        return actor_output["action"].squeeze(0).cpu().numpy().clip(-1.0, 1.0)

    def reset(self):
        self.state = None
        self.previous_action = None
        self.is_first = True

    def cache(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
    ):
        self.buffer.cache_transition(
            state, action, reward, next_state, terminated, truncated
        )

    def process(self):
        return self.buffer.process()

    def step(self, batch_size=None, sequence_length=None):
        batch_size = batch_size or self.config.batch_size
        sequence_length = sequence_length or self.config.sequence_length
        if not self.buffer.can_sample(batch_size, sequence_length):
            return None
        batch = self.buffer.sample(batch_size, sequence_length)

        self.model_opt.zero_grad()
        model_loss, outputs, metrics = self.world_model.loss(batch)
        model_loss.backward()
        model_grad = nn.utils.clip_grad_norm_(
            self.world_model.parameters(), self.config.grad_clip
        )
        self.model_opt.step()

        posterior = outputs["posterior"]
        flat_state = {
            key: value.detach().reshape(-1, *value.shape[2:])
            for key, value in posterior.items()
        }
        valid = batch["terminated"].reshape(-1) < 0.5
        flat_state = {key: value[valid] for key, value in flat_state.items()}
        behavior_metrics = self._train_behavior(flat_state)
        metrics.update(behavior_metrics)
        metrics["model_grad_norm"] = torch.as_tensor(model_grad).detach()
        self.updates += 1
        if self.updates % self.config.slow_target_update == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        return {key: float(value.item()) for key, value in metrics.items()}

    def _train_behavior(self, start):
        if not len(start["deter"]):
            zero = torch.tensor(0.0, device=self.device)
            return {
                "actor_loss": zero,
                "critic_loss": zero,
                "actor_entropy": zero,
                "actor_grad_norm": zero,
                "critic_grad_norm": zero,
            }

        self.actor_opt.zero_grad()
        with freeze_modules(
            self.world_model, self.critic, self.target_critic
        ):
            imagination = self._imagine(start)
            if self.discrete:
                baseline = self.target_critic(
                    imagination["feature"][:-1]
                )
                advantage = (
                    imagination["return"] - baseline
                ).detach()
                objective = imagination["log_prob"] * advantage
                entropy_scale = self.config.discrete_entropy
            else:
                objective = imagination["return"]
                entropy_scale = self.config.continuous_entropy
            actor_loss = -(
                imagination["weight"]
                * (objective + entropy_scale * imagination["entropy"])
            ).mean()
            actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.grad_clip
        )
        self.actor_opt.step()

        self.critic_opt.zero_grad()
        critic_value = self.critic(imagination["feature"][:-1].detach())
        critic_loss = (
            imagination["weight"].detach()
            * (critic_value - imagination["return"].detach()).pow(2)
        ).mean()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip
        )
        self.critic_opt.step()
        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "actor_entropy": imagination["entropy"].mean().detach(),
            "actor_grad_norm": torch.as_tensor(actor_grad).detach(),
            "critic_grad_norm": torch.as_tensor(critic_grad).detach(),
        }

    def _imagine(self, start):
        horizon = self.config.imagination_horizon
        states = [start]
        log_prob, entropy = [], []
        state = start
        for _ in range(horizon):
            feature = self.world_model.rssm.get_feature(state)
            actor_output = self.actor.sample(feature)
            state = self.world_model.rssm.img_step(
                state, actor_output["action"]
            )
            states.append(state)
            log_prob.append(actor_output["log_prob"])
            entropy.append(actor_output["entropy"])

        feature = torch.stack(
            [self.world_model.rssm.get_feature(state) for state in states]
        )
        next_feature = feature[1:]
        reward = self.world_model.reward(next_feature)
        continued = torch.sigmoid(
            self.world_model.continue_head(next_feature)
        )
        discount = self.config.discount * continued
        value = self.target_critic(next_feature)
        # λ-return 同时保留短期模型奖励和末端价值 bootstrap。
        returns = lambda_return(
            reward, value, discount, self.config.discount_lambda
        )
        weight = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0),
            dim=0,
        ).detach()
        return {
            "feature": feature,
            "reward": reward,
            "discount": discount,
            "return": returns,
            "weight": weight,
            "log_prob": torch.stack(log_prob),
            "entropy": torch.stack(entropy),
        }

    def save(self, model="last"):
        path = self._checkpoint_path(model)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "model_opt": self.model_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "updates": self.updates,
            },
            path,
        )
        return path

    def load(self, model="last"):
        path = self._checkpoint_path(model)
        if not path.exists():
            return False
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        for name, optimizer in (
            ("model_opt", self.model_opt),
            ("actor_opt", self.actor_opt),
            ("critic_opt", self.critic_opt),
        ):
            if name in checkpoint:
                optimizer.load_state_dict(checkpoint[name])
        self.updates = int(checkpoint.get("updates", 0))
        return True

    def _checkpoint_path(self, model):
        suffix = "discrete" if self.discrete else "continuous"
        return Path(self.params) / f"{self.name}_{suffix}_{model}.pt"


def make_carracing_env(update, continuous):
    env = gym.make(
        "CarRacing-v3",
        continuous=continuous,
        render_mode=None if update else "human",
    )
    wrapper = (
        wrap_continuous_carracing_observation
        if continuous
        else wrap_carracing_observation
    )
    return wrapper(env)


def build_agent(env, config, device):
    observation, _ = env.reset(seed=config.seed)
    as_chw_uint8(observation)
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete = True
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete = False
        action_dim = int(np.prod(env.action_space.shape))
        if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
            env.action_space.high, 1.0
        ):
            raise ValueError("continuous actions must be rescaled to [-1, 1]")
    else:
        raise TypeError("Dreamer supports only Discrete and Box actions")
    return DreamerV2Agent(
        action_dim,
        discrete,
        config,
        device,
    )


def prefill_buffer(env, agent, config):
    iterator = tqdm(total=config.prefill, desc="random prefill")
    while len(agent.buffer) < config.prefill:
        observation, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(
                action
            )
            agent.cache(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                truncated,
            )
            observation = next_observation
            done = terminated or truncated
        before = len(agent.buffer)
        agent.process()
        iterator.update(min(len(agent.buffer) - before, iterator.total - iterator.n))
    iterator.close()


def train(env, agent, config):
    if len(agent.buffer) < config.prefill:
        prefill_buffer(env, agent, config)
    observation, _ = env.reset(seed=config.seed)
    agent.reset()
    episode_reward = 0.0
    episode_length = 0
    best_reward = -math.inf
    metrics = None
    iterator = tqdm(range(config.total_steps))
    for step in iterator:
        action = agent.action(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        agent.cache(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )
        episode_reward += float(reward)
        episode_length += 1
        observation = next_observation

        if step % config.train_every == 0:
            for _ in range(config.train_steps):
                metrics = agent.step()

        if terminated or truncated:
            agent.process()
            agent.save("last")
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save("best")
            description = (
                f"reward: {episode_reward:.1f}, best: {best_reward:.1f}, "
                f"length: {episode_length}"
            )
            if metrics is not None:
                description += f", model: {metrics['model_loss']:.3f}"
            iterator.set_description(description)
            observation, _ = env.reset()
            agent.reset()
            episode_reward = 0.0
            episode_length = 0


def evaluate(env, agent, episodes):
    rewards = []
    for _ in range(episodes):
        observation, _ = env.reset()
        agent.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.action(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
        print(f"episode reward: {total:.1f}")
    return rewards


if __name__ == "__main__":
    update = 1
    continuous = True
    config = Config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_carracing_env(update, continuous)
    agent = build_agent(env, config, device)
    if bool(update):
        agent.load("last")
        train(env, agent, config)
    else:
        if not agent.load("best") and not agent.load("last"):
            raise FileNotFoundError("no Dreamer V2 checkpoint found")
        evaluate(env, agent, config.eval_episodes)
    env.close()
