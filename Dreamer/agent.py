from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from Dreamer.behavior import Actor, Critic, lambda_return
from Dreamer.config import Config
from Dreamer.replaybuffer import SequenceReplayBuffer, as_chw_uint8
from Dreamer.world_model import WorldModel


def detach_state(state):
    return {key: value.detach() for key, value in state.items()}


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

        self.model_opt = Adam(
            self.world_model.parameters(),
            lr=self.config.model_lr,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        self.actor_opt = Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
        )
        self.critic_opt = Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            eps=self.config.adam_eps,
            weight_decay=self.config.weight_decay,
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
        if self.updates % self.config.slow_target_update == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())
        self.updates += 1
        return {key: float(value.item()) for key, value in metrics.items()}

    def _train_behavior(self, start):
        if not len(start["hidden_state"]):
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
        critic_error = F.mse_loss(
            critic_value,
            imagination["return"].detach(),
            reduction="none",
        )
        critic_loss = (
            imagination["weight"].detach()
            * critic_error
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
