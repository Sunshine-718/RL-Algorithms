import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.auto import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from breakout_env import (
    FRAME_SKIP,
    OBSERVATION_SHAPE,
    REPEAT_ACTION_PROBABILITY,
    STACK_SIZE,
    make_breakout_env,
)
from common import (
    NNBase,
    SoftDQNAgentBase,
    flush_n_step_transitions,
    reset_done_envs,
    reset_env,
    single_spaces,
    step_env,
    store_n_step_transition,
)
from image_replaybuffer import ImageReplayBuffer


@dataclass
class Config:
    discount: float = 0.99
    params: str = "./params"
    tau: float = 3e-2
    # State and next-state frame stacks use about 26.29 GiB as uint8.
    capacity: int = 500_000
    epoch: int = 30
    learning_starts: int = 20_000
    reward_scale: float = 1.0
    n_step: int = 5
    alpha: float = 0.1
    alpha_lr: float = 1e-2


class BreakoutDuelingNetwork(NNBase):
    def __init__(self, lr, num_actions, computes_grad=True, device="cpu"):
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
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, num_actions)
        self.action_dim = num_actions
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

        hidden = self.hidden(self.features(state))
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class BreakoutSoftDQNAgent(SoftDQNAgentBase):
    def __init__(self, name, q_network, config):
        self.net = q_network
        self.target_net = deepcopy(q_network)
        self.target_net.computes_grad(False)
        self.target_net.eval()
        self.buffer = ImageReplayBuffer(
            q_network.obs_shape,
            config.capacity,
            1,
            config.discount,
            config.n_step,
            q_network.device,
        )

        self.name = name
        self.n_actions = q_network.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.learning_starts = config.learning_starts
        self.reward_scale = config.reward_scale
        self._n_step = config.n_step
        self.tau = config.tau
        self.configure_alpha(config.alpha, lr=config.alpha_lr)
        self.target_entropy = float(np.log(q_network.action_dim)) * 0.45
        self.soft_update(tau=1.0)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state)
        single_state = state.ndim == len(self.net.obs_shape)
        state_tensor = torch.as_tensor(state, device=self.net.device)
        if single_state:
            state_tensor = state_tensor.unsqueeze(0)

        training = self.net.training
        self.net.eval()
        q_value = self.net(state_tensor).cpu()
        if deterministic:
            actions = q_value.argmax(dim=-1).numpy()
        else:
            probabilities = torch.softmax(q_value / self.alpha, dim=-1)
            actions = Categorical(probabilities).sample().numpy()
        self.net.train(training)
        return int(actions[0]) if single_state else actions

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated, n):
        training = self.net.training
        self.net.eval()
        online_q = self.net(next_state)
        self.net.train(training)
        target_q = self.target_net(next_state)
        log_probabilities = torch.log_softmax(
            online_q / self.alpha,
            dim=1,
        )
        probabilities = log_probabilities.exp()
        soft_value = (
            probabilities
            * (target_q - self.alpha * log_probabilities)
        ).sum(dim=1, keepdim=True)
        return (
            reward
            + torch.pow(self.discount, n)
            * soft_value
            * (1.0 - terminated)
        )

    def loss(self, state, action, reward, next_state, terminated, truncated,
             n):
        q_values = self.net(state)
        chosen_q = q_values.gather(1, action.long())
        target = self.td_target(reward, next_state, terminated, n)
        return F.smooth_l1_loss(chosen_q, target), q_values.detach()

    def step(self, batch_size=128):
        if len(self.buffer) < max(batch_size, self.learning_starts):
            self.training_metrics()
            return {}

        for _ in range(self.epoch):
            self.net.opt.zero_grad()
            self.target_net.eval()
            self.net.train()
            loss, q_values = self.loss(*self.buffer.sample(batch_size))
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.net.opt.step()

            entropy = self._update_alpha(q_values)
            self.soft_update()

        metrics = self.training_metrics(loss, entropy)
        return {"loss": metrics["loss"], "alpha": metrics["alpha"]}

    def soft_update(self, tau=None):
        super().soft_update(tau)
        for target_buffer, buffer in zip(
            self.target_net.buffers(), self.net.buffers()
        ):
            target_buffer.copy_(buffer)


if __name__ == "__main__":
    update = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4 if bool(update) else 1
    env = make_breakout_env(update, num_envs)
    observation_space, action_space = single_spaces(env, update)
    if observation_space.shape != OBSERVATION_SHAPE:
        raise RuntimeError(
            f"unexpected observation shape: {observation_space.shape}"
        )

    learning_rate = 1e-4
    q_network = BreakoutDuelingNetwork(
        lr=learning_rate,
        num_actions=action_space.n,
        computes_grad=True,
        device=device,
    )
    config = Config()
    agent_name = "softdqn_breakout_v3"
    agent = BreakoutSoftDQNAgent(
        agent_name, q_network, config
    )
    checkpoint_loaded = agent.load(required=not bool(update))
    batch_size = 128
    training_metrics = agent.training_metrics()
    print(
        f"model={agent_name}, checkpoint_loaded={checkpoint_loaded}, "
        f"device={device}, envs={num_envs}, "
        f"observation={observation_space.shape}, actions={action_space.n}, "
        f"frame_skip={FRAME_SKIP}, stack_size={STACK_SIZE}, "
        f"repeat_action_probability={REPEAT_ACTION_PROBABILITY}, "
        f"terminal_on_life_loss={bool(update)}, "
        f"learning_rate={learning_rate}, "
        f"capacity={config.capacity:,}, batch_size={batch_size}, "
        f"learning_starts={config.learning_starts:,}, epoch={config.epoch}, "
        f"n_step={config.n_step}, discount={config.discount}, "
        f"reward_scale={config.reward_scale}, tau={config.tau}, "
        f"alpha={agent.alpha:.4f}, "
        f"alpha_lr={config.alpha_lr}, "
        f"target_entropy={agent.target_entropy:.3f}"
    )

    reward_container = []
    interval = 10
    recent_rewards = np.zeros(interval)
    best_average = -float("inf")
    average_reward = 0.0
    total_episodes = float("inf") if bool(update) else 10_000
    iterator = tqdm(total=total_episodes)
    plt.ion()

    states = reset_env(env, update)
    episode_caches = [[] for _ in range(num_envs)]
    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    completed_episodes = 0

    while completed_episodes < total_episodes:
        actions = agent.action(states, deterministic=not bool(update))
        next_states, rewards, terminated, truncated, _ = step_env(
            env, actions, update
        )
        episode_lengths += 1
        done = np.logical_or(terminated, truncated)

        for env_id in range(num_envs):
            if completed_episodes >= total_episodes:
                break
            if bool(update):
                episode_caches[env_id].append(
                    (
                        np.asarray(states[env_id]).copy(),
                        int(actions[env_id]),
                        float(rewards[env_id]) * agent.reward_scale,
                        np.asarray(next_states[env_id]).copy(),
                        bool(terminated[env_id]),
                        bool(truncated[env_id]),
                    )
                )
                store_n_step_transition(
                    agent, episode_caches[env_id]
                )
                if done[env_id]:
                    flush_n_step_transitions(
                        agent, episode_caches[env_id]
                    )
            episode_rewards[env_id] += float(rewards[env_id])
            if not done[env_id]:
                continue

            if bool(update):
                agent.step(batch_size)
                training_metrics = agent.last_training_metrics

            episode_index = completed_episodes
            episode_reward = float(episode_rewards[env_id])
            episode_length = int(episode_lengths[env_id])
            reward_container.append(episode_reward)
            recent_rewards[episode_index % interval] = episode_reward
            if bool(update):
                agent.save()

            if episode_index % interval == 0 and episode_index != 0:
                average_reward = float(np.mean(recent_rewards))
                if bool(update) and average_reward > best_average:
                    best_average = average_reward
                    agent.save("best")
                plt.clf()
                plt.plot(reward_container, label="Reward")
                plt.title(f"Reward: {episode_reward:.1f}")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.pause(0.1)

            training_status = (
                f", {agent.format_training_metrics(training_metrics)}"
                if bool(update) else ""
            )
            iterator.set_description(
                f"episode reward: {episode_reward: .0f}, "
                f"avg: {average_reward: .1f}, "
                f"best avg: {best_average: .1f}, "
                f"episode length: {episode_length}"
                f"{training_status}"
            )
            iterator.update(1)
            completed_episodes += 1
            episode_rewards[env_id] = 0.0
            episode_lengths[env_id] = 0

        if completed_episodes >= total_episodes:
            break
        states = reset_done_envs(env, next_states, done, update)

    iterator.close()
    env.close()
