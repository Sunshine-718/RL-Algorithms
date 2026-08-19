from copy import deepcopy
from dataclasses import dataclass
from itertools import count

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from common import DQNAgentBase, NNBase, flush_episode
from trajectorybuffer import TrajectoryBuffer


CARTPOLE_OBSERVATION_INDICES = np.asarray([0, 2])


def partial_cartpole_observation(observation):
    observation = np.asarray(observation, dtype=np.float32)
    return np.take(
        observation, CARTPOLE_OBSERVATION_INDICES, axis=-1
    )


def make_partial_cartpole_env(render_mode=None):
    env = gym.make('CartPole-v1', render_mode=render_mode)
    source_space = env.observation_space
    observation_space = gym.spaces.Box(
        low=source_space.low[CARTPOLE_OBSERVATION_INDICES],
        high=source_space.high[CARTPOLE_OBSERVATION_INDICES],
        dtype=source_space.dtype,
    )
    return gym.wrappers.TransformObservation(
        env, partial_cartpole_observation, observation_space
    )


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    tau: float = 3e-2
    capacity: int = 1_000_000
    epoch: int = 10
    noise: float = 0.2
    min_noise: float = 0.02
    decay: float = 0.993
    burn_in: int = 10
    sequence_length: int = 20
    batch_size: int = 128
    max_grad_norm: float = 5
    auxiliary_loss_weight: float = 1
    observation_delta_scale: float = 50.
    learning_starts: int = 500
    evaluation_start: int = 100
    evaluation_interval: int = 25
    evaluation_episodes: int = 20
    solved_score: float = 475.
    seed: int = 43
    evaluation_seed: int = 9_900_000


class RecurrentDuelingDQN(NNBase):
    def __init__(self, lr, obs_dim, h_dim, recurrent_dim, num_actions,
                 recurrent_layers=1, dropout=0., computes_grad=True,
                 device='cpu'):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            h_dim,
            recurrent_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.,
        )
        self.v = nn.Sequential(
            nn.Linear(recurrent_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1),
        )
        self.a = nn.Sequential(
            nn.Linear(recurrent_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, num_actions),
        )
        self.action_dim = num_actions
        self.obs_dim = obs_dim
        self.recurrent_dim = recurrent_dim
        self.recurrent_layers = recurrent_layers
        self.device = torch.device(device)

        self.apply(self.init_weights)
        nn.init.constant_(self.a[-1].weight, 0)
        self.action_embed = nn.Embedding(num_actions, recurrent_dim)
        self.dynamics_head = nn.Sequential(
            nn.Linear(recurrent_dim * 2, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, obs_dim),
        )
        self.action_embed.apply(self.init_weights)
        self.dynamics_head.apply(self.init_weights)
        self.computes_grad(computes_grad)
        self.to(self.device)
        self.opt = self.configure_optimizer(0.01, lr)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.orthogonal_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, parameter in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(parameter, 0)
                elif 'weight_ih' in name:
                    for gate in parameter.chunk(3, dim=0):
                        nn.init.xavier_uniform_(gate)
                elif 'weight_hh' in name:
                    for gate in parameter.chunk(3, dim=0):
                        nn.init.orthogonal_(gate)

    def initial_hidden(self, batch_size):
        return torch.zeros(
            self.recurrent_layers,
            batch_size,
            self.recurrent_dim,
            dtype=next(self.parameters()).dtype,
            device=self.device,
        )

    def _as_sequence(self, state):
        if state.dim() == 1:
            state = state.reshape(1, 1, -1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
        elif state.dim() != 3:
            raise ValueError('state must have shape [O], [B, O] or [B, T, O]')
        if state.shape[-1] != self.obs_dim:
            raise ValueError(
                f'expected observation size {self.obs_dim}, '
                f'got {state.shape[-1]}'
            )
        return state

    def recurrent_features(self, state, hidden=None):
        state = self._as_sequence(state)
        if hidden is None:
            hidden = self.initial_hidden(state.shape[0])
        feature = self.encoder(state)
        self.gru.flatten_parameters()
        return self.gru(feature, hidden)

    def q_values(self, recurrent_feature):
        value = self.v(recurrent_feature)
        advantage = self.a(recurrent_feature)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def predict_observation_delta(self, recurrent_feature, action):
        if action.dim() == recurrent_feature.dim() and action.shape[-1] == 1:
            action = action.squeeze(-1)
        if action.shape != recurrent_feature.shape[:-1]:
            raise ValueError('action shape does not match recurrent features')
        action = self.action_embed(action.long())
        return self.dynamics_head(torch.cat((recurrent_feature, action), -1))

    def forward(self, state, hidden=None):
        feature, next_hidden = self.recurrent_features(state, hidden)
        return self.q_values(feature), next_hidden

    def burn_in(self, observation, valid, hidden=None):
        observation = self._as_sequence(observation)
        if valid.dim() == 3 and valid.shape[-1] == 1:
            valid = valid.squeeze(-1)
        if valid.shape != observation.shape[:2]:
            raise ValueError('burn-in mask does not match observations')
        if hidden is None:
            hidden = self.initial_hidden(observation.shape[0])

        feature = self.encoder(observation)
        for time_idx in range(observation.shape[1]):
            _, candidate_hidden = self.gru(
                feature[:, time_idx:time_idx + 1], hidden
            )
            mask = valid[:, time_idx].reshape(1, -1, 1)
            hidden = torch.where(mask, candidate_hidden, hidden)
        return hidden


class DRQNAgent(DQNAgentBase):
    def __init__(self, name, Q, config):
        self.net = Q
        self.target_net = deepcopy(Q)
        self.target_net.computes_grad(False)
        self.buffer = TrajectoryBuffer(
            Q.obs_dim, config.capacity, 1, Q.device
        )

        self.name = name
        self.n_actions = Q.action_dim
        self.params = config.params
        self.discount = config.discount
        self.epoch = config.epoch
        self.tau = config.tau
        self.noise = config.noise
        self.min_noise = config.min_noise
        self.decay = config.decay
        self.burn_in = config.burn_in
        self.sequence_length = config.sequence_length
        self.batch_size = config.batch_size
        self.max_grad_norm = config.max_grad_norm
        self.auxiliary_loss_weight = config.auxiliary_loss_weight
        self.observation_delta_scale = config.observation_delta_scale
        self.learning_starts = config.learning_starts
        self._n_step = 1
        self._action_hidden = None
        self.soft_update(tau=1)
        self.reset_hidden()

    @property
    def n_step(self):
        return 1

    @n_step.setter
    def n_step(self, value):
        if value != 1:
            raise ValueError('DRQN currently supports one-step TD targets only')

    def reset_hidden(self, done=None, batch_size=1):
        if done is None:
            self._action_hidden = self.net.initial_hidden(batch_size)
            return self._action_hidden

        done = torch.as_tensor(
            done, dtype=torch.bool, device=self.net.device
        ).reshape(-1)
        if (
            self._action_hidden is None
            or self._action_hidden.shape[1] != len(done)
        ):
            self._action_hidden = self.net.initial_hidden(len(done))
        else:
            self._action_hidden = self._action_hidden.clone()
            self._action_hidden[:, done] = 0
        return self._action_hidden

    @torch.no_grad()
    def action(self, state, deterministic=False):
        state = np.asarray(state, dtype=np.float32)
        single_state = state.ndim == 1
        if single_state:
            state = np.expand_dims(state, axis=0)
        if state.ndim != 2:
            raise ValueError('state must have shape [O] or [B, O]')

        state = torch.from_numpy(state).to(self.net.device)
        if (
            self._action_hidden is None
            or self._action_hidden.shape[1] != len(state)
        ):
            self.reset_hidden(batch_size=len(state))

        was_training = self.net.training
        self.net.eval()
        q, self._action_hidden = self.net(state, self._action_hidden)
        self._action_hidden = self._action_hidden.detach()
        if was_training:
            self.net.train()

        greedy_actions = q[:, -1].argmax(dim=-1).cpu().numpy()
        if deterministic:
            actions = greedy_actions
        else:
            explore = np.random.random(len(greedy_actions)) < self.noise
            random_actions = np.random.randint(
                0, self.n_actions, size=len(greedy_actions)
            )
            actions = np.where(explore, random_actions, greedy_actions)
        return int(actions[0]) if single_state else actions

    def td_target(self, reward, next_q, terminated):
        return reward + self.discount * next_q * (~terminated).float()

    def loss(self, burn_observation, burn_mask, observation, action,
             reward, terminated, truncated, loss_mask):
        del truncated
        with torch.no_grad():
            online_hidden = self.net.burn_in(
                burn_observation, burn_mask
            )
            target_hidden = self.target_net.burn_in(
                burn_observation, burn_mask
            )

        recurrent_feature, _ = self.net.recurrent_features(
            observation, online_hidden.detach()
        )
        q = self.net.q_values(recurrent_feature)
        current_q = q[:, :-1].gather(-1, action.long())

        with torch.no_grad():
            target_q, _ = self.target_net(
                observation, target_hidden.detach()
            )
            next_action = q[:, 1:].detach().argmax(
                dim=-1, keepdim=True
            )
            next_q = target_q[:, 1:].gather(-1, next_action)
            target = self.td_target(reward, next_q, terminated)

        elementwise_td_loss = F.smooth_l1_loss(
            current_q, target, reduction='none'
        )
        predicted_delta = self.net.predict_observation_delta(
            recurrent_feature[:, :-1], action
        )
        observation_delta = (
            observation[:, 1:] - observation[:, :-1]
        ) * self.observation_delta_scale
        elementwise_auxiliary_loss = F.smooth_l1_loss(
            predicted_delta, observation_delta, reduction='none'
        ).mean(dim=-1, keepdim=True)
        loss_mask = loss_mask.float()
        valid_count = loss_mask.sum().clamp_min(1.)
        td_loss = (elementwise_td_loss * loss_mask).sum() / valid_count
        auxiliary_loss = (
            (elementwise_auxiliary_loss * loss_mask).sum() / valid_count
        )
        return td_loss + self.auxiliary_loss_weight * auxiliary_loss

    def step(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        loss = None
        ready = max(batch_size, self.learning_starts)
        if len(self.buffer) >= ready:
            for _ in range(self.epoch):
                self.net.opt.zero_grad()
                self.target_net.eval()
                self.net.train()
                loss = self.loss(*self.buffer.sample(
                    batch_size, self.burn_in, self.sequence_length
                ))
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm
                )
                self.net.opt.step()
                self.soft_update()
            self.decay_noise()
        return loss.item() if loss is not None else None


def evaluate_agent(agent, env, episodes, seed):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        agent.reset_hidden()
        episode_reward = 0.
        done = False
        while not done:
            action = agent.action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    return np.asarray(rewards, dtype=np.float32)


if __name__ == '__main__':
    update = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    env = make_partial_cartpole_env(
        render_mode=None if bool(update) else 'human'
    )
    Q = RecurrentDuelingDQN(
        lr=1e-3,
        obs_dim=env.observation_space.shape[0],
        h_dim=128,
        recurrent_dim=128,
        num_actions=env.action_space.n,
        recurrent_layers=1,
        device=device,
    )
    agent = DRQNAgent('cartpole_drqn', Q, config)
    agent.load(required=not bool(update))

    reward_container = []
    loss_container = []
    best_eval_reward = -float('inf')
    last_eval_reward = float('nan')
    episodes = count() if bool(update) else range(10)
    iterator = tqdm(episodes)

    try:
        for episode in iterator:
            episode_seed = (
                config.seed * 10_000 + episode
                if bool(update)
                else config.evaluation_seed + episode
            )
            state, _ = env.reset(seed=episode_seed)
            agent.reset_hidden()
            episode_cache = []
            episode_reward = 0.
            done = False

            while not done:
                action = agent.action(
                    state, deterministic=not bool(update)
                )
                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )
                done = terminated or truncated
                if bool(update):
                    episode_cache.append((
                        state.copy(), action, reward, next_state.copy(),
                        terminated, truncated,
                    ))
                state = next_state
                episode_reward += reward

            if bool(update):
                flush_episode(agent, episode_cache)
                loss = agent.step()
                if loss is not None:
                    loss_container.append(loss)

            reward_container.append(episode_reward)
            average_reward = float(np.mean(reward_container[-10:]))
            if bool(update):
                agent.save()

            solved = False
            evaluation_due = (
                bool(update)
                and episode + 1 >= config.evaluation_start
                and (episode + 1) % config.evaluation_interval == 0
            )
            if evaluation_due:
                evaluation_rewards = evaluate_agent(
                    agent,
                    env,
                    config.evaluation_episodes,
                    config.evaluation_seed,
                )
                last_eval_reward = float(evaluation_rewards.mean())
                if last_eval_reward > best_eval_reward:
                    best_eval_reward = last_eval_reward
                    agent.save('best')
                solved = last_eval_reward >= config.solved_score

            iterator.set_description(
                f'episode reward: {episode_reward: .0f}, '
                f'avg: {average_reward: .1f}, '
                f'eval: {last_eval_reward: .1f}, '
                f'noise: {agent.noise: .3f}'
            )
            if solved:
                iterator.write(
                    f'Solved at episode {episode + 1}: '
                    f'evaluation reward {last_eval_reward:.2f}'
                )
    finally:
        if bool(update):
            agent.save()
        env.close()

    plt.plot(reward_container, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
