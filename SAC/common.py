import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from pathlib import Path


SAC_TEMPERATURE_VERSION = 2


def _unwrap_env(env):
    return env.unwrapped


def make_train_test_env(env_id, update, num_envs=16, unwrap=False,
                        rescale_action=False, **kwargs):
    if bool(update):
        env = gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            },
            wrappers=[_unwrap_env] if unwrap else [],
            **kwargs,
        )
        if rescale_action:
            env = gym.wrappers.vector.RescaleAction(env, -1, 1)
        return env

    env = gym.make(env_id, render_mode="human", **kwargs)
    if unwrap:
        env = env.unwrapped
    if rescale_action:
        env = gym.wrappers.RescaleAction(env, -1, 1)
    return env


def single_spaces(env, update):
    if bool(update):
        return env.single_observation_space, env.single_action_space
    return env.observation_space, env.action_space


def reset_env(env, update):
    state, _ = env.reset()
    if bool(update):
        return state
    return np.expand_dims(state, axis=0)


def step_env(env, actions, update):
    if bool(update):
        return env.step(actions)
    next_state, reward, terminated, truncated, info = env.step(actions[0])
    return (
        np.expand_dims(next_state, axis=0),
        np.asarray([reward], dtype=np.float32),
        np.asarray([terminated], dtype=bool),
        np.asarray([truncated], dtype=bool),
        info,
    )


def reset_done_envs(env, next_states, done, update):
    if not np.any(done):
        return next_states
    if bool(update):
        next_states, _ = env.reset(
            options={"reset_mask": done.astype(np.bool_)},
        )
        return next_states
    next_state, _ = env.reset()
    return np.expand_dims(next_state, axis=0)


def flush_episode(agent, transitions):
    for transition in transitions:
        agent.cache(*transition)
    agent.process()
    transitions.clear()


def continuous_temperature_loss(log_alpha, log_prob, target_entropy):
    entropy_error = (log_prob + target_entropy).detach()
    return -(log_alpha * entropy_error).mean()


def discrete_temperature_loss(
    log_alpha, log_probs, probabilities, target_entropy
):
    entropy_error = (log_probs + target_entropy).detach()
    return -(
        probabilities.detach() * log_alpha * entropy_error
    ).sum(dim=-1).mean()


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    """Pairwise quantile Huber loss for prediction and target distributions."""
    td_error = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_error = td_error.abs()
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    weight = torch.abs(tau.unsqueeze(-1) - (td_error.detach() < 0).float())
    return (weight * huber).mean()


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.proj(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        self.glu = GLU(in_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim)
        self.norm = nn.RMSNorm(in_dim, 1e-5)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.residual = in_dim == out_dim

    def forward(self, x):
        residual = 0
        if self.residual:
            residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.linear(x)
        return self.dropout(x) + residual


class NetworkBase(nn.Module):
    def computes_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def save(self, path=None):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            state_dict = {"model": self.state_dict(),
                          "actor_opt": self.actor_opt.state_dict(),
                          "critic_opt": self.critic_opt.state_dict(),
                          "alpha_opt": self.alpha_opt.state_dict(),
                          "temperature_version": SAC_TEMPERATURE_VERSION}
            torch.save(state_dict, path)

    def load(self, path=None):
        try:
            if path is None:
                return False
            state_dict = torch.load(
                path, map_location=self.device, weights_only=True
            )
            model_state = dict(state_dict["model"])
            temperature_version = int(
                state_dict.get("temperature_version", 1)
            )
            reset_temperature = (
                temperature_version < SAC_TEMPERATURE_VERSION
            )
            if reset_temperature:
                model_state["alpha"] = self.alpha.detach().clone()
            elif "alpha" not in model_state and "alpha" in state_dict:
                model_state["alpha"] = state_dict["alpha"].detach()
            self.load_state_dict(model_state)
            self.actor_opt.load_state_dict(state_dict["actor_opt"])
            self.critic_opt.load_state_dict(state_dict["critic_opt"])
            if reset_temperature:
                print("Reset legacy SAC temperature state.")
            else:
                self.alpha_opt.load_state_dict(state_dict["alpha_opt"])
        except FileNotFoundError:
            print(f'Checkpoint not found: {path}')
            return False
        finally:
            self.to(self.device)
        return True


class AgentBase:
    @property
    def n_step(self):
        return self._n_step

    @n_step.setter
    def n_step(self, val):
        assert val >= 1 and isinstance(val, int)
        self._n_step = val
        self.buffer.n_step = val

    def cache(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.cache_transition(state, action, reward, next_state, terminated, truncated)

    def process(self):
        self.buffer.process()

    def save(self, model='last'):
        if self.params is not None:
            self.net.save(f'{self.params}/{self.name}_{model}.pt')
        else:
            self.net.save(f'{self.name}_{model}.pt')

    def load(self, model='last', required=False):
        path = f'{self.name}_{model}.pt'
        if self.params is not None:
            path = f'{self.params}/{path}'
        loaded = self.net.load(path)
        if not loaded:
            if required:
                raise FileNotFoundError(path)
            return False
        self.soft_update(tau=1)
        return True

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
